// Copyright Axelera AI, 2024
// Optimized Facenet decoder

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaClassification.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iterator>
#include <limits>
#include <ranges>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace facenet_recog
{

struct properties {
  std::string embeddings_file{};
  float distance_threshold{ 0.5 };
  int metric_type{ ax_utils::EUCLIDEAN_DISTANCE };
  bool pair_validation{ true };
  int top_k{ 1 };
  bool update_embeddings{ false };

  std::string meta_name{};
  std::string master_meta{};
  std::string association_meta{};
  std::string decoder_name{};

  // For update mode: labels to be processed in sequence
  std::vector<std::string> labels_for_update{};
  mutable size_t current_update_index{ 0 };

  mutable Eigen::MatrixXf embeddings{};
  mutable std::vector<std::string> labels{};
};

} // namespace facenet_recog

extern "C" void
decode_to_meta(const AxTensorsInterface &in_tensors, const facenet_recog::properties *prop,
    unsigned int current_frame, unsigned int total_frames,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    const AxDataInterface &video_interface, Ax::Logger &logger)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  if (!prop) {
    logger(AX_ERROR) << "decode_to_meta : properties not set" << std::endl;
    throw std::runtime_error("decode_to_meta : properties not set");
  }
  auto tensors = in_tensors;
  if (tensors.size() != 1) {
    throw std::runtime_error("facenet_recog_to_meta : Number of input tensors must be 1");
  }
  auto *data = static_cast<float *>(tensors[0].data);
  const auto total_size = tensors[0].total();

  // Normalize the embedding
  std::vector<std::vector<float>> embeddings_vec;
  std::vector<float> normalized_data(data, data + total_size);
  float norm = std::sqrt(std::inner_product(normalized_data.begin(),
      normalized_data.end(), normalized_data.begin(), 0.0f));
  if (norm > 0) {
    std::transform(normalized_data.begin(), normalized_data.end(),
        normalized_data.begin(), [norm](float val) { return val / norm; });
  }
  embeddings_vec.emplace_back(std::move(normalized_data));

  if (prop->pair_validation) {
    ax_utils::insert_meta<AxMetaEmbeddings>(map, prop->meta_name, prop->master_meta,
        current_frame, total_frames, std::move(embeddings_vec), prop->decoder_name);
  } else if (prop->update_embeddings) {
    if (prop->labels_for_update.empty()) {
      logger(AX_ERROR) << "facenet_recog: labels_for_update must be provided when update_embeddings is true"
                       << std::endl;
      throw std::runtime_error(
          "facenet_recog: labels_for_update must be provided when update_embeddings is true");
    }

    if (prop->current_update_index >= prop->labels_for_update.size()) {
      logger(AX_ERROR) << "facenet_recog: Processed more frames than labels_for_update provided"
                       << std::endl;
      throw std::runtime_error(
          "facenet_recog: Processed more frames than labels_for_update provided");
    }

    auto &current_embedding = embeddings_vec[0];
    const std::string &person_name = prop->labels_for_update[prop->current_update_index];

    // Check if person already exists in embeddings
    auto label_it = std::find(prop->labels.begin(), prop->labels.end(), person_name);

    if (label_it != prop->labels.end()) {
      // Person exists - update their embedding
      size_t person_idx = std::distance(prop->labels.begin(), label_it);

      // Replace the existing embedding row
      for (size_t i = 0; i < current_embedding.size(); ++i) {
        prop->embeddings(person_idx, i) = current_embedding[i];
      }

      logger(AX_INFO) << "facenet_recog: Updated embedding for person: '"
                      << person_name << "'" << std::endl;
    } else {
      // Person doesn't exist - add new person
      ax_utils::add_vec_to_matrix(current_embedding, prop->embeddings);
      prop->labels.push_back(person_name);

      logger(AX_INFO) << "facenet_recog: Added new person: '" << person_name
                      << "'" << std::endl;
    }

    // Increment update index for next frame
    prop->current_update_index++;

    // Always write embeddings file after each update/addition
    try {
      ax_utils::write_embedding_json(
          prop->embeddings, prop->labels, prop->embeddings_file, logger);
      logger(AX_DEBUG) << "facenet_recog: Successfully saved embeddings to: "
                       << prop->embeddings_file << std::endl;
    } catch (const std::exception &e) {
      logger(AX_ERROR) << "facenet_recog: Failed to save embeddings: " << e.what()
                       << std::endl;
      throw;
    }

  } else {
    // Recognition mode - never modify embeddings or file
    const auto &current_embedding = embeddings_vec[0];
    const auto &const_embeddings = prop->embeddings;
    const auto &const_labels = prop->labels;

    if (const_labels.empty()) {
      logger(AX_ERROR)
          << "facenet_recog: No embeddings loaded for recognition" << std::endl;
      throw std::runtime_error("facenet_recog: No embeddings loaded for recognition");
    }

    std::vector<float> distance;
    switch (prop->metric_type) {
      case ax_utils::EUCLIDEAN_DISTANCE:
        distance = ax_utils::embeddings_euclidean_distance(current_embedding, const_embeddings);
        break;
      case ax_utils::SQUARED_EUCLIDEAN_DISTANCE:
        distance = ax_utils::embeddings_squared_euclidean_distance(
            current_embedding, const_embeddings);
        break;
      case ax_utils::COSINE_DISTANCE:
        distance = ax_utils::embeddings_cosine_distance(current_embedding, const_embeddings);
        break;
      case ax_utils::COSINE_SIMILARITY:
        distance = ax_utils::embeddings_cosine_similarity(current_embedding, const_embeddings);
        break;
      default:
        {
          std::stringstream ss;
          ss << "facenet_recog_to_meta : Unsupported metric id: " << prop->metric_type;
          logger(AX_ERROR) << ss.str() << std::endl;
          throw std::runtime_error(ss.str());
        }
    }

    std::vector<size_t> indices(distance.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto top_k = std::min(indices.size(), static_cast<size_t>(prop->top_k));

    // Sort the indices based on the distance metric - match Python logic
    std::partial_sort(indices.begin(), std::next(indices.begin(), top_k),
        indices.end(), [&](size_t a, size_t b) {
          return (prop->metric_type == ax_utils::COSINE_SIMILARITY) ?
                     distance[a] > distance[b] :
                     distance[a] < distance[b];
        });

    std::vector<float> top_scores(top_k);
    std::vector<std::string> top_labels(top_k);
    std::vector<int> top_ids;
    top_ids.reserve(top_k);

    for (size_t i = 0; i < top_k; i++) {
      const auto idx = indices[i];
      const bool invalid_match = (prop->metric_type == ax_utils::COSINE_SIMILARITY) ?
                                     distance[idx] < prop->distance_threshold :
                                     distance[idx] > prop->distance_threshold;

      if (invalid_match) {
        top_scores[i] = -1.0f; // Mark as invalid
        top_labels[i] = "";
        top_ids.push_back(-1);
        continue;
      }
      top_ids.push_back(static_cast<int>(idx));

      top_scores[i] = distance[idx];
      top_labels[i] = const_labels[idx];
    }

    logger(AX_INFO) << "facenet_recognition: Recognized person: '" << top_labels[0]
                    << "(" << top_ids[0] << ")' with score: " << top_scores[0]
                    << " (threshold: " << prop->distance_threshold << ")" << std::endl;

    ax_utils::insert_and_associate_meta<AxMetaClassification>(map,
        prop->meta_name, prop->master_meta, current_frame, total_frames,
        prop->association_meta, AxMetaClassification::scores_vec{ top_scores },
        AxMetaClassification::classes_vec{ top_ids },
        AxMetaClassification::labels_vec{ top_labels });
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  logger(AX_DEBUG) << "decode_to_meta : Decoding facenet_recog: " << duration.count()
                   << " microseconds" << std::endl;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "embeddings_file",
    "distance_threshold", "metric_type", "meta_key", "master_meta",
    "association_meta", "pair_validation", "top_k", "decoder_name",
    "update_embeddings", "labels_for_update" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto props = std::make_shared<facenet_recog::properties>();
  props->meta_name = Ax::get_property(
      input, "meta_key", "recog_static_properties", props->meta_name);
  props->master_meta = Ax::get_property(
      input, "master_meta", "recog_static_properties", props->master_meta);
  props->association_meta = Ax::get_property(input, "association_meta",
      "recog_static_properties", props->association_meta);
  props->decoder_name = Ax::get_property(
      input, "decoder_name", "recog_static_properties", props->decoder_name);
  props->distance_threshold = Ax::get_property(input, "distance_threshold",
      "recog_static_properties", props->distance_threshold);
  props->metric_type = Ax::get_property(
      input, "metric_type", "recog_static_properties", props->metric_type);

  props->pair_validation = Ax::get_property(input, "pair_validation",
      "recog_static_properties", props->pair_validation);
  props->update_embeddings = Ax::get_property(input, "update_embeddings",
      "recog_static_properties", props->update_embeddings);
  props->embeddings_file = Ax::get_property(input, "embeddings_file",
      "recog_static_properties", props->embeddings_file);

  if (!props->embeddings_file.empty() && std::ifstream(props->embeddings_file).good()) {
    auto [embeddings, labels]
        = ax_utils::read_embedding_json(props->embeddings_file, false, logger);
    props->embeddings = std::move(embeddings);
    props->labels = std::move(labels);
    logger(AX_INFO)
        << "facenet_recog: Loaded " << props->labels.size()
        << " existing embeddings from: " << props->embeddings_file << std::endl;
  }

  bool embeddings_required = !props->pair_validation && !props->update_embeddings;
  if (embeddings_required && props->labels.empty()) {
    throw std::runtime_error(
        "facenet_recog: Embeddings must be provided when update_embeddings is false and pair_validation is false");
  }

  if (props->update_embeddings) {
    auto labels_for_update = Ax::get_property(input, "labels_for_update",
        "recog_static_properties", std::vector<std::string>());
    if (labels_for_update.empty()) {
      throw std::runtime_error(
          "facenet_recog: labels_for_update must be provided when update_embeddings is true");
    }
    props->labels_for_update = std::move(labels_for_update);
    props->current_update_index = 0;

    logger(AX_INFO) << "facenet_recog: Prepared to update/add "
                    << props->labels_for_update.size() << " people: ";
    for (size_t i = 0; i < props->labels_for_update.size(); ++i) {
      logger(AX_INFO) << (i > 0 ? ", " : "") << "'" << props->labels_for_update[i] << "'";
    }
    logger(AX_INFO) << std::endl;
  }

  props->top_k
      = Ax::get_property(input, "top_k", "recog_static_properties", props->top_k);

  return props;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> &input,
    facenet_recog::properties *prop, Ax::Logger &logger)
{
}
