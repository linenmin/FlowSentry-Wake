// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaStreamId.hpp"
#include "AxMetaTracker.hpp"
#include "MultiObjTracker.hpp"
#include "TrackerFactory.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_set>

struct PerStreamTracker {
  std::unique_ptr<ax::MultiObjTracker> tracker;
  std::unordered_map<int, TrackingDescriptor> static_tracker_id_to_tracking_descriptor;

  PerStreamTracker(std::string algorithm, TrackerParams algo_params)
      : tracker(CreateMultiObjTracker(algorithm, algo_params))
  {
  }
};

TrackerParams
DeserializeToTrackerParams(const nlohmann::json &j)
{
  TrackerParams params;

  for (const auto &el : j.items()) {
    if (el.value().is_boolean()) {
      params[el.key()] = el.value().get<bool>();
    } else if (el.value().is_number_integer()) {
      params[el.key()] = el.value().get<int>();
    } else if (el.value().is_number_float()) {
      params[el.key()] = el.value().get<float>();
    } else if (el.value().is_string()) {
      params[el.key()] = el.value().get<std::string>();
    } else {
      throw std::runtime_error("Unsupported type in JSON");
    }
  }
  return params;
}

template <typename T>
void
insert_into_callback_map(std::string_view options_string,
    std::unordered_map<std::string, T> &callback_map, Ax::Logger &logger)
{
  auto split_options = Ax::Internal::split(options_string, "?");
  for (const auto &split_option : split_options) {
    if (split_option.empty()) {
      continue;
    }
    auto options = Ax::extract_secondary_options(logger, std::string(split_option));
    auto found_key = options.find("key");
    if (found_key == options.end()) {
      throw std::runtime_error("Callback key not found");
    }
    auto found_lib = options.find("lib");
    if (found_lib == options.end()) {
      throw std::runtime_error("Callback lib not found");
    }
    auto key = std::move(found_key->second);
    options.erase(found_key);
    auto lib = std::move(found_lib->second);
    options.erase(found_lib);
    auto returned
        = callback_map.try_emplace(std::move(key), std::move(lib), options, logger);
    if (!returned.second) {
      throw std::runtime_error("Callback key already exists");
    }
  }
}

struct tracker_properties {
  std::string tracker_meta_key{};
  std::string input_meta_key{};
  std::string output_meta_key{};
  std::string streamid_meta_key{ "stream_id" };
  size_t history_length{ 1 };
  std::string algorithm{ "oc-sort" };
  TrackerParams algo_params{};
  std::unordered_map<std::string, KeepBoxCallback> keep_box_callback_map;
  std::unordered_map<std::string, DetermineObjectAttributeCallback> determine_object_attribute_callback_map;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "tracker_meta_key",
    "input_meta_key", "output_meta_key", "streamid_meta_key", "history_length", "algorithm",
    "algo_params_json", "filter_callbacks", "determine_object_attribute_callbacks" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<tracker_properties>();
  prop->tracker_meta_key = Ax::get_property(
      input, "tracker_meta_key", "tracker_properties", prop->tracker_meta_key);
  prop->input_meta_key = Ax::get_property(
      input, "input_meta_key", "tracker_properties", prop->input_meta_key);
  prop->output_meta_key = Ax::get_property(
      input, "output_meta_key", "tracker_properties", prop->output_meta_key);
  prop->streamid_meta_key = Ax::get_property(
      input, "streamid_meta_key", "tracker_properties", prop->streamid_meta_key);
  prop->history_length = Ax::get_property(
      input, "history_length", "tracker_properties", prop->history_length);
  prop->algorithm
      = Ax::get_property(input, "algorithm", "tracker_properties", prop->algorithm);

  auto filename = Ax::get_property(
      input, "algo_params_json", "tracker_properties", std::string{});
  if (!filename.empty()) {
    std::ifstream file(filename);
    if (!file) {
      logger(AX_ERROR) << "tracker_properties : algo_params_json not found" << std::endl;
      throw std::runtime_error("tracker_properties : algo_params_json not found");
    }
    nlohmann::json j;
    file >> j;
    prop->algo_params = DeserializeToTrackerParams(j);
  }

  if (auto found = input.find("filter_callbacks"); found != input.end()) {
    insert_into_callback_map(found->second, prop->keep_box_callback_map, logger);
  }

  if (auto found = input.find("determine_object_attribute_callbacks");
      found != input.end()) {
    insert_into_callback_map(
        found->second, prop->determine_object_attribute_callback_map, logger);
  }

  return prop;
}

void
create_box_meta(const std::string &output_meta_key,
    const std::unordered_map<int, TrackingDescriptor> &track_id_to_tracking_descriptor,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map,
    int width, int height, const KeepBoxCallback *callback_ptr = nullptr)
{
  BboxXyxyVector bboxvec;
  std::vector<int> ids;
  std::vector<int> class_ids;
  std::vector<float> scores;
  for (const auto &[track_id, tracking_descriptor] : track_id_to_tracking_descriptor) {
    if (callback_ptr && !(*callback_ptr)(tracking_descriptor)) {
      continue;
    }
    const auto &bbox
        = tracking_descriptor.collection->get_frame(tracking_descriptor.frame_id)
              .bbox;
    if (bbox.x2 < 0 || bbox.x1 >= width || bbox.y2 < 0 || bbox.y1 >= height) {
      continue;
    }
    bboxvec.push_back({ std::max(bbox.x1, 0), std::max(bbox.y1, 0),
        std::min(bbox.x2, width - 1), std::min(bbox.y2, height - 1) });
    ids.push_back(track_id);
    class_ids.push_back(tracking_descriptor.collection->detection_class_id);
    scores.push_back(tracking_descriptor.collection->detection_score);
  }
  ax_utils::insert_meta<AxMetaObjDetection>(map, output_meta_key, "", 0, 1,
      std::move(bboxvec), std::move(scores), std::move(class_ids), std::move(ids));
  map[output_meta_key]->enable_extern = false;
}

extern "C" void
inplace(const AxDataInterface &data, const tracker_properties *prop,
    unsigned int subframe_index, unsigned int number_of_subframes,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &map, Ax::Logger &logger)
{
  // Static map to hold Tracker Records for each stream_id
  static std::unordered_map<int, PerStreamTracker> stream_tracker_map;

  if (!std::holds_alternative<AxVideoInterface>(data)) {
    throw std::runtime_error("inplace_tracker: buffer not of type AxVideoInterface");
  }
  auto &video_info = std::get<AxVideoInterface>(data).info;

  int stream_id = 0;
  if (!prop->streamid_meta_key.empty()) {
    auto *stream_id_meta = ax_utils::get_meta<AxMetaStreamId>(
        prop->streamid_meta_key, map, "inplace_tracker");
    stream_id = stream_id_meta->stream_id;
  }

  auto res = map.try_emplace(prop->tracker_meta_key,
      std::make_unique<AxMetaTracker>(prop->history_length));
  if (!res.second) {
    logger(AX_ERROR) << "inplace_tracker: tracker_meta_key already exists" << std::endl;
    throw std::runtime_error("inplace_tracker: tracker_meta_key already exists");
  }
  auto *tracker_meta = dynamic_cast<AxMetaTracker *>(res.first->second.get());

  auto *box_meta = ax_utils::get_meta<AxMetaBbox>(prop->input_meta_key, map, "inplace_tracker");
  const auto *det_meta = dynamic_cast<const AxMetaObjDetection *>(box_meta);
  const auto *kpts_meta = dynamic_cast<const AxMetaKptsDetection *>(box_meta);
  if (det_meta == nullptr && kpts_meta == nullptr) {
    logger(AX_ERROR) << "inplace_tracker: input_meta_key not of type AxMetaObjDetectionMeta or AxMetaKptsDetectionMeta"
                     << std::endl;
    throw std::runtime_error(
        "inplace_tracker: input_meta_key not of type AxMetaObjDetection or AxMetaKptsDetection");
  }

  std::vector<ax::ObservedObject> convertedDetections(box_meta->num_elements());
  for (int i = 0; i < box_meta->num_elements(); ++i) {
    const auto &[x1, y1, x2, y2] = box_meta->get_box_xyxy(i);
    auto &det = convertedDetections[i];
    det.bbox.x1 = x1 / static_cast<float>(video_info.width);
    det.bbox.y1 = y1 / static_cast<float>(video_info.height);
    det.bbox.x2 = x2 / static_cast<float>(video_info.width);
    det.bbox.y2 = y2 / static_cast<float>(video_info.height);
    if (det_meta) {
      det.class_id = det_meta->class_id(i);
      det.score = det_meta->score(i);
    } else {
      det.class_id = kpts_meta->class_id(i);
      det.score = kpts_meta->score(i);
    }
  }

  auto &per_stream_tracker = stream_tracker_map
                                 .try_emplace(stream_id, prop->algorithm, prop->algo_params)
                                 .first->second;

  std::vector<ax::TrackedObject> trackers
      = per_stream_tracker.tracker->Update(convertedDetections);
  for (const auto &tracker : trackers) {
    const auto &xyxy = tracker.GetXyxy(video_info.width, video_info.height);

    auto [itr, success]
        = per_stream_tracker.static_tracker_id_to_tracking_descriptor.try_emplace(
            tracker.track_id, tracker.track_id, tracker.class_id, tracker.score,
            tracker_meta->history_length, prop->determine_object_attribute_callback_map);

    if (!success) {
      ++itr->second.frame_id;
    }
    itr->second.detection_meta_id = tracker.latest_detection_id;
    itr->second.collection->set_frame(itr->second.frame_id,
        TrackingElement{ BboxXyxy{ std::get<0>(xyxy), std::get<1>(xyxy),
                             std::get<2>(xyxy), std::get<3>(xyxy) },
            {} });
    tracker_meta->track_id_to_tracking_descriptor.insert(*itr);
  }
  for (auto itr = per_stream_tracker.static_tracker_id_to_tracking_descriptor.begin();
       itr != per_stream_tracker.static_tracker_id_to_tracking_descriptor.end();) {
    if (tracker_meta->track_id_to_tracking_descriptor.count(itr->first)) {
      ++itr;
    } else {
      itr = per_stream_tracker.static_tracker_id_to_tracking_descriptor.erase(itr);
    }
  }

  if (!prop->output_meta_key.empty()) {
    create_box_meta(prop->output_meta_key, tracker_meta->track_id_to_tracking_descriptor,
        map, video_info.width, video_info.height);
  }
  for (const auto &[callback_key, callback] : prop->keep_box_callback_map) {
    std::string output_meta_key
        = prop->tracker_meta_key + "_adapted_as_input_for_" + callback_key;
    create_box_meta(output_meta_key, tracker_meta->track_id_to_tracking_descriptor,
        map, video_info.width, video_info.height, &callback);
  }

  if (kpts_meta) {
    auto kpts_shape = kpts_meta->get_kpts_shape();
    int kpts_per_box = kpts_shape[0];

    for (auto &[track_id, tracking_descriptor] : tracker_meta->track_id_to_tracking_descriptor) {
      if (tracking_descriptor.detection_meta_id < 0) {
        continue;
      }
      if (tracking_descriptor.detection_meta_id >= kpts_meta->num_elements()) {
        throw std::runtime_error("inplace_tracker: detection_meta_id out of bounds");
      }
      KptXyvVector kpts = kpts_meta->get_kpts_xyv(
          tracking_descriptor.detection_meta_id * kpts_per_box, kpts_per_box);
      float score = kpts_meta->score(tracking_descriptor.detection_meta_id);

      tracking_descriptor.collection->set_frame_data_map(tracking_descriptor.frame_id,
          prop->input_meta_key,
          std::make_shared<AxMetaKptsDetection>(std::vector<box_xyxy>(1), std::move(kpts),
              std::vector<float>{ score }, kpts_shape, kpts_meta->get_decoder_name()));
    }
  }
}
