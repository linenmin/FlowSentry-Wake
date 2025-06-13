// Copyright Axelera AI, 2023
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaSegmentsDetection.hpp"
#include "AxOpUtils.hpp"
#include "AxUtils.hpp"

#include <unordered_set>

enum class Which { None, Score, Center, Area };

struct filterdetections_properties {
  std::string input_meta_key{};
  std::string output_meta_key{};
  bool hide_output_meta = true;
  Which which = Which::None;
  int top_k = 0;
  std::vector<int> classes_to_keep{};
  int min_width = 0;
  int min_height = 0;
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "input_meta_key",
    "output_meta_key", "hide_output_meta", "which", "top_k", "classes_to_keep",
    "min_width", "min_height" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<filterdetections_properties>();
  prop->input_meta_key = Ax::get_property(input, "input_meta_key",
      "filterdetections_properties", prop->input_meta_key);
  if (prop->input_meta_key.empty()) {
    throw std::runtime_error("filterdetections : input_meta_key is empty");
  }
  prop->output_meta_key = Ax::get_property(input, "output_meta_key",
      "filterdetections_properties", prop->output_meta_key);
  if (prop->output_meta_key.empty()) {
    throw std::runtime_error("filterdetections : output_meta_key is empty");
  }
  prop->hide_output_meta = Ax::get_property(input, "hide_output_meta",
      "filterdetections_properties", prop->hide_output_meta);
  prop->top_k = Ax::get_property(input, "top_k", "filterdetections_properties", prop->top_k);
  std::string which = Ax::get_property(
      input, "which", "filterdetections_properties", std::string{ "NONE" });
  if (which == "NONE") {
    prop->which = Which::None;
  } else if (which == "SCORE") {
    prop->which = Which::Score;
  } else if (which == "CENTER") {
    prop->which = Which::Center;
  } else if (which == "AREA") {
    prop->which = Which::Area;
  } else {
    throw std::runtime_error("filterdetections : 'which' must be one of NONE, SCORE, CENTER, AREA, instead received "
                             + which);
  }
  if (prop->top_k && prop->which == Which::None) {
    throw std::runtime_error("filterdetections : top_k is set but which is NONE");
  }
  if (!prop->top_k && prop->which != Which::None) {
    throw std::runtime_error(
        "filterdetections : which is set to one of SCORE, CENTER, or AREA, but top_k is not set");
  }
  prop->classes_to_keep = Ax::get_property(input, "classes_to_keep",
      "filterdetections_properties", prop->classes_to_keep);
  prop->min_width = Ax::get_property(
      input, "min_width", "filterdetections_properties", prop->min_width);
  prop->min_height = Ax::get_property(
      input, "min_height", "filterdetections_properties", prop->min_height);
  return prop;
}

extern "C" void
inplace(const AxDataInterface &interface,
    const filterdetections_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  auto meta_itr = meta_map.find(prop->input_meta_key);
  if (meta_itr == meta_map.end()) {
    throw std::runtime_error("inplace_filterdetections.cc: " + prop->input_meta_key
                             + " not found in meta map");
  }
  auto *base_meta = meta_itr->second.get();
  auto *box_meta = dynamic_cast<AxMetaBbox *>(base_meta);
  if (!box_meta) {
    throw std::runtime_error("inplace_filterdetections.cc: " + prop->input_meta_key
                             + " is not derived from AxMetaBbox");
  }
  auto *obj_det_meta = dynamic_cast<AxMetaObjDetection *>(base_meta);
  auto *kpt_det_meta = dynamic_cast<AxMetaKptsDetection *>(base_meta);
  auto *seg_det_meta = dynamic_cast<AxMetaSegmentsDetection *>(base_meta);

  int top_k = prop->top_k;

  std::vector<box_xyxy> boxes{};
  std::vector<int> ids{};
  std::vector<float> scores{};
  std::vector<int> classes{};
  KptXyvVector kpts{};
  std::vector<ax_utils::segment> segments{};

  for (size_t i = 0; i < box_meta->num_elements(); ++i) {
    auto box = box_meta->get_box_xyxy(i);
    int id = box_meta->get_id(i);
    id = id == -1 ? i : id;
    if ((prop->min_width && box.x2 - box.x1 + 1 < prop->min_width)
        || (prop->min_height && box.y2 - box.y1 + 1 < prop->min_height)) {
      continue;
    }
    if (!prop->classes_to_keep.empty()) {
      int class_id;
      if (obj_det_meta) {
        class_id = obj_det_meta->class_id(i);

      } else if (seg_det_meta && seg_det_meta->is_multi_class()) {
        class_id = seg_det_meta->class_id(i);
      } else {
        throw std::runtime_error(
            "filterdetections : classes_to_keep is set but no class id found");
      }

      if (std::find(prop->classes_to_keep.begin(), prop->classes_to_keep.end(), class_id)
          == prop->classes_to_keep.end()) {
        continue;
      }
    }

    boxes.push_back(box);
    ids.push_back(id);
    if (obj_det_meta) {
      scores.push_back(obj_det_meta->score(i));
      classes.push_back(obj_det_meta->class_id(i));
    } else if (kpt_det_meta) {
      scores.push_back(kpt_det_meta->score(i));
      int nk = kpt_det_meta->get_kpts_shape()[0];
      for (int j = 0; j < nk; ++j) {
        kpts.push_back(kpt_det_meta->get_kpt_xy(nk * i + j));
      }
    } else if (seg_det_meta) {
      scores.push_back(seg_det_meta->score(i));
      auto good_segment = std::move(
          const_cast<AxMetaSegmentsDetection *>(seg_det_meta)->get_segment(i));
      segments.push_back(std::move(good_segment));
      if (seg_det_meta->is_multi_class()) {
        classes.push_back(seg_det_meta->class_id(i));
      }
    }
  }

  if (prop->which != Which::None && top_k < static_cast<int>(boxes.size())) {
    std::vector<int> indices;
    if (prop->which == Which::Score) {
      if (!kpt_det_meta && !obj_det_meta && !seg_det_meta) {
        throw std::runtime_error("filterdetections : SCORE requires meta with scores");
      }
      indices = ax_utils::indices_for_topk(scores, top_k);
    } else if (prop->which == Which::Center) {
      if (!std::holds_alternative<AxVideoInterface>(interface)) {
        throw std::runtime_error("filterdetections : CENTER requires video interface");
      }
      const auto &info = std::get<AxVideoInterface>(interface).info;
      indices = ax_utils::indices_for_topk_center(boxes, top_k, info.width, info.height);
    } else if (prop->which == Which::Area) {
      indices = ax_utils::indices_for_topk_area(boxes, top_k);
    } else {
      throw std::logic_error("filterdetections : 'which' is NONE but top_k is set");
    }

    std::vector<box_xyxy> new_boxes{};
    std::vector<int> new_ids{};
    std::vector<float> new_scores{};
    std::vector<int> new_classes{};
    KptXyvVector new_kpts{};
    std::vector<ax_utils::segment> new_segments{};
    for (auto idx : indices) {
      new_boxes.push_back(boxes[idx]);
      new_ids.push_back(ids[idx]);
      if (!scores.empty()) {
        new_scores.push_back(scores[idx]);
      }
      if (!classes.empty()) {
        new_classes.push_back(classes[idx]);
      }
      if (!kpts.empty()) {
        int nk = kpt_det_meta->get_kpts_shape()[0];
        for (int j = 0; j < nk; ++j) {
          new_kpts.push_back(kpts[nk * idx + j]);
        }
      }
      if (!segments.empty()) {
        new_segments.push_back(segments[idx]);
      }
    }
    boxes.swap(new_boxes);
    ids.swap(new_ids);
    scores.swap(new_scores);
    classes.swap(new_classes);
    kpts.swap(new_kpts);
    segments.swap(new_segments);
  }

  bool enable_extern = base_meta->enable_extern;
  std::unique_ptr<AxMetaBbox> new_meta;
  if (obj_det_meta) {
    new_meta = std::make_unique<AxMetaObjDetection>(
        std::move(boxes), std::move(scores), std::move(classes));
  } else if (kpt_det_meta) {
    new_meta = std::make_unique<AxMetaKptsDetection>(std::move(boxes),
        std::move(kpts), std::move(scores), kpt_det_meta->get_kpts_shape(),
        kpt_det_meta->get_decoder_name());
  } else if (seg_det_meta) {
    auto shape = seg_det_meta->get_segments_shape();
    auto sizes = SegmentShape{ shape[2], shape[1] };
    new_meta = std::make_unique<AxMetaSegmentsDetection>(std::move(boxes),
        std::move(segments), std::move(scores), std::move(classes), sizes,
        seg_det_meta->get_base_box(), std::move(seg_det_meta->get_decoder_name()));
  } else if (box_meta) {
    new_meta = std::make_unique<AxMetaBbox>(std::move(boxes));
  } else {
    throw std::logic_error("inplace_filterdetections.cc: No appropriate meta type");
  }

  for (int i = 0; i < ids.size(); ++i) {
    new_meta->set_id(i, ids[i]);
  }
  new_meta->enable_extern = enable_extern;
  if (prop->hide_output_meta) {
    new_meta->enable_extern = false;
  }
  auto ret = meta_map.insert_or_assign(prop->output_meta_key, std::move(new_meta));
  if (!ret.second) {
    logger(AX_INFO) << "inplace_filterdetections.cc: key "
                    << prop->output_meta_key << " replaced" << std::endl;
  }
}
