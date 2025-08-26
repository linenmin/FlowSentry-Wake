// Copyright Axelera AI, 2025
#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "AxDataInterface.h"
#include "AxMeta.hpp"
#include "AxUtils.hpp"

enum class box_format { xyxy = 0, xywh = 1, ltxywh = 2 };

struct box_xyxy {
  int x1;
  int y1;
  int x2;
  int y2;
};

struct box_xywh {
  int x;
  int y;
  int w;
  int h;
};

struct box_ltxywh {
  int x;
  int y;
  int w;
  int h;
};

using BboxXyxy = box_xyxy;
using BboxXyxyVector = std::vector<BboxXyxy>;

class AxMetaBbox : public virtual AxMetaBase
{
  public:
  AxMetaBbox() = default;

  explicit AxMetaBbox(BboxXyxyVector boxes, std::vector<float> scores,
      std::vector<int> classes, std::vector<int> ids)
      : bboxvec(std::move(boxes)), scores_(std::move(scores)),
        classes_(std::move(classes)), ids(std::move(ids))
  {
    if (num_elements() != classes_.size() && !classes_.empty()) {
      throw std::logic_error(
          "AxMetaObjDetection: scores and classes must have the same size as boxes");
    }

    if (!ids.empty() && ids.size() != bboxvec.size()) {
      throw std::runtime_error(
          "When constructing AxMetaBbox with ids, the number of ids must match the number of boxes");
    }
    if (num_elements() != scores_.size()) {
      throw std::logic_error(
          "AxMetaObjDetection: scores and classes must have the same size as boxes");
    }
  }

  void extend(const BboxXyxyVector &boxes, const std::vector<float> &scores,
      const std::vector<int> &classes)
  {
    if (boxes.size() != scores.size()) {
      throw std::runtime_error("Boxes and scores must have the same size in extend of AxMetaBbox");
    }
    if (classes_.empty() && !classes.empty()) {
      throw std::runtime_error(
          "Classes must not be provided in extend of AxMetaBbox if the existing meta does not have classes");
    } else if (!classes_.empty() && classes.size() != boxes.size()) {
      throw std::runtime_error(
          "Boxes and classes must have the same size in extend of AxMetaBbox");
    }
    bboxvec.insert(bboxvec.end(), boxes.begin(), boxes.end());
    scores_.insert(scores_.end(), scores.begin(), scores.end());
    classes_.insert(classes_.end(), classes.begin(), classes.end());
  }

  void extend(const AxMetaBbox &other)
  {
    if (other.bboxvec.size() != other.scores_.size()) {
      throw std::runtime_error(
          "Other bbox and scores must have the same size in extend of AxMetaBbox");
    }
    bboxvec.insert(bboxvec.end(), other.bboxvec.begin(), other.bboxvec.end());
    scores_.insert(scores_.end(), other.scores_.begin(), other.scores_.end());
    classes_.insert(classes_.end(), other.classes_.begin(), other.classes_.end());
    ids.insert(ids.end(), other.ids.begin(), other.ids.end());
  }

  void draw(const AxVideoInterface &video,
      const std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map) override
  {
    if (video.info.format != AxVideoFormat::RGB && video.info.format != AxVideoFormat::RGBA) {
      throw std::runtime_error("Boxes can only be drawn on RGB or RGBA");
    }
    cv::Mat mat(cv::Size(video.info.width, video.info.height),
        Ax::opencv_type_u8(video.info.format), video.data, video.info.stride);
    for (auto i = size_t{}; i < bboxvec.size(); ++i) {
      cv::rectangle(mat,
          cv::Rect(cv::Point(bboxvec[i].x1, bboxvec[i].y1),
              cv::Point(bboxvec[i].x2, bboxvec[i].y2)),
          cv::Scalar(0, 0, 0));
    }
  }

  ///
  /// @brief Get the number of boxes in the metadata
  /// @return The number of boxes
  ///
  size_t num_elements() const
  {
    return bboxvec.size();
  }

  std::vector<extern_meta> get_extern_meta() const override
  {
    return { { "bbox", "bbox", int(bboxvec.size() * sizeof(BboxXyxy)),
        reinterpret_cast<const char *>(bboxvec.data()) } };
  }

  ///
  /// @brief Get the elements score at the given index. No bounds checking is
  /// performed. The behaviour is undefined if idx is out of range
  /// @param idx - Index of the requested element
  /// @return The score of the given element
  ///
  float score(size_t idx) const
  {
    return scores_[idx];
  }
  ///
  /// @brief As score, but if idx is out of bounds std::out_of_range is thrown.
  /// @param idx - Index of the requested element
  /// @return The score of the given element
  ///
  float score_at(size_t idx) const
  {
    return scores_.at(idx);
  }

  void set_score(size_t idx, float score)
  {
    scores_[idx] = score;
  }


  ///
  /// @brief Get the class id of the given element. No bounds checking is
  /// performed. The behaviour is undefined if idx is out of range
  /// @param idx - Index of the requested element
  /// @return The class id of the given element
  ///
  int class_id(size_t idx) const
  {
    return classes_.empty() ? -1 : classes_[idx];
  }
  ///
  /// @brief As class_id, but if idx is out of bounds std::out_of_range is thrown.
  /// @param idx - Index of the requested element
  /// @return The class id of the given element
  ///
  int class_id_at(size_t idx) const
  {
    return classes_.empty() ? -1 : classes_.at(idx);
  }
  // true if the metadata has class ids
  bool has_class_id() const
  {
    return !classes_.empty();
  }

  // true if the metadata has multiple class ids,
  // this is currenty an alias for has_class_id but this may change in the future
  bool is_multi_class() const
  {
    return has_class_id();
  }

  ///
  /// @brief Get the box at the given index in xyxy format. No bounds checking
  /// is performed, if idx is out of bounds, the behaviour is undefined.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  BboxXyxy get_box_xyxy(size_t idx) const
  {
    return bboxvec[idx];
  }

  void set_box_xyxy(size_t idx, const BboxXyxy &box)
  {
    bboxvec[idx] = box;
  }

  ///
  /// @brief Get the box at the given index in ltxyxy format. No bounds checking
  /// is performed, if idx is out of bounds, the behaviour is undefined.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  box_ltxywh get_box_ltxywh(size_t idx) const
  {
    const box_xyxy &box = bboxvec[idx];
    return { box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1 };
  }

  ///
  /// @brief Get the box at the given index in xywh format. No bounds checking
  /// is performed, if idx is out of bounds, the behaviour is undefined.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  box_xywh get_box_xywh(size_t idx) const
  {
    const box_xyxy &box = bboxvec[idx];
    return { (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, box.x2 - box.x1,
      box.y2 - box.y1 };
  }

  ///
  /// @brief Get the box at the given index in xyxy format.Bounds checking
  /// is performed, if idx is out of bounds, std::out_of_range is thrown.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  BboxXyxy get_box_xyxy_at(size_t idx) const
  {
    return bboxvec.at(idx);
  }

  ///
  /// @brief Get the box at the given index in ltxyxy format.Bounds checking
  /// is performed, if idx is out of bounds, std::out_of_range is thrown.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  box_ltxywh get_box_ltxywh_at(size_t idx) const
  {
    const box_xyxy &box = bboxvec.at(idx);
    return { box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1 };
  }

  ///
  /// @brief Get the box at the given index in xywh format.Bounds checking
  /// is performed, if idx is out of bounds, std::out_of_range is thrown.
  /// @param idx - Index of the requested box
  /// @return - The box at the given index
  ///
  box_xywh get_box_xywh_at(size_t idx) const
  {
    const box_xyxy &box = bboxvec.at(idx);
    return { (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, box.x2 - box.x1,
      box.y2 - box.y1 };
  }

  const BboxXyxy *get_boxes_data() const
  {
    return bboxvec.data();
  }

  ///
  /// @brief Set box id
  /// @param idx - Index of the box
  /// @param id - Id to set
  ///
  void set_id(size_t idx, int id)
  {
    if (ids.size() < bboxvec.size()) {
      ids.resize(bboxvec.size(), -1);
    }
    ids[idx] = id;
  }

  ///
  /// @brief Get box id
  /// @param idx - Index of the box
  /// @return - Id of the box
  ///
  int get_id(size_t idx) const
  {
    if (idx >= bboxvec.size()) {
      throw std::out_of_range("Index out of range");
    }
    if (idx >= ids.size()) {
      return -1;
    }
    return ids[idx];
  }

  ///
  /// @brief Get the number of boxes in the metadata
  /// @return Get the number of subframes (number of boxes in the metadata)
  ///
  size_t get_number_of_subframes() const override
  {
    return bboxvec.size();
  }

  protected:
  BboxXyxyVector bboxvec;
  std::vector<float> scores_;
  std::vector<int> classes_;
  std::vector<int> ids;
};
