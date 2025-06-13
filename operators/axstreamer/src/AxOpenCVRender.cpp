#include "AxOpenCVRender.hpp"
#include "AxStreamerUtils.hpp"

template <typename BoxyMeta>
static void
draw_bounding_boxes(BoxyMeta bboxes, cv::Mat &buffer, const Ax::OpenCV::RenderOptions &options)
{
  for (auto i = size_t{}; i < bboxes.num_elements(); ++i) {
    auto box = bboxes.get_box_xyxy(i);
    if (options.render_labels) {
      auto id = bboxes.class_id(i);
      auto label = id == 0                    ? "" :
                   id < options.labels.size() ? options.labels[id] + " " :
                                                "cls=" + std::to_string(id) + " ";
      auto msg = label + std::to_string(int(bboxes.score(i) * 100)) + "%";
      cv::putText(buffer, msg, cv::Point(box.x1, box.y1 - 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0xff, 0xff), 2);
    }
    if (options.render_bboxes) {
      cv::rectangle(buffer,
          cv::Rect(cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2)),
          cv::Scalar(0, 0xff, 0xff), 2);
    }
  }
}

void
Ax::OpenCV::render(const AxMetaObjDetection &detections, cv::Mat &buffer,
    const RenderOptions &options)
{
  draw_bounding_boxes(detections, buffer, options);
}


void
Ax::OpenCV::render(const AxMetaKptsDetection &detections, cv::Mat &buffer,
    const RenderOptions &options)
{
  draw_bounding_boxes(detections, buffer, options);
  auto kpts = detections.get_kpts();
  constexpr auto num_keypoints = 17;
  std::vector<cv::Point *> polylines;
  std::vector<int> npts;
  std::vector<cv::Point> points;
  for (auto i = size_t{}; i < detections.num_elements(); ++i) {
    if (options.render_keypoint_lines) {
      points.clear();
      npts.clear();
      polylines.clear();
      for (auto &&idx : options.keypoint_lines) {
        if (idx > -1) {
          points.emplace_back(kpts[idx].x, kpts[idx].y);
        }
      }
      auto *first_point = &points[0];
      auto *current_point = first_point;
      for (auto &&idx : options.keypoint_lines) {
        if (idx == -1) {
          polylines.push_back(first_point);
          npts.push_back(current_point - first_point);
          first_point = current_point;
        } else {
          ++current_point;
        }
      }
      cv::polylines(buffer, polylines.data(), npts.data(), polylines.size(),
          false, cv::Scalar(0, 0xff, 0));
    }
    if (options.render_keypoints) {
      for (auto k = 0; k != num_keypoints; ++k) {
        auto &kpt = kpts[k + i * num_keypoints];
        cv::rectangle(buffer, cv::Point(kpt.x - 1, kpt.y - 1),
            cv::Point(kpt.x + 1, kpt.y + 1), cv::Scalar(0, 0xff, 0));
      }
    }
  }
}

static box_xyxy
translate_image_space_rect(const auto &bbox, const box_xyxy &input_roi, int mask_width)
{
  // note this assumes a square mask shape, which will not be true for a rect model
  int input_w = input_roi.x2 - input_roi.x1;
  int input_h = input_roi.y2 - input_roi.x1;
  int longest_edge = std::max(input_w, input_h);
  float scale_factor = static_cast<float>(longest_edge) / mask_width;
  int xoffset = static_cast<int>((longest_edge - input_w) / 2.0);
  int yoffset = static_cast<int>((longest_edge - input_h) / 2.0);
  return {
    static_cast<int>(bbox.x1 * scale_factor - xoffset + input_roi.x1),
    static_cast<int>(bbox.y1 * scale_factor - yoffset + input_roi.y1),
    static_cast<int>(bbox.x2 * scale_factor - xoffset + input_roi.x1),
    static_cast<int>(bbox.y2 * scale_factor - yoffset + input_roi.y1),
  };
}

void
Ax::OpenCV::render(const AxMetaSegmentsDetection &segs, cv::Mat &buffer,
    const RenderOptions &options)
{
  draw_bounding_boxes(segs, buffer, options);
  if (!options.render_segments) {
    return;
  }
  const auto shape = segs.get_segments_shape();
  for (auto i = size_t{}; i < segs.num_elements(); ++i) {
    auto seg = segs.get_segment(i);
    const auto height = seg.y2 - seg.y1;
    const auto width = seg.x2 - seg.x1;
    assert(seg.map.size() == height * width);
    auto dest = translate_image_space_rect(
        seg, { 0, 0, buffer.cols, buffer.rows }, shape[1]);
    auto scale = static_cast<float>(dest.x2 - dest.x1) / width;
    auto fmask = cv::Mat(height, width, CV_32F, seg.map.data());
    for (auto y = dest.y1; y < dest.y2; ++y) {
      for (auto x = dest.x1; x < dest.x2; ++x) {
        const auto b = buffer.at<cv::Vec3b>(y, x)[0];
        const auto g = buffer.at<cv::Vec3b>(y, x)[1];
        const auto r = buffer.at<cv::Vec3b>(y, x)[2];
        const auto gray = (r * 0.299f) + (g * 0.587f) + (b * 0.114f);
        const auto grayness = options.segments_in_grayscale ? fmask.at<float>(
                                  (y - dest.y1) / scale, (dest.x1 - x) / scale) :
                                                              0.0;
        buffer.at<cv::Vec3b>(y, x)[0] = (grayness * gray) + ((1.0f - grayness) * b);
        buffer.at<cv::Vec3b>(y, x)[1] = (grayness * gray) + ((1.0f - grayness) * g);
        buffer.at<cv::Vec3b>(y, x)[2] = (grayness * gray) + ((1.0f - grayness) * r);
      }
    }
  }
}

void
Ax::OpenCV::render(const AxMetaBase &detections, cv::Mat &buffer, const RenderOptions &options)
{
  const auto *bboxes = dynamic_cast<const AxMetaBbox *>(&detections);
  if (const auto *obj = dynamic_cast<const AxMetaObjDetection *>(&detections)) {
    render(*obj, buffer, options);
  } else if (const auto *kpts = dynamic_cast<const AxMetaKptsDetection *>(&detections)) {
    render(*kpts, buffer, options);
  } else if (const auto *segs = dynamic_cast<const AxMetaSegmentsDetection *>(&detections)) {
    render(*segs, buffer, options);
  }

  if (!options.render_submeta) {
    return;
  }
  const auto submeta_names = detections.submeta_names();
  for (auto &&submeta_name : submeta_names) {
    auto submetas = detections.get_submetas(submeta_name);
    for (auto &&[n, sub] : Ax::Internal::enumerate(submetas)) {
      auto roi = bboxes ? bboxes->get_box_xyxy(n) :
                          BboxXyxy{ 0, 0, buffer.cols, buffer.rows };
      cv::Mat subbuffer(
          buffer, cv::Rect(roi.x1, roi.y1, roi.x2 - roi.x1, roi.y2 - roi.y1));
      render(*sub, subbuffer, options);
    }
  }
}
