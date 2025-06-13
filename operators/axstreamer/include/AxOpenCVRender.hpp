
#include <string>
#include <vector>
#include "AxMeta.hpp"
#include "AxMetaKptsDetection.hpp"
#include "AxMetaObjectDetection.hpp"
#include "AxMetaSegmentsDetection.hpp"
#include "opencv2/opencv.hpp"

namespace Ax
{
// This file provides some simple OpenCV rendering functions for keypoint
// detection, detection, and object detection. As well as a generic render
// function that will dispatch to the correct overload based on the dynamic type
// and submetas of the AxMetaBase object passed to it.

// These renderers are not higly optimised and are meant to be used as a
// debugging tool rather than a production renderer.

namespace OpenCV
{
struct RenderOptions {
  //! \brief The labels to use for the classes. If empty, the class id will be used in the form `cls=17`
  std::vector<std::string> labels;

  //! \brief Whether to render the bounding boxes of detections
  bool render_bboxes = true;

  //! \brief Whether to render the labels of detections
  bool render_labels = true;

  //! \brief Whether to render the keypoints of keypoint detections
  bool render_keypoints = true;

  //! \brief Whether to render the connecting lines of keypoint detections (see keypoint_lines)
  bool render_keypoint_lines = true;

  //! \brief Whether to render the segments of segmentations
  bool render_segments = true;

  //! \brief Whether to render the segments of segmentations in grayscale (render_segments must be true)
  bool segments_in_grayscale = true;

  //! \brief Whether to render the submetas in a cascaded network
  bool render_submeta = true;

  // Determine how keypoints should be connected by lines, if empty no lines will be drawn
  // A -1 indicates a stop in the line. The default is suitable for yolov8Xpose-coco
  std::vector<int> keypoint_lines = {
    15, 13, 11, 5, 6, 12, 14, 16, -1, // both legs + shoulders
    5, 7, 9, -1, // left arm
    6, 8, 10, -1 // right arm
  };
};

void render(const AxMetaBase &detections, cv::Mat &buffer, const RenderOptions &options);
void render(const AxMetaSegmentsDetection &segs, cv::Mat &buffer,
    const RenderOptions &options);
void render(const AxMetaObjDetection &segs, cv::Mat &buffer, const RenderOptions &options);
void render(const AxMetaKptsDetection &detections, cv::Mat &buffer,
    const RenderOptions &options);
} // namespace OpenCV
} // namespace Ax
