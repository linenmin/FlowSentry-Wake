// Copyright Axelera AI, 2025
// An example program showing how to use the AxInferenceNet class to run
// inference on a video stream and obtain raw tensor outputs.
// This allows users to implement their own postprocessing logic
// instead of relying on built-in metadata or postprocessing.

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "AxInferenceNet.hpp"
#include "AxMetaRawTensor.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"
#include "axruntime/axruntime.hpp"
#include "opencv2/opencv.hpp"

using namespace std::string_literals;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
constexpr auto DEFAULT_LABELS = "ax_datasets/labels/coco.names";

namespace
{
auto
micro_duration(high_resolution_clock::time_point start, high_resolution_clock::time_point now)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(now - start);
}

auto
read_labels(const std::string &path)
{
  std::vector<std::string> labels;
  std::ifstream file(path);
  for (std::string line; std::getline(file, line);) {
    labels.push_back(line);
  }
  return labels;
}

std::tuple<std::string, std::vector<std::string>, std::string>
parse_args(int argc, char **argv)
{
  std::string model_properties;
  std::vector<std::string> labels;
  std::string input;
  for (auto arg = 1; arg != argc; ++arg) {
    auto s = std::string(argv[arg]);
    if (s.ends_with(".axnet")) {
      model_properties = s;
    } else if (s.ends_with(".txt") || s.ends_with(".names")) {
      labels = read_labels(s);
    } else if (!std::filesystem::exists(s)) {
      std::cerr << "Warning: Path does not exist: " << s << std::endl;
    } else if (!input.empty()) {
      std::cerr << "Warning: Multiple input files specified: " << s << std::endl;
    } else {
      input = s;
    }
  }
  if (model_properties.empty() || input.empty()) {
    std::cerr
        << "Usage: " << argv[0] << " <model>.axnet [labels.txt] input-source\n"
        << "  <model>.axnet: path to the model axnet file\n"
        << "  labels.txt: path to the labels file (default: " << DEFAULT_LABELS << ")\n"
        << "  input-source: path to video source\n"
        << "\n"
        << "The <model>.axnet file describes the model and the pipeline stages.\n"
        << "In the axinferencenet_example.cpp, AxInferenceNet runs the full end-to-end pipeline, including preprocessing, inference, and postprocessing.\n"
        << "\n"
        << "In this example, AxInferenceNet is used to perform only preprocessing and main inference.\n"
        << "Users receive the raw tensor outputs and are responsible for implementing their own postprocessing (such as NMS and other steps).\n"
        << "This allows for custom postprocessing logic and more flexible integration.\n"
        << "\n"
        << "The first step is to deploy the model and pipeline. This can be done by running:\n"
        << "\n"
        << "  ./inference.py yolov8n-output-tensor fakevideo --frames=1 --no-display\n"
        << "\n"
        << "This will deploy the model and pipeline (may take some time on first run) and generate the .axnet file in the build folder.\n"
        << "You can use any media file as input (e.g., fakevideo or a real video file), and only need to infer 1 frame for deployment.\n"
        << "\n"
        << "To run inference, use:\n"
        << "\n"
        << "  examples/bin/axinferencenet_tensor build/yolov8n-output-tensor/yolov8n-output-tensor.axnet </path/to/the/media>\n"
        << std::endl;
    std::exit(1);
  }
  if (labels.empty()) {
    const auto root = std::getenv("AXELERA_FRAMEWORK");
    labels = read_labels(root[0] == '\0' ? DEFAULT_LABELS : root + "/"s + DEFAULT_LABELS);
  }

  return { model_properties, labels, input };
}

struct Frame {
  cv::Mat bgr;
  Ax::MetaMap meta;
};

void
reader_thread(cv::VideoCapture &input, Ax::InferenceNet &net)
{
  while (true) {
    auto frame = std::make_shared<Frame>();
    if (!input.read(frame->bgr)) {
      // Signal to AxInferenceNet that there is no more input and it should
      // flush any buffers still in the pipeline.
      net.end_of_input();
      break;
    }
    auto video = Ax::video_from_cvmat(frame->bgr, AxVideoFormat::BGR);
    net.push_new_frame(frame, video, frame->meta);
  }
}

struct Detection {
  cv::Rect bbox;
  int label;
  float score;
};

std::vector<Detection>
postprocess_yolov8(const float *data, const std::vector<int64_t> &dims,
    float prob_threshold, float nms_threshold, int top_k, int img_width,
    int img_height, int model_input_width, int model_input_height, bool letterboxed)
{
  std::vector<Detection> detections;
  if (dims.size() != 3)
    return detections;
  int batch = static_cast<int>(dims[0]);
  int num_channels = static_cast<int>(dims[1]);
  int num_anchors = static_cast<int>(dims[2]);
  int num_classes = num_channels - 4;
  if (batch != 1)
    return detections;

  cv::Mat output(num_channels, num_anchors, CV_32F, const_cast<float *>(data));
  output = output.t();

  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<int> indices;

  // Calculate scale and padding for letterbox
  float scale = std::min(float(model_input_width) / img_width,
      float(model_input_height) / img_height);
  float new_unpad_w = img_width * scale;
  float new_unpad_h = img_height * scale;
  float pad_w = (model_input_width - new_unpad_w) / 2.0f;
  float pad_h = (model_input_height - new_unpad_h) / 2.0f;

  for (int i = 0; i < num_anchors; ++i) {
    const float *row = output.ptr<float>(i);
    const float *bboxesPtr = row;
    const float *scoresPtr = row + 4;
    auto maxSPtr = std::max_element(scoresPtr, scoresPtr + num_classes);
    float score = *maxSPtr;
    if (score > prob_threshold) {
      float x = *bboxesPtr++;
      float y = *bboxesPtr++;
      float w = *bboxesPtr++;
      float h = *bboxesPtr;

      float x0 = x - 0.5f * w;
      float y0 = y - 0.5f * h;
      float x1 = x + 0.5f * w;
      float y1 = y + 0.5f * h;

      if (letterboxed) {
        // Undo letterbox: remove padding, then scale to original image
        x0 = (x0 - pad_w) / scale;
        y0 = (y0 - pad_h) / scale;
        x1 = (x1 - pad_w) / scale;
        y1 = (y1 - pad_h) / scale;
      } else {
        // Simple resize
        x0 = x0 * img_width / model_input_width;
        y0 = y0 * img_height / model_input_height;
        x1 = x1 * img_width / model_input_width;
        y1 = y1 * img_height / model_input_height;
      }

      x0 = std::clamp(x0, 0.f, float(img_width));
      y0 = std::clamp(y0, 0.f, float(img_height));
      x1 = std::clamp(x1, 0.f, float(img_width));
      y1 = std::clamp(y1, 0.f, float(img_height));

      int label = int(maxSPtr - scoresPtr);
      cv::Rect bbox{ cv::Point(int(x0), int(y0)), cv::Point(int(x1), int(y1)) };
      bboxes.push_back(bbox);
      labels.push_back(label);
      scores.push_back(score);
    }
  }
  // NMS
  cv::dnn::NMSBoxes(bboxes, scores, prob_threshold, nms_threshold, indices, 1.f, top_k);
  for (int idx : indices) {
    Detection det;
    det.bbox = bboxes[idx];
    det.label = labels[idx];
    det.score = scores[idx];
    detections.push_back(det);
  }
  return detections;
}

void
render_detections(const std::vector<Detection> &detections, cv::Mat &buffer,
    const std::vector<std::string> &labels)
{
  for (const auto &det : detections) {
    std::string label = (det.label >= 0 && det.label < int(labels.size())) ?
                            labels[det.label] :
                            "Unknown";
    std::string msg = label + " " + std::to_string(int(det.score * 100)) + "%";
    cv::putText(buffer, msg, cv::Point(det.bbox.x, det.bbox.y - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0xff, 0xff), 2);
    cv::rectangle(buffer, det.bbox, cv::Scalar(0, 0xff, 0xff), 2);
  }
}

} // namespace

int
main(int argc, char **argv)
{
  Ax::Logger logger;
  const auto [model_properties, labels, input] = parse_args(argc, argv);
  cv::VideoCapture cap(input);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video source: " << input << std::endl;
    return 1;
  }

  // We use BlockingQueue to communicate between the frame_completed callback and the main loop
  Ax::BlockingQueue<std::shared_ptr<Frame>> ready;
  auto props = Ax::read_inferencenet_properties(model_properties, logger);
  auto net = Ax::create_inference_net(props, logger, Ax::forward_to(ready));

  // Start the reader thread which reads frames from the video source and pushes
  // them to the inference network
  std::thread reader(reader_thread, std::ref(cap), std::ref(*net));

  // Use OpenCV window to display the results
  const std::string wndname = "AxInferenceNet Tensor Output Demo";
  cv::namedWindow(wndname, cv::WINDOW_AUTOSIZE);
  cv::setWindowProperty(wndname, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);

  int model_input_width = 640;
  int model_input_height = 640;
  bool letterboxed = true;

  const auto start = high_resolution_clock::now();
  int num_frames = 0;
  while (1) {
    auto frame = ready.wait_one();
    if (!frame) {
      break;
    }
    if ((num_frames % 10) == 0) {
      // OpenCV is quite slow at rendering results, so we only render every 10th frame
      auto &tensor_wrapper = dynamic_cast<AxMetaRawTensor &>(*frame->meta["detections"]);
      if (tensor_wrapper.get_tensor() && tensor_wrapper.get_tensor()->num_tensors() > 0) {
        std::vector<int64_t> dims = get_tensor_shape(tensor_wrapper);
        const float *data = get_tensor_data<float>(tensor_wrapper);
        auto detections = postprocess_yolov8(data, dims, 0.25f, 0.45f, 100,
            frame->bgr.cols, frame->bgr.rows, model_input_width,
            model_input_height, letterboxed);
        render_detections(detections, frame->bgr, labels);
      }
      cv::imshow(wndname, frame->bgr);
      cv::waitKey(1);
    }
    ++num_frames;
  }
  const auto end = high_resolution_clock::now();

  // Wait for AxInferenceNet to complete and join its threads, before joining the reader thread
  net->stop();
  reader.join();

  // Output some statistics
  const auto duration = micro_duration(start, end);
  const auto taken = static_cast<float>(duration.count()) / 1e6;
  std::cout << "Executed " << num_frames << " frames in " << taken
            << "s : " << num_frames / taken << "fps" << std::endl;
  cv::destroyWindow(wndname);
}
