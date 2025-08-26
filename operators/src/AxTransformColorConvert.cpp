// Copyright Axelera AI, 2025
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

struct ColorConversionCodesTableEntry {
  AxVideoFormat in_fmt;
  AxVideoFormat out_fmt;
  cv::ColorConversionCodes code;
};

static cv::ColorConversionCodes
format2format(AxVideoFormat in_format, AxVideoFormat out_format)
{
  static constexpr std::array<ColorConversionCodesTableEntry, 36> lut = { {
      { AxVideoFormat::RGB, AxVideoFormat::RGBA, cv::COLOR_RGB2RGBA },
      { AxVideoFormat::RGB, AxVideoFormat::BGRA, cv::COLOR_RGB2BGRA },
      { AxVideoFormat::RGB, AxVideoFormat::BGR, cv::COLOR_RGB2BGR },
      { AxVideoFormat::RGB, AxVideoFormat::GRAY8, cv::COLOR_RGB2GRAY },

      { AxVideoFormat::BGR, AxVideoFormat::RGBA, cv::COLOR_BGR2RGBA },
      { AxVideoFormat::BGR, AxVideoFormat::BGRA, cv::COLOR_BGR2BGRA },
      { AxVideoFormat::BGR, AxVideoFormat::RGB, cv::COLOR_BGR2RGB },
      { AxVideoFormat::BGR, AxVideoFormat::GRAY8, cv::COLOR_BGR2GRAY },

      { AxVideoFormat::RGBA, AxVideoFormat::RGB, cv::COLOR_RGBA2RGB },
      { AxVideoFormat::RGBA, AxVideoFormat::BGR, cv::COLOR_RGBA2BGR },
      { AxVideoFormat::RGBA, AxVideoFormat::BGRA, cv::COLOR_RGBA2BGRA },
      { AxVideoFormat::RGBA, AxVideoFormat::GRAY8, cv::COLOR_RGBA2GRAY },

      { AxVideoFormat::BGRA, AxVideoFormat::RGB, cv::COLOR_BGRA2RGB },
      { AxVideoFormat::BGRA, AxVideoFormat::BGR, cv::COLOR_BGRA2BGR },
      { AxVideoFormat::BGRA, AxVideoFormat::RGBA, cv::COLOR_BGRA2RGBA },
      { AxVideoFormat::BGRA, AxVideoFormat::GRAY8, cv::COLOR_BGRA2GRAY },

      { AxVideoFormat::GRAY8, AxVideoFormat::RGB, cv::COLOR_GRAY2RGB },
      { AxVideoFormat::GRAY8, AxVideoFormat::BGR, cv::COLOR_GRAY2BGR },
      { AxVideoFormat::GRAY8, AxVideoFormat::RGBA, cv::COLOR_GRAY2RGBA },
      { AxVideoFormat::GRAY8, AxVideoFormat::BGRA, cv::COLOR_GRAY2BGRA },

      { AxVideoFormat::YUY2, AxVideoFormat::RGB, cv::COLOR_YUV2RGB_YUY2 },
      { AxVideoFormat::YUY2, AxVideoFormat::BGR, cv::COLOR_YUV2BGR_YUY2 },
      { AxVideoFormat::YUY2, AxVideoFormat::RGBA, cv::COLOR_YUV2RGBA_YUY2 },
      { AxVideoFormat::YUY2, AxVideoFormat::BGRA, cv::COLOR_YUV2BGRA_YUY2 },
      { AxVideoFormat::YUY2, AxVideoFormat::GRAY8, cv::COLOR_YUV2GRAY_YUY2 },

      { AxVideoFormat::NV12, AxVideoFormat::RGB, cv::COLOR_YUV2RGB_NV12 },
      { AxVideoFormat::NV12, AxVideoFormat::BGR, cv::COLOR_YUV2BGR_NV12 },
      { AxVideoFormat::NV12, AxVideoFormat::RGBA, cv::COLOR_YUV2RGBA_NV12 },
      { AxVideoFormat::NV12, AxVideoFormat::BGRA, cv::COLOR_YUV2BGRA_NV12 },
      { AxVideoFormat::NV12, AxVideoFormat::GRAY8, cv::COLOR_YUV2GRAY_NV12 },

      { AxVideoFormat::I420, AxVideoFormat::RGB, cv::COLOR_YUV2RGB_I420 },
      { AxVideoFormat::I420, AxVideoFormat::BGR, cv::COLOR_YUV2BGR_I420 },
      { AxVideoFormat::I420, AxVideoFormat::RGBA, cv::COLOR_YUV2RGBA_I420 },
      { AxVideoFormat::I420, AxVideoFormat::BGRA, cv::COLOR_YUV2BGRA_I420 },
      { AxVideoFormat::I420, AxVideoFormat::GRAY8, cv::COLOR_YUV2GRAY_I420 },
  } };
  auto it = std::find_if(
      lut.begin(), lut.end(), [&](const ColorConversionCodesTableEntry &e) {
        return e.in_fmt == in_format && e.out_fmt == out_format;
      });
  if (it != lut.end()) {
    return it->code;
  }
  throw std::runtime_error("OpenCV color conversion not supported for "
                           + AxVideoFormatToString(in_format) + " to "
                           + AxVideoFormatToString(out_format));
}

struct cc_ocv_properties {
  AxVideoFormat format{ AxVideoFormat::UNDEFINED };
};

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "format",
  };
  return allowed_properties;
}

struct {
  std::string color;
  AxVideoFormat format;
} supported_output_formats[] = {
  { "rgba", AxVideoFormat::RGBA },
  { "bgra", AxVideoFormat::BGRA },
  { "rgb", AxVideoFormat::RGB },
  { "bgr", AxVideoFormat::BGR },
  { "gray", AxVideoFormat::GRAY8 },
};

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  auto prop = std::make_shared<cc_ocv_properties>();
  std::string fmt_string
      = Ax::get_property(input, "format", "ColorConvertProperties", std::string{});
  auto fmt_itr = std::find_if(std::begin(supported_output_formats),
      std::end(supported_output_formats),
      [&](auto f) { return f.color == fmt_string; });
  if (fmt_itr == std::end(supported_output_formats)) {
    logger(AX_ERROR) << "Invalid output format given in color conversion: " << fmt_string
                     << std::endl;
    throw std::runtime_error("Invalid output format given in color conversion: " + fmt_string);
  }
  prop->format = fmt_itr->format;
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const cc_ocv_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(interface)) {
    logger(AX_ERROR) << "color_convert works on video input only" << std::endl;
    throw std::runtime_error("color_convert works on video input only");
  }
  auto out = std::get<AxVideoInterface>(interface);
  out.info.format = prop->format;
  return out;
}

/// @brief  Check if the plugin has any work to do
/// @param input
/// @param output
/// @param logger
/// @return true if the plugin can pass through the input to output
extern "C" bool
can_passthrough(const AxDataInterface &input, const AxDataInterface &output,
    const cc_ocv_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxVideoInterface>(input)) {
    throw std::runtime_error("color_convert works on video input only");
  }

  if (!std::holds_alternative<AxVideoInterface>(output)) {
    throw std::runtime_error("color_convert works on video input only");
  }
  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("color_convert works on single video (possibly batched) input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error("color_convert works on single video (possibly batched) output only");
  }
  return (input_details[0].format == output_details[0].format);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const cc_ocv_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  auto in_video = std::get<AxVideoInterface>(input);
  auto out_video = std::get<AxVideoInterface>(output);
  if (in_video.info.cropped) {
    throw std::runtime_error("OpenCV color conversion does not support cropped input");
  }
  if (in_video.info.format != AxVideoFormat::I420
      && in_video.info.format != AxVideoFormat::NV12) {
    if (in_video.strides.size() != 1) {
      throw std::runtime_error("OpenCV color conversion does not support multiple strides");
    }
    if (in_video.strides[0] != in_video.info.stride) {
      throw std::runtime_error(
          "OpenCV color conversion does not support inconsistent input stride");
    }
    for (size_t i = 0; i < in_video.offsets.size(); i++) {
      if (in_video.offsets[i] != 0) {
        throw std::runtime_error("OpenCV color conversion does not support non-zero offset "
                                 + std::to_string(in_video.offsets[i])
                                 + " at plane " + std::to_string(i));
      }
    }
  }

  AxVideoFormat in_format = in_video.info.format;
  int input_opencv_type = Ax::opencv_type_u8(in_video.info.format);
  int height = in_video.info.height;
  int stride = in_video.info.stride;
  if (in_video.info.format == AxVideoFormat::YUY2) {
    input_opencv_type = CV_MAKETYPE(CV_8U, 2);
    stride = 0;
  } else if (in_video.info.format == AxVideoFormat::NV12
             || in_video.info.format == AxVideoFormat::I420) {
    input_opencv_type = CV_MAKETYPE(CV_8U, 1);
    height = height * 3 / 2;
    stride = 0;
  }
  cv::Mat input_mat(cv::Size(in_video.info.width, height), input_opencv_type,
      in_video.data, stride);

  AxVideoFormat out_format = out_video.info.format;
  cv::Mat output_mat(cv::Size(out_video.info.width, out_video.info.height),
      Ax::opencv_type_u8(out_format), out_video.data, out_video.info.stride);

  cv::cvtColor(input_mat, output_mat, format2format(in_format, out_format));
}
