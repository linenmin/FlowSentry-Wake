// Copyright Axelera AI, 2023
#include <array>
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"
#include "AxOpenCl.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"

const char *kernel_cl = R"##(

uchar3 yuv_to_rgb(uchar y, uchar u, uchar v, char bgr) {
    uchar3 result = YUV_to_RGB(y, u, v);
    return bgr ? result.zyx : result;
}

uchar4 yuv_to_rgba(uchar y, uchar u, uchar v, char bgr) {
    return (uchar4)(yuv_to_rgb(y, u, v, bgr), 255);
}

uchar4 convert_to_rgba(float y, float u, float v, char bgr) {
    float r = y + 1.402f * (v - 0.5f);
    float g = y - 0.344f * (u - 0.5f) - 0.714f * (v - 0.5f);
    float b = y + 1.772f * (u - 0.5f);
    return bgr ?  (uchar4)(convert_uchar_sat(b * 255.0f),
                    convert_uchar_sat(g * 255.0f),
                    convert_uchar_sat(r * 255.0f),

                    255) :
                (uchar4)(convert_uchar_sat(r * 255.0f),
                    convert_uchar_sat(g * 255.0f),
                    convert_uchar_sat(b * 255.0f),
                    255);
}

__kernel void nv12_to_rgb_image(read_only image2d_t y_plane, read_only image2d_t uv_plane,
    __global uchar4 *rgb_out, int width, int stride, int height, char is_bgr) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= width || col >= height) {
      return;
    }

    int x = col * 2;
    int y = row * 2;
    int2 coord = (int2)(x , y);
    int2 uv_coord = (int2)(col, row);
    int2 y_coord = (int2)(x, y);
    float2 UV = read_imagef(uv_plane, CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST, uv_coord).xy;
    float Y = read_imagef(y_plane, CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST, y_coord).x;
    int idx = y * stride + x;
    rgb_out[idx] = convert_to_rgba(Y, UV.x, UV.y, is_bgr);
}

__kernel void nv12_planes_to_rgba(int width, int height, int strideInY, int strideUV, int strideOut,
    char is_bgr, __global const uchar *iny, __global const uchar *inuv, __global uchar4 *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    uchar y = iny[top_left.y * strideInY + top_left.x];
    int uv_idx = top_left.y / 2 * strideUV + (top_left.x & ~1); ;
    uchar u = inuv[uv_idx];
    uchar v = inuv[uv_idx + 1];
    __global uchar4* prgb = advance_uchar4_ptr(rgb, row * strideOut);
    prgb[col] = yuv_to_rgba(y, u, v, is_bgr);
}

__kernel void nv12_planes_to_rgb(int width, int height, int strideInY, int strideUV, int strideOut,
    char is_bgr, __global const uchar *iny, __global const uchar *inuv, __global uchar *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    uchar y = iny[top_left.y * strideInY + top_left.x];
    int uv_idx = top_left.y / 2 * strideUV + (top_left.x & ~1); ;
    uchar u = inuv[uv_idx];
    uchar v = inuv[uv_idx + 1];
    __global uchar* prgb = advance_uchar_ptr(rgb, row * strideOut);
    vstore3(yuv_to_rgb(y, u, v, is_bgr), col, prgb);
}

__kernel void nv12_to_rgba(int width, int height, int strideInY, int strideUV, int strideOut,
    int uv_offset, char is_bgr, __global const uchar *iny, __global uchar4 *rgb) {

  return nv12_planes_to_rgba(width, height, strideInY, strideUV, strideOut, is_bgr, iny, iny + uv_offset, rgb);
}

__kernel void nv12_to_rgb(int width, int height, int strideInY, int strideUV, int strideOut,
    int uv_offset, char is_bgr, __global const uchar *iny, __global uchar *rgb) {

  return nv12_planes_to_rgb(width, height, strideInY, strideUV, strideOut, is_bgr, iny, iny + uv_offset, rgb);
}


__kernel void i420_planes_to_rgba(int width, int height, int strideInY, int strideU, int strideV, int strideOut,
    char is_bgr, __global const uchar *iny, __global const uchar *inu , __global const uchar *inv, __global uchar4 *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar4* prgb = advance_uchar4_ptr(rgb, strideOut * row);
    uchar y = iny[top_left.y * strideInY + top_left.x];
    uchar u = inu[top_left.y / 2 * strideU + top_left.x / 2];
    uchar v = inv[top_left.y / 2 * strideV + top_left.x / 2];
    prgb[col] = yuv_to_rgba(y, u, v, is_bgr);
}

__kernel void i420_planes_to_rgb(int width, int height, int strideInY, int strideU, int strideV, int strideOut,
    char is_bgr, __global const uchar *iny, __global const uchar *inu , __global const uchar *inv, __global uchar *rgb) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= height || col >= width) {
        return;
    }
    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar* prgb = advance_uchar_ptr(rgb, strideOut * row);
    uchar y = iny[top_left.y * strideInY + top_left.x];
    uchar u = inu[top_left.y / 2 * strideU + top_left.x / 2];
    uchar v = inv[top_left.y / 2 * strideV + top_left.x / 2];
    vstore3(yuv_to_rgb(y, u, v, is_bgr), col, prgb);
}


__kernel void i420_to_rgba(int width, int height, int strideInY, int strideU, int strideV, int strideOut,
    int u_offset, int v_offset, char is_bgr, __global const uchar *iny, __global uchar4 *rgb) {
  return i420_planes_to_rgba(width, height, strideInY, strideU, strideV, strideOut, is_bgr, iny, iny + u_offset, iny + v_offset, rgb);
}

__kernel void i420_to_rgb(int width, int height, int strideInY, int strideU, int strideV, int strideOut,
    int u_offset, int v_offset, char is_bgr, __global const uchar *iny, __global uchar *rgb) {
  return i420_planes_to_rgb(width, height, strideInY, strideU, strideV, strideOut, is_bgr, iny, iny + u_offset, iny + v_offset, rgb);
}

__kernel void YUYV_to_rgba(int width, int height, int strideIn, int strideOut,
    char is_bgr, __global const uchar4 *in, __global uchar4 *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }
    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar4 *p_in = advance_uchar4_ptr(in, top_left.y * strideIn);
    __global uchar4* prgb = advance_uchar4_ptr(rgb, strideOut * row);

    uchar4 i = p_in[top_left.x / 2];
    uchar y =  top_left.x % 2 == 0 ? i.x : i.z;
    uchar u = i.y;
    uchar v = i.w;
    prgb[col] = yuv_to_rgba(y, u, v, is_bgr);
}

__kernel void YUYV_to_rgb(int width, int height, int strideIn, int strideOut,
    char is_bgr, __global const uchar4 *in, __global uchar *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);

    __global uchar4 *p_in = advance_uchar4_ptr(in, top_left.y * strideIn);
    __global uchar* prgb = advance_uchar_ptr(rgb, strideOut * row);

    uchar4 i = p_in[top_left.x / 2];
    uchar y =  top_left.x % 2 == 0 ? i.x : i.z;
    uchar u = i.y;
    uchar v = i.w;
    vstore3(yuv_to_rgb(y, u, v, is_bgr), col, prgb);
}

__kernel void bgra_to_rgba(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar4 *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    __global uchar4* prgb = advance_uchar4_ptr(rgb, row * strideOut);
    uchar4 i = in[top_left.y * strideI + top_left.x];
    prgb[col] = i.zyxw;
}

__kernel void bgr_to_rgb(int width, int height, int strideIn, int strideOut,
    __global const uchar *in, __global uchar *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    __global uchar *p_in = advance_uchar_ptr(in, top_left.y * strideIn);
    __global uchar* prgb = advance_uchar_ptr(rgb, row * strideOut);
    uchar3 i = vload3(top_left.x, p_in);
    vstore3(i.zyx, col, prgb);
}

__kernel void rgba_to_bgr(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    __global uchar* prgb = advance_uchar_ptr(rgb, row * strideOut);
    uchar4 i = in[top_left.y * strideI + top_left.x];
    vstore3(i.zyx, col, prgb);
}

__kernel void rgba_to_rgb(int width, int height, int strideIn, int strideOut,
    __global const uchar4 *in, __global uchar *rgb) {

    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= height || col >= width) {
        return;
    }

    int2 top_left = get_input_coords(row, col, width, height);
    int strideI = strideIn / 4;   //  Stride is in bytes, but pointer is uchar4
    __global uchar* prgb = advance_uchar_ptr(rgb, row * strideOut);
    uchar4 i = in[top_left.y * strideI + top_left.x];
    vstore3(i.xyz, col, prgb);
}

)##";

using ax_utils::buffer_details;
using ax_utils::CLProgram;

class CLColorConvert
{
  public:
  using buffer = CLProgram::ax_buffer;
  using kernel = CLProgram::ax_kernel;

  CLColorConvert(std::string source, int flip_type, void *display, Ax::Logger &logger)
      : program(ax_utils::get_kernel_utils(flip_type) + source, display, logger),
        nv12_to_rgba_image{ program.get_kernel("nv12_to_rgb_image") },
        nv12_to_rgba{ program.get_kernel("nv12_to_rgba") },
        i420_to_rgba{ program.get_kernel("i420_to_rgba") },
        YUYV_to_rgba{ program.get_kernel("YUYV_to_rgba") }, //
        nv12_to_rgb{ program.get_kernel("nv12_to_rgb") }, //
        i420_to_rgb{ program.get_kernel("i420_to_rgb") }, //
        YUYV_to_rgb{ program.get_kernel("YUYV_to_rgb") }, //
        bgra_to_rgba{ program.get_kernel("bgra_to_rgba") }, //
        rgba_to_rgb{ program.get_kernel("rgba_to_rgb") }, //
        rgba_to_bgr{ program.get_kernel("rgba_to_bgr") }, //
        bgr_to_rgb{ program.get_kernel("bgr_to_rgb") }
  {
  }

  CLProgram::flush_details run_kernel(kernel &k, const buffer_details &out, buffer &outbuf)
  {
    size_t global_work_size[3] = { 1, 1, 1 };
    const int numpix_per_kernel = 1;
    global_work_size[0] = out.width;
    global_work_size[1] = out.height;
    error = program.execute_kernel(k, 2, global_work_size);
    if (error != CL_SUCCESS) {
      return {};
    }
    return program.flush_output_buffer_async(outbuf, ax_utils::determine_buffer_size(out));
  }

  std::function<void()> run_kernel(
      kernel &k, const buffer_details &out, buffer &inbuf, buffer &outbuf)
  {
    auto [error, event, mapped] = run_kernel(k, out, outbuf);
    if (error != CL_SUCCESS) {
      throw std::runtime_error(
          "Unable to map output buffer, error = " + std::to_string(error));
    }
    return [this, event, mapped, inbuf, outbuf]() {
      program.unmap_buffer(event, outbuf, mapped);
    };
  }

  std::function<void()> run_kernel(kernel &k, const buffer_details &out,
      buffer &inbuf1, buffer &inbuf2, buffer &outbuf, const cl_extensions extensions)
  {
    std::array input_buffers = { *inbuf1, *inbuf2 };
    program.acquireva(input_buffers);
    auto [error, event, mapped] = run_kernel(k, out, outbuf);
    if (error != CL_SUCCESS) {
      throw std::runtime_error(
          "Unable to map output buffer, error = " + std::to_string(error));
    }
    return [this, extensions, event, mapped, inbuf1, inbuf2, outbuf]() {
      program.unmap_buffer(event, outbuf, mapped);
      std::array input_buffers = { *inbuf1, *inbuf2 };
      program.releaseva(input_buffers);
    };
  }

  std::function<void()> run_nv12_to_rgba(const buffer_details &in,
      const buffer_details &out, cl_char is_bgr, const cl_extensions &extensions)
  {
    const int rgba_size = 4;
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);
    auto inpbuf = program.create_buffers(1, ax_utils::determine_buffer_size(in),
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, in.data, in.offsets.size());

    if (inpbuf.size() == 1) {
      cl_int y_stride = in.strides[0];
      cl_int uv_stride = in.strides[1];
      cl_int uv_offset = in.offsets[1];
      //  Set the kernel arguments
      auto kernel = AxVideoFormatNumChannels(out.format) == 3 ? nv12_to_rgb : nv12_to_rgba;
      program.set_kernel_args(kernel, 0, out.width, out.height, y_stride,
          uv_stride, out.stride, uv_offset, is_bgr, *inpbuf[0], *outbuf);
      return run_kernel(kernel, out, inpbuf[0], outbuf);
    }

    program.set_kernel_args(nv12_to_rgba_image, 0, *inpbuf[0], *inpbuf[1],
        *outbuf, in.width, out.stride / rgba_size, in.height, is_bgr);
    return run_kernel(nv12_to_rgba_image, out, inpbuf[0], inpbuf[1], outbuf, extensions);
  }

  std::function<void()> run_i420_to_rgba(
      const buffer_details &in, const buffer_details &out, cl_char is_bgr)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_int y_stride = in.strides[0];
    cl_int u_stride = in.strides[1];
    cl_int v_stride = in.strides[2];
    cl_int u_offset = in.offsets[1];
    cl_int v_offset = in.offsets[2];
    //  Set the kernel arguments
    auto kernel = AxVideoFormatNumChannels(out.format) == 3 ? i420_to_rgb : i420_to_rgba;
    program.set_kernel_args(kernel, 0, out.width, out.height, y_stride, u_stride,
        v_stride, out.stride, u_offset, v_offset, is_bgr, *inpbuf, *outbuf);
    return run_kernel(kernel, out, inpbuf, outbuf);
  }

  std::function<void()> run_YUYV_to_rgba(
      const buffer_details &in, const buffer_details &out, cl_char is_bgr)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    cl_int y_stride = in.strides[0];
    //  Set the kernel arguments
    auto kernel = AxVideoFormatNumChannels(out.format) == 3 ? YUYV_to_rgb : YUYV_to_rgba;
    program.set_kernel_args(kernel, 0, out.width, out.height, y_stride,
        out.stride, is_bgr, *inpbuf, *outbuf);
    return run_kernel(kernel, out, inpbuf, outbuf);
  }

  std::function<void()> run_bgra_to_rgba(const buffer_details &in, const buffer_details &out)
  {
    auto inpbuf = program.create_buffer(in, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    auto outbuf = program.create_buffer(out, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    auto kernel = bgra_to_rgba;
    if ((in.format == AxVideoFormat::RGBA && out.format == AxVideoFormat::RGB)
        || (in.format == AxVideoFormat::BGRA && out.format == AxVideoFormat::BGR)) {
      kernel = rgba_to_rgb;
    } else if ((in.format == AxVideoFormat::RGBA && out.format == AxVideoFormat::BGR)
               || (in.format == AxVideoFormat::BGRA && out.format == AxVideoFormat::RGB)) {
      kernel = rgba_to_bgr;
    }
    cl_int y_stride = in.stride;
    //  Set the kernel arguments
    program.set_kernel_args(kernel, 0, out.width, out.height, y_stride,
        out.stride, *inpbuf, *outbuf);
    return run_kernel(kernel, out, inpbuf, outbuf);
  }

  std::function<void()> run(const buffer_details &in, const buffer_details &out,
      const std::string &format)
  {
    cl_char is_bgr = format == "bgra" || format == "bgr";
    if (in.format == AxVideoFormat::NV12) {
      return run_nv12_to_rgba(in, out, is_bgr, program.extensions);
    } else if (in.format == AxVideoFormat::I420) {
      return run_i420_to_rgba(in, out, is_bgr);
    } else if (in.format == AxVideoFormat::YUY2) {
      return run_YUYV_to_rgba(in, out, is_bgr);
    } else if (in.format == AxVideoFormat::RGBA || in.format == AxVideoFormat::BGRA) {
      return run_bgra_to_rgba(in, out);
    }
    return {};
  }

  bool can_use_dmabuf() const
  {
    return program.can_use_dmabuf();
  }

  bool can_use_vaapi() const
  {
    //  Once we enable this, just replace the return false with the following
    //  return program.can_use_va();
    return false;
  }

  private:
  CLProgram program;
  int error{};
  kernel nv12_to_rgba_image;
  kernel nv12_to_rgba;
  kernel i420_to_rgba;
  kernel YUYV_to_rgba;
  kernel nv12_to_rgb;
  kernel i420_to_rgb;
  kernel YUYV_to_rgb;
  kernel bgra_to_rgba;
  kernel rgba_to_rgb;
  kernel rgba_to_bgr;
  kernel bgr_to_rgb;
};

struct cc_properties {
  std::string format{ "rgba" };
  std::string flip_method{ "none" };
  mutable std::unique_ptr<CLColorConvert> color_convert;
  mutable int total_time{};
  mutable int num_calls{};
};

std::string_view flips[] = {
  "none",
  "clockwise",
  "rotate-180",
  "counterclockwise",
  "horizontal-flip",
  "vertical-flip",
  "upper-left-diagonal",
  "upper-right-diagonal",
};

int
determine_flip_type(std::string_view flip)
{
  auto it = std::find(std::begin(flips), std::end(flips), flip);
  return it != std::end(flips) ? std::distance(std::begin(flips), it) : -1;
}

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "format",
    "flip_method",
  };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  void *display = nullptr;
  auto prop = std::make_shared<cc_properties>();
  prop->format = Ax::get_property(input, "format", "ColorConvertProperties", prop->format);
  prop->flip_method = Ax::get_property(
      input, "flip_method", "ColorConvertProperties", prop->flip_method);
  auto flip_type = determine_flip_type(prop->flip_method);
  if (flip_type == -1) {
    logger(AX_ERROR) << "Invalid flip_method type: " << prop->flip_method
                     << " defaulting to none" << std::endl;
    flip_type = 0;
  }
  prop->color_convert
      = std::make_unique<CLColorConvert>(kernel_cl, flip_type, display, logger);
  return prop;
}

extern "C" void
set_dynamic_properties(const std::unordered_map<std::string, std::string> & /*input*/,
    cc_properties * /*prop*/, Ax::Logger & /*logger*/)
{
}

bool
is_a_rotate(std::string_view flip_type)
{
  static std::string_view rotates[] = {
    "clockwise",
    "counterclockwise",
    "upper-left-diagonal",
    "upper-right-diagonal",
  };
  return std::find(std::begin(rotates), std::end(rotates), flip_type) != std::end(rotates);
}

struct {
  std::string color;
  AxVideoFormat format;
} valid_formats[] = {
  { "rgba", AxVideoFormat::RGBA },
  { "bgra", AxVideoFormat::BGRA },
  { "rgb", AxVideoFormat::RGB },
  { "bgr", AxVideoFormat::BGR },
};

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const cc_properties *prop, Ax::Logger &logger)
{
  AxDataInterface output{};
  if (std::holds_alternative<AxVideoInterface>(interface)) {
    auto in_info = std::get<AxVideoInterface>(interface);
    auto out_info = in_info;
    if (is_a_rotate(prop->flip_method)) {
      std::swap(out_info.info.width, out_info.info.height);
      out_info.info.stride = out_info.info.width * 4;
      out_info.strides = { size_t(out_info.info.stride) };
    };
    auto fmt_found = std::find_if(std::begin(valid_formats), std::end(valid_formats),
        [fmt = prop->format](auto f) { return f.color == fmt; });
    if (fmt_found == std::end(valid_formats)) {
      logger(AX_ERROR)
          << "Invalid output format given in color conversion: " << prop->format
          << std::endl;
      throw std::runtime_error(
          "Invalid output format given in color conversion: " + prop->format);
    }
    logger(AX_INFO) << "Setting output format to " << prop->format << std::endl;
    out_info.info.format = fmt_found->format;
    output = out_info;
  }
  return output;
}

/// @brief  Check if the plugin has any work to do
/// @param input
/// @param output
/// @param logger
/// @return true if the plugin can pass through the input to output
extern "C" bool
can_passthrough(const AxDataInterface &input, const AxDataInterface &output,
    const cc_properties *prop, Ax::Logger &logger)
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

  return input_details[0].format == output_details[0].format
         && input_details[0].width == output_details[0].width
         && input_details[0].height == output_details[0].height;
}


extern "C" std::function<void()>
transform_async(const AxDataInterface &input, const AxDataInterface &output,
    const cc_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  //  These must be video interfaces as we have already checked in can_passthrough
  auto in_info = std::get<AxVideoInterface>(input);
  auto out_info = std::get<AxVideoInterface>(output);

  //  Validate input and output formats

  auto input_details = ax_utils::extract_buffer_details(input);
  if (input_details.size() != 1) {
    throw std::runtime_error("color_convert works on single tensor (possibly batched) input only");
  }

  auto output_details = ax_utils::extract_buffer_details(output);
  if (output_details.size() != 1) {
    throw std::runtime_error(
        "color_convert works on single tensor (possibly batched) output only");
  }
  if (std::holds_alternative<void *>(input_details[0].data)) {
    const int pagesize = 4096;
    auto ptr = std::get<void *>(input_details[0].data);
    if ((reinterpret_cast<uintptr_t>(ptr) & (pagesize - 1)) != 0) {
      logger(AX_DEBUG) << "Input buffer is not page aligned" << std::endl;
    }
  }

  return prop->color_convert->run(input_details[0], output_details[0], prop->format);
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const cc_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &meta_map,
    Ax::Logger &logger)
{
  logger(AX_WARN) << "Running in synchronous mode, possible performance degradation"
                  << std::endl;
  auto completer = transform_async(input, output, prop, 0, 0, meta_map, logger);
  completer();
}

extern "C" bool
can_use_dmabuf(const cc_properties *prop, Ax::Logger &logger)
{
  return prop->color_convert->can_use_dmabuf();
}

extern "C" bool
can_use_vaapi(const cc_properties *prop, Ax::Logger &logger)
{
  return prop->color_convert->can_use_vaapi();
}
