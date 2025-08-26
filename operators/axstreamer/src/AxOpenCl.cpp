// Copyright Axelera AI, 2025
#include "AxOpenCl.hpp"

#include <iostream>
#include <vector>


namespace ax_utils
{
struct local_size {
  size_t width;
  size_t height;
};
local_size
determine_local_work_size(size_t width, size_t height, size_t max_work_size)
{
  while (width * height > max_work_size) {
    if (width > height)
      width /= 2;
    else
      height /= 2;
  }
  return { width, height };
}


output_format
get_output_format(AxVideoFormat format, bool ignore_alpha)
{
  switch (format) {
    case AxVideoFormat::RGBA:
      return ignore_alpha ? RGB_OUTPUT : RGBA_OUTPUT;
    case AxVideoFormat::BGRA:
      return ignore_alpha ? BGR_OUTPUT : BGRA_OUTPUT;
    case AxVideoFormat::RGB:
      return RGB_OUTPUT;
    case AxVideoFormat::BGR:
      return BGR_OUTPUT;
    case AxVideoFormat::GRAY8:
      return GRAY_OUTPUT;
    default:
      throw std::runtime_error(
          "Unsupported output format: " + AxVideoFormatToString(format));
  }
}

std::mutex CLProgram::cl_mutex;

CLProgram::CLProgram(const std::string &source, void *display, Ax::Logger &log)
    : logger(log)
{
  std::lock_guard<std::mutex> lock(cl_mutex);

  // Get OpenCL preference from environment variable
  // AX_OPENCL_PREFERENCE can be: "INTEL", "CPU", "GPU", "AUTO" (default)
  std::string preference = std::getenv("AX_OPENCL_PREFERENCE") ?
                               std::getenv("AX_OPENCL_PREFERENCE") :
                               "AUTO";

  // Get all available platforms
  cl_uint numPlatforms;
  auto error = clGetPlatformIDs(0, nullptr, &numPlatforms);
  if (error != CL_SUCCESS) {
    logger.throw_error("OpenCL not functional: Failed to get platform count! Error = "
                       + std::to_string(error));
  }

  std::vector<cl_platform_id> platforms(numPlatforms);
  error = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  if (error != CL_SUCCESS) {
    logger.throw_error("OpenCL not functional: Failed to get platform IDs! Error = "
                       + std::to_string(error));
  }

  // Test platform functionality
  auto test_platform_functionality = [this](cl_platform_id platform) -> bool {
    cl_uint num_devices = 0;
    cl_device_id test_device;
    cl_int test_error
        = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &test_device, &num_devices);
    if (test_error != CL_SUCCESS || num_devices == 0) {
      return false;
    }

    // Try to create a context to verify the platform works
    cl_context test_context
        = clCreateContext(nullptr, 1, &test_device, nullptr, nullptr, &test_error);
    if (test_error != CL_SUCCESS) {
      return false;
    }
    clReleaseContext(test_context);
    return true;
  };

  cl_platform_id selected_platform = nullptr;
  std::string selected_platform_name;

  // Define preference order for each setting
  std::vector<std::string> preference_order;
  if (preference == "INTEL") {
    preference_order = { "Intel" };
  } else if (preference == "CPU") {
    preference_order = { "Intel", "Portable Computing Language" };
  } else if (preference == "GPU") {
    preference_order = { "NVIDIA" };
  } else { // AUTO or any other value
    preference_order = { "Intel", "Portable Computing Language", "NVIDIA" };
  }

  // Try platforms in preference order
  for (const std::string &preferred_platform : preference_order) {
    for (cl_platform_id platform : platforms) {
      char platform_name[256];
      clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
          platform_name, nullptr);
      std::string platform_str(platform_name);

      logger(AX_DEBUG) << "Checking OpenCL platform: " << platform_str << std::endl;

      if (platform_str.find(preferred_platform) != std::string::npos) {
        logger(AX_DEBUG) << "Testing platform functionality: " << platform_str << std::endl;

        if (test_platform_functionality(platform)) {
          selected_platform = platform;
          selected_platform_name = platform_str;
          logger(AX_INFO) << "Selected OpenCL platform: " << selected_platform_name
                          << " (preference: " << preference << ")" << std::endl;
          goto platform_selected; // Break out of nested loops
        } else {
          logger(AX_DEBUG) << "Platform " << platform_str
                           << " is not functional, skipping" << std::endl;
        }
      }
    }
  }

  // If no preferred platform works, handle based on preference type
  if (!selected_platform) {
    if (preference == "AUTO") {
      // For AUTO mode, try any functional platform in remaining order
      logger(AX_DEBUG) << "No preferred platform functional in AUTO mode, trying remaining platforms"
                       << std::endl;
      for (cl_platform_id platform : platforms) {
        char platform_name[256];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
            platform_name, nullptr);
        std::string platform_str(platform_name);

        if (test_platform_functionality(platform)) {
          selected_platform = platform;
          selected_platform_name = platform_str;
          logger(AX_INFO) << "Selected functional OpenCL platform: " << selected_platform_name
                          << " (AUTO mode)" << std::endl;
          break;
        }
      }
    } else {
      // For explicit preferences (INTEL, GPU, CPU), fail with clear error
      std::string available_platforms;
      for (cl_platform_id platform : platforms) {
        char platform_name[256];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
            platform_name, nullptr);
        if (!available_platforms.empty())
          available_platforms += ", ";
        available_platforms += platform_name;
      }

      logger.throw_error("Requested OpenCL platform '" + preference + "' is not available or functional. "
                         + "Available platforms: [" + available_platforms + "]. "
                         + "Use AX_OPENCL_PREFERENCE=AUTO for automatic selection.");
    }
  }

platform_selected:

  if (!selected_platform) {
    logger.throw_error("No functional OpenCL platform found. Available platforms may be installed but not working properly.");
  }

  extensions = init_extensions(selected_platform, display);

  cl_uint num_devices;
  error = get_device_id(selected_platform, &device_id, &num_devices, extensions);
  if (error != CL_SUCCESS) {
    logger.throw_error("OpenCL not functional: Failed to get device! Error = "
                       + std::to_string(error));
  }

  context = create_context(selected_platform, device_id, extensions);
  if (error != CL_SUCCESS) {
    logger.throw_error("OpenCL not functional: Failed to create OpenCL context, error = "
                       + std::to_string(error));
  }
  commands = clCreateCommandQueue(context, device_id, 0, &error);
  if (error != CL_SUCCESS) {
    logger.throw_error("OpenCL not functional: Failed to create OpenCL command queue, error = "
                       + std::to_string(error));
  }

  const char *sources[] = { source.c_str() };
  program = error == CL_SUCCESS ?
                clCreateProgramWithSource(context, 1, sources, NULL, &error) :
                cl_program{};
  error = error == CL_SUCCESS ? clBuildProgram(program, 0, NULL, NULL, NULL, NULL) : error;
  if (error != CL_SUCCESS) {
    size_t param_value_size_ret;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
        &param_value_size_ret);
    std::vector<char> build_log(param_value_size_ret + 1);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
        param_value_size_ret, build_log.data(), NULL);
    std::cerr << "Build log:\n" << build_log.data() << std::endl;
    throw std::runtime_error("Failed to create OpenCL program");
  }
  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
      sizeof max_work_group_size, &max_work_group_size, NULL);
}

CLProgram::ax_kernel
CLProgram::get_kernel(const std::string &kernel_name) const
{
  int error = CL_SUCCESS;
  auto kernel = clCreateKernel(program, kernel_name.c_str(), &error);
  if (error != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL kernel " + kernel_name
                             + ", error = " + std::to_string(error));
  }
  return ax_kernel{ kernel };
}

std::vector<CLProgram::ax_buffer>
CLProgram::create_buffers(int elem_size, int num_elems, int flags,
    const std::variant<void *, int, VASurfaceID_proxy *> &ptr, int num_planes) const
{
  cl_int error = CL_SUCCESS;
  auto buffers = create_optimal_buffer(
      context, extensions, elem_size, num_elems, flags, ptr, num_planes, error);
  auto wrapped_buffers = std::vector<ax_buffer>{};
  std::transform(buffers.begin(), buffers.end(), std::back_inserter(wrapped_buffers),
      [this](cl_mem buffer) { return ax_buffer{ buffer }; });
  return wrapped_buffers;
}

CLProgram::ax_buffer
CLProgram::create_buffer(int elem_size, int num_elems, int flags,
    const std::variant<void *, int, VASurfaceID_proxy *> &ptr, int num_planes) const
{
  return create_buffers(elem_size, num_elems, flags, ptr, num_planes)[0];
}

CLProgram::ax_buffer
CLProgram::create_buffer(const buffer_details &details, int flags)
{
  return create_buffer(1, ax_utils::determine_buffer_size(details), flags,
      details.data, details.offsets.size());
}

int
CLProgram::write_buffer(const ax_buffer &buffer, int elem_size, int num_elems, const void *data)
{
  return clEnqueueWriteBuffer(
      commands, *buffer, CL_TRUE, 0, elem_size * num_elems, data, 0, NULL, NULL);
}

int
CLProgram::read_buffer(const ax_buffer &buffer, int elem_size, int num_elems, void *data)
{
  return clEnqueueReadBuffer(
      commands, *buffer, CL_TRUE, 0, elem_size * num_elems, data, 0, NULL, NULL);
}

CLProgram::flush_details
CLProgram::flush_output_buffer_async(const ax_buffer &out, int size)
{
  int ret = CL_SUCCESS;
  auto event = cl_event{};
  auto mapped = clEnqueueMapBuffer(
      commands, *out, CL_FALSE, CL_MAP_READ, 0, size, 0, NULL, &event, &ret);
  return { ret, event, mapped };
}

int
CLProgram::unmap_buffer(const ax_buffer &out, void *mapped)
{
  auto ret = clEnqueueUnmapMemObject(commands, *out, mapped, 0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    throw std::runtime_error(
        "Failed to unmap output buffer, error = " + std::to_string(ret));
  }
  return ret;
}

int
CLProgram::unmap_buffer(cl_event event, const ax_buffer &out, void *mapped)
{
  auto ret = clWaitForEvents(1, &event);
  if (ret != CL_SUCCESS) {
    throw std::runtime_error("Failed to wait for event, error = " + std::to_string(ret));
  }
  clReleaseEvent(event);
  return unmap_buffer(out, mapped);
}

int
CLProgram::flush_output_buffer(const ax_buffer &out, int size)
{
  auto [result, event, mapped] = flush_output_buffer_async(out, size);
  if (result != CL_SUCCESS) {
    throw std::runtime_error(
        "Failed to map output buffer, error = " + std::to_string(result));
    return result;
  }

  int ret = clWaitForEvents(1, &event);
  return unmap_buffer(out, mapped);
}

int
CLProgram::acquireva(std::span<cl_mem> input_buffers)
{
  return acquire_va(commands, extensions, input_buffers);
}

int
CLProgram::releaseva(std::span<cl_mem> input_buffers)
{
  return release_va(commands, extensions, input_buffers);
}

int
CLProgram::execute_kernel(const ax_kernel &kernel, int num_dims, size_t global_work_size[3])
{
  //  TODO: determine local work size
  size_t local[3] = { 16, 16, 1 };
  size_t global[3] = { 0 };
  global[0] = (global_work_size[0] + local[0] - 1) & ~(local[0] - 1);
  global[1] = (global_work_size[1] + local[1] - 1) & ~(local[1] - 1);
  global[2] = global_work_size[2];
  return clEnqueueNDRangeKernel(
      commands, *kernel, num_dims, NULL, global, local, 0, NULL, NULL);
}

CLProgram::~CLProgram()
{
  if (program)
    clReleaseProgram(program);
  if (commands)
    clReleaseCommandQueue(commands);
  if (context)
    clReleaseContext(context);
}

std::string
get_kernel_utils(int rotate_type)
{

  std::string utils = R"##(

#define advance_uchar_ptr(ptr, offset) ((__global uchar *)ptr + offset)
#define advance_uchar2_ptr(ptr, offset) ((__global uchar2 *)((__global uchar *)ptr + offset))
#define advance_uchar3_ptr(ptr, offset) ((__global uchar3 *)((__global uchar *)ptr + offset))
#define advance_uchar4_ptr(ptr, offset) ((__global uchar4 *)((__global uchar *)ptr + offset))

typedef enum : int {
      RGBA_OUTPUT = 0,
      BGRA_OUTPUT = 1,
      RGB_OUTPUT = 3,
      BGR_OUTPUT = 4,
      GRAY_OUTPUT = 5
} output_format;


uchar RGB_to_GRAY(uchar3 rgb) {
    // Convert RGB to grayscale using the formula: Y = 0.299*R + 0.587*G + 0.114*B
    return (uchar)((rgb.x * 77 + rgb.y * 150 + rgb.z * 29) >> 8);
}


uchar3 YUV_to_RGB(uchar Y, uchar U, uchar V) {
    int C = (Y - 16) * 19071; // 1.164f * 16384
    int D = U - 128;
    int E = V - 128;

    // Integer approximation of YUV to RGB conversion
    int3 rgb;
    rgb.x = (C + (26149 * E)) >> 14;            // 1.596 * 16384
    rgb.y = (C - (6406 * D + 13320 * E)) >> 14; // 0.391F * 16384, 0.813 * 16384
    rgb.z = (C + (33063 * D)) >> 14;            // 2.018 * 16384
    return convert_uchar3_sat_rte(rgb);
}

//  Convert YUV values to RGBA
uchar4 convert_YUV2RGBA(float3 yuv) {

    float Y = yuv.x - 16.0f;
    float U = yuv.y - 128.0f;
    float V = yuv.z - 128.0f;

    Y *= 1.164f;
    float4 rgba = (float4)(Y + 1.596F * V, Y - 0.391F * U - 0.813F * V, Y + 2.018F * U, 255.0f);
    return convert_uchar4_sat_rte(rgba);
}

float3 bilinear(float3 p00, float3 p01, float3 p10, float3 p11, float xfrac, float yfrac) {
    float3 i1 = mad((p01 - p00), xfrac, p00);
    float3 i2 = mad((p11 - p10), xfrac, p10);
    return mad((i2 - i1), yfrac, i1);
}

typedef struct nv12_image {
    int width;
    int height;
    int ystride;
    int uvstride;
    int crop_x;
    int crop_y;
} nv12_image;

uchar4 nv12_sampler(__global const uchar *y_image, __global uchar2 *uv, float fx, float fy, const nv12_image *img) {
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;
  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  x1 = max(x1, 0);
  y1 = max(y1, 0);
  int x2 = min(x1 + 1,img->width - 1);
  int y2 = min(y1 + 1,img->height - 1);
  x1 += img->crop_x;
  y1 += img->crop_y;
  x2 += img->crop_x;
  y2 += img->crop_y;

  float y00 = convert_float(y_image[y1 * img->ystride + x1]);
  float y01 = convert_float(y_image[y1 * img->ystride + x2]);
  float y10 = convert_float(y_image[y2 * img->ystride + x1]);
  float y11 = convert_float(y_image[y2 * img->ystride + x2]);

  int ux1 = x1 / 2;
  int uy1 = y1 / 2;
  int ux2 = x2 / 2;
  int uy2 = y2 / 2;

  bool need_right = ux1 == ux2;
  bool need_bottom = uy1 == uy2;
#define NV12_READ(x, y, p, stride) convert_float2(p[y * stride + x])

  float2 uv00 = NV12_READ(ux1, uy1, uv, img->uvstride);
  float2 uv01 = need_right ? NV12_READ(ux2, uy1, uv, img->uvstride) : uv00;
  float2 uv10 = need_bottom ? NV12_READ(ux1, uy2, uv, img->uvstride) : uv00;
  float2 uv11 = need_right ? (need_bottom ? NV12_READ(ux2, uy2, uv, img->uvstride) : uv01) : uv10;

  float3 yuv0 = (float3)(y00, uv00);
  float3 yuv1 = (float3)(y01, uv01);
  float3 yuv2 = (float3)(y10, uv10);
  float3 yuv3 = (float3)(y11, uv11);

  float3 yuv = bilinear(yuv0, yuv1, yuv2, yuv3, xfrac, yfrac);
  return convert_YUV2RGBA(yuv);
}

typedef struct i420_image {
    int width;
    int height;
    int ystride;
    int ustride;
    int vstride;
    int crop_x;
    int crop_y;
} i420_image;

uchar4 i420_sampler(__global const uchar *y_image, __global const uchar *u, __global const uchar *v, float fx, float fy, const i420_image *img) {
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;
  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  x1 = max(x1, 0);
  y1 = max(y1, 0);
  int x2 = min(x1 + 1,img->width - 1);
  int y2 = min(y1 + 1,img->height - 1);
  x1 += img->crop_x;
  y1 += img->crop_y;
  x2 += img->crop_x;
  y2 += img->crop_y;

  float y00 = convert_float(y_image[y1 * img->ystride + x1]);
  float y01 = convert_float(y_image[y1 * img->ystride + x2]);
  float y10 = convert_float(y_image[y2 * img->ystride + x1]);
  float y11 = convert_float(y_image[y2 * img->ystride + x2]);

  int ux1 = x1 / 2;
  int uy1 = y1 / 2;
  int ux2 = x2 / 2;
  int uy2 = y2 / 2;

  bool need_right = ux1 == ux2;
  bool need_bottom = uy1 == uy2;

#define I420_READ(x, y, pu, pv, ustride, vstride) convert_float2((uchar2)(pu[y * ustride + x], pv[y * vstride + x]))

  float2 uv00 = I420_READ(ux1, uy1, u, v, img->ustride, img->vstride);
  float2 uv01 = need_right ? I420_READ(ux2, uy1, u, v, img->ustride, img->vstride) : uv00;
  float2 uv10 = need_bottom ? I420_READ(ux1, uy2, u, v, img->ustride, img->vstride) : uv00;
  float2 uv11 = need_right ? (need_bottom ? I420_READ(ux2, uy2, u, v, img->ustride, img->vstride) : uv01) : uv10;

  float3 yuv0 = (float3)(y00, uv00);
  float3 yuv1 = (float3)(y01, uv01);
  float3 yuv2 = (float3)(y10, uv10);
  float3 yuv3 = (float3)(y11, uv11);

  float3 yuv = bilinear(yuv0, yuv1, yuv2, yuv3, xfrac, yfrac);
  return convert_YUV2RGBA(yuv);
}


typedef struct yuyv_image {
    int width;
    int height;
    int stride;
    int crop_x;
    int crop_y;
} yuyv_image;

uchar4  yuyv_sampler(__global uchar4 *in, float fx, float fy, const yuyv_image *img) {
  float xpixel_left = fx - 0.5f;
  float ypixel_top = fy - 0.5f;

  int x1 = xpixel_left;
  int y1 = ypixel_top;
  float xfrac = xpixel_left - x1;
  float yfrac = ypixel_top - y1;

  int x2 = min(x1 + 1, img->width - 1);
  int y2 = min(y1 + 1, img->height - 1);

  x1 += img->crop_x;
  y1 += img->crop_y;
  x2 += img->crop_x;
  y2 += img->crop_y;

  // Each YUYV pixel pair is stored as Y1 U Y2 V
  int idx1 = y1 * img->stride + (x1 >> 1);
  int idx2 = y1 * img->stride + (x2 >> 1);
  int idx3 = y2 * img->stride + (x1 >> 1);
  int idx4 = y2 * img->stride + (x2 >> 1);

  float4 p00 = convert_float4(in[idx1]);
  float4 p01 = convert_float4(in[idx2]);
  float4 p10 = convert_float4(in[idx3]);
  float4 p11 = convert_float4(in[idx4]);

  // Select correct Y and UV values based on even/odd position
  float3 in00 = (x1 & 1) ? (float3)(p00.z, p00.y, p00.w) : (float3)(p00.x, p00.y, p00.w);
  float3 in01 = (x2 & 1) ? (float3)(p01.z, p01.y, p01.w) : (float3)(p01.x, p01.y, p01.w);
  float3 in10 = (x1 & 1) ? (float3)(p10.z, p10.y, p10.w) : (float3)(p10.x, p10.y, p10.w);
  float3 in11 = (x2 & 1) ? (float3)(p11.z, p11.y, p11.w) : (float3)(p11.x, p11.y, p11.w);

  float3 yuv = bilinear(in00, in01, in10, in11, xfrac, yfrac);
  return convert_YUV2RGBA(yuv);
}

typedef struct rgb_image {
    int width;
    int height;
    int stride;
    int crop_x;
    int crop_y;
} rgb_image;

typedef struct gray8_image {
    int width;
    int height;
    int stride;
    int crop_x;
    int crop_y;
} gray8_image;

uchar gray8_sampler_bl(__global const uchar *image, float fx, float fy, const gray8_image *img) {
    //  Here we add in the offsets to the pixel from the crop meta
    float xpixel_left = clamp(fx - 0.5f, 0.0f, (float)(img->width - 1));
    float ypixel_top = clamp(fy - 0.5f, 0.0f, (float)(img->height - 1));

    int x1 = (int)floor(xpixel_left);
    int y1 = (int)floor(ypixel_top);
    float xfrac = xpixel_left - x1;
    float yfrac = ypixel_top - y1;

    int x2 = min(x1 + 1, img->width - 1);
    int y2 = min(y1 + 1, img->height - 1);

    x1 += img->crop_x;
    y1 += img->crop_y;
    x2 += img->crop_x;
    y2 += img->crop_y;

    float p00 = (float)image[y1 * img->stride + x1];
    float p01 = (float)image[y1 * img->stride + x2];
    float p10 = (float)image[y2 * img->stride + x1];
    float p11 = (float)image[y2 * img->stride + x2];

    //  Performs bilinear interpolation with higher precision
    float i1 = mix(p00, p01, xfrac);
    float i2 = mix(p10, p11, xfrac);
    float value = mix(i1, i2, yfrac);

    return convert_uchar_sat_rte(value);
}
uchar4 rgba_sampler_bl(__global const uchar4 *image, float fx, float fy, const rgb_image *img ) {
    //  Here we add in the offsets to the pixel from the crop meta
    float xpixel_left = fx - 0.5f;
    float ypixel_top = fy - 0.5f;

    int x1 = xpixel_left;
    int y1 = ypixel_top;
    float xfrac = xpixel_left - x1;
    float yfrac = ypixel_top - y1;

    x1 = max(x1, 0);
    y1 = max(y1, 0);
    int x2 = min(x1 + 1,img->width - 1);
    int y2 = min(y1 + 1,img->height - 1);

    x1 += img->crop_x;
    y1 += img->crop_y;
    x2 += img->crop_x;
    y2 += img->crop_y;
    float4 p00 = convert_float4(image[y1 * img->stride + x1]);
    float4 p01 = convert_float4(image[y1 * img->stride + x2]);
    float4 p10 = convert_float4(image[y2 * img->stride + x1]);
    float4 p11 = convert_float4(image[y2 * img->stride + x2]);

    //  Performs bilinear interpolation
    //  frac is the fraction of the pixel that is color2
    //  color = color1 + (color2 - color1) * frac

    float4 i1 = mad((p01 - p00), xfrac, p00);
    float4 i2 = mad((p11 - p10), xfrac, p10);
    uchar4 result = convert_uchar4_sat_rte(mad((i2 - i1), yfrac, i1));
    return result;
}

uchar4 rgb_sampler_bl(__global const uchar *image, float fx, float fy, const rgb_image *img ) {
    //  Here we add in the offsets to the pixel from the crop meta
    float xpixel_left = fx - 0.5f;
    float ypixel_top = fy - 0.5f;

    int x1 = xpixel_left;
    int y1 = ypixel_top;
    float xfrac = xpixel_left - x1;
    float yfrac = ypixel_top - y1;

    x1 = max(x1, 0);
    y1 = max(y1, 0);
    int x2 = min(x1 + 1,img->width - 1);
    int y2 = min(y1 + 1,img->height - 1);

    x1 += img->crop_x;
    y1 += img->crop_y;
    x2 += img->crop_x;
    y2 += img->crop_y;
    __global const uchar * p_in = advance_uchar_ptr(image, y1 * img->stride);
    float3 p00 = convert_float3(vload3(x1, p_in));
    float3 p01 = convert_float3(vload3(x2, p_in));
    p_in = advance_uchar_ptr(image, y2 * img->stride);
    float3 p10 = convert_float3(vload3(x1, p_in));
    float3 p11 = convert_float3(vload3(x2, p_in));

    //  Performs bilinear interpolation
    //  frac is the fraction of the pixel that is color2
    //  color = color1 + (color2 - color1) * frac

    float3 i1 = mad((p01 - p00), xfrac, p00);
    float3 i2 = mad((p11 - p10), xfrac, p10);
    uchar4 result = (uchar4)(convert_uchar3_sat_rte(mad((i2 - i1), yfrac, i1)), 255);
    return result;
}

//  upper-right-diagonal
inline int2 get_input_coords_urd(int row, int col, int width, int height) {
  int new_x = (height - row) - 1;
  int new_y = (width - col) - 1;
  return (int2)(new_x, new_y);
}

//  upper-left-diagonal
inline int2 get_input_coords_uld(int row, int col, int width, int height) {
  int new_x = row;
  int new_y = col;
  return (int2)(new_x, new_y);
}


//  Rotate 90 clockwise
inline int2 get_input_coords_clockwise(int row, int col, int width, int height) {
  int new_x = row;
  int new_y = (width - col) - 1;
  return (int2)(new_x, new_y);
}

//  Rotate 90 counter clockwise
inline int2 get_input_coords_counter_clockwise(int row, int col, int width, int height) {
  int new_x = (height - row) - 1;
  int new_y = col;
  return (int2)(new_x, new_y);
}

//  Rotate 180
inline int2 get_input_coords_rotate180(int row, int col, int width, int height) {
  int new_x = (width - col) - 1;
  int new_y = (height - row) - 1;
  return (int2)(new_x, new_y);
}

//  Vertical flip
inline int2 get_input_coords_vertical(int row, int col, int width, int height) {
  int new_x = col;
  int new_y = (height - row) - 1;
  return (int2)(new_x, new_y);
}

//  Horizontal flip
inline int2 get_input_coords_horizontal(int row, int col, int width, int height) {
  int new_x = (width - col) - 1;
  int new_y = row;
  return (int2)(new_x, new_y);
}

//  Do nothing
inline int2 get_input_coords_none(int row, int col, int width, int height) {
  return (int2)(col, row);
}


)##";

  std::array xlate_names = {
    "get_input_coords_none"s,
    "get_input_coords_clockwise"s,
    "get_input_coords_rotate180"s,
    "get_input_coords_counter_clockwise"s,
    "get_input_coords_horizontal"s,
    "get_input_coords_vertical"s,
    "get_input_coords_uld"s,
    "get_input_coords_urd"s,
  };

  if (0 <= rotate_type && rotate_type < xlate_names.size()) {
    utils += "#define get_input_coords " + xlate_names[rotate_type];
  } else {
    utils += "#define get_input_coords " + xlate_names[0];
  }
  return utils;
}

} // namespace ax_utils
