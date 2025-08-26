#include "unittest_transform_common.h"

#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace
{
bool has_opencl_platform = [] {
  cl_platform_id platformId;
  cl_uint numPlatforms;

  auto error = clGetPlatformIDs(1, &platformId, &numPlatforms);
  if (error == CL_SUCCESS) {
    cl_uint num_devices = 0;
    cl_device_id device_id;
    error = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
  }
  return error == CL_SUCCESS;
}();

class ColorFormatFixture : public ::testing::TestWithParam<FormatParam>
{
};

INSTANTIATE_TEST_SUITE_P(PerspectiveTestSuite, ColorFormatFixture,
    ::testing::Values(
        FormatParam{ AxVideoFormat::RGB, 3 }, FormatParam{ AxVideoFormat::BGR, 4 },
        FormatParam{ AxVideoFormat::RGB, 5 }, FormatParam{ AxVideoFormat::GRAY8, 5 }/*,
        FormatParam{ AxVideoFormat::I420, 0 }, FormatParam{ AxVideoFormat::YUY2, 0 }*/ ));

TEST_P(ColorFormatFixture, color_fusing_test)
{
  FormatParam format = GetParam();
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "matrix", "1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0" },
    { "out_format", std::to_string(format.out_format) },
  };

  Transformer perspective("libtransform_perspective_cl.so", input);

  auto out_bbp = (format.out_format == 5) ? 1 : 3;
  auto in_bbp = (format.format == AxVideoFormat::GRAY8) ? 1 : 3;
  std::vector<int8_t> in_buf(1920 * 1080 * in_bbp, 0);
  std::vector<int8_t> out_buf(1920 * 1080 * out_bbp, 0);
  std::iota(in_buf.begin(), in_buf.end(), 1);

  std::vector<size_t> strides;
  std::vector<size_t> offsets;
  if (format.format == AxVideoFormat::NV12) {
    strides = { 1920, 1920 };
    offsets = { 0, 1920 * 1080 };
  } else if (format.format == AxVideoFormat::I420) {
    strides = { 1920, 1920 / 2, 1920 / 2 };
    offsets = { 0, 1920 * 1080, 1920 * 1080 * 5 / 4 };
  } else if (format.format == AxVideoFormat::YUY2) {
    strides = { 1920 * 2 };
    offsets = { 0 };
  } else if (format.format == AxVideoFormat::RGB || format.format == AxVideoFormat::BGR) {
    strides = { 1920 * 3 };
    offsets = { 0 };
  } else if (format.format == AxVideoFormat::GRAY8) {
    strides = { 1920 };
    offsets = { 0 };
  } else {
    strides = { 1920 * 4 };
    offsets = { 0 };
  }

  auto in = AxVideoInterface{ { 1920, 1080, int(strides[0]), 0, format.format },
    in_buf.data(), strides, offsets, -1 };

  auto out = AxVideoInterface{ { 1920, 1080, 1920 * out_bbp, 0,
                                   format.out_format == 4 ? AxVideoFormat::BGR :
                                   format.out_format == 5 ? AxVideoFormat::GRAY8 :
                                                            AxVideoFormat::RGB },
    out_buf.data(), { 1920 * static_cast<size_t>(out_bbp) }, { 0 }, -1 };

  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  EXPECT_NO_THROW({ perspective.transform(in, out, metadata, 0, 1); });
  if (format.format == AxVideoFormat::RGB && format.out_format == 3) {
    EXPECT_EQ(in_buf, out_buf);
  } else if (format.format == AxVideoFormat::BGR && format.out_format == 4) {
    EXPECT_EQ(in_buf, out_buf);
  } else if (format.format == AxVideoFormat::GRAY8 && format.out_format == 5) {
    EXPECT_EQ(in_buf, out_buf);
  } else {
    EXPECT_NE(in_buf, out_buf);
  }
}

TEST(perspective_cl, identity_test)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "matrix", "1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0" },
  };

  Transformer perspective("libtransform_perspective_cl.so", input);
  std::vector<int8_t> in_buf(16 * 16 * 3);
  std::iota(in_buf.begin(), in_buf.end(), 1);
  std::vector<int8_t> out_buf(16 * 16 * 3, 0);

  auto in = AxVideoInterface{ { 16, 16, 16 * 3, 0, AxVideoFormat::RGB }, in_buf.data() };
  auto out
      = AxVideoInterface{ { 16, 16, 16 * 3, 0, AxVideoFormat::RGB }, out_buf.data() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  EXPECT_NO_THROW({ perspective.transform(in, out, metadata, 0, 1); });

  EXPECT_TRUE(in_buf == out_buf);
}

TEST(perspective_cl, translation_test)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "matrix", "1.0,0.0,-2.0,0.0,1.0,-2.0,0.0,0.0,1.0" },
  };

  Transformer perspective("libtransform_perspective_cl.so", input);
  std::vector<uint8_t> in_buf(4 * 4 * 3);
  std::iota(in_buf.begin(), in_buf.end(), 1);
  std::vector<uint8_t> out_buf(4 * 4 * 3, 0);

  auto in = AxVideoInterface{ { 4, 4, 4 * 3, 0, AxVideoFormat::RGB }, in_buf.data() };
  auto out = AxVideoInterface{ { 4, 4, 4 * 3, 0, AxVideoFormat::RGB }, out_buf.data() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  EXPECT_NO_THROW({ perspective.transform(in, out, metadata, 0, 1); });

  auto expected = std::vector<uint8_t>{
    // clang-format off
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,
     0,  0,  0,  0,  0,  0, 13, 14, 15, 16, 17, 18,
    // clang-format on
  };
  EXPECT_EQ(out_buf, expected);
}

TEST(perspective_cl, happy_path_test)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "matrix", "2.0,0.0,-860,0.0,2.0,-540,0.0,0.0,1.0" },
  };

  Transformer perspective("libtransform_perspective_cl.so", input);
  std::vector<int8_t> in_buf(16 * 16 * 3);
  std::iota(in_buf.begin(), in_buf.end(), 1);
  std::vector<int8_t> out_buf(16 * 16 * 3, 0);

  auto in = AxVideoInterface{ { 16, 16, 16 * 3, 0, AxVideoFormat::RGB }, in_buf.data() };
  auto out
      = AxVideoInterface{ { 16, 16, 16 * 3, 0, AxVideoFormat::RGB }, out_buf.data() };
  std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> metadata;
  EXPECT_NO_THROW({ perspective.transform(in, out, metadata, 0, 1); });

  EXPECT_TRUE(in_buf != out_buf);
}

TEST(perspective_cl, invalid_matrix_test)
{
  if (!has_opencl_platform) {
    GTEST_SKIP();
  }
  std::unordered_map<std::string, std::string> input = {
    { "matrix", "1.0,0.0,0.0,0.0,0.0,1.0" },
  };

  EXPECT_THROW(Transformer perspective("libtransform_perspective_cl.so", input),
      std::runtime_error);
}

} // namespace
