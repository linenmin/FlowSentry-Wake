// Copyright Axelera AI, 2025
#include <algorithm>
#include <gmock/gmock.h>
#include "unittest_transform_common.h"

TEST(transform_paddingdequantize, non_tensor_input)
{
  Transformer transformer(
      "libtransform_paddingdequantize.so", { { "dequant_scale", "0" } });
  AxDataInterface inp_empty;
  EXPECT_THROW(transformer.set_output_interface(inp_empty), std::runtime_error);
  AxVideoInterface inp_video{ {}, nullptr };
  EXPECT_THROW(transformer.set_output_interface(inp_video), std::runtime_error);
}
TEST(transform_paddingdequantize, test_dequantize)
{
  std::unordered_map<std::string, std::string> input = {
    { "dequant_scale", "0.5,0.25" },
    { "dequant_zeropoint", "1,2" },
    { "transpose", "0" },
    { "padding", "0,0,0,0,0,0,0,0|0,0,0,0,0,0,0,0" },
  };
  Transformer transformer("libtransform_paddingdequantize.so", input);
  int size = 1 * 2 * 3 * 4 * 2;
  auto inp_data = std::vector<int8_t>(size);
  std::iota(inp_data.begin(), inp_data.begin() + 1 * 2 * 3 * 4, 0);
  std::iota(inp_data.begin() + 1 * 2 * 3 * 4, inp_data.end(), 0);
  auto out_data = std::vector<float>(size);
  AxTensorsInterface inp{ { { 1, 2, 3, 4 }, 1, inp_data.data() },
    { { 1, 2, 3, 4 }, 1, inp_data.data() + 1 * 2 * 3 * 4 } };

  auto out = std::get<AxTensorsInterface>(transformer.set_output_interface(inp));
  ASSERT_EQ(out.size(), 2);
  EXPECT_EQ(out[0].sizes, std::vector<int>({ 1, 2, 3, 4 }));
  EXPECT_EQ(out[0].bytes, 4);
  out[0].data = out_data.data();
  out[1].data = out_data.data() + 1 * 2 * 3 * 4;
  transformer.transform(inp, out);

  auto expected = std::vector<float>(size);
  std::iota(expected.begin(), expected.begin() + 1 * 2 * 3 * 4, 0);
  std::iota(expected.begin() + 1 * 2 * 3 * 4, expected.end(), 0);
  std::transform(expected.begin(), expected.begin() + 1 * 2 * 3 * 4,
      expected.begin(), [](auto x) { return 0.5 * (x - 1); });
  std::transform(expected.begin() + 1 * 2 * 3 * 4, expected.end(),
      expected.begin() + 1 * 2 * 3 * 4, [](auto x) { return 0.25 * (x - 2); });
  EXPECT_EQ(expected, out_data);
}

TEST(transform_paddingdequantize, test_transpose)
{
  std::unordered_map<std::string, std::string> input = {
    { "dequant_scale", "1" },
    { "dequant_zeropoint", "0" },
    { "transpose", "1" },
    { "padding", "0,0,0,0,0,0,0,0" },
  };
  Transformer transformer("libtransform_paddingdequantize.so", input);
  int size = 1 * 2 * 3 * 4;
  auto inp_data = std::vector<int8_t>(size);
  std::iota(inp_data.begin(), inp_data.end(), 0);
  auto out_data = std::vector<float>(size);
  AxTensorsInterface inp{ { { 1, 2, 3, 4 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(transformer.set_output_interface(inp));
  ASSERT_EQ(out.size(), 1);
  EXPECT_EQ(out[0].sizes, std::vector<int>({ 1, 4, 2, 3 }));
  EXPECT_EQ(out[0].bytes, 4);
  out[0].data = out_data.data();
  transformer.transform(inp, out);

  auto expected = std::vector<float>({ 0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21,
      2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23 });
  EXPECT_EQ(expected, out_data);
}

TEST(transform_paddingdequantize, test_padding_transform_dequantize)
{
  std::unordered_map<std::string, std::string> input = {
    { "dequant_scale", "0.5" },
    { "dequant_zeropoint", "1" },
    { "transpose", "1" },
    { "padding", "0,0,0,0,0,0,0,-1" },
  };
  Transformer transformer("libtransform_paddingdequantize.so", input);
  int size = 1 * 2 * 3 * 4;
  auto inp_data = std::vector<int8_t>(size);
  std::iota(inp_data.begin(), inp_data.end(), 0);
  auto out_data = std::vector<float>(1 * 3 * 2 * 3);
  AxTensorsInterface inp{ { { 1, 2, 3, 4 }, 1, inp_data.data() } };

  auto out = std::get<AxTensorsInterface>(transformer.set_output_interface(inp));
  ASSERT_EQ(out.size(), 1);
  EXPECT_EQ(out[0].sizes, std::vector<int>({ 1, 3, 2, 3 }));
  EXPECT_EQ(out[0].bytes, 4);
  out[0].data = out_data.data();
  transformer.transform(inp, out);

  auto expected = std::vector<float>({ -0.5, 1.5, 3.5, 5.5, 7.5, 9.5, 0, 2, 4,
      6, 8, 10, 0.5, 2.5, 4.5, 6.5, 8.5, 10.5 });
  EXPECT_EQ(expected, out_data);
}
