// Copyright Axelera AI, 2025
#include "AxInference.hpp"
#include "AxStreamerUtils.hpp"

#include <algorithm>
#include <fstream>

#if defined(AXELERA_ENABLE_AXRUNTIME)
#include <axruntime/axruntime.hpp>

using namespace std::string_literals;
using axr::to_ptr;

namespace
{

template <size_t N>
void
fill_char_array(char (&arr)[N], const std::string &str)
{
  std::fill(std::begin(arr), std::end(arr), 0);
  std::copy(str.begin(), str.begin() + std::min(N - 1, str.size()), arr);
}

AxTensorInterface
to_axtensorinfo(const axrTensorInfo &info)
{
  AxTensorInterface tensor;
  tensor.sizes.assign(info.dims, info.dims + info.ndims);
  tensor.bytes = info.bits / 8;
  return tensor;
}

axr::ptr<axrProperties>
create_properties(axrContext *context, bool input_dmabuf, bool output_dmabuf,
    bool double_buffer, int num_sub_devices)
{
  std::string s;
  s += "input_dmabuf=" + std::to_string(int(input_dmabuf)) + "\n";
  s += "output_dmabuf=" + std::to_string(int(output_dmabuf)) + "\n";
  s += "num_sub_devices=" + std::to_string(num_sub_devices) + "\n";
  s += "aipu_cores=" + std::to_string(num_sub_devices) + "\n";
  s += "double_buffer=" + std::to_string(int(double_buffer)) + "\n";
  return to_ptr(axr_create_properties(context, s.c_str()));
}

axr::ptr<axrProperties>
create_conn_properties(axrContext *context)
{
  std::string s;
  s += "device_firmware_check=0"; // AF checks this further up
  return to_ptr(axr_create_properties(context, s.c_str()));
}

static std::mutex second_slice_workaround_mutex;

class AxRuntimeInference : public Ax::Inference
{
  public:
  AxRuntimeInference(Ax::Logger &logger, axrContext *ctx, axrModel *model,
      const Ax::InferenceProperties &props)
      : logger(logger)
  {
    // level-zero/triton/kmd has issues if we try to load the model from
    // multiple threads. So lock here to load a model at a time. This is a
    // workaround for the issue, and it needs fixing lower down.
    // Proper fix tracked here https://axeleraai.atlassian.net/browse/SDK-6708
    std::lock_guard lock(second_slice_workaround_mutex);
    auto inputs = axr_num_model_inputs(model);
    for (int n = 0; n != inputs; ++n) {
      input_shapes_.push_back(to_axtensorinfo(axr_get_model_input(model, n)));
    }
    input_args.resize(inputs);
    auto outputs = axr_num_model_outputs(model);
    for (int n = 0; n != outputs; ++n) {
      output_shapes_.push_back(to_axtensorinfo(axr_get_model_output(model, n)));
    }
    output_args.resize(outputs);
    logger(AX_INFO) << "Loaded model " << props.model << " with " << inputs
                    << " inputs and " << outputs << " outputs" << std::endl;

    auto device = axrDeviceInfo{};
    fill_char_array(device.name, props.devices);
    const auto *pdevice = props.devices.empty() ? nullptr : &device;
    // (batch_size is virtual so don't use it)
    const auto num_sub_devices = input_shapes_.front().sizes.front();
    const auto conn_props = create_conn_properties(ctx);
    connection = to_ptr(
        axr_device_connect(ctx, pdevice, num_sub_devices, conn_props.get()));
    if (!connection) {
      throw std::runtime_error(
          "axr_device_connect failed : "s + axr_last_error_string(AXR_OBJECT(ctx)));
    }
    const auto load_props = create_properties(ctx, props.dmabuf_inputs,
        props.dmabuf_outputs, props.double_buffer, num_sub_devices);
    instance
        = to_ptr(axr_load_model_instance(connection.get(), model, load_props.get()));
    if (!instance) {
      throw std::runtime_error("axr_load_model_instance failed : "s
                               + axr_last_error_string(AXR_OBJECT(ctx)));
    }
  }

  int batch_size() const override
  {
    return input_shapes_.front().sizes.front();
  }

  const AxTensorsInterface &input_shapes() const override
  {
    return input_shapes_;
  }

  const AxTensorsInterface &output_shapes() const override
  {
    return output_shapes_;
  }

  void dispatch(const std::vector<std::shared_ptr<void>> &input_ptrs,
      const std::vector<Ax::SharedFD> &input_fds,
      const std::vector<std::shared_ptr<void>> &output_ptrs,
      const std::vector<Ax::SharedFD> &output_fds) override
  {
    const auto use_fds = input_ptrs.empty();
    assert(use_fds ? input_fds.size() == input_shapes().size() :
                     input_ptrs.size() == input_shapes().size());
    assert(use_fds ? input_ptrs.size() == 0 : input_fds.size() == 0);
    if (use_fds) {
      for (auto &&[i, shared_fd] : Ax::Internal::enumerate(input_fds)) {
        input_args[i].fd = shared_fd->fd;
        input_args[i].ptr = nullptr;
        input_args[i].offset = 0;
      }
    } else {
      for (auto &&[i, ptr] : Ax::Internal::enumerate(input_ptrs)) {
        input_args[i].fd = 0;
        input_args[i].ptr = ptr.get();
        input_args[i].offset = 0;
      }
    }
    if (!output_ptrs.empty()) {
      assert(output_ptrs.size() == output_shapes().size());
      for (auto &&[i, ptr] : Ax::Internal::enumerate(output_ptrs)) {
        output_args[i].fd = 0;
        output_args[i].ptr = ptr.get();
        output_args[i].offset = 0;
      }
    } else if (!output_fds.empty()) {
      assert(output_fds.size() == output_shapes().size());
      int i = 0;
      for (auto &&[i, shared_fd] : Ax::Internal::enumerate(output_fds)) {
        output_args[i].fd = shared_fd->fd;
        output_args[i].ptr = nullptr;
        output_args[i].offset = 0;
      }
    }
    auto res = axr_run_model_instance(instance.get(), input_args.data(),
        input_args.size(), output_args.data(), output_args.size());
    if (res != AXR_SUCCESS) {
      throw std::runtime_error("axr_run_model failed with "s
                               + axr_last_error_string(AXR_OBJECT(instance.get())));
    }
  }

  void collect(const std::vector<std::shared_ptr<void>> &output_ptrs) override
  {
  }

  private:
  Ax::Logger &logger;
  axr::ptr<axrConnection> connection;
  axr::ptr<axrModelInstance> instance;
  std::vector<axrArgument> input_args;
  std::vector<axrArgument> output_args;
  AxTensorsInterface input_shapes_;
  AxTensorsInterface output_shapes_;
};
} // namespace

std::unique_ptr<Ax::Inference>
Ax::create_axruntime_inference(Ax::Logger &logger, axrContext *ctx,
    axrModel *model, const InferenceProperties &props)
{
  return std::make_unique<AxRuntimeInference>(logger, ctx, model, props);
}
#else
std::unique_ptr<Ax::Inference>
Ax::create_axruntime_inference(Ax::Logger &logger, axrContext *ctx,
    axrModel *model, const InferenceProperties &props)
{
  throw std::runtime_error("Axelera AI runtime not installed at compile time");
}
#endif
