// Copyright Axelera AI, 2025
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric> // For std::iota
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOnnxRuntimeHelper.hpp" // Ensure this is included
#include "AxUtils.hpp"

// Properties for the postamble processor
struct postamble_properties {
  std::string onnx_path{}; // Path to ONNX model for postamble processing
  std::vector<int> tensor_selection_plan{}; // Pre-calculated indices of which tensors to use as ONNX inputs
  std::unique_ptr<ax_onnxruntime::OnnxRuntimeInference> onnx_runtime_; // ONNX runtime engine
};

// Define allowed properties that can be set from Python
extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{
    "onnx_path",
    "tensor_selection_plan", // Format: comma-separated integers "0,2,5"
  };
  return allowed_properties;
}

// Helper for logging shapes
std::string
shape_to_string(const std::vector<int64_t> &shape)
{
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    ss << shape[i];
    if (i < shape.size() - 1)
      ss << ",";
  }
  ss << "]";
  return ss.str();
}
// Overload for int shapes
std::string
shape_to_string(const std::vector<int> &shape)
{
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    ss << shape[i];
    if (i < shape.size() - 1)
      ss << ",";
  }
  ss << "]";
  return ss.str();
}


// Initialize properties from configuration
extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<postamble_properties> prop = std::make_shared<postamble_properties>();

  // Get ONNX model path
  prop->onnx_path
      = Ax::get_property(input, "onnx_path", "transform_postamble", std::string{});

  // Get tensor selection plan (which tensors to use for ONNX inference)
  prop->tensor_selection_plan = Ax::get_property(
      input, "tensor_selection_plan", "transform_postamble", std::vector<int>{});

  if (!prop->tensor_selection_plan.empty()) {
    std::stringstream ss;
    ss << "Tensor selection plan: ";
    for (size_t i = 0; i < prop->tensor_selection_plan.size(); ++i) {
      ss << prop->tensor_selection_plan[i];
      if (i < prop->tensor_selection_plan.size() - 1)
        ss << ", ";
    }
    logger(AX_INFO) << ss.str();
  }

  // Initialize ONNX runtime if path is provided
  if (!prop->onnx_path.empty()) {
    try {
      // Initialize ONNX Runtime
      prop->onnx_runtime_ = std::make_unique<ax_onnxruntime::OnnxRuntimeInference>(
          prop->onnx_path, logger);
      logger(AX_INFO) << "Initialized ONNX Runtime for postamble: " << prop->onnx_path;

      // Log ONNX model information
      const auto &input_names = prop->onnx_runtime_->get_input_node_names();
      const auto &input_dims = prop->onnx_runtime_->get_input_node_dims();
      const auto &output_names = prop->onnx_runtime_->get_output_node_names();
      const auto &output_dims = prop->onnx_runtime_->get_output_node_dims(); // Get expected output dims


      logger(AX_INFO) << "ONNX model has " << input_names.size()
                      << " inputs and " << output_names.size() << " outputs";

      // If tensor selection plan is empty, use default (first N tensors)
      if (prop->tensor_selection_plan.empty()) {
        logger(AX_INFO) << "No tensor selection plan provided, will use first "
                        << input_names.size() << " tensors as input";
        prop->tensor_selection_plan.resize(input_names.size());
        std::iota(prop->tensor_selection_plan.begin(),
            prop->tensor_selection_plan.end(), 0);
      } else if (prop->tensor_selection_plan.size() != input_names.size()) {
        // This is now an error, not just a warning, for I/O Binding consistency
        logger(AX_ERROR)
            << "Tensor selection plan has " << prop->tensor_selection_plan.size()
            << " indices but ONNX model requires " << input_names.size() << " inputs.";
        throw std::runtime_error("Mismatch between tensor selection plan and ONNX model inputs");
      }

      // Log input shapes for debugging
      for (size_t i = 0; i < input_names.size(); ++i) {
        const auto &dims = input_dims[i];
        logger(AX_INFO) << "ONNX input " << i << " (" << input_names[i]
                        << "): Expected Shape " << shape_to_string(dims);
      }
      // Log output shapes for debugging
      for (size_t i = 0; i < output_names.size(); ++i) {
        const auto &dims = output_dims[i];
        logger(AX_INFO) << "ONNX output " << i << " (" << output_names[i]
                        << "): Expected Shape " << shape_to_string(dims);
      }

    } catch (const std::exception &e) {
      logger(AX_ERROR) << "Failed to initialize ONNX Runtime: " << e.what();
      throw;
    }
  } else {
    logger(AX_WARN) << "No ONNX model path provided. This transform will pass through tensors unchanged.";
  }

  return prop;
}

// Set output interface based on ONNX model and input tensors
extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const postamble_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxTensorsInterface>(interface)) {
    throw std::runtime_error("transform_postamble works on tensor input only");
  }

  // If we have no ONNX model, output interface is the same as input
  if (!prop->onnx_runtime_) {
    logger(AX_INFO) << "No ONNX runtime, output interface matches input.";
    return interface;
  }

  try {
    const auto &input_tensors = std::get<AxTensorsInterface>(interface);

    // Get expected output shapes from the initialized ONNX runtime
    const auto &output_names = prop->onnx_runtime_->get_output_node_names();
    const auto &output_dims = prop->onnx_runtime_->get_output_node_dims(); // Use expected dims
    size_t num_onnx_outputs = output_names.size();

    // Create a set of used input tensor indices for quick lookup
    std::unordered_set<int> used_indices(
        prop->tensor_selection_plan.begin(), prop->tensor_selection_plan.end());

    // Count unused input tensors
    size_t num_unused_inputs = 0;
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      if (used_indices.find(i) == used_indices.end()) {
        num_unused_inputs++;
      }
    }

    // Create output interface with the right number of tensors
    AxTensorsInterface output_tensors;
    output_tensors.resize(num_onnx_outputs + num_unused_inputs);
    logger(AX_INFO) << "Setting output interface: " << num_onnx_outputs
                    << " ONNX outputs + " << num_unused_inputs
                    << " unused inputs = " << output_tensors.size() << " total outputs.";


    // Configure ONNX output tensors based on model info
    for (size_t i = 0; i < num_onnx_outputs; ++i) {
      // Get expected dims (int64_t) and convert to int for AxTensorInterface
      const auto &dims_int64 = output_dims[i];
      output_tensors[i].sizes.clear();
      output_tensors[i].sizes.reserve(dims_int64.size());
      for (const auto &dim : dims_int64) {
        if (dim <= 0) {
          // This case should ideally be handled better, maybe by requiring fixed output shapes
          // or allowing dynamic allocation later. For now, error out or default to 1.
          logger(AX_WARN)
              << "Output tensor " << i << " has dynamic dimension (" << dim
              << "). I/O Binding might not work correctly. Defaulting dim to 1.";
          output_tensors[i].sizes.push_back(1);
        } else {
          output_tensors[i].sizes.push_back(static_cast<int>(dim));
        }
      }

      // ONNX outputs are assumed float (4 bytes) for this implementation
      // TODO: Could check prop->onnx_runtime_->get_output_node_types() if needed
      output_tensors[i].bytes = sizeof(float);
      output_tensors[i].fd = -1; // Not file-based
      output_tensors[i].data = nullptr; // Data pointer will be set by framework allocator

      logger(AX_INFO)
          << "Output tensor " << i << " (from ONNX " << output_names[i]
          << ") configured with shape: " << shape_to_string(output_tensors[i].sizes)
          << ", bytes: " << output_tensors[i].bytes;
    }

    // Configure unused input tensors (passthrough)
    size_t output_idx = num_onnx_outputs;
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      if (used_indices.find(i) == used_indices.end()) {
        if (output_idx < output_tensors.size()) {
          output_tensors[output_idx] = input_tensors[i]; // Copy interface info
          output_tensors[output_idx].data
              = nullptr; // Data pointer will be set by framework allocator
          logger(AX_INFO) << "Output tensor " << output_idx << " (passthrough from input "
                          << i << ") configured with shape: "
                          << shape_to_string(output_tensors[output_idx].sizes)
                          << ", bytes: " << output_tensors[output_idx].bytes;
          output_idx++;
        } else {
          logger(AX_ERROR) << "Logic error: Not enough space allocated for unused input tensors in output interface.";
          // This shouldn't happen if resize calculation was correct
          break;
        }
      }
    }

    return AxDataInterface{ output_tensors };

  } catch (const std::exception &e) {
    logger(AX_ERROR) << "Error determining output interface: " << e.what();
    throw;
  }
}


// Convert an Ax tensor to ONNX format (for input)
Ort::Value
convert_tensor_to_onnx_input(const AxTensorInterface &ax_tensor, Ax::Logger &logger)
{
  // Assume ONNX input needs float
  if (ax_tensor.bytes != sizeof(float)) {
    logger(AX_ERROR) << "ONNX input requires float (4 bytes). Found tensor with "
                     << ax_tensor.bytes << " bytes.";
    throw std::runtime_error("Invalid tensor format for ONNX input");
  }

  // Memory info for CPU allocation
  Ort::MemoryInfo memory_info
      = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Get data pointer as float
  const float *data_ptr = static_cast<const float *>(ax_tensor.data);

  // Convert sizes to int64_t for ONNX
  std::vector<int64_t> dims;
  dims.reserve(ax_tensor.sizes.size());
  for (int size : ax_tensor.sizes) {
    dims.push_back(static_cast<int64_t>(size));
  }

  // Calculate total size in bytes
  size_t total_bytes = ax_tensor.total() * sizeof(float);

  // Create ONNX tensor (non-owning - using original data)
  // NOTE: const_cast is necessary because CreateTensor takes non-const data*,
  // but ONNX Runtime typically treats input tensors as read-only.
  return Ort::Value::CreateTensor<float>(memory_info,
      const_cast<float *>(data_ptr), total_bytes, dims.data(), dims.size());
}


// Main transform function using I/O Binding
extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const postamble_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  auto &input_tensors = std::get<AxTensorsInterface>(input);
  auto &output_tensors = std::get<AxTensorsInterface>(
      output); // This is const&, but we need to modify data

  // If no ONNX model, just copy input to output (passthrough)
  if (!prop->onnx_runtime_) {
    logger(AX_DEBUG) << "No ONNX model. Passing tensors through unchanged.";
    for (size_t i = 0; i < std::min(input_tensors.size(), output_tensors.size()); ++i) {
      const auto &in_tensor = input_tensors[i];
      // Need non-const access to output tensor data
      auto &out_tensor = const_cast<AxTensorInterface &>(output_tensors[i]);

      size_t input_bytes = in_tensor.total() * in_tensor.bytes;
      size_t output_bytes = out_tensor.total() * out_tensor.bytes;

      if (input_bytes != output_bytes || in_tensor.sizes != out_tensor.sizes) {
        logger(AX_WARN) << "Input/Output mismatch during passthrough for tensor "
                        << i << ". In: " << shape_to_string(in_tensor.sizes)
                        << " bytes=" << in_tensor.bytes
                        << ", Out: " << shape_to_string(out_tensor.sizes)
                        << " bytes=" << out_tensor.bytes
                        << ". Attempting copy anyway based on input size.";
        // Ensure output buffer is large enough
        if (input_bytes > output_bytes) {
          logger(AX_ERROR) << "Output tensor " << i << " is smaller than input. Cannot copy.";
          throw std::runtime_error("Output tensor size mismatch during passthrough");
        }
      }


      if (!in_tensor.data || !out_tensor.data) {
        logger(AX_ERROR)
            << "Null data pointer found during passthrough copy for tensor " << i;
        throw std::runtime_error("Null data pointer in passthrough");
      }
      std::memcpy(out_tensor.data, in_tensor.data, input_bytes);
    }
    return;
  }

  // --- ONNX Processing with I/O Binding ---

  // Get ONNX model information needed for run
  const auto &input_names = prop->onnx_runtime_->get_input_node_names();
  const auto &output_names = prop->onnx_runtime_->get_output_node_names();
  size_t num_onnx_inputs = input_names.size();
  size_t num_onnx_outputs = output_names.size();


  // Verify tensor selection plan validity (should match number of ONNX inputs)
  if (prop->tensor_selection_plan.size() != num_onnx_inputs) {
    logger(AX_ERROR)
        << "Invalid tensor selection plan size for ONNX model. Expected "
        << num_onnx_inputs << ", got " << prop->tensor_selection_plan.size();
    throw std::runtime_error("Invalid tensor selection plan size");
  }

  // Verify we have enough input tensors based on selection plan
  int max_index = -1;
  if (!prop->tensor_selection_plan.empty()) {
    max_index = *std::max_element(
        prop->tensor_selection_plan.begin(), prop->tensor_selection_plan.end());
  }
  if (input_tensors.size() <= max_index) {
    logger(AX_ERROR) << "Not enough input tensors for tensor selection plan. "
                     << "Need at least " << (max_index + 1) << " but have "
                     << input_tensors.size();
    throw std::runtime_error("Not enough input tensors for selection plan");
  }

  // Prepare ONNX input tensors (Ort::Value wrappers around input data)
  std::vector<Ort::Value> onnx_inputs;
  onnx_inputs.reserve(num_onnx_inputs);
  try {
    for (size_t i = 0; i < num_onnx_inputs; ++i) {
      int tensor_idx = prop->tensor_selection_plan[i];
      const auto &tensor = input_tensors[tensor_idx];
      if (!tensor.data) {
        logger(AX_ERROR) << "Input tensor " << tensor_idx
                         << " selected for ONNX has null data pointer.";
        throw std::runtime_error("Null data pointer in selected input tensor");
      }
      onnx_inputs.push_back(convert_tensor_to_onnx_input(tensor, logger));
      logger(AX_DEBUG) << "Using input tensor " << tensor_idx << " ("
                       << shape_to_string(tensor.sizes) << ") as ONNX input "
                       << i << " (" << input_names[i] << ")";
    }
  } catch (const std::exception &e) {
    logger(AX_ERROR) << "Failed to convert input tensors to ONNX format: " << e.what();
    throw;
  }

  // Prepare pointers to the output AxTensorInterface objects for I/O Binding
  // These correspond to the first 'num_onnx_outputs' tensors in the 'output_tensors' vector.
  std::vector<AxTensorInterface *> onnx_output_ax_tensors;
  onnx_output_ax_tensors.reserve(num_onnx_outputs);
  if (output_tensors.size() < num_onnx_outputs) {
    logger(AX_ERROR)
        << "Framework did not provide enough output tensors for the ONNX model. "
        << "Expected " << num_onnx_outputs << ", got " << output_tensors.size();
    throw std::runtime_error("Insufficient output tensors allocated");
  }
  for (size_t i = 0; i < num_onnx_outputs; ++i) {
    // Need non-const pointer to modify the tensor interface (specifically its data buffer)
    auto &mutable_out_tensor = const_cast<AxTensorInterface &>(output_tensors[i]);
    if (!mutable_out_tensor.data) {
      logger(AX_ERROR) << "Output tensor " << i << " for ONNX binding has null data pointer.";
      throw std::runtime_error("Null data pointer in output tensor for binding");
    }
    onnx_output_ax_tensors.push_back(&mutable_out_tensor);
    logger(AX_DEBUG)
        << "Binding output tensor " << i << " (buffer for " << output_names[i]
        << ", shape " << shape_to_string(mutable_out_tensor.sizes) << ")";
  }


  // Run ONNX inference using I/O Binding
  try {
    logger(AX_DEBUG)
        << "Running ONNX inference with I/O Binding (" << onnx_inputs.size()
        << " inputs, " << onnx_output_ax_tensors.size() << " outputs).";

    prop->onnx_runtime_->run_with_io_binding(onnx_inputs, onnx_output_ax_tensors);

    // Inference is complete. Data is directly in the output_tensors[0..num_onnx_outputs-1] buffers.
    // The run_with_io_binding implementation assumes output shapes matched expectations.
    // If dynamic shapes were possible, we might need to update tensor sizes here.

    logger(AX_INFO) << "ONNX inference completed using I/O Binding.";
    // Log final shapes placed in output buffers
    for (size_t i = 0; i < num_onnx_outputs; ++i) {
      logger(AX_INFO) << "Final data for ONNX output " << i << " ("
                      << output_names[i] << ") placed in output tensor " << i
                      << " with shape " << shape_to_string(output_tensors[i].sizes); // Use the (const) output_tensors view
    }

  } catch (const std::exception &e) {
    logger(AX_ERROR) << "ONNX inference with I/O Binding failed: " << e.what();
    throw;
  }


  // --- Handle unused input tensors (passthrough) ---

  // Create a set of used input tensor indices for quick lookup
  std::unordered_set<int> used_indices(
      prop->tensor_selection_plan.begin(), prop->tensor_selection_plan.end());

  // Copy any unused input tensors to the remaining output tensor slots
  size_t output_idx = num_onnx_outputs; // Start filling after the ONNX outputs
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    // Skip tensors used for ONNX input
    if (used_indices.find(i) != used_indices.end()) {
      continue;
    }

    // Check if we have space in the output array
    if (output_idx < output_tensors.size()) {
      const auto &in_tensor = input_tensors[i];
      // Get non-const access to the target output tensor
      auto &mutable_out_tensor
          = const_cast<AxTensorInterface &>(output_tensors[output_idx]);

      // Verify compatibility (should match from set_output_interface)
      size_t input_bytes = in_tensor.total() * in_tensor.bytes;
      size_t output_bytes = mutable_out_tensor.total() * mutable_out_tensor.bytes;

      if (input_bytes != output_bytes || in_tensor.sizes != mutable_out_tensor.sizes) {
        logger(AX_WARN) << "Input/Output mismatch during passthrough for unused input "
                        << i << " to output " << output_idx
                        << ". In: " << shape_to_string(in_tensor.sizes)
                        << " bytes=" << in_tensor.bytes
                        << ", Out: " << shape_to_string(mutable_out_tensor.sizes)
                        << " bytes=" << mutable_out_tensor.bytes
                        << ". Attempting copy anyway based on input size.";
        // Ensure output buffer is large enough
        if (input_bytes > output_bytes) {
          logger(AX_ERROR) << "Output tensor " << output_idx
                           << " is smaller than unused input tensor " << i
                           << ". Cannot copy.";
          throw std::runtime_error(
              "Output tensor size mismatch during passthrough (unused inputs)");
        }
      }

      if (!in_tensor.data || !mutable_out_tensor.data) {
        logger(AX_ERROR) << "Null data pointer found during passthrough copy for unused input "
                         << i << " to output " << output_idx;
        throw std::runtime_error("Null data pointer in passthrough (unused inputs)");
      }

      // Copy data
      std::memcpy(mutable_out_tensor.data, in_tensor.data, input_bytes);

      logger(AX_DEBUG) << "Copied unused input tensor " << i << " ("
                       << shape_to_string(in_tensor.sizes)
                       << ") to output tensor " << output_idx;

      // Increment output index
      output_idx++;
    } else {
      // This indicates a potential mismatch between calculation in set_output_interface and here
      logger(AX_WARN) << "Not enough output tensors allocated to store all unused input tensors. "
                      << "Stopped after output index " << (output_idx - 1);
      break;
    }
  }
}
