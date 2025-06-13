# Copyright Axelera AI, 2024
# Inference Operator Implementation
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Any, Optional, Union, get_args, get_origin

import numpy as np

from axelera import types

from .. import compile, config, constants, gst_builder, logging_utils, torch_utils
from ..torch_utils import torch
from .base import AxOperator, builtin
from .custom_preprocessing import PermuteChannels

if TYPE_CHECKING:
    from axelera import runtime

    from . import PipelineContext
    from .. import device_manager, gst_builder
    from ..pipe import graph

LOG = logging_utils.getLogger(__name__)
logging_utils.getLogger('axelera.runtime').setLevel(logging.WARNING)


def _build_mock_options() -> str:
    options = {}
    if mock := config.env.inference_mock:
        m = re.match(r'(load|save)-([^@\n]+)(?:@(\d+))?', mock)
        if not m:
            raise ValueError(f"Invalid mock option: {mock}, see AXELERA_HELP=1")
        if config.env.UseDmaBuf.OUTPUTS in config.env.use_dmabuf:
            raise ValueError(
                "Cannot use mock with output dmabufs, please disable with AXELERA_USE_DMABUF=1"
            )
        path = Path(m.group(2))
        if m.group(1) == 'load':
            if not path.is_dir():
                raise ValueError(f"Mock path {path} must exist and be a directory")
            if not (path / "shapes.txt").is_file():
                raise ValueError(f"Mock path {path} must exist and be a directory")
            options['mock-load'] = str(path)
            options['mock-fps'] = m.group(3) or '500'
        else:
            if not path.is_dir():
                LOG.info("Creating mock directory %s", path)
                path.mkdir(parents=True)
            options['mock-save'] = str(path)
    return ';'.join(f"{k}:{v}" for k, v in options.items())


def dequantize_single(np_array, dequant_params):
    """Dequantize a single np array."""
    scale, zero_point = dequant_params
    dequantized_array = (np_array - zero_point) * scale
    return dequantized_array


def dequantize(np_arrays, dequantize_params):
    """Dequantize a list of np arrays."""
    return [
        dequantize_single(np_array, params)
        for np_array, params in zip(np_arrays, dequantize_params)
    ]


def pad_and_quantize(np_array, quant_params, n_padded_ch, tensor_layout):
    """Pad and quantize a single numpy array. Quantization must be
    done after padding to have the same zeropoint for the padded pixels.
    """
    scale, zero_point = quant_params[0]
    quantized = np.round(np_array / scale + zero_point).clip(-128, 127).astype(np.int8)

    if n_padded_ch:
        n_low, n_high = compile.get_padded_low_high(n_padded_ch, tensor_layout, 'N')[0]
        top, bottom = compile.get_padded_low_high(n_padded_ch, tensor_layout, 'H')[0]
        left, right = compile.get_padded_low_high(n_padded_ch, tensor_layout, 'W')[0]
        c_low, c_high = compile.get_padded_low_high(n_padded_ch, tensor_layout, 'C')[0]

        pad_width = (
            (n_low, n_high),
            (top, bottom),
            (left, right),
            (c_low, c_high),
        )  # Assuming the order is batch, height, width, channels (NHWC)
        quantized = np.pad(quantized, pad_width, mode='constant', constant_values=zero_point)

    return quantized


def _convert_output_arrays(output_arrays, n_padded_ch, current_layout, expected_layout):
    if not all(isinstance(array, np.ndarray) for array in output_arrays):
        raise TypeError("All output arrays must be NumPy arrays")
    if len(output_arrays) != len(n_padded_ch):
        raise ValueError("Length of output_arrays and n_padded_ch must be the same")

    # Get the original shapes using get_original_shape
    output_shapes = tuple(array.shape for array in output_arrays)
    original_shapes = compile.get_original_shape(
        output_shapes, n_padded_ch, current_layout, expected_layout
    )

    # Convert the arrays according to the new shapes and layout
    converted_arrays = []
    for array, original_shape in zip(output_arrays, original_shapes):
        # Assuming the padding is only applied to the channels
        # and the rest of the dimensions are the same
        if current_layout == 'NHWC' and expected_layout == 'NCHW':
            array = np.moveaxis(array, -1, 1)
        elif current_layout == 'NCHW' and expected_layout == 'NHWC':
            array = np.moveaxis(array, 1, -1)

        # Slicing the array to remove padding if necessary
        converted_array = array[tuple(slice(dim) for dim in original_shape)]
        converted_arrays.append(converted_array)

    return converted_arrays


def _convert_to_tensors(data_input) -> torch.Tensor:
    """
    Convert a list containing torch.Tensors or numpy.ndarrays to a single stacked tensor.
    If a single tensor is provided, it is cloned and detached.

    Parameters:
    - data_input (torch Tensor or list): A tensor or a list of torch Tensors or numpy.ndarrays

    Returns:
    - torch Tensor: Resulting tensor
    """

    if isinstance(data_input, torch.Tensor):
        return data_input.clone().detach()
    if isinstance(data_input, list):
        if not data_input:
            return torch.empty(0)
        else:
            tensor_list = []
            for item in data_input:
                if isinstance(item, torch.Tensor):
                    tensor_list.append(item.clone().detach())
                elif isinstance(item, np.ndarray):
                    tensor_list.append(torch.from_numpy(item))
                else:
                    raise ValueError(f"Unsupported data type {type(item)} in the list")
            return tensor_list
    elif isinstance(data_input, np.ndarray):
        if data_input.size == 0:
            return torch.empty(0)
        else:
            return torch.from_numpy(data_input)
    else:
        raise ValueError(f"Unsupported data type {type(data_input)} in the list")


def _reshape_to_target_shapes(data, output_shapes):
    reshaped_data = []

    for item, shape in zip(data, output_shapes):
        # Convert item to a NumPy array if it's not already
        if not isinstance(item, np.ndarray):
            item = np.array(item)

        # Check if the number of elements matches
        if item.size == np.prod(shape):
            reshaped_item = item.reshape(shape)
            reshaped_data.append(reshaped_item)
        else:
            raise ValueError(
                f"Cannot reshape data of shape {item.shape} "
                f"to target shape {shape} due to mismatch in number of elements."
            )

    return reshaped_data


def convert_to_rgba(
    tensor: torch.Tensor, input_layout: types.TensorLayout = types.TensorLayout.NCHW
) -> torch.Tensor:
    """
    Converts an RGB tensor to RGBA with alpha set to 0 based on the layout.

    :param tensor: The input tensor in RGB format.
    :param input_layout: One of 'NCHW', 'NHWC', or 'CHWN'.
    :return: A tensor in RGBA format with the same layout.
    """
    if input_layout == types.TensorLayout.NCHW:
        if tensor.shape[1] != 3:
            raise ValueError(f"Expected tensor with 3 channels but got {tensor.shape[1]}")
        alpha_channel = torch.zeros(
            (tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, alpha_channel], dim=1)
    elif input_layout == types.TensorLayout.NHWC:
        if tensor.shape[3] != 3:
            raise ValueError(f"Expected tensor with 3 channels but got {tensor.shape[3]}")
        alpha_channel = torch.zeros(
            (tensor.shape[0], tensor.shape[1], tensor.shape[2], 1),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, alpha_channel], dim=3)
    elif input_layout == types.TensorLayout.CHWN:
        if tensor.shape[0] != 3:
            raise ValueError(f"Expected tensor with 3 channels but got {tensor.shape[0]}")
        alpha_channel = torch.zeros(
            (1, tensor.shape[1], tensor.shape[2], tensor.shape[3]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, alpha_channel], dim=0)
    else:
        raise ValueError("Invalid layout. Expected one of 'NCHW', 'NHWC', or 'CHWN'.")


def _add_batch_channel(
    tensor: torch.Tensor, input_tensor_layout: types.TensorLayout
) -> torch.Tensor:
    """
    Checks the dimensions of a tensor and adds a batch channel if necessary.

    If the tensor has shape (height, width, channels), this function will add a
    new dimension of size 1 to the beginning of the tensor, effectively adding
    a batch channel. If the tensor already has a batch channel (i.e., has shape
    (batch_size, height, width, channels)), this function does nothing.

    Args:
        tensor: The torch.Tensor to modify.

    Returns:
        The modified tensor.
    """

    if tensor.dim() == 3:
        if input_tensor_layout == types.TensorLayout.CHWN:
            tensor = tensor.unsqueeze(-1)
        else:
            tensor = tensor.unsqueeze(0)
    return tensor


def _determine_device(model):
    return 'aipu' if isinstance(model, types.Manifest) else torch_utils.device_name()


def _get_io_shapes_from_tvm_mod(quant_model_file: Path):
    """
    Given a TVM IRModule file, this function extracts the input and output shapes.
    """
    import tvm
    from tvm import relay

    mod = tvm.ir.load_json(quant_model_file.read_text())
    main_func = mod["main"]

    input_shapes = []
    for arg in main_func.params:
        input_shapes.append([int(dim) for dim in arg.type_annotation.shape])

    output_type = main_func.ret_type
    if isinstance(output_type, relay.TensorType):
        output_shape = [[int(dim) for dim in output_type.shape]]
    elif isinstance(output_type, relay.TupleType):
        output_shape = [[int(dim) for dim in typ.shape] for typ in output_type.fields]
    else:
        raise ValueError("The main function does not return a tensor or tuple of tensors")
    return input_shapes, output_shape


def _remove_trailing_singletons(shape_list):
    """
    Remove trailing singleton dimensions from each shape in a list of shapes.

    Parameters:
    - shape_list: list of list of ints, each inner list is a shape

    Returns:
    - adjusted_shapes: list of adjusted shapes with trailing singletons removed
    """
    adjusted_shapes = []

    for shape in shape_list:
        # Find the last dimension that is not 1 (i.e., not a singleton)
        last_non_singleton_dim = next(
            (i for i, size in reversed(list(enumerate(shape))) if size != 1), -1
        )
        # Include all dimensions up to the last non-singleton dimension
        new_shape = shape[: last_non_singleton_dim + 1] if last_non_singleton_dim != -1 else shape
        adjusted_shapes.append(new_shape)
    return adjusted_shapes


def build_onnx_inferencer(model: str):
    import onnxruntime as rt

    preferred_providers = [
        # onnxruntime-gpu
        'CUDAExecutionProvider',
        # onnxruntime-gpu
        'MPSExecutionProvider' if sys.platform == 'darwin' else None,
        # onnxruntime-openvino
        'OpenVINOExecutionProvider',
        'CPUExecutionProvider',
    ]

    # Filter out unavailable providers
    available_providers = rt.get_available_providers()
    providers = [provider for provider in preferred_providers if provider in available_providers]
    LOG.debug(f"Available ONNX runtime providers: {available_providers}")

    # Create and return the session
    session = rt.InferenceSession(model, providers=providers)

    post_input_names = [node.name for node in session.get_inputs()]
    post_output_names = [node.name for node in session.get_outputs()]
    post_input_shapes = [node.shape for node in session.get_inputs()]
    return session, post_input_names, post_output_names, post_input_shapes


def _run_onnx_session(session, input_names, output_names, inputs):
    """Run ONNX inference session with proper input handling.

    Args:
        session: ONNX inference session
        input_names: List of input names expected by the session
        output_names: List of output names to retrieve
        inputs: Single tensor/array or list of tensors/arrays

    Returns:
        Output from ONNX session run
    """
    # Convert to list if single input
    inputs_list = inputs if isinstance(inputs, (list, tuple)) else [inputs]

    # Convert all inputs to numpy arrays
    numpy_inputs = []
    for inp in inputs_list:
        if isinstance(inp, torch.Tensor):
            numpy_inputs.append(inp.cpu().numpy())
        elif isinstance(inp, np.ndarray):
            numpy_inputs.append(inp)
        else:
            raise ValueError(f"Unsupported input type: {type(inp)}")

    # Create input dictionary and run session
    input_dict = dict(zip(input_names, numpy_inputs))
    return session.run(output_names, input_dict)


def _get_model_path(model_root: Path, model_file) -> Path | None:
    '''Return a checked path to the model.'''
    json_path = model_root / model_file
    if not json_path.exists():
        raise RuntimeError(f"{json_path!s} does not exist, cannot proceed")
    return json_path


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration settings related to inference processing.
    Initialized from model info and potentially updated by a postprocess operator.
    """

    # Field determined during initial creation from model info
    cpp_focus_layer_on_host: bool = True
    # Internal flags updated later by the operator
    _cpp_decoder_does_dequantization_and_depadding: bool = dataclasses.field(
        init=False, default=False
    )
    _cpp_decoder_does_transpose: bool = dataclasses.field(init=False, default=True)
    _cpp_decoder_does_postamble: bool = dataclasses.field(init=False, default=False)

    def _validate_and_set_cpp_flags(self, depadding: bool, transpose: bool, postamble: bool):
        """Validates the gst flags and sets the internal state.
        If postamble is False, depadding and transpose must also be False (will warn and force).
        If postamble is True, depadding and transpose can be either True or False.
        """
        # TODO: Once manifest_postamble_graph is available, enable this logic.
        # For classifier models, explicit transposition is unnecessary. Additionally, postamble is set to False but actually there is no postamble graph.
        # if self._postamble_graph and not postamble:
        #     if depadding or transpose:
        #         LOG.warning(
        #             "Configuration conflict: 'cpp_decoder_does_postamble' is False, "
        #             "so 'cpp_decoder_does_dequantization_and_depadding' and 'cpp_decoder_does_transpose' "
        #             "must also be False. Forcing both to False."
        #         )
        #     depadding = False
        #     transpose = False

        self._cpp_decoder_does_dequantization_and_depadding = depadding
        self._cpp_decoder_does_transpose = transpose
        self._cpp_decoder_does_postamble = postamble

    @classmethod
    def from_model_info(cls, model_info_kwargs: dict):  # Runtime model info
        """
        Creates an InferenceConfig instance based solely on model information.
        The cpp_* flags related to the operator will be default until updated.

        TODO: add manifest_postamble_graph; ideally, when we build AxTask, we should know if the model has been deployed or not; if deployed, we should take manifest to determine how we want to configure the inference config

        Args:
            model_info_kwargs: Dictionary containing model-specific information.
        """
        instance = cls()  # Create an instance with default values for all fields

        # --- Determine settings SOLELY from model_info ---
        instance.cpp_focus_layer_on_host = bool(
            model_info_kwargs.get('YOLO', {}).get('focus_layer_replacement', True)
        )
        return instance

    def update_from_decoder_op(self, postprocess_op: AxOperator | None):
        """
        Updates the GST-related flags based on the postprocess operator.
        This should be called after the instance is created, typically
        once the specific postprocess operator is known.

        Args:
            postprocess_op: The postprocessing operator instance (or None).
        """
        if not postprocess_op:
            return
        elif postprocess_op.cpp_decoder_does_all:
            self._validate_and_set_cpp_flags(True, True, True)
            return

        depadding = getattr(
            postprocess_op,
            "cpp_decoder_does_dequantization_and_depadding",
            self._cpp_decoder_does_dequantization_and_depadding,
        )
        if depadding is None:
            depadding = self._cpp_decoder_does_dequantization_and_depadding

        transpose = getattr(
            postprocess_op, "cpp_decoder_does_transpose", self._cpp_decoder_does_transpose
        )
        if transpose is None:
            transpose = self._cpp_decoder_does_transpose

        postamble = getattr(
            postprocess_op, "cpp_decoder_does_postamble", self._cpp_decoder_does_postamble
        )
        if postamble is None:
            postamble = self._cpp_decoder_does_postamble

        self._validate_and_set_cpp_flags(depadding, transpose, postamble)

    @property
    def cpp_decoder_does_dequantization_and_depadding(self) -> bool:
        return self._cpp_decoder_does_dequantization_and_depadding

    @property
    def cpp_decoder_does_transpose(self) -> bool:
        return self._cpp_decoder_does_transpose

    @property
    def cpp_decoder_does_postamble(self) -> bool:
        return self._cpp_decoder_does_postamble


class Inference:
    # assume that the input tensor is well prepared according to its required order
    # but still lack of the batch channel

    # device can be AUTO, AIPU, CPU, CUDA
    # model is the model instance in Python and model manifest for AIPU
    # if device is AIPU, model must be a Manifest instance

    def __init__(
        self,
        device_man: device_manager.DeviceManager,
        compiled_model_dir: Path,
        model_name: str,
        model: Union[types.Manifest, types.Model],
        input_tensor_layout: Optional[types.TensorLayout],
        inference_op_config: InferenceConfig,
    ):
        self.compiled_model_dir = compiled_model_dir
        self.model_name = model_name
        self.model = model
        self.input_tensor_layout = input_tensor_layout
        self._output_shape0 = []
        self._permute_op = None
        self._device_man = device_man
        self._axr_conn: runtime.Connection = None
        self._axr_model: runtime.Model = None
        self._axr_modeli: runtime.ModelInstance = None
        self._inf_config = inference_op_config
        self.pre_ort_sess, self.post_ort_sess = None, None
        self.devices = []

        self.device = _determine_device(self.model)
        if model := self._try_load_quantized_model(self.model):
            self.model = model
        elif self.device == 'aipu':
            self.devices = self._device_man.devices
            if not isinstance(self.model, types.Manifest):
                raise ValueError('AIPU device requires model to be a Manifest instance')

        elif isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.to_device(torch.device(self.device))

            if self.input_tensor_layout and self.input_tensor_layout != types.TensorLayout.NCHW:
                self._permute_op = PermuteChannels(
                    input_layout=types.TensorLayout.NCHW,
                    output_layout=self.input_tensor_layout,
                )
        elif isinstance(self.model, types.ONNXModel):
            self.ort_sess, self.input_names, self.output_names, _ = build_onnx_inferencer(
                self.model.onnx_model.SerializeToString()
            )

            if self.input_tensor_layout and self.input_tensor_layout != types.TensorLayout.NCHW:
                self._permute_op = PermuteChannels(
                    input_layout=types.TensorLayout.NCHW,
                    output_layout=self.input_tensor_layout,
                )
        else:
            raise ValueError(f'Unsupported model type {type(self.model)}')
        super().__init__()

    def release(self):
        if self._axr_modeli:
            self._axr_modeli.release()
        if self._axr_conn:
            self._axr_conn.release()

    def _try_load_quantized_model(
        self, model: Union[types.Manifest, types.Model]
    ) -> torch.fx.GraphModule | None:
        if isinstance(model, types.Manifest):
            if model.model_lib_file is None:
                LOG.trace(
                    f"Manifest {self.model_name} does not have model_lib_file, assume quantized model"
                )
                try:
                    from qtoolsv2.utils.graph.graph_save import load_qtools_graphmodule

                    model_path = self.compiled_model_dir / constants.K_MODEL_QUANTIZED_FOR_DEBUG
                    if not model_path.is_file():
                        raise ValueError(
                            f"Please run ./deploy.py {self.model_name} --mode=quantize_debug to generate the quantized model file"
                        )
                    model = load_qtools_graphmodule(model_path)
                    model.eval()
                except Exception as e:
                    raise ValueError(
                        f"Failed to load quantized model from manifest {self.model_name}: {e}"
                    )
                self.device = torch_utils.device_name()
                self._init_pre_and_post()
                return model
        return None

    def _get_core_model_output_shapes(self):
        try:
            return self._core_model_output_shapes
        except AttributeError:
            pass
        output_shapes = None
        try:
            quant_model_file = self.compiled_model_dir / self.model.quantized_model_file
            _, output_shapes = _get_io_shapes_from_tvm_mod(quant_model_file)

            # this is for classification but not for other models; simply check if len(output_shapes)==1
            if len(output_shapes) == 1:
                output_shapes = _remove_trailing_singletons(output_shapes)
            LOG.debug(f"Expected core model output shape: {output_shapes}")
        except Exception as e:
            LOG.warning(f"Failed to get input and output shapes from TVM IRModule file: {e}")
        self._core_model_output_shapes = output_shapes
        return output_shapes

    def _init_pre_and_post(self):
        self.pre_ort_sess, self.post_ort_sess = None, None

        # torch preprocessing gives NCHW - always permute to NHWC for AIPU if torch-aipu
        # (we will address TF2 NHWC input in SDK-2649)
        if self.device == 'aipu':
            self._permute_op = PermuteChannels(
                input_layout=types.TensorLayout.NCHW,
                output_layout=types.TensorLayout.NHWC,
            )

        if self.model.postprocess_graph:
            (
                self.post_ort_sess,
                self.post_input_names,
                self.post_output_names,
                self.post_input_shapes,
            ) = build_onnx_inferencer(self.compiled_model_dir / self.model.postprocess_graph)
            LOG.trace("Expected input for postprocess graph:")
            LOG.trace(
                [f"{input.name}: {input.shape}" for input in self.post_ort_sess.get_inputs()]
            )
        if self.model.preprocess_graph:
            (
                self.pre_ort_sess,
                self.pre_input_names,
                self.pre_output_names,
                self.pre_input_shapes,
            ) = build_onnx_inferencer(self.compiled_model_dir / self.model.preprocess_graph)
            LOG.trace("Expected input for preprocess graph:")
            LOG.trace([f"{input.name}: {input.shape}" for input in self.pre_ort_sess.get_inputs()])

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path,
        task_graph: graph.DependencyGraph,
    ):
        self.task_name = task_name
        self.model_category = model_info.task_category
        if self.device == 'aipu':
            self._model_cores = model_info.manifest.input_shapes[0][0]
            self._output_shapes = model_info.manifest.output_shapes
        self._taskn = taskn

        if context.color_format != model_info.input_color_format:
            raise ValueError(
                f"Input color format mismatch in {task_name}. Expected {model_info.input_color_format}, but got {context.color_format}"
            )

    def check_focus_layer_on_host(self):
        if hasattr(self.model, 'preprocess_graph') and self.model.preprocess_graph:
            preprocess_path = Path(self.model.preprocess_graph)
            if not preprocess_path.is_absolute():
                preprocess_path = self.compiled_model_dir / preprocess_path
            return has_focus_layer_onnx(preprocess_path)

    @property
    def config(self):
        return self._inf_config

    def _do_pads_or_preproc(self, gst: gst_builder.Builder):
        from ..pipe.gst import generate_padding

        padding = generate_padding(self.model)
        if self.config.cpp_focus_layer_on_host and self.check_focus_layer_on_host():
            gst.axtransform(
                lib='libtransform_yolopreproc.so',
                options=f'padding:{padding}',
                batch=self._model_cores,
            )
        else:
            gst.axtransform(
                lib='libtransform_padding.so',
                options=f'padding:{padding};fill:{0}',
                batch=self._model_cores,
            )

    def build_inference_gst(self, gst: gst_builder.Builder, num_cores: int):
        self._do_pads_or_preproc(gst)

        num_children = 0
        if self._model_cores < num_cores:
            if num_cores % self._model_cores == 0:
                num_children = num_cores // self._model_cores
                nd = len(self.devices)
                LOG.debug(
                    f"Enabled {num_children}x{nd} inference queues for {self.model_name} because "
                    f"model_cores={self._model_cores} and num_cores={num_cores}"
                )
            else:
                LOG.info(
                    f"This model is restricted to run on up to {self._model_cores} cores (not {num_cores})"
                )

        model = _get_model_path(self.compiled_model_dir, self.model.model_lib_file)

        name = f'inference-task{self._taskn}'
        options = _build_mock_options()
        inf = dict(
            name=name,
            model=str(model),
            devices=','.join(d.name for d in self.devices),
            double_buffer=config.env.use_double_buffer,
            dmabuf_inputs=config.env.UseDmaBuf.INPUTS in config.env.use_dmabuf,
            dmabuf_outputs=config.env.UseDmaBuf.OUTPUTS in config.env.use_dmabuf,
            num_children=num_children,
        )
        if gst.tile:
            inf['meta'] = 'axelera-tiles-internal'
        if options:
            inf['options'] = options
        sopts = ' '.join(f"{k}={v}" for k, v in inf.items())
        LOG.debug(f"Using inferencenet {sopts}")
        gst.axinference(**inf)

    def exec_torch(self, image, result, meta):
        # result is the input tensor which changes in-place
        result = _add_batch_channel(result, self.input_tensor_layout)
        if isinstance(self.model, torch.nn.Module):
            if self._permute_op:
                result = self._permute_op.exec_torch(result)
            if self.pre_ort_sess:
                result = _run_onnx_session(
                    self.pre_ort_sess, self.pre_input_names, self.pre_output_names, result
                )
            result = result.to(self.device)
            with torch.no_grad():
                result = self.model(result)
            result = self._run_post_processing(result)
        elif isinstance(self.model, types.ONNXModel):
            if self._permute_op:
                input_array = self._permute_op.exec_torch(result).numpy()
            else:
                input_array = result.numpy()
            result = _run_onnx_session(
                self.ort_sess, self.input_names, self.output_names, input_array
            )
            result = _convert_to_tensors(result if len(result) > 1 else result[0])
        elif isinstance(self.model, types.Manifest):
            # pipe==torch-aipu
            if self._axr_conn is None:
                from axelera import runtime

                model_path = _get_model_path(self.compiled_model_dir, self.model.model_lib_file)
                c = self._device_man.context
                self._axr_model = c.load_model(model_path)
                self._axr_conn = c.device_connect(None, 1)
                self._axr_modeli = self._axr_conn.load_model_instance(self._axr_model)

                LOG.debug(f"Loaded model : {model_path}")
                self._init_pre_and_post()

            if self.pre_ort_sess:
                input_dict = dict(zip(self.pre_input_names, [result.numpy()]))
                input_array = self.pre_ort_sess.run(self.pre_output_names, input_dict)
                assert len(input_array) == 1, (
                    f"Support only one input tensor as AIPU input for "
                    f"now, got {len(input_array)}"
                )
                input_array = torch.from_numpy(input_array[0])
            else:
                input_array = result

            # suppose the model has only one input, pad and quantize it
            # TODO: support multiple inputs
            # input_array = convert_to_rgba(input_array, self.input_tensor_layout)

            input_array = self._permute_op.exec_torch(input_array).numpy()
            input_array = pad_and_quantize(
                input_array,
                self.model.quantize_params,
                self.model.n_padded_ch_inputs,
                'NHWC',
            )

            outputs = [np.empty(t.shape, np.int8) for t in self._axr_model.outputs()]
            inputs = [input_array]
            self._axr_modeli.run(inputs, outputs)

            if len(outputs) != len(self.model.dequantize_params):
                raise ValueError(
                    f"Got number of output arrays {len(outputs)} != number of dequantize params "
                    f"{len(self.model.dequantize_params)}"
                )

            outputs = [o.astype(np.float32) for o in outputs]
            if True:  # was self.need_padding_and_layout_transform_of_inputs_outputs
                # this should be factored out into a different fn
                if self.model.n_padded_ch_outputs:
                    outputs = _convert_output_arrays(
                        outputs,
                        self.model.n_padded_ch_outputs,
                        self.model.input_tensor_layout,
                        "NCHW",
                    )
                    # reshape the output tensors according to the expected shapes
                    if out_shapes := self._get_core_model_output_shapes():
                        outputs = _reshape_to_target_shapes(outputs, out_shapes)
                outputs = dequantize(outputs, self.model.dequantize_params)
                result = self._run_post_processing(outputs)

        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        return image, result, meta

    def _run_post_processing(self, outputs):
        if not isinstance(outputs, list):
            outputs = [outputs]

        # Create a mapping of shapes to arrays
        shape_to_array = {tuple(arr.shape): arr for arr in outputs}
        unused_shapes = set(shape_to_array.keys())

        # Run post-processing if applicable
        if self.post_ort_sess:
            # Reorder arrays based on post-processing input shapes
            reordered_array_list = []
            for shape in self.post_input_shapes:
                shape_tuple = tuple(shape)
                try:
                    reordered_array_list.append(shape_to_array[shape_tuple])
                    unused_shapes.remove(shape_tuple)
                except KeyError:
                    LOG.warning(f"Expected shape {shape} not found in model outputs")

            post_processed_result = _run_onnx_session(
                self.post_ort_sess,
                self.post_input_names,
                self.post_output_names,
                reordered_array_list,
            )
            result = post_processed_result
        else:
            result = []

        # Add unused arrays to the result
        unused_arrays = [shape_to_array[shape] for shape in unused_shapes]
        if unused_arrays:
            LOG.trace(f"Found {len(unused_arrays)} unused output arrays")
            result = result if isinstance(result, list) else [result]
            result.extend(unused_arrays)
        return _convert_to_tensors(result if len(result) > 1 else result[0])


def _calculate_tensor_selection_plan(model, postprocess_graph_path, compiled_model_dir=None):
    """
    Calculate tensor selection plan for postamble ONNX model.

    This determines which tensors from model output should be used
    as inputs to the postamble model.

    Args:
        model: Model manifest containing information about expected input shapes
        postprocess_graph_path: Path to the ONNX postprocessing graph
        compiled_model_dir: Optional directory where compiled models are stored

    Returns:
        List of tensor indices to use for ONNX postamble input
    """
    import os

    import onnx

    # Default to empty plan if no postprocess graph
    if not postprocess_graph_path:
        return []

    try:
        # Determine the full path to the ONNX model
        model_path = postprocess_graph_path
        if compiled_model_dir and not os.path.isabs(postprocess_graph_path):
            model_path = os.path.join(compiled_model_dir, postprocess_graph_path)

        # Check if the file exists
        if not os.path.exists(model_path):
            LOG.warning(f"Postprocess graph not found at: {model_path}")
            return []

        # Load ONNX model to get input shapes
        onnx_model = onnx.load(model_path)

        # Get expected input shapes from the ONNX model
        postamble_inputs = []
        for input_info in onnx_model.graph.input:
            postamble_inputs.append(input_info.name)

        # Get original output shapes from the model manifest
        if not model or not hasattr(model, 'output_format'):
            return list(range(len(postamble_inputs)))  # Default to sequential indices

        # Match shapes between model outputs and postamble inputs
        indices = []
        model_shapes = []
        if hasattr(model, 'get_original_shape'):
            model_shapes = model.get_original_shape() or []

        # If we have shape information, use it to match tensors
        if model_shapes:
            for i in range(len(postamble_inputs)):
                # For simplicity, just use sequential mapping if counts match
                if i < len(model_shapes):
                    indices.append(i)
                else:
                    # If postamble expects more inputs than model provides,
                    # just use what we have (will likely fail at runtime)
                    break
        else:
            # Fallback: use sequential indices
            output_layers = getattr(model, 'output_layers', [])
            indices = list(range(min(len(postamble_inputs), len(output_layers))))

        return indices
    except Exception as e:
        LOG.error(f"Error calculating tensor selection plan: {e}")
        return []  # Return empty plan on error


def _generate_depadding(manifest: types.Manifest) -> str:
    return '|'.join(
        ','.join(str(-num) for num in sublist) for sublist in manifest.n_padded_ch_outputs
    )


@builtin
class AxeleraDequantize(AxOperator):
    model: Union[types.Manifest, types.Model]
    inference_op_config: InferenceConfig
    num_classes: int
    task_category: types.TaskCategory
    assigned_model_name: str
    manifest_dir: Path
    taskn: int = 0

    def _post_init(self):
        self._model_name = self.assigned_model_name

    def exec_torch(self, *args):
        raise NotImplementedError()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        connections = dict(src=f'decoder_task{self.taskn}{stream_idx}.sink_1')

        # If all post-processing is handled by the decoder, nothing to do here.
        if (
            self.inference_op_config.cpp_decoder_does_dequantization_and_depadding
            and self.inference_op_config.cpp_decoder_does_transpose
            and self.inference_op_config.cpp_decoder_does_postamble
        ):
            return
        deq_scales, deq_zeropoints = zip(*self.model.dequantize_params)
        scales = ','.join(str(s) for s in deq_scales)
        zeros = ','.join(str(s) for s in deq_zeropoints)

        # If postamble is not handled by the decoder, but a postprocess graph exists,
        # we must add all transforms: depadding, dequantize+transpose, and postamble.
        if (
            not self.inference_op_config.cpp_decoder_does_postamble
            and self.model.postprocess_graph
        ):
            if stream_idx:
                gst.queue(name=f'stream_queue{stream_idx}')
            if self.model.n_padded_ch_outputs and any(self.model.n_padded_ch_outputs):
                gst.axtransform(
                    lib='libtransform_paddingdequantize.so',
                    options=f'padding:{_generate_depadding(self.model)};'
                    f'dequant_scale:{scales};dequant_zeropoint:{zeros};transpose:1',
                )
            else:
                gst.axtransform(
                    lib=f'libtransform_dequantize.so',
                    options=f'dequant_scale:{scales};dequant_zeropoint:{zeros};transpose:1',
                    connections=connections,
                )
            tensor_selection_plan = _calculate_tensor_selection_plan(
                self.model, self.model.postprocess_graph, self.manifest_dir
            )
            options = f'onnx_path:{self.manifest_dir / self.model.postprocess_graph}'
            if tensor_selection_plan:
                options += f';tensor_selection_plan:{",".join(map(str, tensor_selection_plan))}'
            LOG.debug(f"Using tensor selection plan for postamble: {tensor_selection_plan}")
            gst.axtransform(
                lib='libtransform_postamble.so',
                options=options,
            )
            return

        # If postamble is handled by the decoder, but not all transforms are,
        # add only the transforms not handled by the decoder.
        if stream_idx:
            gst.queue(name=f'stream_queue{stream_idx}')
        deq_scales, deq_zeropoints = zip(*self.model.dequantize_params)
        scales = ','.join(str(s) for s in deq_scales)
        zeros = ','.join(str(s) for s in deq_zeropoints)
        # Only add dequantize if not handled by decoder
        if not self.inference_op_config.cpp_decoder_does_dequantization_and_depadding:
            if self.model.n_padded_ch_outputs and any(self.model.n_padded_ch_outputs):

                gst.axtransform(
                    lib='libtransform_paddingdequantize.so',
                    options=f'padding:{_generate_depadding(self.model)};'
                    f'dequant_scale:{scales};dequant_zeropoint:{zeros};transpose:{int(not self.inference_op_config.cpp_decoder_does_transpose)}',
                    connections=connections,
                )
            else:
                gst.axtransform(
                    lib=f'libtransform_dequantize.so',
                    options=f'dequant_scale:{scales};dequant_zeropoint:{zeros};transpose:{int(not self.inference_op_config.cpp_decoder_does_transpose)}',
                    connections=connections,
                )


def has_focus_layer_onnx(onnx_model_path):
    """
    Detects a focus-layer pattern in an ONNX model by looking for a group of Slice nodes
    operating on the input, followed by a Concat along the channel axis.
    """
    try:
        import onnx
        import onnx.numpy_helper
    except ImportError:
        # ONNX is not installed; cannot analyze focus layer; this is for passing CI runtime tests.
        # Ideally, this step belongs to ahead of time AxInferenceNet generation, so runtime won't need this function.
        LOG.warning("ONNX is not installed; skipping focus layer analysis.")
        return False
    try:
        model = onnx.load(onnx_model_path)
        graph = model.graph
        if not graph.input:
            return False
        input_name = graph.input[0].name
        # Find all nodes that take the input tensor as input
        first_nodes = [node for node in graph.node if input_name in node.input]
        # Look for a group of Slice nodes that all take the input tensor
        slice_nodes = [node for node in first_nodes if node.op_type == "Slice"]
        if not slice_nodes:
            return False
        # Find concat nodes that take outputs of these slice nodes
        for node in graph.node:
            if node.op_type == "Concat":
                # Check if all inputs to this concat come from our slice nodes
                if all(any(inp in s.output for s in slice_nodes) for inp in node.input):
                    # Optionally, check axis attribute (should be channel axis, e.g., 1 for NCHW, 3 for NHWC)
                    axis = None
                    for attr in node.attribute:
                        if attr.name == "axis":
                            axis = attr.i
                    if axis in (1, 3):
                        return True
        return False
    except Exception:
        return False
