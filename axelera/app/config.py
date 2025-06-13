# Copyright Axelera AI, 2024
from __future__ import annotations

import argparse
import collections
import dataclasses
import enum
import os
from pathlib import Path
import re
import textwrap
from typing import Any, Callable, Optional, Tuple

from . import environ, logging_utils, utils, yaml_parser

LOG = logging_utils.getLogger(__name__)
MODEL_FOR_HELP_EXAMPLE = 'yolov8n-coco'


class DeployMode(enum.Enum):
    QUANTIZE = enum.auto()
    # this will be invoked automatically when inference is called with --pipe quantized
    QUANTIZE_DEBUG = enum.auto()
    QUANTCOMPILE = enum.auto()
    PREQUANTIZED = enum.auto()


class EmulatedBackend(enum.Enum):
    DEFAULT = 'NONE'
    CPU = 'CPU'
    QEMU = 'QEMU'
    AIPU = 'AIPU'


class Metis(enum.Enum):
    none = enum.auto()
    pcie = enum.auto()
    m2 = enum.auto()


class HardwareEnable(enum.Enum):
    detect = enum.auto()
    enable = enum.auto()
    disable = enum.auto()

    def __bool__(self) -> bool:
        return self == HardwareEnable.enable


env = environ.env
'''Access environment variable configuration switches in a consistent way.'''

_DETECTABLE_CAPS = ('vaapi', 'opencl', 'opengl')
_DETECTABLE_CAPS_AVAILABLE_ARGS = collections.defaultdict(list)
_DETECTABLE_CAPS_AVAILABLE_ARGS['opengl'] = [env.opengl_backend]

DEFAULT_MAX_EXECUTION_CORES = 4
'''The number of cores to execute on, this is the default for the AIPU.'''

DEFAULT_CORE_CLOCK = 800
'''The default core clock frequency to use for the AIPU.'''


class _HardwareEnableAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)
            if option_string.startswith('--enable-'):
                _option_strings.append('--disable-' + option_string[9:])
                _option_strings.append('--auto-' + option_string[9:])

        if help is not None and default is not None:
            help += f" (default: {default.name})"

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            val = HardwareEnable.enable
            if option_string.startswith('--disable-'):
                val = HardwareEnable.disable
            elif option_string.startswith('--auto-'):
                val = HardwareEnable.detect
            setattr(namespace, self.dest, val)

    def format_usage(self):
        return ' | '.join(self.option_strings)


@dataclasses.dataclass
class HardwareCaps:
    vaapi: HardwareEnable = HardwareEnable.disable
    opencl: HardwareEnable = HardwareEnable.disable
    opengl: HardwareCaps = HardwareEnable.detect
    aipu_cores: int = 4

    def detect_caps(self) -> HardwareCaps:
        '''Return a new HardwareCaps with any 'detect' value resolved.'''
        vals = [
            (n, getattr(self, n), getattr(utils, f'is_{n}_available')) for n in _DETECTABLE_CAPS
        ]
        conv = {True: HardwareEnable.enable, False: HardwareEnable.disable}
        new = {
            n: conv[detect(*_DETECTABLE_CAPS_AVAILABLE_ARGS[n])]
            for n, v, detect in vals
            if v == HardwareEnable.detect
        }
        return dataclasses.replace(self, **new)

    def as_argv(self) -> str:
        '''Convert the HardwareCaps to a string of command line arguments.'''
        values = [(n, getattr(self, n)) for n in _DETECTABLE_CAPS]
        defaults = HardwareCaps()
        conv = dict(detect='auto', disable='disable', enable='enable')
        enables = [f"--{conv[v.name]}-{n}" for n, v in values if v != getattr(defaults, n)]
        if self.aipu_cores != defaults.aipu_cores:
            enables += [f'--aipu-cores={self.aipu_cores}']
        return ' '.join(enables)

    @classmethod
    def from_parsed_args(cls, args: argparse.Namespace) -> HardwareCaps:
        '''Construct HardwareCaps from the parsed command line arguments.'''
        return cls(
            args.enable_vaapi,
            args.enable_opencl,
            args.enable_opengl,
            args.aipu_cores,
        )

    @classmethod
    def add_to_argparser(
        cls, parser: argparse.ArgumentParser, defaults: Optional[HardwareCaps] = None
    ) -> None:
        '''Add hardware caps arguments to the given argparse parser.'''
        defaults = defaults or cls.DETECT_ALL
        for cap in _DETECTABLE_CAPS:
            parser.add_argument(
                f'--enable-{cap}',
                dest=f'enable_{cap}',
                action=_HardwareEnableAction,
                default=getattr(defaults, cap),
                help=f'enable/disable/detect {cap} acceleration',
            )
        parser.add_argument(
            '--aipu-cores',
            type=int,
            choices=range(0, 5),
            default=defaults.aipu_cores,
            help='number of AIPU cores to use; supported options are %(choices)s; default is %(default)s; 0 means this model is not a deep learning model',
        )

    def enabled(self, cap: str) -> bool:
        '''Return True if the given cap is enabled.'''
        return getattr(self, cap) == HardwareEnable.enable


def range_check(min: int, max: int):
    def _range_check(value: str) -> int:
        value = int(value)
        if value < min or value > max:
            raise argparse.ArgumentTypeError(f"must be in range {min} to {max}")
        return value

    return _range_check


def add_compile_extras(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--num-cal-images',
        type=range_check(2, 2000),
        default=200,
        help='Specify the required number of images for model quantization. '
        'This value is rounded up to a multiple of the batch size if necessary. '
        'Minimum is 2 images, and the default is %(default)s.',
    )
    parser.add_argument(
        '--calibration-batch',
        type=int,
        default=1,
        help=argparse.SUPPRESS,  #'specify batch size for model quantization (default: %(default)s)',
    )
    parser.add_argument(
        '--cal-seed',
        type=int,
        default=0,
        # Internal use: Specify the seed for the torch.manual_seed which will affect the dataset shuffling.
        # We use it to experiment with different seeds to see the impact on the accuracy.
        help=argparse.SUPPRESS,
    )


HardwareCaps.ALL = HardwareCaps(
    HardwareEnable.enable, HardwareEnable.enable, HardwareEnable.enable
)
HardwareCaps.AIPU = HardwareCaps(
    HardwareEnable.disable, HardwareEnable.disable, HardwareEnable.enable
)
HardwareCaps.NONE = HardwareCaps(
    HardwareEnable.disable, HardwareEnable.disable, HardwareEnable.disable
)
HardwareCaps.DETECT_ALL = HardwareCaps(
    HardwareEnable.detect, HardwareEnable.detect, HardwareEnable.detect
)
HardwareCaps.OPENCL = HardwareCaps(
    HardwareEnable.disable, HardwareEnable.enable, HardwareEnable.enable
)


def gen_compilation_config(deploy_cores, user_cfg, deploy_mode):
    """Generate the compilation configuration based on the user configuration.

    NOTE: CompilerConfig is a Pydantic model that will validate the configuration
    when it is created or when a field is set. If the configuration is invalid,
    a ValueError or ValidationError will be raised.
    """

    from axelera.compiler.config import CompilerConfig, HostArch, HostOS, MulticoreMode

    # Deploy cores may be 0 for classical cv models. This is unsupported by the
    # CompilerConfig. This is a workaround since the CompilerConfig is discarded
    # ultimately for these models, and this avoids a larger refactor.
    deploy_cores = max(1, deploy_cores)

    compiler_config = CompilerConfig(
        multicore_mode=MulticoreMode.BATCH,  # Default to batch mode
        aipu_cores_used=deploy_cores,
        resources_used=0.25 * deploy_cores,
        host_processes_used=1,  # Will disable compiler-internal resource validation
        host_arch=HostArch.auto_detect(),
        host_os=HostOS.auto_detect(),
        # NOTE: If we enable this and pass the correct model name, the compiler will
        # automatically determine optimal settings based on the model name in the config.
        # This could replace the current model-card specific configuration, and would
        # probably help align the compiler behavior more closely with the app framework.
        configure_model_specific=False,
        model_name="model",
    )

    # Apply any manual/user overrides
    user_overrides = user_cfg.get('compilation_config', {})
    for key, value in user_overrides.items():
        setattr(compiler_config, key, value)

    if deploy_mode == DeployMode.QUANTIZE:
        return compiler_config

    elif deploy_mode == DeployMode.QUANTIZE_DEBUG:
        if not compiler_config.quantization_debug:
            compiler_config.quantization_debug = True
        return compiler_config

    else:
        return compiler_config


def positive_int_for_argparse(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"cannot be negative: {value}")
    return ivalue


def _window_size(value: str) -> Tuple[int, int]:
    if value == 'fullscreen':
        from . import display  # lazy import to avoid circular import

        return display.FULL_SCREEN
    m = re.match(r'(\d+)(?:[,x](\d+))?', value)
    if not m:
        raise argparse.ArgumentTypeError(f"cannot parse {value} as window size")
    w = int(m.group(1))
    h = int(m.group(2)) if m.group(2) else (w * 10 // 16)
    return max(100, w), max(100, h)


if not hasattr(argparse, 'BooleanOptionalAction'):

    class _BooleanOptionalAction(argparse.Action):
        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith('--'):
                    option_string = '--no-' + option_string[2:]
                    _option_strings.append(option_string)

            if help is not None and default is not None:
                help += f" (default: {default})"

            super().__init__(
                option_strings=_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string in self.option_strings:
                setattr(namespace, self.dest, not option_string.startswith('--no-'))

        def format_usage(self):
            return ' | '.join(self.option_strings)

    argparse.BooleanOptionalAction = _BooleanOptionalAction


def default_build_root() -> Path:
    return env.build_root


def default_data_root() -> Path:
    return env.data_root


def default_exported_root() -> Path:
    return env.exported_root


def add_nn_and_network_arguments(
    parser: argparse.ArgumentParser,
    network_yaml_info: yaml_parser.NetworkYamlInfo,
    default_network: str | None = None,
) -> None:
    example_yaml = next(iter(network_yaml_info.get_all_info())).yaml_path
    # Format the list of available networks without breaking names across lines
    all_yaml_names = sorted(network_yaml_info.get_all_yaml_names())
    # Safer approach to format the list with one model name per line
    valid_nets = '\n    '.join(all_yaml_names)
    nninfo = f"""network to run, this can be a path to a pipeline file, e.g.
    {example_yaml}
or it can be a shorthand name for a network YAML file, e.g. one of:
    {valid_nets}
"""
    netopts = {'default': '', 'nargs': '?'} if default_network else {}
    parser.add_argument('network', help=nninfo, **netopts)
    parser.add_argument(
        '--build-root',
        default=default_build_root(),
        type=str,
        metavar='PATH',
        help='specify build directory',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=default_data_root(),
        metavar='PATH',
        help='specify dataset download directory, or point to your existing dataset directory',
    )


class _MetisAction(argparse.Action):
    AUTO = 'auto'
    CHOICES = (
        [AUTO]
        + [m.name for m in Metis]
        + [m.name.replace('_', '-') for m in Metis if '_' in m.name]
    )

    def __call__(self, parser, namespace, value, option_string=None):
        v = Metis.none if value == self.AUTO else Metis[value.lower().replace('-', '_')]
        setattr(namespace, self.dest, v)


def add_metis_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--metis',
        default=Metis.none,
        action=_MetisAction,
        choices=_MetisAction.CHOICES,
        help=f'specify metis target for deployment (default: detect)',
    )


class _LlmArgumentParser(argparse.ArgumentParser):
    def __init__(
        self,
        network_yaml_info: yaml_parser.NetworkYamlInfo,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._network_yaml_info = network_yaml_info

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        ns = super().parse_args(args, namespace)
        _resolve_network(
            self,
            ns,
            self._network_yaml_info,
        )
        ns.build_root = Path(ns.build_root).expanduser().absolute()
        return ns


GENAI_SYSTEM_PROMPT = "Be concise. You are a chatbot for Axelera AI, a start-up building a state-of-the-art AI accelerator chip, called Metis. Answer user questions accurately and to the best of your ability. You run on the Metis chip."


def create_llm_argparser(
    network_yaml_info: yaml_parser.NetworkYamlInfo, **kwargs
) -> argparse.ArgumentParser:
    """Create an argument parser for LLM tasks."""
    parser = _LlmArgumentParser(network_yaml_info, **kwargs)
    add_nn_and_network_arguments(parser, network_yaml_info)

    # --- Simplified CLI modes and prompt handling ---
    parser.add_argument(
        '--ui',
        nargs='?',
        const='share',
        default=None,
        choices=[None, 'local', 'share', 'local_simple', 'share_simple'],
        help='Enable Gradio web UI mode. Use --ui local or --ui share for classic UI, --ui local_simple or --ui share_simple for the simplified native UI. If not set, runs in CLI mode.',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default=None,
        help='Prompt for single-prompt CLI mode. If not provided, runs in interactive CLI mode.',
    )
    parser.add_argument(
        '--no-history',
        action='store_true',
        default=False,
        help='Disable conversation history (stateless mode). Each message will be processed independently.',
    )
    parser.add_argument(
        '--pipeline',
        type=str.lower,
        default='transformers-aipu',
        choices=['transformers', 'transformers-aipu'],
        help=argparse.SUPPRESS,  # we hide this for now as it requires a different version of pytorch
        # help='Specify pipeline backend:\n'
        # '  - transformers: Model runs on CPU/GPU based on the Hugging Face Transformers library\n'
        # '  - transformers-aipu: Model runs on Axelera AIPU with Transformer\'s tokenizer on the host (default)\n',
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default=GENAI_SYSTEM_PROMPT,
        metavar='STR',
        help='Specify system prompt',
    )
    parser.add_argument(
        '--temperature', type=float, default=0, metavar='INT', help='Logits temperature'
    )
    parser.add_argument(
        '--show-stats',
        action='store_true',
        default=False,
        help='show performance statistics',
    )
    # Add CPU and temperature monitoring flags (implicitly enabled when --show-stats is used with transformers-aipu)
    parser.add_argument(
        '--show-cpu-usage',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable/disable CPU usage monitoring (default: on when --show-stats is used)',
    )
    parser.add_argument(
        '--show-temp',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable/disable temperature monitoring (default: on when --show-stats is used)',
    )
    parser.add_argument(
        '--rich-cli',
        action='store_true',
        default=False,
        help='Enable beautiful Rich-based CLI chat experience (colors, panels, markdown).',
    )
    parser.add_argument(
        '--tokenizer-dir',
        type=str,
        default=None,
        help='Path to a local directory containing tokenizer files (overrides tokenizer_url and HuggingFace)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to connect to the server (default: 7860)',
    )
    logging_utils.add_logging_args(parser)
    return parser


def create_inference_argparser(
    network_yaml_info: yaml_parser.NetworkYamlInfo | None = None,
    default_caps: HardwareCaps | None = None,
    default_network: str | None = None,
    default_show_stats=False,
    default_show_system_fps=True,
    default_show_device_fps=False,
    default_show_host_fps=False,
    default_show_cpu_usage=True,
    default_show_temp=True,
    default_show_stream_timing=False,
    unsupported_yaml_cond: Optional[Callable[[Any], bool]] = None,
    unsupported_reason: str = '',
    port=None,
    **kwargs,
) -> argparse.ArgumentParser:
    network_yaml_info = network_yaml_info or yaml_parser.get_network_yaml_info()
    parser = _InferenceArgumentParser(
        network_yaml_info,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f'\nExample: ./%(prog)s {MODEL_FOR_HELP_EXAMPLE} usb',
        unsupported_yaml_cond=unsupported_yaml_cond,
        unsupported_reason=unsupported_reason,
        default_network=default_network,
        **kwargs,
    )
    add_nn_and_network_arguments(parser, network_yaml_info, default_network=default_network)
    source_help = '''source input device(s); one or more of:
   video file (filename.mp4)
   usb camera (usb, usb:0, usb:1)
   uri (http://, https://, rtsp://)
   fakevideo (fakevideo:widthxheight)'''
    parser.add_argument('sources', default=[], nargs='*', help=source_help)
    parser.add_argument(
        '--pipe',
        type=str.lower,
        default='gst',
        choices=['gst', 'torch', 'torch-aipu', 'quantized'],
        help='specify pipeline type:\n'
        '  - gst: C++ pipeline based on GStreamer (uses AIPU)\n'
        '  - torch-aipu: PyTorch pre/post-processing with model on AIPU\n'
        '  - torch: PyTorch pipeline in FP32 for accuracy baseline (uses ONNXRuntime for ONNX models)\n'
        '  - quantized: PyTorch pipeline using Axelera mixed-precision quantized model on host',
    )
    parser.add_argument(
        '--frames',
        type=positive_int_for_argparse,
        default=0,
        help='Specify number of frames to process (0 for all frames).\n'
        'When using multiple sources, this is the total number of '
        'frames to process across all sources combined.',
    ),
    parser.add_argument(
        '--display',
        choices=['none', 'opengl', 'opencv', 'console', 'auto'],
        default='auto',
        help='display the results of the inference in a window. The window can be opengl, opencv,\n'
        'console (using ANSI control codes) or none. If auto then if DISPLAY is set then OpenGL\n'
        'is preferred over OpenCV, and if DISPLAY is not set then a console display is used.',
    )
    parser.add_argument(
        '--no-display',
        dest='display',
        action='store_const',
        const='none',
        help='This is an alias for --display=none',
    )
    default_window_size = (900, 600)
    parser.add_argument(
        '--window-size',
        type=_window_size,
        metavar='WxH | W | fullscreen',
        default=default_window_size,
        help=(
            'If --display sets the size of the window. Default is {}x{}.\n'
            'Size can be given as 800x600, just a width or fullscreen.'
        ).format(*default_window_size),
    )
    parser.add_argument(
        '--save-output',
        default='',
        metavar='PATH',
        help='save the inference result with annotations to a video file.\n'
        'For multiple streams use a format string with %%d for the stream index.\n'
        'e.g. "output_%%02d.mp4"',
    )
    parser.add_argument(
        '--timeout',
        default=5,
        type=float,
        metavar="SEC",
        help="specify timeout to wait for next inference in seconds",
    )
    HardwareCaps.add_to_argparser(parser, default_caps)
    add_compile_extras(parser)
    parser.add_argument(
        '--enable-hardware-codec',
        action='store_true',
        default=False,
        help='enable hardware video codec in an optimized GST pipeline',
    )
    parser.add_argument('--ax-precompiled-gst', type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        '--show-stats',
        action='store_true',
        default=default_show_stats,
        help='show performance statistics',
    )
    on_off = lambda x: 'on' if x else 'off'
    parser.add_argument(
        '--show-system-fps',
        action=argparse.BooleanOptionalAction,
        default=default_show_system_fps,
        help=f'show system FPS (default {on_off(default_show_system_fps)})',
    )
    parser.add_argument(
        '--show-device-fps',
        action=argparse.BooleanOptionalAction,
        default=default_show_device_fps,
        help=f'show device FPS (default {on_off(default_show_device_fps)})',
    )
    parser.add_argument(
        '--show-host-fps',
        action=argparse.BooleanOptionalAction,
        default=default_show_host_fps,
        help=f'show host FPS (default {on_off(default_show_host_fps)})',
    )
    parser.add_argument(
        '--show-temp',
        action=argparse.BooleanOptionalAction,
        default=default_show_temp,
        help=f'show AI Core temperatures (default {on_off(default_show_temp)})',
    )
    parser.add_argument(
        '--show-cpu-usage',
        action=argparse.BooleanOptionalAction,
        default=default_show_cpu_usage,
        help=f'show CPU usage (default {on_off(default_show_cpu_usage)})',
    )
    parser.add_argument(
        '--show-stream-timing',
        action=argparse.BooleanOptionalAction,
        default=default_show_stream_timing,
        help=f'show stream timing (latency and jitter) (default {on_off(default_show_stream_timing)})',
    )
    parser.add_argument(
        '--rtsp-latency',
        default=500,
        type=int,
        metavar="MSEC",
        help="specify latency for rtsp input in milliseconds",
    )
    parser.add_argument(
        '--frame-rate',
        default=0,
        type=int,
        metavar="FPS",
        help="""\
for gst-pipe only. Specify the frame rate for all of the input streams. If the input source is
unable to provide the video at the specified frame rate, the pipeline will drop or duplicate frames
from the input source to produce a stream of frames at the specified frame rate. If the frame rate
is set to 0, the pipeline will use the frame rate of each individual input sources.
""",
    )
    parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default='',
        help="""\
Comma separated list of devices to run on. Identifiers can be zero-based index, e.g. -d0,1 or by name, e.g.
-dmetis-0:1:0.  Use the tool axdevice to enumerate available devices.

$ axdevice
Device 0: metis-0:1:0 board_type=pcie fwver=v1.0.0-a6-15-g9d681b7bcfe9 clock=800MHz
Device 1: metis-0:3:0 board_type=pcie fwver=v1.0.0-a6-15-g9d681b7bcfe9 clock=800MHz
$ %(prog)s yolov5s-v7-coco media/traffic3_480p.mp4 -d0,1
$ %(prog)s yolov5s-v7-coco media/traffic3_480p.mp4 -dmetis-0:3:0

By default all devices available will be used.""",
    )

    parser.add_argument(
        "--tiled",
        default='',
        type=str,
        help=argparse.SUPPRESS,
        # "Enable tiled inference and specify the size of the tile. Default is (disabled).",
    )

    parser.add_argument(
        "--tile-overlap",
        default=0,
        type=int,
        help=argparse.SUPPRESS,
        # "Specify minimum amount of overlap as a percentage. Default is 0.",
    )

    parser.add_argument(
        "--tile-position",
        default='none',
        type=str,
        choices=['none', 'left', 'right', 'bottom', 'top'],
        help=argparse.SUPPRESS,
        # "Specify the position of the tile. Default is none.",
    )

    parser.add_argument(
        "--show-tiles",
        action='store_true',
        help=argparse.SUPPRESS,
        # "Specify whether tiles should be shouwn. Default is False.",
    )

    add_metis_arg(parser)

    if port is not None:
        parser.add_argument(
            '--port',
            default=port,
            help=f'port to connect to the server (default: {port})',
        )

    logging_utils.add_logging_args(parser)
    return parser


def determine_tile_options(args: argparse.Namespace) -> Optional[dict[str, Any]]:
    '''Convert the options from the command line into a dictionary of tile options or None.'''
    if not args.tiled and not args.tile_overlap and args.tile_position == "none":
        return None

    return {
        "tile": args.tiled if args.tiled else "default",
        "tile_overlap": args.tile_overlap if args.tile_overlap else 0,
        "tile_position": args.tile_position if args.tile_position else "none",
        "show_tiles": args.show_tiles,
    }


def _is_dataset(x: str) -> bool:
    return x == 'dataset' or x.startswith('dataset:') or x == 'server' or x.startswith('server:')


def _is_network(x: str, known_networks: list[str]) -> bool:
    return x in known_networks or (x.endswith('.yaml') and os.path.exists(x))


def _expand_source_path(x: str) -> str:
    if _is_dataset(x) or re.match(r'^\w+://', x):
        return x
    return str(Path(x).expanduser())


def _resolve_network(
    parser: argparse.ArgumentParser,
    ns: argparse.Namespace,
    network_yaml_info: yaml_parser.NetworkYamlInfo,
    unsupported_yaml_cond: Optional[Callable[[Any], bool]] = None,
    unsupported_reason: str = '',
    default_network: str | None = None,
) -> None:
    valid_nets = sorted(network_yaml_info.get_all_yaml_names())
    if not ns.network:
        if default_network:
            ns.network = default_network
        else:
            parser.error(
                f"The network argument is required, consider one of: {', '.join(valid_nets)}"
            )
    elif default_network and not _is_network(ns.network, valid_nets):
        ns.sources.insert(0, ns.network)
        ns.network = default_network

    try:
        yaml_info = network_yaml_info.get_info(ns.network)
    except KeyError as e:
        parser.error(str(e))
    if unsupported_yaml_cond and unsupported_yaml_cond(yaml_info):
        parser.error(
            f"Unsupported network '{ns.network}'{': ' + unsupported_reason if unsupported_reason else ''}"
        )
    ns.network = yaml_info.yaml_path


class _InferenceArgumentParser(argparse.ArgumentParser):
    '''A subclass of argparse.ArgumentParser that does some extra transforms
    after parsing inference arguments.
    '''

    def __init__(
        self,
        network_yaml_info: yaml_parser.NetworkYamlInfo,
        unsupported_yaml_cond: Optional[Callable[[Any], bool]] = None,
        unsupported_reason: str = '',
        default_network: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._network_yaml_info = network_yaml_info
        self._unsupported_yaml_cond = unsupported_yaml_cond
        self._unsupported_reason = unsupported_reason
        self._default_network = default_network

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        '''Parse the given args, and do some extra validation.'''
        ns = super().parse_args(args, namespace)
        if ns.pipe == 'torch' and ns.show_stats:
            LOG.info('Do not support torch pipeline for show-stats')
            ns.show_stats = False
        if ns.pipe == 'torch-aipu' and ns.aipu_cores > 1:
            LOG.info('torch-aipu pipeline supports aipu-cores=1 only')
            ns.aipu_cores = 1

        _resolve_network(
            self,
            ns,
            self._network_yaml_info,
            self._unsupported_yaml_cond,
            self._unsupported_reason,
            self._default_network,
        )
        ns.sources = [_expand_source_path(s) for s in ns.sources]
        ns.data_root = Path(ns.data_root).expanduser().absolute()
        ns.build_root = Path(ns.build_root).expanduser().absolute()

        if not ns.sources:
            self.error('No source provided')

        if len(ns.sources) > 1 and ns.save_output and '%' not in ns.save_output:
            self.error(
                f'--save-output requires a pattern for multistream input e.g. "output_%02d.mp4" (got: {ns.save_output})'
            )
        if len(ns.sources) > 1 and any(_is_dataset(x) for x in ns.sources):
            self.error(f"Dataset sources cannot be used with multistream")
        if _is_dataset(ns.sources[0]) or ns.pipe in ('torch', 'torch-aipu', 'quantized'):
            try:
                import torch  # noqa: just ensure torch is available
            except ImportError as e:
                if _is_dataset(ns.sources[0]) and ns.pipe.startswith('torch'):
                    msg = f'Dataset source and {ns.pipe} pipeline require torch to be installed'
                elif _is_dataset(ns.sources[0]):
                    msg = 'Dataset source requires torch to be installed'
                else:
                    msg = f'{ns.pipe} pipeline requires torch to be installed'
                self.error(f"{msg} : {e}")

        if ns.display == 'none':
            ns.display = False
            ns.enable_opengl = HardwareEnable.disable
        elif ns.display == 'opengl':
            ns.enable_opengl = HardwareEnable.enable
        elif not os.environ.get('DISPLAY') and ns.enable_opengl == HardwareEnable.detect:
            LOG.debug('DISPLAY not set, disabling OpenGL detection')
            ns.enable_opengl = HardwareEnable.disable
        return ns


class CompileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = DeployMode[values]
        setattr(namespace, 'mode', values)

        if values in (DeployMode.QUANTIZE, DeployMode.QUANTIZE_DEBUG):
            if not namespace.model and not namespace.models_only:
                setattr(namespace, 'models_only', True)
            setattr(namespace, 'pipeline_only', False)


def create_deploy_argparser(
    network_yaml_info: yaml_parser.NetworkYamlInfo,
    **kwargs,
) -> argparse.ArgumentParser:
    parser = _DeployArgumentParser(
        network_yaml_info,
        description='Deploy a model to Axelera platforms',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f'\nExample: ./%(prog)s {MODEL_FOR_HELP_EXAMPLE}',
        **kwargs,
    )
    add_nn_and_network_arguments(parser, network_yaml_info)

    deploy_group = parser.add_mutually_exclusive_group()
    deploy_group.add_argument(
        '--model',
        type=str,
        default='',
        help='compile a specific model in the given network YAML without deploying pipeline',
    )
    deploy_group.add_argument(
        '--models-only',
        action='store_true',
        default=False,
        help='compile all models in the given network YAML without deploying pipeline',
    )
    deploy_group.add_argument(
        '--pipeline-only',
        action='store_true',
        default=False,
        help='compile pipeline in network YAML with pre-compiled models',
    )
    parser.add_argument(
        "--mode",
        type=str.upper,
        default=DeployMode.PREQUANTIZED,
        choices=DeployMode.__members__,
        action=CompileAction,
        help="Specify the model deployment mode:\n"
        " - QUANTIZE: Quantize the model. (Will NOT deploy pipeline)\n"
        " - QUANTCOMPILE: Quantize and compile the model.\n"
        " - PREQUANTIZED: Compile from a pre-quantized model (default).\n",
    )
    add_metis_arg(parser)
    parser.add_argument(
        '--pipe',
        type=str.lower,
        required=False,
        default='gst',
        choices=['gst', 'torch', 'torch-aipu'],
        help='specify pipeline type; gst is always with AIPU',
    )
    parser.add_argument(
        "--export",
        action='store_true',
        default=False,
        help="Export quantized/compiled model to exported/<model_name>.zip",
    )
    logging_utils.add_logging_args(parser)
    HardwareCaps.add_to_argparser(parser)
    add_compile_extras(parser)
    return parser


class _DeployArgumentParser(argparse.ArgumentParser):
    '''A subclass of argparse.ArgumentParser that does some extra transforms
    after parsing deploy arguments.
    '''

    def __init__(
        self,
        network_yaml_info: yaml_parser.NetworkYamlInfo,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._network_yaml_info = network_yaml_info

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        ns = super().parse_args(args, namespace)

        if ns.pipe == 'torch-aipu' and ns.aipu_cores > 1:
            LOG.info('torch-aipu pipeline supports aipu-cores=1 only')
            ns.aipu_cores = 1

        _resolve_network(self, ns, self._network_yaml_info)

        ns.data_root = Path(ns.data_root).expanduser().absolute()
        ns.build_root = Path(ns.build_root).expanduser().absolute()
        return ns
