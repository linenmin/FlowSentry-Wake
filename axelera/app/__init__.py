# Copyright Axelera AI, 2024
import warnings

warnings.filterwarnings(
    "ignore", module="torchvision", message="Failed to load image Python extension"
)
warnings.filterwarnings(
    "ignore",
    module="pytools",
    message="Unable to import recommended hash",
)

try:
    # noqa - we do this really early on to avoid some issues when imported later
    # gi.repository.Gst is required to set mainloop configs
    import gi

    gi.require_version('Gst', '1.0')
    gi.require_version("GstApp", "1.0")  # for try_pull_sample
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst, GstApp, GstVideo  # noqa
except ImportError:
    pass

try:
    import onnxruntime  # noqa - prevents a crash in the compiler
except ImportError:
    pass
