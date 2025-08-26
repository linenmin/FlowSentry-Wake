# Copyright Axelera AI, 2025
# Inference stream class
from __future__ import annotations

import collections
import copy
import dataclasses
import io
import itertools
import queue
import signal
import sys
import threading
from typing import TYPE_CHECKING, Iterable

from . import config, logging_utils, pipe, utils

LOG = logging_utils.getLogger(__name__)
_INTERRUPT_RAISED = object()  # sentinel for interrupt raised

if TYPE_CHECKING:
    from .inf_tracers import TraceMetric, Tracer


class InterruptHandler:
    def __init__(self, stream=None):
        self.stream = stream
        self._interrupted = threading.Event()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self)
            signal.signal(signal.SIGTERM, self)

    def __call__(self, *args):
        self._interrupted.set()
        if self.stream is not None:
            LOG.info("Interrupting the inference stream")
            return self.stream.stop()
        else:
            LOG.error('Unable to stop stream')
            sys.exit(1)

    def is_interrupted(self):
        return self._interrupted.is_set()


def _create_manager(
    system: config.SystemConfig,
    stream: config.InferenceStreamConfig,
    pipeline: config.PipelineConfig,
    deploy: config.DeployConfig,
    tracers: list[Tracer],
    render_config: config.RenderConfig,
    id_allocator: pipe.SourceIdAllocator,
) -> pipe.PipeManager:
    if not pipeline.network:
        raise ValueError("network must be specified")
    if not pipeline.sources:
        raise ValueError("sources must be specified")
    if pipeline.pipe_type not in ['gst', 'torch', 'torch-aipu', 'quantized']:
        raise ValueError(
            f"Invalid pipe type: {pipeline.pipe_type}, valid types are gst, torch, torch-aipu, quantized"
        )
    try:
        return pipe.PipeManager(
            system,
            stream,
            pipeline,
            deploy,
            tracers,
            render_config,
            id_allocator,
        )
    except TypeError as e:
        raise TypeError(f"Invalid PipeManager configuration: {str(e)}") from e


def _calculate_num_frames(pipelines, requested_frames):
    nframes = [p.number_of_frames for p in pipelines]
    combined = [x for x in itertools.chain(nframes, [requested_frames]) if x > 0]
    return min(combined) if combined else 0


class InferenceStream:
    """An iterator that launches the inference pipeline locally
    and yields inference results for each frame"""

    def __init__(
        self,
        system_config: config.SystemConfig,
        stream_config: config.InferenceStreamConfig,
        deploy_config: config.DeployConfig,
        default_pipeline_config: config.PipelineConfig,
        pipeline_configs: list[config.PipelineConfig] | None = None,
        tracers: list[Tracer] | None = None,
    ):
        self._system_config = system_config
        self._stream_config = stream_config
        self._deploy_config = deploy_config
        self._default_pipeline_config = default_pipeline_config
        any_eval_mode = any(p.eval_mode for p in pipeline_configs or [])
        self.timeout = (
            stream_config.timeout if stream_config.timeout > 0 and not any_eval_mode else None
        )
        self._queue = queue.Queue(maxsize=10)
        self._interrupt_raised = False
        self._pipelines: list[pipe.PipeManager] = []
        self._timer = utils.Timer()
        self._frames_requested = stream_config.frames
        self._frames_executed = 0
        self._stream_lock = threading.Lock()
        self._interrupt_handler = InterruptHandler(self)
        self._id_allocator = pipe.SourceIdAllocator()
        self.hardware_caps = system_config.hardware_caps.detect_caps()
        self._done = False
        self._queue_blocked = False
        self._tracers: list[Tracer] = tracers or []
        for pc in pipeline_configs:
            self.add_pipeline(pc)
        self._frames = _calculate_num_frames(self._pipelines, stream_config.frames)

    def pipeline_by_stream_id(self, stream_id: int) -> pipe.PipeManager:
        """Find a pipeline by its stream ID.

        If not found, raises ValueError.
        """
        for pipeline in self._pipelines:
            if stream_id in pipeline.sources:
                return pipeline
        raise ValueError(f"Stream ID {stream_id} not found in any pipeline")

    def add_pipeline(
        self, pipeline_config: config.PipelineConfig | None = None, **kwargs
    ) -> pipe.PipeManager:
        '''Add a new pipeline to the stream.

        This pipeline will execute in parallel with any existing pipelines, and
        results will be yielded as they become available in the same inference
        loop.

        The configuration can either be provided as a PipelineConfig object, or
        as keyword arguments that will be used to create a PipelineConfig object.
        '''
        pipeline_config = pipeline_config or copy.deepcopy(self._default_pipeline_config)
        pipeline_config.update_from_kwargs(kwargs)
        manager = _create_manager(
            self._system_config,
            self._stream_config,
            pipeline_config,
            self._deploy_config,
            self._tracers,
            None,
            self._id_allocator,
        )
        manager.init_pipe()
        manager.setup_callback(lambda result: self._feed_result(manager, result))
        self._pipelines.append(manager)
        return manager

    @property
    def manager(self):
        if len(self._pipelines) > 1:
            raise RuntimeError(
                "More than one pipeline in the stream, please use pipelines property with multiple pipelines"
            )
        return self._pipelines[0]

    @property
    def pipelines(self):
        return self._pipelines[:]

    @property
    def sources(self) -> dict[int, config.Source]:
        '''Return the list of input sources'''
        return collections.ChainMap(*(p.sources for p in self._pipelines))

    @property
    def frames(self) -> int:
        '''Return the number of frames to process, or 0 if there is no bounding condition.

        If all inputs are unbounded and the stream is not configured with a frame limit, this will
        be 0.

        If any input is bounded (eg a filesrc), or if the stream was configured to stop after a
        number this will return the minimum of all the non-zero boundaries.

        Typically used to implement a progress bar, it shold not be considered a reliable indicator
        of when the stream will finish, as the stream may be interrupted by a signal or other
        event.
        '''
        return self._frames_requested

    def __len__(self):
        return self._frames

    def _feed_result(self, pipeline: pipe.PipeManager, result: pipe.FrameResult | None):
        while not self._done:
            try:
                self._queue.put((pipeline, result), timeout=0.2)
                if self._queue_blocked and self._queue.qsize() < self._queue.maxsize // 2:
                    LOG.info(
                        f"InferencedStream is being processed quickly enough again (backlog={self._queue.qsize()})"
                    )
                    self._queue_blocked = False
                break
            except queue.Full:
                if not self._queue_blocked:
                    LOG.warning(
                        f"New inference data is ready, but the InferencedStream is not being processed fast enough (backlog={self._queue.qsize()})"
                    )
                self._queue_blocked = True

    def __iter__(self) -> Iterable[pipe.FrameResult]:
        if not self._pipelines:
            raise ValueError(
                "No pipeline configs provided, please add a pipeline using add_pipeline()"
            )
        for pipeline in self._pipelines:
            pipeline.run_pipe()
        self._timer.reset()
        try:
            n = 0
            while (not self.frames or n < self.frames) and self._pipelines:
                if self.is_interrupted:
                    LOG.debug("Stream interrupted, stopping...")
                    break
                try:
                    pipeline, result = self._queue.get(timeout=self.timeout)
                    if result is None and pipeline is not None:
                        if pipeline in self._pipelines:
                            LOG.debug(
                                f"Received None from {pipeline.network.name} removing pipeline"
                            )
                            self._pipelines.remove(pipeline)
                            self.report_summary(pipeline)
                        else:
                            LOG.warning(f"Received None from unknown pipeline {pipeline}")
                        continue
                    elif result is _INTERRUPT_RAISED or self._interrupt_raised:
                        LOG.debug("Stream interrupted, stopping...")
                        break
                    if pipeline.evaluator:
                        pipeline.evaluator.append_new_sample(result.meta)

                except queue.Empty:  # timeout
                    LOG.warning("Timeout for querying an inference")
                    raise RuntimeError('timeout for querying an inference') from None
                if n == 1:
                    # Reset the timer after the first two frames which are usually very slow
                    self._timer.reset()
                yield result
                n += 1
        finally:
            self._done = True
            self._frames_executed = max(n - 2, 0)
            self._timer.stop()
            for pipeline in self._pipelines:
                pipeline.stop_pipe()
                self.report_summary(pipeline)

    def stream_select(self, streams: Iterable[int] | str) -> None:
        '''Configure streams to be in paused or resumed state.

        Args:
            streams: A list of stream IDs that should be in the playing state.

        NOTE: This is only supported by GStreamer pipelines.
        NOTE: For compatiblity reasons a string of the form '0,2,3' is also supported, but
              this is deprecated and will raise a DeprecationWarning.
        '''

        used = set()
        all = set(streams)
        with self._stream_lock:
            for pipeline in self._pipelines:
                this_pipeline = pipeline.sources.keys() & all
                pipeline.stream_select(list(this_pipeline))
                used |= this_pipeline
        if used != all:
            LOG.warning("Did not find the correct pipeline for the following stream_id ")

    def get_stream_select(self) -> list[int]:
        '''Get the list of currently playing streams, as configured by stream_select().

        NOTE: This is only supported by GStreamer pipelines.
        '''

        sids = set()
        with self._stream_lock:
            for p in self._pipelines:
                sids |= p.get_stream_select()
        valid_ids = set(self.sources.keys())
        if paused := valid_ids - sids:
            spaused = ' '.join(str(x) for x in paused)
            LOG.debug(f"stream ids in source but not in stream_select (paused): {spaused}")
        if unknown := sids - valid_ids:
            sunknown = ' '.join(str(x) for x in unknown)
            LOG.error(f"stream ids in stream_select but not in sources (unknown): {sunknown}")
        return sorted(sids)

    def add_source(
        self,
        source: str | config.Source,
        source_id: int = -1,
        pipeline: pipe.PipeManager | int = 0,
    ) -> int:
        '''Add a new source to the pipeline.

        Args:
            source: The source to add.
            source_id: The source id of the source to add. If -1, a new source_id will be assigned.
            pipeline: Reference to the pipeline to add, or the index of the pipeline to add to.
        Returns:
            The source id of the new source.
        NOTE: This is only supported by GStreamer pipelines.
        '''
        source = config.Source(source)
        pipeline = pipeline or self._pipelines[pipeline]
        with self._stream_lock:
            pipeline.pause_pipe()
            source_id = pipeline.add_source(source, source_id)
            pipeline.play_pipe()
        LOG.info(f"Added new source: {source} as {source_id=}")
        return source_id

    def remove_source(self, source_id: int):
        '''Remove a source from the pipeline.

        Args:
            source_id: The source id of the source to remove.
        NOTE: This is only supported by GStreamer pipelines.
        '''
        pipelines = {sid: p for p in self._pipelines for sid in p.sources.keys()}
        pipeline = pipelines[source_id]
        with self._stream_lock:
            pipeline.pause_pipe()
            pipeline.remove_source(source_id)
            pipeline.play_pipe()

    @property
    def is_interrupted(self) -> bool:
        '''Returns True if the stream has been interrupted by a signal.'''
        return self._interrupt_raised or self._interrupt_handler.is_interrupted()

    def stop(self):
        self._interrupt_raised = True
        # put a None, but if the queue is full flush it, do this in a loop until we succeed
        # otherwise the queue may be being fed by the producer
        while 1:
            try:
                self._queue.put((None, _INTERRUPT_RAISED), timeout=0)  # unblock the queue
                break
            except queue.Full:
                pass
            while not self._queue.empty():
                try:
                    self._queue.get(timeout=0)
                except queue.Empty:
                    break

    def is_single_image(self) -> bool:
        '''True if any input stream is a single image.'''
        return any(p.is_single_image() for p in self._pipelines)

    def report_summary(self, pipeline: pipe.PipeManager) -> None:
        '''When evaluating, report the summary of the evaluation.'''
        if evaluator := pipeline.evaluator:
            duration_s = self._timer.time
            evaluator.evaluate_metrics(duration_s)
            output = io.StringIO()
            evaluator.write_metrics(output)
            LOG.info(output.getvalue().strip())

    def get_all_metrics(self) -> dict[str, TraceMetric]:
        '''Return all tracer metrics.

        The available tracer metrics will depend on those that were passed to the PipeManager
        (or create_inference_stream) at construction.

        See examples/application.py for an example of how to use this method.
        '''
        metrics = {}
        for p in self._pipelines:
            # TODO should we have tracers per pipeline or per InferenceStream?
            for t in p.tracers:
                metrics.update({m.key.strip('_'): m for m in t.get_metrics()})
        return metrics


def create_inference_stream(
    system_config: config.SystemConfig | None = None,
    stream_config: config.InferenceStreamConfig | None = None,
    pipeline_configs: config.PipelineConfig | list[config.PipelineConfig] | None = None,
    logging_config: config.LoggingConfig | None = None,
    deploy_config: config.DeployConfig | None = None,
    *,
    log_level: int | None = None,
    tracers: list[Tracer] | None = None,
    **kwargs,
) -> InferenceStream:
    """Factory function to create appropriate stream type.

    Args:
        system_config: Optional SystemConfig object, if not provided, a default will be created.
        stream_config: Optional InferenceConfig object, if not provided, a default will be created.
        pipeline_configs: Optional PipelineConfig objects, if not provided, and suitable kwargs are found
        logging_config: Optional LoggingConfig object, if not provided, a default will be created.
        deploy_config: Optional DeployConfig object, if not provided, a default will be created.
        then a PipelineConfig will be created.

    Additional keyword only arguments:
        log_level: Optional log level, if provided this will override that set in logging_config.
        tracers: Optional list of Tracer objects, if not provided no tracers will be configured.

    Additional keyword arguments:
        All other keyword arguments are passed to the SystemConfig, InferenceStreamConfig and
        PipelineConfig as appropriate this allows you to override any of the default values in the
        configs.  For example `allow_hardware_codec=True` will override the value of
        `allow_hardware_codec` in the SystemConfig (whether SystemConfig was passed in or a default
        created).

        **kwargs: these kwargs will override any of the settings in the above configs
    Returns:
        InferenceStream: Configured inference stream

    TODO        Blah de blah
    For example:

            parser = config.create_inference_argparser()
            args = parser.parse_args()
            stream = stream.create_inference_stream(
                config.SystemConfig.from_parsed_args(args),
                config.InferenceStreamConfig.from_parsed_args(args),
                config.PipelineConfig.from_parsed_args(args),
            )


    """
    if logging_config is None:
        logging_config = config.LoggingConfig()
    if log_level is not None:
        logging_config.console_level = log_level
    logging_utils.configure_logging(logging_config)

    system_config = system_config or config.SystemConfig()
    system_config.update_from_kwargs(kwargs)
    system_config.hardware_caps = system_config.hardware_caps.detect_caps()
    stream_config = stream_config or config.InferenceStreamConfig()
    stream_config.update_from_kwargs(kwargs)
    deploy_config = deploy_config or config.DeployConfig()
    deploy_config.update_from_kwargs(kwargs)
    default = config.PipelineConfig.from_kwargs(kwargs)
    if isinstance(pipeline_configs, config.PipelineConfig):
        # if a single pipeline config is passed, convert it to a list
        pipeline_configs = [pipeline_configs]
    pipeline_configs = pipeline_configs or []
    if default.sources or default.ax_precompiled_gst:
        # create a pipeline config from kwargs
        pipeline_configs.append(default)
        # then reset the defaults for the active fields
        d = config.PipelineConfig()
        default = dataclasses.replace(
            default, sources=d.sources, ax_precompiled_gst=d.ax_precompiled_gst
        )

    if kwargs:
        unexp = ', '.join(kwargs.keys())
        all_valid = (
            config.SystemConfig.valid_kwargs()
            | config.InferenceStreamConfig.valid_kwargs()
            | config.PipelineConfig.valid_kwargs()
            | config.DeployConfig.valid_kwargs()
        )
        valid = ', '.join(f"{k}" for k in all_valid)
        raise ValueError(f"Unexpected keyword arguments: {unexp}, valid kwargs are {valid}")
    return InferenceStream(
        system_config, stream_config, deploy_config, default, pipeline_configs, tracers
    )
