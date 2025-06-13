# Copyright Axelera AI, 2024
# helper functions for building gst pipelines
from __future__ import annotations

import enum
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable

from gi.repository import GObject, Gst

from .. import logging_utils

LOG = logging_utils.getLogger(__name__)
AGGREGATE_NAME = 'inference-task0'
_counters = {}


def _iter_to_list(name: str) -> Callable[..., list[Gst.Object]]:
    def _iter(o: Gst.Object, *args) -> list[Gst.Object]:
        gst_iter = getattr(o, name)(*args)
        items = []
        while 1:
            ok, value = gst_iter.next()
            if ok == Gst.IteratorResult.DONE:
                break
            items.append(value)
        return items

    return _iter


# equivalent to o.iterate_xyx(*args)
list_src_pads = _iter_to_list('iterate_src_pads')
list_sink_pads = _iter_to_list('iterate_sink_pads')
list_elements = _iter_to_list('iterate_elements')
list_all_by_element_factory_name = _iter_to_list('iterate_all_by_element_factory_name')


class InitState(enum.IntEnum):
    started = 0
    pipeline_created = enum.auto()
    connecting_elements = enum.auto()
    stream_starting = enum.auto()
    stream_ready = enum.auto()
    stream_paused = enum.auto()
    stream_playing = enum.auto()
    first_frame_received = enum.auto()


def _on_pad_added(element, newPad, sinkPad, prop, key):
    src_name = f"{element.get_name()}.{newPad.get_name()}"
    LOG.trace(f"Received new pad {src_name}")
    if sinkPad.is_linked() or newPad.is_linked():
        LOG.trace("We are already linked. Ignoring.")
        return
    template = newPad.get_pad_template()
    if template.get_name() != key:
        tname = template.get_name()
        LOG.trace(f"Pad template name {tname} does not match requested {key}. Ignoring.")
        return
    sink_parent = sinkPad.get_parent()
    sink_parent_name = "<none>" if sink_parent is None else sink_parent.get_name()
    sink_name = f"{sink_parent_name}.{sinkPad.get_name()}"
    linked_ok = newPad.link(sinkPad) == Gst.PadLinkReturn.OK
    log = LOG.trace if linked_ok else LOG.debug
    verb = "Linked" if linked_ok else "Failed to link"
    log(f"{verb} {src_name} with {sink_name}")
    if linked_ok:
        _set_pad_props(newPad, prop)


def _set_pad_props(pad, props):
    props = {k: v for k, v in props.items() if "." in k}
    if props:
        sprops = ', '.join(f"{k}={v}" for k, v in props.items())
        LOG.trace(f"Setting pad properties {pad.get_name()} : {sprops}")
    for k, v in props.items():
        pad_name, prop_name = k.split(".", 1)
        if pad.get_name() != pad_name:
            continue
        _set_element_or_pad_properties(pad, prop_name, v)


def _set_element_or_pad_properties(element_or_pad, propKey, propValueStr):
    if (prop := element_or_pad.find_property(propKey)) is None:
        raise RuntimeError(f'Failed to find property {propKey} in {element_or_pad.get_name()}')
    property_type = GObject.type_name(prop.value_type)
    if property_type == 'GstCaps':
        propValue = Gst.Caps.from_string(propValueStr)
    elif property_type == 'GstFraction':
        fracs = propValueStr.split('/')
        propValue = Gst.Fraction(int(fracs[0]), int(fracs[1]))
    elif property_type == 'GstElement':
        propValue = Gst.ElementFactory.make(propValueStr)
    elif property_type == 'GstValueArray':
        vals = propValueStr.split('/')
        propValue = Gst.ValueArray(vals)
    elif property_type == 'GValueArray':
        values = propValueStr.split(',')
        value_array = GObject.ValueArray.new(0)
        for value in values:
            gvalue = GObject.Value()
            gvalue.init(GObject.TYPE_DOUBLE)  # TODO add support for other gvalue types
            gvalue.set_double(float(value))
            value_array.append(gvalue)
        propValue = value_array

    else:
        propValue = propValueStr
    element_or_pad.set_property(propKey, propValue)


def _connect(key, connectionKey, element, elements, props):
    if "." in key:
        otherElementName, otherElementPadName = key.split(".", 1)
        otherElement = elements[otherElementName]
    else:
        otherElementName = key
        otherElement = elements[otherElementName]
    if otherElement is None:
        raise ValueError(f"{otherElementName} not found")

    if "%" in connectionKey:
        srcPadTemplate = element.get_pad_template(connectionKey)
        if srcPadTemplate is None:
            raise ValueError(f"Failed to find pad template {connectionKey} in {element.name}")
        sinkPad = otherElement.get_static_pad(otherElementPadName)
        _set_pad_props(sinkPad, props[otherElementName])
        if srcPadTemplate.presence == Gst.PadPresence.REQUEST:
            srcPad = element.request_pad(srcPadTemplate, None, None)
            LOG.trace(
                "Linking to request pad %s.%s with %s.%s ",
                element.get_name(),
                srcPad.get_name(),
                otherElementName,
                sinkPad.get_name(),
            )
            srcPad.link(sinkPad)
            _set_pad_props(srcPad, props[element.get_name()])
        else:
            LOG.trace("Deferring linking %s with %s ", element.get_name(), otherElementName)
            element.connect(
                "pad-added", _on_pad_added, sinkPad, props[element.get_name()], connectionKey
            )
    else:
        srcPad = element.get_static_pad(connectionKey)
        if "%" in otherElementPadName:
            sinkPadTemplate = otherElement.get_pad_template(otherElementPadName)
            sinkPad = otherElement.request_pad(sinkPadTemplate, None, None)
        else:
            sinkPad = otherElement.get_static_pad(otherElementPadName)
        LOG.trace(
            f"Explicit linking {element.get_name()}.{srcPad.get_name()} with {otherElementName}.{sinkPad.get_name()} "
        )
        srcPad.link(sinkPad)
        _set_pad_props(srcPad, props[element.get_name()])
        _set_pad_props(sinkPad, props[otherElementName])


_PREVIOUS = "__prev__"
_PENDING_CONNECTIONS = dict[str, Gst.Element | dict[str, str]]


def _update_counters(fullname):
    if not bool(re.search(r"\d+$", fullname)):
        _counters[fullname] = 0
        return

    match = re.match(r"^(.*?)(\d+)$", fullname)

    name = match.group(1)
    id = int(match.group(2))
    if name not in _counters:
        _counters[name] = id
    elif id > _counters[name]:
        _counters[name] = id


def _generate_name(element: dict[str, Any]):
    if name := element.get("name"):
        _update_counters(name)
        return name
    inst = element["instance"]
    if inst not in _counters:
        _counters[inst] = 0
    else:
        _counters[inst] += 1
    name = inst + str(_counters[inst])
    while name in _counters:
        _counters[inst] += 1
        name = inst + str(_counters[inst])
    element["name"] = name
    return name


def set_state_and_wait(element: Gst.Element, state: Gst.State, timeout=10):
    '''Set the state of a GStreamer element, and wait for it to take effect.'''
    name = element.get_name()
    LOG.debug(f"Setting state of {name} to {state}")
    res = element.set_state(state)
    if res == Gst.StateChangeReturn.FAILURE:
        # TODO should this be a warning?
        raise RuntimeError(f"Failed to set state of {name} to {state}")
    elif res in (Gst.StateChangeReturn.NO_PREROLL, Gst.StateChangeReturn.SUCCESS):
        return
    elif res == Gst.StateChangeReturn.ASYNC:
        start, taken = time.time(), 0
        while taken < timeout and (new_state := element.get_state(timeout=Gst.SECOND)[1]) != state:
            taken = time.time() - start
            LOG.debug(f"{name} is taking some time to reach {state} ({taken:.1f}s)")
        if new_state != state:
            raise RuntimeError(f"{name} failed to reach {state} state in {timeout} seconds")
    LOG.trace(f"State of {name} now {state}")


def remove_source(pipeline: Gst.Pipeline, agg_pad_name: str):
    def traverse_pad(pad):
        if not pad:
            return
        peer_pad = pad.get_peer()
        if peer_pad:
            parent_element = peer_pad.get_parent_element()
            if not parent_element:
                return
            if peer_pad.is_linked() and Gst.PadDirection.SRC == peer_pad.get_direction():
                parent_name = pad.get_parent_element().get_name()
                LOG.trace(f"Unlinking {parent_name} and {parent_element.get_name()}")
                peer_pad.unlink(pad)

            for npad in list_sink_pads(parent_element):
                traverse_pad(npad)

            set_state_and_wait(parent_element, Gst.State.NULL)
            LOG.debug(f"Removing {parent_element.get_name()}")
            pipeline.remove(parent_element)
            LOG.trace(f"Removed {parent_element.get_name()}")

    agg = pipeline.get_by_name(AGGREGATE_NAME)
    if agg is None:
        raise RuntimeError("Aggregator not found")

    for pad in list_sink_pads(agg):
        if pad.get_name() == agg_pad_name:
            traverse_pad(pad)
            break

    agg.release_request_pad(pad)


def _create_element(element: dict[str, Any]) -> Gst.Element:
    PSEUDO = ["instance", "name", "connections"]
    instance, name = element["instance"], element["name"]
    gst_element = Gst.ElementFactory.make(instance, name)
    if gst_element is None:
        raise RuntimeError(f"Failed to create element of type {instance} ({name})")

    props = [(k, v) for k, v in element.items() if k not in PSEUDO and "." not in k]
    sprops = ', '.join(f"{k}={v}" for k, v in props)
    LOG.trace(f"Creating {name} {instance}({sprops})")
    for k, v in props:
        _set_element_or_pad_properties(gst_element, k, v)
    return gst_element


def _connect_previous(
    pending_connections: _PENDING_CONNECTIONS,
    gst_element: Gst.Element,
    element: dict[str, Any],
    all_element_properties: dict[str, Any],
) -> None:
    '''Connect gst_element to the previous element in the pipeline.

    Previous element is retrieved from the pending_connections dict with the key _PREVIOUS.
    After the call the current element is saved to the pending_connections dict with the key
    _PREVIOUS if it does not a 'connections' property.

    Additionally we set any pad properties that are defined in the element properties, both for
    this element and the previous one, because its pad might be newly created.
    '''
    if (prev := pending_connections.pop(_PREVIOUS, None)) is not None:
        assert isinstance(prev, Gst.Element)
        prev.link(gst_element)
        for src_pad in list_src_pads(prev):
            _set_pad_props(src_pad, all_element_properties.get(prev.get_name(), {}))
            sink = src_pad.get_peer()
            if not sink:
                LOG.warning(f"Failed to get peer pad for {prev.get_name()}.{src_pad.get_name()}")
                continue
            _set_pad_props(sink, all_element_properties.get(gst_element.get_name(), {}))
    try:
        pending_connections[element["name"]] = element["connections"]
    except KeyError:
        pending_connections[_PREVIOUS] = gst_element


def _add_elements_to_pipeline(
    yaml_elements: list[dict],
    pipeline: Gst.Pipeline,
    play_rtsp: bool = False,
) -> Gst.Element:
    '''Add new elements to the pipeline. Connect them up as appropriate.'''
    gst_elements: dict[str, Gst.Element] = {}
    pending_connections: _PENDING_CONNECTIONS = {}
    props = {}
    is_live_src = False
    for element in yaml_elements:
        element_name = _generate_name(element)
        props[element_name] = element
        gst_element = _create_element(element)
        pipeline.add(gst_element)
        gst_elements[element_name] = gst_element
        _connect_previous(pending_connections, gst_element, element, props)
        if play_rtsp and (is_live_src or element["instance"].startswith('rtsp')):
            is_live_src = True
            gst_element.set_state(Gst.State.PLAYING)
    last_element = gst_element

    # the last element is either a sink or needs explicit connection
    pending_connections.pop(_PREVIOUS, None)
    gst_elements[AGGREGATE_NAME] = pipeline.get_by_name(AGGREGATE_NAME)
    props[AGGREGATE_NAME] = {}
    for element_name, element_connections in pending_connections.items():
        gst_element = gst_elements[element_name]
        for src_pad_name, connectTo in element_connections.items():
            _connect(connectTo, src_pad_name, gst_element, gst_elements, props)

    return last_element


def add_input(yaml_elements: list[dict], pipeline: Gst.Pipeline) -> str:
    '''Add a gst input to the pipeline. The input is defined in the yaml_elements dict.

    The pad name on the aggregate (axinferencenet) is returned.
    '''
    last_element = _add_elements_to_pipeline(yaml_elements, pipeline, play_rtsp=True)

    # the last element is the one that links to the aggregate
    src = list_src_pads(last_element)
    if len(src) != 1:
        raise RuntimeError(f"Expected 1 src pad, but got {len(src)}")
    peer = src[0].get_peer()
    if not peer:
        raise RuntimeError("Source pad is not connected to any sink pad")
    return peer.get_name()


def get_agg_pads(pipeline: Gst.Pipeline):
    agg = pipeline.get_by_name(AGGREGATE_NAME)
    return [pad.get_name() for pad in list_sink_pads(agg)]


def build_pipeline(yaml_elements: list[dict]):
    if not Gst.is_initialized():
        Gst.init(None)

    gst_pipeline = Gst.Pipeline()
    _add_elements_to_pipeline(yaml_elements, gst_pipeline)
    return gst_pipeline


def _dump_pipeline_graph(pipeline: Gst.Pipeline, pipeline_dot_file: Path) -> Path:
    pipeline_dot_file.write_text(Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL))
    out = pipeline_dot_file.with_suffix(".svg")
    try:
        # use svg because it is ~x10 faster than png
        subprocess.run(["dot", "-Tsvg", str(pipeline_dot_file), "-o", str(out)], check=True)
    except subprocess.CalledProcessError:
        LOG.warning("Error occurred while converting .dot to .svg")
    return out


def on_bus_message(message: Gst.Message, pipeline: Gst.Pipeline, logging_dir: Path = Path.cwd()):
    '''Callback function for watching the GST pipeline. Dump the pipeline graph to a .dot file
    and convert it to a .svg file once the stream starts playing.'''
    mType = message.type
    if mType == Gst.MessageType.EOS:
        LOG.debug("End of stream")
        exit(0)
    elif mType == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        LOG.error('%s: %s\n -> %s', message.src.get_name(), err.message, debug)
        try:
            pipeline.set_state(Gst.State.NULL)
        except Exception as e:
            # we really don't want to propagate any other error here
            LOG.error("Failed to set pipeline NULL state: %s", str(e))
        exit(1)
    elif mType == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        LOG.warning('%s: %s\n (%s)', message.src.get_name(), err.message, debug)
    elif mType == Gst.MessageType.STATE_CHANGED:
        old, new, pending = message.parse_state_changed()
        if LOG.isEnabledFor(logging_utils.TRACE) and pipeline == message.src:
            pipeline_dot_file = (
                logging_dir
                / f"pipeline_graph_{Gst.Element.state_get_name(old)}_to_{Gst.Element.state_get_name(new)}.dot"
            )
            graph = _dump_pipeline_graph(pipeline, pipeline_dot_file)
            LOG.trace(f"Pipeline state change: {pipeline.get_name()}, written graph to {graph}")
    elif mType == Gst.MessageType.STREAM_START and LOG.isEnabledFor(logging_utils.TRACE):
        pipeline_dot_file = logging_dir / "pipeline_graph_STREAM_START.dot"
        graph = _dump_pipeline_graph(pipeline, pipeline_dot_file)
        LOG.trace(f"Pipeline playing: {pipeline.get_name()}, written graph to {graph}")
    return True
