#!/usr/bin/env python
# Copyright Axelera AI, 2025

import os
import threading
import time

try:
    import wx
except ImportError:
    print(
        "[ERROR] The 'wxPython' module is not installed.\n"
        "Please install it using pip.\n"
        "Run: pip install wxpython\n"
        "Exiting."
    )
    exit(1)
# Try to import CLIP, install if missing
try:
    import clip
except ImportError:
    print("Installing CLIP...")
    os.system("pip install git+https://github.com/openai/CLIP.git")
    import clip

import numpy as np
from scipy.ndimage import zoom
import torch

from axelera import types
from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.meta.segmentation import InstanceSegmentationMeta
from axelera.app.stream import create_inference_stream

W = 1280
H = 720
SURFACE_SIZE = (W, H)

SURFACE_REFRESH_INTERVAL = 1000 // 30  # ~30 FPS

tracers = inf_tracers.create_tracers('core_temp', 'cpu_usage', 'end_to_end_fps')
stream = create_inference_stream(
    network="fastsams-rn50x4-onnx",
    sources=[config.env.framework / "media/bowl-of-fruit.mp4@auto"],
    pipe_type='gst',
    log_level=logging_utils.INFO,  # INFO, DEBUG, TRACE
    hardware_caps=config.HardwareCaps(
        vaapi=config.HardwareEnable.detect,
        opencl=config.HardwareEnable.detect,
        opengl=config.HardwareEnable.detect,
    ),
    tracers=tracers,
    specified_frame_rate=7,
)


class WxViewer(wx.Frame):
    def __init__(self, app: display.App, size, stop: threading.Event, surface: display.Surface):
        super().__init__(parent=None, title="FastSAM demo", size=size)
        self._app = app
        self._stop = stop
        self._surface = surface
        self._sizer = wx.BoxSizer(wx.VERTICAL)
        self._prompt = 'fruits and vegetables'
        self._topk = 1

        self._bitmap_panel = wx.Panel(self)
        self._bitmap_sizer = wx.BoxSizer(wx.VERTICAL)
        self._bmp_ctrl = wx.StaticBitmap(self._bitmap_panel, -1)
        self._bitmap_sizer.Add(self._bmp_ctrl, 0, wx.ALIGN_CENTER)
        self._bitmap_panel.SetSizer(self._bitmap_sizer)

        self._sizer.Add(self._bitmap_panel, 1, wx.ALL | wx.EXPAND, 5)

        controls = wx.BoxSizer(wx.HORIZONTAL)

        prompt_label = wx.StaticText(self, label="Text prompt:")
        controls.Add(prompt_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.prompt_entry = wx.TextCtrl(self, value=self._prompt)
        controls.Add(self.prompt_entry, 3, wx.ALL | wx.EXPAND, 5)

        topk_label = wx.StaticText(self, label="Top-K:")
        controls.Add(topk_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.topk_entry = wx.TextCtrl(self, value=str(self._topk))
        controls.Add(self.topk_entry, 1, wx.ALL | wx.EXPAND, 5)

        update_btn = wx.Button(self, label="Update")
        update_btn.Bind(wx.EVT_BUTTON, self.on_update)
        controls.Add(update_btn, 0, wx.ALL, 5)

        self._sizer.Add(controls, 0, wx.ALL | wx.EXPAND, 5)
        self.SetSizer(self._sizer)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Bind(wx.EVT_SIZE, self._on_resize)

        self.Show()

    def _get_bitmap_size(self):
        client_size = self.GetClientSize()
        controls_height = 0
        if self._sizer.GetItemCount() > 1:
            controls_item = self._sizer.GetItem(1)
            if controls_item:
                controls_height = controls_item.GetMinSize().height

        w = max(1, client_size.width)
        h = max(1, client_size.height - controls_height)

        return w, h

    def _on_resize(self, event):
        self._bitmap_panel.Layout()
        event.Skip()

    def _scale_image(self, np_img):
        target_w, target_h = self._get_bitmap_size()
        img_h, img_w = np_img.shape[:2]
        scale_w = target_w / img_w
        scale_h = target_h / img_h
        scale = min(scale_w, scale_h)
        if scale != 1.0:
            np_img = zoom(np_img, (scale, scale, 1), order=1)
        return np_img

    def update_surfaces(self):
        called_at = time.time()
        if self._stop.is_set():
            self._on_close(wx.CloseEvent())
            return

        if new := self._surface.pop_latest():
            np_img = new.asarray(types.ColorFormat.RGB)
            np_img = self._scale_image(np_img)
            h, w = np_img.shape[:2]
            bmp = wx.Bitmap.FromBuffer(w, h, np_img)
            self._bmp_ctrl.SetBitmap(bmp)
            self._bitmap_panel.Layout()

        delay = max(1, SURFACE_REFRESH_INTERVAL - int((time.time() - called_at) * 1000))
        wx.CallLater(delay, self.update_surfaces)

    def _on_close(self, evt):
        self._stop.set()
        self.Destroy()

    def on_update(self, event):
        self._prompt = self.prompt_entry.GetValue()
        self._topk = self.topk_entry.GetValue()
        try:
            self._topk = int(self.topk_entry.GetValue())
        except ValueError:
            print("[Warning] Invalid topk value, must be integer.")
        print(f"[Prompt Updated] -> {self._prompt} | topk = {self._topk}")

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def topk(self) -> int:
        return self._topk


def _main(stream, surface: display.Surface, window: WxViewer):
    nr_boxes_to_process = 15
    last_prompt = None
    text_features = None
    model, preprocess = clip.load('RN50x4')

    wx.CallAfter(window.update_surfaces)  # Start polling for rendered frames in the UI

    surface.options(0, bbox_label_format="{scorep:.0f}{scoreunit}")

    for frame_result in stream:
        current_prompt = window.prompt

        if current_prompt != last_prompt:
            print(f"Updating text prompt to: {current_prompt}")
            text_input = clip.tokenize(current_prompt)
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            last_prompt = current_prompt

        meta = frame_result.meta['master_detections']
        nr_boxes = meta.boxes.shape[0]
        nr_boxes = min(nr_boxes, nr_boxes_to_process)

        topk = window.topk

        det_idxs = meta.secondary_frame_indices.get('detections', [])
        no_det_idxs = len(det_idxs)

        topk = min(no_det_idxs, topk)

        if no_det_idxs > 1:
            emb_tensors = []
            for idx in det_idxs:
                emb = meta.get_secondary_meta('detections', idx).embedding
                emb_tensor = torch.tensor(emb)
                emb_tensors.append(emb_tensor.squeeze())

            img_features = torch.stack(emb_tensors)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            similarity = 100.0 * img_features @ text_features.T

            idxs = torch.argsort(similarity.squeeze(), descending=True)
            top_idxs = idxs[0:topk]

            newmasks = [meta.masks[idx] for idx in top_idxs]
            newboxes = [meta.boxes[idx] for idx in top_idxs]
            newids = top_idxs
            newscores = [0] * topk

        elif no_det_idxs == 0:  # No detections
            newmasks = newboxes = newids = newscores = []

        else:  # Single detection, no filtering needed
            pass

        filtered_meta = InstanceSegmentationMeta(seg_shape=meta.seg_shape, labels=meta.labels)
        filtered_meta.add_results(
            newmasks,
            np.array(newboxes) if len(newboxes) else np.array([]).reshape(0, 4),
            np.array(newids),
            np.array(newscores),
        )
        frame_result.meta.delete_instance('master_detections')
        frame_result.meta.add_instance('master_detections', filtered_meta)

        surface.push(frame_result.image, frame_result.meta)


def main(stream, stop: threading.Event, surface: display.Surface, window: WxViewer):
    try:
        _main(stream, surface, window)
    finally:
        stop.set()


with display.App(renderer='opencv') as app:
    surface = app.create_surface(SURFACE_SIZE)
    stop = threading.Event()
    wx_app = wx.App(False)
    wx_wnd = WxViewer(app, (W, H), stop, surface)
    app.start_thread(main, (stream, stop, surface, wx_wnd), name="InferenceThread")
    wx_app.MainLoop()
stream.stop()
