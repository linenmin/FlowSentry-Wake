#!/usr/bin/env python
# Copyright Axelera AI, 2025

import os
import threading

try:
    import tkinter as tk
    import tkinter.font as tkFont
except ImportError:
    print(
        "[ERROR] The 'tkinter' module is not installed.\n"
        "Please install it using your system package manager.\n"
        "For Ubuntu/Debian, run: sudo apt-get update && sudo apt-get install python3-tk\n"
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
import torch

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.meta.segmentation import InstanceSegmentationMeta
from axelera.app.stream import create_inference_stream

tracers = inf_tracers.create_tracers('core_temp', 'cpu_usage', 'end_to_end_fps')
stream = create_inference_stream(
    network="fastsams-rn50x4-onnx",
    sources=[config.env.framework / "media/bowl-of-fruit.mp4"],
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

shared_text_prompt = {"prompt": "fruits and vegetables", "topk": 1}  # default value


def prompt_input_window():
    """Creates a simple Tkinter window for text input with larger font."""

    def update_prompt():
        shared_text_prompt["prompt"] = prompt_entry.get()
        try:
            shared_text_prompt["topk"] = int(topk_entry.get())
        except ValueError:
            print("[Warning] Invalid topk value, must be integer.")
        print(
            f"[Prompt Updated] -> {shared_text_prompt['prompt']} | topk = {shared_text_prompt['topk']}"
        )

    root = tk.Tk()
    root.title("CLIP Prompt & Top-K Settings")

    # Define custom font
    custom_font = tkFont.Font(family="Helvetica", size=16)

    tk.Label(root, text="Text prompt for CLIP:", font=custom_font).pack(padx=10, pady=5)
    prompt_entry = tk.Entry(root, width=40, font=custom_font)
    prompt_entry.insert(0, shared_text_prompt["prompt"])
    prompt_entry.pack(padx=10, pady=5)

    tk.Label(root, text="Top-K value:", font=custom_font).pack(padx=10, pady=5)
    topk_entry = tk.Entry(root, width=10, font=custom_font)
    topk_entry.insert(0, str(shared_text_prompt["topk"]))
    topk_entry.pack(padx=10, pady=5)

    tk.Button(root, text="Update", command=update_prompt, font=custom_font).pack(pady=10)

    root.mainloop()


def main(window, stream):
    nr_boxes_to_process = 15
    last_prompt = None
    text_features = None
    model, preprocess = clip.load('RN50x4')

    for frame_result in stream:
        current_prompt = shared_text_prompt["prompt"]

        if current_prompt != last_prompt:
            print(f"Updating text prompt to: {current_prompt}")
            text_input = clip.tokenize(current_prompt)
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            last_prompt = current_prompt

        meta = frame_result.meta['master_detections']
        nr_boxes = meta.boxes.shape[0]
        nr_boxes = min(nr_boxes, nr_boxes_to_process)

        topk = shared_text_prompt.get("topk", 1)

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

        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)


# Start Tkinter prompt window in a background thread
tk_thread = threading.Thread(target=prompt_input_window, daemon=True)
tk_thread.start()

with display.App(
    visible=True,
    opengl=True,
    buffering=not stream.is_single_image(),
) as app:
    wnd = app.create_window("FastSAM demo", (1800, 1200))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run(interval=1 / 10)
stream.stop()
