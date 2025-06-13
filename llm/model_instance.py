# Copyright Axelera AI, 2025
"""
model_instance.py
Unified model instance interface for LLM inference (AxInstance, TorchInstance, etc.)
"""

import hashlib
import json
from pathlib import Path

from llm.embedding_processor import EmbeddingProcessor
import numpy as np

from axelera.app import logging_utils, utils

LOG = logging_utils.getLogger(__name__)


class AxInstance:
    """
    Axelera AI accelerator model instance for LLM inference.
    """

    def __init__(self, yaml, build_root, ddr_requirement_gb):
        from axelera.runtime import Context

        # Extract number of cores from model name, default to 1 if not specified
        model_name = yaml.get('name', '')
        context = Context()
        available_devices = context.list_devices()

        qualified_devices = [
            d for d in available_devices if d.max_memory >= ddr_requirement_gb * (1024**3)
        ]
        if not qualified_devices:
            raise RuntimeError(
                f"{model_name} requires a card with at least {ddr_requirement_gb} GB of DDR memory."
                f"No device with the required memory is available. "
            )

        # Check if any devices are already in use (try to detect busy devices)
        best_device = None
        for device in qualified_devices:
            try:
                # Try to temporarily connect to the device to see if it's available
                temp_conn = context.device_connect(device=device)
                temp_conn.release()
                best_device = device
                LOG.info(
                    f"Selected available device {device.name} with {device.max_memory/(1024**3):.1f} GB memory"
                )
                break
            except Exception as e:
                LOG.warning(f"Device {device.name} appears to be in use or unavailable: {e}")

        # If we couldn't find an available device, use the first qualified one
        if not best_device:
            best_device = qualified_devices[0]
            LOG.warning(
                f"All qualified devices appear to be in use. Will try to use {best_device.name} anyway."
            )

        model_path = self._download_precompiled_network(yaml, build_root)
        embedding_file = self._download_embedding_file(yaml, build_root)
        self.ctx = Context()
        model = self.ctx.load_model(model_path)
        # we always allocate 4 sub-devices for LLMs because we want all L1/L2 caches to be used
        self.connection = self.ctx.device_connect(device=best_device, num_sub_devices=4)
        self.instance = self.connection.load_model_instance(model)

        self._embedding_processor = EmbeddingProcessor(embedding_file)
        self._outputs_logits = np.zeros(
            self._embedding_processor.get_embedding_shape()[0], dtype=np.int8
        )
        LOG.info(f"AxInstance initialized with model: {model_path} on device: {best_device.name}")

    def run(self, inputs):
        # inputs: (input_ids, embedding_features)
        input_ids, input_tokens = inputs
        inputs_int8 = input_tokens[0].view(np.int8)
        self.instance.run([inputs_int8], [self._outputs_logits])
        return [self._outputs_logits]

    @property
    def embedding_processor(self):
        return self._embedding_processor

    def _download_precompiled_network(self, yaml: dict, build_root: Path) -> Path:
        if isinstance(yaml['models'], list):
            raise ValueError("Generative AI network should have only one task")
        model_name, model = next(iter(yaml['models'].items()))

        build_root = build_root or config.default_build_root()
        precompiled_path = model['precompiled_path']
        if precompiled_path.startswith("build/"):
            precompiled_path = precompiled_path[len("build/") :]
        precompiled_path = build_root / precompiled_path
        model_path = build_root / f'{model_name}/model/model.json'

        log_path = build_root / "downloaded_models.json"
        log_data = {}
        if log_path.exists():
            with open(log_path, "r") as f:
                log_data = json.load(f)

        entry = log_data.get(model['precompiled_md5'])
        needs_download = True
        if model_path.exists() and entry:
            try:
                current_md5 = utils.generate_md5(model_path)
                if current_md5 == entry["model_md5"]:
                    LOG.info(f"Model already downloaded and verified: {model_path}")
                    needs_download = False
                else:
                    LOG.warning(
                        f"Model file exists but MD5 mismatch, will re-download: {model_path}"
                    )
            except Exception as e:
                LOG.warning(f"Failed to check MD5: {e}, will re-download.")
        elif not model_path.exists():
            LOG.info(f"Model file does not exist, will download: {model_path}")

        if needs_download:
            try:
                utils.download_and_extract_asset(
                    model['precompiled_url'], precompiled_path, model['precompiled_md5']
                )
                try:
                    model_md5 = utils.generate_md5(model_path)
                    log_data[model['precompiled_md5']] = {
                        "model_path": str(model_path),
                        "model_md5": model_md5,
                    }
                    with open(log_path, "w") as f:
                        json.dump(log_data, f, indent=2)
                except Exception as e:
                    LOG.warning(f"Failed to log model MD5: {e}")
            except Exception as e:
                LOG.error(e)
                LOG.trace_exception()
                raise e
        return model_path

    def _download_embedding_file(self, yaml: dict, build_root: Path) -> Path:
        model_name, model = next(iter(yaml['models'].items()))
        extra_kwargs = model['extra_kwargs']['llm']
        embeddings_url = extra_kwargs['embeddings_url']
        embeddings_name = embeddings_url.split('/')[-1]
        embedding_path = build_root / f"{model_name}/{embeddings_name}"
        utils.download(embeddings_url, embedding_path)
        return embedding_path


class TorchInstance:
    """
    Unified Torch model instance for LLM inference (using HuggingFace Transformers, auto device selection).
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()
        self.torch = torch
        LOG.info(f"TorchInstance initialized with model: {model_name} on device: {self.device}")

    def run(self, inputs):
        # inputs: (input_ids, embedding_features) or just input_ids
        try:
            input_ids = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            input_ids = self.torch.tensor(input_ids, device=self.device)
            with self.torch.no_grad():
                outputs = self.model(input_ids)
                return outputs.logits[:, -1, :].cpu().numpy()
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            LOG.error("GPU out of memory during generation")
            return None
