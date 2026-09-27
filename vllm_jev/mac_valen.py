"""Apple Silicon inference with the original Valen prompts and decision head."""

import json
import math
import os
import time
from collections import OrderedDict
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from threading import Event

import mlx.core as mx
from mlx_vlm.utils import load_model

from .endpoint import SystemOneRequest
from .mac import _MacService
from .valen import ValenService, _answer


class MacValenService(_MacService):
    def __init__(self, checkpoint: Path, model_id: str, executor: ThreadPoolExecutor):
        super().__init__(executor)
        # Leave room for CPU applications in Apple's shared memory pool.
        mx.set_cache_limit(1024**3)
        manifest = json.loads((checkpoint / "valen_manifest.json").read_text())
        self.compiler = ValenService(
            None, checkpoint, model_id, manifest.get("max_length", 8192)
        )
        self.model = load_model(checkpoint, strict=True)
        head = mx.load(str(checkpoint / "valen_head.safetensors"))
        self.head = {key: value.astype(mx.float32) for key, value in head.items()}
        self.model_id = model_id
        self._image_cache_size = (
            2 if os.environ.get("VLLM_JEV_MAC_IMAGE_CACHE") == "1" else 0
        )
        self._image_cache = OrderedDict()

    def _image_key(self, images):
        if not self._image_cache_size or len(images) != 1:
            return None
        image = images[0]
        if image.width * image.height * len(image.getbands()) > 8 * 1024**2:
            return None
        pixels = image.tobytes()
        if len(pixels) > 8 * 1024**2:
            return None
        return image.mode, image.size, pixels

    def _systemone(self, payload: SystemOneRequest, cancelled: Event) -> dict:
        started = time.perf_counter()
        if cancelled.is_set():
            raise CancelledError()
        questions, images, logical_tokens, compute_tokens = self.compiler._compile(
            payload
        )
        pixels = grid = image_features = None
        if images:
            key = self._image_key(images)
            cached = self._image_cache.get(key) if key is not None else None
            if cancelled.is_set():
                raise CancelledError()
            if cached is not None:
                pixels, grid, image_features = cached
                self._image_cache.move_to_end(key)
            else:
                processed = self.compiler.processor.image_processor(
                    images=images,
                    return_tensors="np",
                    **self.compiler.media_kwargs.get("images_kwargs", {}),
                )
                pixels = mx.array(processed["pixel_values"])
                grid = mx.array(processed["image_grid_thw"], dtype=mx.int32)
                dtype = self.model.vision_tower.patch_embed.proj.weight.dtype
                image_features, _ = self.model.vision_tower(pixels.astype(dtype), grid)
                mx.eval(image_features)
                if key is not None and (
                    pixels.nbytes + grid.nbytes + image_features.nbytes <= 64 * 1024**2
                ):
                    self._image_cache[key] = pixels, grid, image_features
                    if len(self._image_cache) > self._image_cache_size:
                        self._image_cache.popitem(last=False)

        answers = {}
        for identifier, kind, pairs, branches in questions:
            logits = []
            for ids, positions, decision_position in branches:
                if cancelled.is_set():
                    raise CancelledError()
                input_ids = mx.array([ids], dtype=mx.int32)
                features = self.model.get_input_embeddings(
                    input_ids,
                    pixel_values=pixels,
                    image_grid_thw=grid,
                    cached_image_features=image_features,
                )
                output = self.model.language_model(
                    input_ids,
                    **features.to_dict(),
                    return_hidden=True,
                    skip_logits=True,
                )
                hidden = output.hidden_states[-1][0]
                candidate = (
                    hidden[mx.array(positions)].astype(mx.float32)
                    @ self.head["candidate.weight"].T
                )
                decision = (
                    hidden[decision_position].astype(mx.float32)
                    @ self.head["decision.weight"].T
                )
                values = (candidate * decision).sum(-1) / math.sqrt(candidate.shape[-1])
                mx.eval(values)
                logits.extend(values.tolist())
            answers[identifier] = _answer(
                kind,
                [key for key, _ in pairs],
                [description for _, description in pairs],
                logits,
            )
        if cancelled.is_set():
            raise CancelledError()
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": logical_tokens, "output_tokens": 0},
            "internal_usage": {"compute_tokens": compute_tokens},
            "metadata": {"inference_seconds": time.perf_counter() - started},
        }

    async def systemone(self, payload: SystemOneRequest) -> dict:
        return await self._run(self._systemone, payload)

    def close(self) -> None:
        self._image_cache.clear()
        del self.model, self.head, self.compiler
        mx.clear_cache()
