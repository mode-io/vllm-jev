"""Bounded image data URLs shared by image-aware decision adapters."""

import base64
import binascii
import io
import math

from PIL import Image

MAX_IMAGES_PER_REQUEST = 8


def validate_text(value: str) -> str:
    if any(
        marker in value
        for marker in (
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        )
    ):
        raise ValueError("text must not contain reserved image or video markers")
    return value


def load_image(value):
    if not isinstance(value, str) or not value.startswith(
        ("data:image/png;base64,", "data:image/jpeg;base64,")
    ):
        raise ValueError("images must be PNG/JPEG data URLs")
    encoded = value.split(",", 1)[1]
    if len(encoded) > 4 * math.ceil(8 * 1024 * 1024 / 3):
        raise ValueError("image exceeds 8 MiB")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("image exceeds 8 MiB")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("images must be PNG or JPEG")
            if image.width * image.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("image exceeds the pixel limit")
            return image.convert("RGB")
    except (ValueError, OSError, binascii.Error, Image.DecompressionBombError) as error:
        raise ValueError("invalid image") from error
