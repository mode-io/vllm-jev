"""Bounded decoding and presentation-time sampling for short MP4 clips."""

import base64
import binascii
import io
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np


@dataclass
class VideoClip:
    frames: np.ndarray
    metadata: dict


def load_video(value, num_frames=8):
    """Sample one MP4 without fetching remote URLs or retaining its contents."""
    if type(num_frames) is not int or not 2 <= num_frames <= 16 or num_frames % 2:
        raise ValueError("video num_frames must be an even integer from 2 to 16")
    if not isinstance(value, str) or not value.startswith("data:video/mp4;base64,"):
        raise ValueError("videos must be MP4 data URLs")
    encoded = value.split(",", 1)[1]
    if len(encoded) > 4 * math.ceil(16 * 1024**2 / 3):
        raise ValueError("video exceeds 16 MiB")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as error:
        raise ValueError("invalid MP4") from error
    if len(raw) > 16 * 1024**2 or raw[4:8] != b"ftyp":
        raise ValueError("invalid MP4")
    # Isolate decoding to enforce a wall-time limit and contain decoder failures.
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            source.write(raw)
            source.flush()
            result = subprocess.run(
                [sys.executable, "-m", "vllm_jev.video", source.name, str(num_frames)],
                capture_output=True,
                timeout=15,
            )
    except subprocess.TimeoutExpired as error:
        raise ValueError("video decoding exceeded 15 seconds") from error
    if result.returncode:
        lines = result.stderr.decode(errors="replace").strip().splitlines()
        message = lines[-1] if lines else "video decoding failed"
        raise ValueError(message if result.returncode == 2 else "video decoding failed")
    with np.load(io.BytesIO(result.stdout), allow_pickle=False) as data:
        return VideoClip(data["frames"], json.loads(str(data["metadata"])))


def _decode(filename, num_frames):
    import cv2

    capture = cv2.VideoCapture(filename, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    try:
        if not capture.isOpened():
            raise ValueError("invalid MP4 video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not 0 < fps <= 60 or not 2 <= total <= 1800 or total / fps > 30:
            raise ValueError("video must be at most 30 seconds and 60 fps")
        if (
            min(width, height) <= 0
            or width * height > 1920 * 1080
            or max(width, height) > 1920
        ):
            raise ValueError("video resolution exceeds 1080p")
        if total < num_frames:
            num_frames = total - total % 2
        indices = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
        selected = []
        first_time = None
        timestamps = []
        constant_rate = True
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index = count
            count += 1
            if count > total or frame.shape[:2] != (height, width):
                raise ValueError("video frame metadata changed during decoding")
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(timestamp):
                raise ValueError("video must have frame timestamps")
            if first_time is None:
                first_time = timestamp
            timestamp -= first_time
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError("video timestamps must increase")
            timestamps.append(timestamp)
            # Keep only numerical rounding tolerance: small real timing jitter
            # must use presentation-time sampling as well.
            if not math.isclose(timestamp, index / fps, rel_tol=0, abs_tol=1e-6):
                constant_rate = False
            if index in indices:
                selected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if count != total or len(selected) != num_frames:
            raise ValueError("incomplete video")
    except cv2.error as error:
        raise ValueError("invalid MP4 video") from error
    finally:
        capture.release()
    if not constant_rate:
        duration = total / fps
        if timestamps[-1] >= duration:
            raise ValueError("video duration metadata is inconsistent")
        # Observe the frame displayed at evenly spaced times. This gives the
        # native video processor an actual uniform timeline, not guessed PTS.
        times = np.arange(num_frames) * duration / num_frames
        # A rounded boundary (e.g. 0.8999999999999999 vs 0.9) is the same instant.
        source_indices = np.searchsorted(timestamps, times + 1e-9, side="right") - 1
        selected.clear()
        frames = {}
        capture = cv2.VideoCapture(
            filename, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2]
        )
        try:
            for index in range(int(source_indices[-1]) + 1):
                ok, frame = capture.read()
                if not ok or frame.shape[:2] != (height, width):
                    raise ValueError("incomplete video")
                if index in source_indices:
                    frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as error:
            raise ValueError("invalid MP4 video") from error
        finally:
            capture.release()
        selected = [frames[index] for index in source_indices]
        total = num_frames
        fps = num_frames / duration
        indices = list(range(num_frames))
    return VideoClip(
        np.stack(selected),
        {
            "fps": fps,
            "total_num_frames": total,
            "duration": total / fps,
            "frames_indices": indices,
            "video_backend": "opencv",
        },
    )


if __name__ == "__main__":
    try:
        clip = _decode(sys.argv[1], int(sys.argv[2]))
        buffer = io.BytesIO()
        np.savez(buffer, frames=clip.frames, metadata=json.dumps(clip.metadata))
        sys.stdout.buffer.write(buffer.getvalue())
    except ValueError as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
