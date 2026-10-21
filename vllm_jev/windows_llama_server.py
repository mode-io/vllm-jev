"""Own a pinned CPU llama-server process; expose only tokenization and readout.

The Windows job closes with this process, including after forced termination.
It owns only the child we create, never a pre-existing inference server.
"""

import ctypes
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from ctypes import wintypes


class _Job:
    def __init__(self):
        class Basic(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t),
                ("active_process_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class IO(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in (
                    "read_operations",
                    "write_operations",
                    "other_operations",
                    "read_bytes",
                    "write_bytes",
                    "other_bytes",
                )
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("basic", Basic),
                ("io", IO),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, arguments, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            (
                "SetInformationJobObject",
                [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
                wintypes.BOOL,
            ),
            (
                "AssignProcessToJobObject",
                [wintypes.HANDLE, wintypes.HANDLE],
                wintypes.BOOL,
            ),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ):
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class CPUServer:
    """Private authenticated loopback endpoint with fixed, bounded CPU settings."""

    def __init__(self, executable, model, threads, *, timeout=180):
        if sys.platform != "win32" or type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError(
                "llama-server requires native Windows and 1 to 4 CPU threads"
            )
        self.process = self.job = self.log = None
        self.threads = threads
        self.key = secrets.token_urlsafe(32)
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith(("LLAMA_", "GGML_", "MTMD_"))
        }
        environment.update(LLAMA_API_KEY=self.key, OMP_NUM_THREADS=str(threads))
        self.command = [
            str(executable),
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--device",
            "none",
            "--gpu-layers",
            "0",
            "--ctx-size",
            "2048",
            "--parallel",
            "1",
            "--threads",
            str(threads),
            "--threads-batch",
            str(threads),
            "--threads-http",
            "2",
            "--batch-size",
            "1024",
            "--ubatch-size",
            "512",
            "--predict",
            "1",
            "--cache-ram",
            "0",
            "--no-cache-prompt",
            "--no-cont-batching",
            "--no-context-shift",
            "--no-warmup",
            "--no-webui",
            "--no-agent",
            "--offline",
            "--fit",
            "off",
        ]
        try:
            self.job = _Job()
            self.log = tempfile.TemporaryFile(mode="w+b")
            self.process = subprocess.Popen(
                self.command,
                cwd=executable.parent,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            self.job.assign(self.process)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    detail = self.log.read()[-6000:].decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"CPU llama-server exited during startup: {detail}"
                    )
                try:
                    # Authenticated request: a competing listener cannot be mistaken
                    # for our ready server after the ephemeral-port reservation.
                    response = self.post(
                        "/tokenize",
                        {"content": "A", "add_special": False, "parse_special": True},
                        timeout=1,
                    )
                    if response.get("tokens"):
                        return
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                time.sleep(0.1)
            raise TimeoutError("CPU llama-server did not become ready")
        except BaseException:
            self.close()
            raise

    def post(self, path, body, *, timeout=600):
        if path not in ("/tokenize", "/completion"):
            raise ValueError("unsupported internal llama-server operation")
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("CPU llama-server is closed")
        request = urllib.request.Request(
            self.url + path,
            json.dumps(body, allow_nan=False).encode("utf-8"),
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def close(self):
        if self.job is not None:
            self.job.close()
            self.job = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=15)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
