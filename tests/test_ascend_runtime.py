"""Reject mixed dependency rows and malformed acceptance responses."""

import importlib.util
from pathlib import Path

import pytest

from vllm_jev.ascend_runtime import RUNTIMES, cann_version, runtime_issues


@pytest.mark.parametrize("native", list(RUNTIMES))
def test_complete_runtime_row(native):
    row = RUNTIMES[native]
    packages = {**row, "vllm": native + "+empty"}
    packages["torch"] += " +cpu".replace(" ", "")
    assert runtime_issues(packages, row["cann"]) == []


@pytest.mark.parametrize(
    "component",
    ["vllm-ascend", "torch", "torch-npu", "transformers", "triton-ascend", "cann"],
)
def test_mismatched_runtime_component_is_not_reported_ready(component):
    row = RUNTIMES["0.22.1"]
    packages = {**row, "vllm": "0.22.1"}
    cann = row["cann"]
    if component == "cann":
        cann = "8.5.0"
    else:
        packages[component] = "0.0.0"
    assert any(component in issue for issue in runtime_issues(packages, cann))


def test_toolkit_version_is_read_from_selected_runtime(tmp_path):
    (tmp_path / "opp").mkdir()
    (tmp_path / "opp/version.info").write_text("Version=9.0.0\nrequired_other=8.0\n")
    assert cann_version({"ASCEND_HOME_PATH": str(tmp_path)}) == "9.0.0"
    assert cann_version({}) is None


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/ascend_acceptance.py"
    spec = importlib.util.spec_from_file_location("ascend_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_answers_are_accepted():
    body = {
        "questions": {
            "c": {"type": "choice", "criteria": {"a": "first", "b": "second"}},
            "n": {"type": "noul"},
            "s": {"type": "score", "criteria": ["low", "high"]},
        }
    }
    result = {
        "answers": {
            "c": {
                "type": "choice",
                "probabilities": {"a": 0.4, "b": 0.6},
                "choice": "b",
            },
            "n": {"type": "noul", "noul": 0.7},
            "s": {"type": "score", "probabilities": {"0": 0.1, "1": 0.9}, "score": 0.9},
        }
    }
    runner().validate_response(result, body)


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"type": "choice"},
        {"type": "choice", "probabilities": {}, "choice": "a"},
        {
            "type": "choice",
            "probabilities": {"a": float("nan"), "b": 0.0},
            "choice": "a",
        },
        {"type": "choice", "probabilities": {"a": True, "b": False}, "choice": "a"},
        {"type": "choice", "probabilities": {"a": 0.1, "b": 0.2}, "choice": "a"},
        {"type": "choice", "probabilities": {"a": 0.4, "b": 0.6}, "choice": "unknown"},
    ],
)
def test_invalid_answers_cannot_pass_pressure_acceptance(answer):
    with pytest.raises(ValueError):
        runner().validate_response(
            {"answers": {"c": answer}},
            {
                "questions": {
                    "c": {"type": "choice", "criteria": {"a": "first", "b": "second"}}
                }
            },
        )


def test_doctor_records_monitor_timeout_instead_of_crashing(monkeypatch):
    import subprocess

    from vllm_jev import doctor

    monkeypatch.setattr(doctor.shutil, "which", lambda _: "/usr/local/bin/npu-smi")
    monkeypatch.setattr(doctor.platform, "platform", lambda: "test Linux")
    monkeypatch.setattr(doctor.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(doctor.platform, "python_version", lambda: "3.11.4")

    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired("npu-smi", 20)

    monkeypatch.setattr(doctor.subprocess, "run", timeout)
    monkeypatch.setattr(doctor, "version", lambda _: None)
    result = doctor.diagnose()
    assert not result["prerequisites_present"]
    assert result["npu_smi"]["returncode"] is None
    assert "npu-smi could not complete" in result["issues"]


def test_doctor_emits_json_and_nonzero_when_not_ready(monkeypatch, capsys):
    import json

    from vllm_jev import doctor

    monkeypatch.setattr(
        doctor,
        "diagnose",
        lambda smoke: {"prerequisites_present": False, "issues": ["test failure"]},
    )
    with pytest.raises(SystemExit) as error:
        doctor.main([])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["issues"] == ["test failure"]


def test_acceptance_failure_writes_a_report_even_in_optimized_python(tmp_path):
    import json
    import subprocess
    import sys
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"answers":{"route":{},"refund":{},"urgency":{}}}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "result.json"
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-O",
                str(
                    Path(__file__).resolve().parents[1] / "scripts/ascend_acceptance.py"
                ),
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--requests",
                "8",
                "--concurrency",
                "4",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert (
        report["successful"] == 0
        and report["failed"] == 8
        and report["status"] == "failed"
    )


@pytest.mark.parametrize(
    "arguments",
    [["--api-server-count", "2"], ["--api-server-count=2"], ["--api-server-count"]],
)
def test_legacy_api_worker_fanout_rejected(arguments):
    from vllm_jev.ascend_server import validate_api_count

    with pytest.raises(ValueError, match="one API-server"):
        validate_api_count(arguments)


@pytest.mark.parametrize(
    "arguments", [[], ["--api-server-count", "1"], ["--api-server-count=1"]]
)
def test_legacy_single_api_worker_supported(arguments):
    from vllm_jev.ascend_server import validate_api_count

    validate_api_count(arguments)
