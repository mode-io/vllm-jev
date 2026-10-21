import subprocess
import sys

import pytest

from vllm_jev.windows_llama_server import _Job


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job object lifecycle")
def test_job_close_stops_its_child_and_leaves_an_unassigned_process_alive():
    """Use two test-owned sleepers; never enumerate or stop unrelated processes."""
    children = []
    job = _Job()
    try:
        for _ in range(2):
            children.append(
                subprocess.Popen(
                    [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            )
        assigned, control = children
        job.assign(assigned)
        job.close()
        assert assigned.wait(timeout=5) is not None
        assert control.poll() is None
        job.close()
    finally:
        job.close()
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
