"""Build the dependency-free Rust index into a caller-owned runtime directory."""

import argparse
import shutil
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--rustc", default=shutil.which("rustc"))
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if not a.rustc:
    p.error("a Rust compiler is required; pass --rustc /path/to/rustc")
source = Path(__file__).resolve().parents[1] / "rust/radixflow/src/lib.rs"
a.output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        a.rustc,
        "--edition=2021",
        "--crate-type",
        "cdylib",
        "-O",
        str(source),
        "-o",
        str(a.output),
    ],
    check=True,
)
print(a.output.resolve())
