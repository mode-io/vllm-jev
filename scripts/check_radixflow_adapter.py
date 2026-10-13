"""Validate actual native source structure without importing GPU runtime code."""

import argparse
import ast
import json
from pathlib import Path
from unittest.mock import patch

from vllm_jev.radixflow import adapter

p = argparse.ArgumentParser()
p.add_argument("scheduler_source", type=Path)
a = p.parse_args()
source = a.scheduler_source.read_text()
tree = ast.parse(source)
cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
)
method = next(
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule"
)
method.decorator_list = []
method.returns = None
for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
    arg.annotation = None
function_source = ast.unparse(method)
namespace = {}
exec(compile(function_source, str(a.scheduler_source), "exec"), namespace)
with patch.object(adapter.inspect, "getsource", return_value=function_source):
    _, result = adapter.adapt_schedule(namespace["schedule"])
print(json.dumps(result, indent=2))
