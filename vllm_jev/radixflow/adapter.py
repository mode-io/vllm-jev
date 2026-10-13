"""Validated, instance-local adaptation of the pinned native waiting scan."""

import ast
import hashlib
import inspect
import textwrap


def adapt_schedule(original):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    allocations = gates = zeroes = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        waiting_failure = (
            ast.unparse(node.test) == "new_blocks is None"
            and node.body
            and isinstance(node.body[-1], ast.Break)
            and any(
                isinstance(n, ast.If)
                and ast.unparse(n.test) == "request.has_encoder_inputs"
                for n in node.body
            )
        )
        zero = (
            ast.unparse(node.test) == "num_new_tokens == 0"
            and node.body
            and isinstance(node.body[-1], ast.Break)
        )
        if waiting_failure or zero:
            site = "allocation" if waiting_failure else "zero_tokens"
            node.body[-1] = ast.parse(f'''
if self._radixflow_reject(request, "{site}", num_scheduled_tokens):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
else:
    break
''').body[0]
            allocations += int(waiting_failure)
            zeroes += int(zero)
    # Admission happens at the native queue visit, retaining native queue ordering.
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        for i, stmt in enumerate(list(node.body)):
            if isinstance(stmt, ast.Assign) and ast.unparse(stmt) == (
                "request = request_queue.peek_request()"
            ):
                gate = ast.parse("""
if self._radixflow_hold(request):
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    continue
""").body[0]
                node.body.insert(i + 1, gate)
                gates += 1
    if allocations != 1 or gates != 1 or zeroes not in (0, 2):
        raise RuntimeError(
            f"unsupported native scheduler structure: "
            f"allocation={allocations}, gate={gates}, zero={zeroes}"
        )
    fn = tree.body[0]
    fn.decorator_list = []
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<radixflow-native-adapter>", "exec"), namespace)
    return namespace[fn.name], dict(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        allocation_sites=allocations,
        admission_sites=gates,
        zero_token_sites=zeroes,
    )
