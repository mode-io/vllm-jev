"""External candidate for dense, common-stride, layer-major alias layouts only."""


class UnsupportedGeometry(ValueError):
    pass


def covering_views(views, num_blocks):
    if type(num_blocks) is not int or num_blocks <= 0:
        raise UnsupportedGeometry("block count")
    groups = {}
    storage_steps = {}
    proof = []
    for v in views:
        if v.ndim < 1 or v.shape[0] <= 0 or v.shape[0] % num_blocks:
            raise UnsupportedGeometry("block folding")
        sub = v.shape[0] // num_blocks
        row = v.unflatten(0, (num_blocks, sub))[0]
        element = v.element_size()
        size = element
        axes = sorted(
            (int(s) * element, int(n)) for n, s in zip(row.shape, row.stride()) if n > 1
        )
        for stride, n in axes:
            if stride != size:
                raise UnsupportedGeometry("non-dense or overlapping row")
            size *= n
        if size != row.numel() * element:
            raise UnsupportedGeometry("extent")
        step = int(v.stride(0)) * sub * element
        if step < size:
            raise UnsupportedGeometry("overlapping blocks")
        storage = v.untyped_storage()
        offset = v.storage_offset() * element
        end = offset + (num_blocks - 1) * step + size
        if offset < 0 or end > storage.nbytes():
            raise UnsupportedGeometry("storage bounds")
        sk = (str(v.device), storage.data_ptr())
        if sk in storage_steps and storage_steps[sk] != step:
            raise UnsupportedGeometry("mixed block strides in storage")
        storage_steps[sk] = step
        key = (str(v.device), v.data_ptr())
        item = {
            "view": v,
            "storage": sk,
            "start": offset,
            "end": end,
            "bytes": size,
            "step": step,
        }
        if key in groups:
            old = groups[key]
            if old["storage"] != sk or old["start"] != offset or old["step"] != step:
                raise UnsupportedGeometry("alias contract")
            if size > old["bytes"]:
                groups[key] = item
        else:
            groups[key] = item
    selected = list(groups.values())
    for sk in storage_steps:
        intervals = sorted(
            (x["start"], x["end"]) for x in selected if x["storage"] == sk
        )
        if any(b > c for (a, b), (c, d) in zip(intervals, intervals[1:])):
            raise UnsupportedGeometry("different-base ranges overlap")
    for x in selected:
        proof.append({k: v for k, v in x.items() if k not in ("view", "storage")})
    return [x["view"] for x in selected], proof


def guarded_copy(views, num_blocks, pairs, native_copy):
    selected, proof = covering_views(list(views), num_blocks)
    native_copy(selected, num_blocks, pairs)
    return proof
