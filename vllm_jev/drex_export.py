"""Experimental merged Drex checkpoint preparation; no LoRA merge or remote code."""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from .checkpoint import sha256, verify_files

MODEL_ID = "nace-ai/drex-v2.0"
REVISION = "3e6d164cb5c8cf264eec167905cc2448d89ea97d"
PROTOCOL = "drex_pointer_v1"
ARCHITECTURE = "VllmDrexQwen35MoeForTokenEmbedding"
HEAD_FILE = "decision_head.safetensors"
MANIFEST_FILE = "decision_manifest.json"
MIN_FREE_BYTES = 32 * 1024**3
MARKERS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)


def validate_template(template):
    if template.config not in ({}, {"layout": "native"}):
        raise ValueError(
            "Experimental Drex supports only its unchanged native template"
        )


def _configuration(config):
    text = config.get("text_config", {})
    if (
        config.get("model_type") != "qwen3_5_moe"
        or text.get("model_type") != "qwen3_5_moe_text"
        or "linear_attention" not in text.get("layer_types", [])
        or config.get("auto_map")
        or text.get("auto_map")
    ):
        raise ValueError("Drex requires the native Qwen3.5 MoE hybrid architecture")
    if type(text.get("hidden_size")) is not int or text["hidden_size"] <= 0:
        raise ValueError("Invalid Drex hidden size")
    if (
        type(text.get("vocab_size")) is not int
        or text["vocab_size"] <= 0
        or type(text.get("max_position_embeddings")) is not int
        or text["max_position_embeddings"] <= 0
    ):
        raise ValueError("Invalid Drex vocabulary or context limit")
    return text


def _index(source):
    data = json.loads(
        (source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weights = data.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Drex requires a safetensors shard index")
    if any(
        not isinstance(k, str)
        or not isinstance(v, str)
        or Path(v).name != v
        or not v.endswith(".safetensors")
        for k, v in weights.items()
    ):
        raise ValueError("Drex shard index must use local safetensors filenames")
    for name in set(weights.values()):
        p = source / name
        if not p.is_file() or p.is_symlink():
            raise ValueError(f"Drex shard missing or symlinked: {name}")
    return data


def _head(meta, hidden):
    if not isinstance(meta, dict) or meta.get("state_bidir") is not False:
        raise ValueError(
            "Drex requires explicit state_bidir=false; bidirectional state is unsupported"
        )
    if (
        meta.get("option_isolation", False)
        or meta.get("special_embeddings", False)
        or meta.get("lora", 0)
    ):
        raise ValueError("Drex requires merged weights and native causal row readouts")
    dim = meta.get("head_dim", 256)
    temperature = meta.get("temperature", 1.0)
    if (
        type(dim) is not int
        or dim <= 0
        or type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Invalid Drex head dimension or temperature")
    shapes = {
        "q.weight": (dim, hidden),
        "q.bias": (dim,),
        "k.weight": (dim, hidden),
        "k.bias": (dim,),
    }
    head = meta.get("head")
    if not isinstance(head, dict) or set(head) != set(shapes):
        raise ValueError("Drex pointer head is incomplete")
    if any(
        not isinstance(head[k], torch.Tensor)
        or tuple(head[k].shape) != shape
        or head[k].dtype != torch.float32
        or not torch.isfinite(head[k]).all()
        for k, shape in shapes.items()
    ):
        raise ValueError(
            "Drex requires finite FP32 pointer weights with matching shapes"
        )
    return head, dim, float(temperature)


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        if (
            shutil.disk_usage(destination.parent).free
            < source.stat().st_size + MIN_FREE_BYTES
        ):
            raise OSError(
                "Insufficient disk headroom for cross-filesystem Drex weight copy"
            ) from None
        shutil.copy2(source, destination)


def export_drex(source, output, model_id=MODEL_ID, revision=REVISION):
    """Link original shards unchanged and publish a separate pointer head/config.

    The caller owns the empty output directory and its atomic publication. Local
    inputs are trusted by the operator; automatic downloads additionally use the
    pinned release file digests in prepare_drex.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text = _configuration(config)
    index = _index(source)
    meta = torch.load(source / "head.pt", map_location="cpu", weights_only=True)
    head, dim, temperature = _head(meta, text["hidden_size"])
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    markers = [tokenizer.convert_tokens_to_ids(t) for t in MARKERS]
    if len(set(markers)) != 5 or any(
        type(t) is not int
        or t < 0
        or t >= text["vocab_size"]
        or t == tokenizer.unk_token_id
        for t in markers
    ):
        raise ValueError("Drex control tokens are missing from the tokenizer")
    output.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(index["weight_map"].values())):
        _link_or_copy(source / name, output / name)
    # No downloaded Python, adapter or model-card code is copied/executed.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
        "generation_config.json",
        "LICENSE",
    ):
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    config["architectures"] = [ARCHITECTURE]
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    save_file({k: t.contiguous() for k, t in head.items()}, str(output / HEAD_FILE))
    manifest = dict(
        format="vllm-jev-decision-v1",
        prompt_protocol=PROTOCOL,
        architecture=ARCHITECTURE,
        source_repository=model_id,
        source_revision=revision,
        experimental=True,
        validation="synthetic_only",
        state_bidir=False,
        hidden_size=text["hidden_size"],
        head_dim=dim,
        temperature=temperature,
        markers=markers,
        max_options=255,
        max_length=min(131072, text["max_position_embeddings"]),
        default_max_length=min(16384, text["max_position_embeddings"]),
        prefix_caching=False,
        files={p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    )
    (output / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    verify_drex(output, full=True)
    return manifest


def verify_drex(path, *, full=True):
    path = Path(path).resolve()
    m = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    text = _configuration(config)
    if (
        m.get("format") != "vllm-jev-decision-v1"
        or m.get("prompt_protocol") != PROTOCOL
        or m.get("architecture") != ARCHITECTURE
        or config.get("architectures") != [ARCHITECTURE]
        or m.get("state_bidir") is not False
        or m.get("prefix_caching") is not False
        or m.get("hidden_size") != text["hidden_size"]
        or m.get("max_options") != 255
        or type(m.get("max_length")) is not int
        or not 1 <= m["max_length"] <= min(131072, text["max_position_embeddings"])
        or type(m.get("default_max_length")) is not int
        or not 1 <= m["default_max_length"] <= m["max_length"]
    ):
        raise ValueError("Invalid Drex checkpoint configuration")
    _head({**m, "head": load_file(str(path / HEAD_FILE))}, text["hidden_size"])
    markers = m.get("markers")
    if (
        not isinstance(markers, list)
        or len(markers) != 5
        or len(set(markers)) != 5
        or any(type(t) is not int or not 0 <= t < text["vocab_size"] for t in markers)
    ):
        raise ValueError("Invalid Drex control token IDs")
    n = verify_files(
        path,
        m,
        _index(path),
        (
            "config.json",
            "model.safetensors.index.json",
            HEAD_FILE,
            "tokenizer.json",
            "tokenizer_config.json",
        ),
        full=full,
    )
    return {
        "format": m["format"],
        "architecture": ARCHITECTURE,
        "checked_files": n,
        "full": full,
    }


def _release_digest(path, spec):
    import hashlib

    if spec.get("sha256"):
        return sha256(path)
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def prepare_drex(model_id, workspace, protocol="auto"):
    from huggingface_hub import snapshot_download

    from .drex_release import FILES
    from .native_models import _publish

    if model_id != MODEL_ID or protocol not in ("auto", PROTOCOL):
        raise ValueError("Unsupported Drex model or protocol")
    output = Path(workspace) / "checkpoint" / model_id
    if output.exists():
        m = json.loads((output / MANIFEST_FILE).read_text(encoding="utf-8"))
        if (
            m.get("source_repository") != MODEL_ID
            or m.get("source_revision") != REVISION
        ):
            raise ValueError("Cached Drex checkpoint has different provenance")
        verify_drex(output, full=True)
        return output
    source = Path(workspace) / "public" / model_id
    source.parent.mkdir(parents=True, exist_ok=True)
    # A Git LFS pointer or corrupt cache entry is not a downloaded weight.
    missing = sum(
        spec["size"]
        for name, spec in FILES.items()
        if not (source / name).is_file()
        or (source / name).stat().st_size != spec["size"]
        or _release_digest(source / name, spec) != (spec.get("sha256") or spec["oid"])
    )
    if shutil.disk_usage(source.parent).free < missing + MIN_FREE_BYTES:
        raise OSError(
            "Drex download needs its missing release bytes plus 32 GiB free headroom; use a larger --workspace"
        )
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        token=False,
        local_dir=source,
        allow_patterns=list(FILES),
    )
    for name, spec in FILES.items():
        p = source / name
        if p.stat().st_size != spec["size"]:
            raise ValueError(f"Drex release file size mismatch: {name}")
        if _release_digest(p, spec) != (spec.get("sha256") or spec["oid"]):
            raise ValueError(f"Drex release checksum mismatch: {name}")
    return _publish(
        output,
        model_id,
        REVISION,
        PROTOCOL,
        lambda d: export_drex(source, d, model_id, REVISION),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from .native_models import _publish

    print(
        _publish(
            args.output,
            MODEL_ID,
            REVISION,
            PROTOCOL,
            lambda d: export_drex(args.source, d),
        )
    )


if __name__ == "__main__":
    main()
