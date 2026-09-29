"""Prepare released pointer and vocabulary-readout decision checkpoints."""

import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from .checkpoint import sha256, verify_files

PROTOCOLS = (
    "kev_pointer_v1",
    "decider_slot_v1",
    "task_json_v1",
    "thisthat_slot_v1",
    "mica_labels_v1",
    "jevk5_letters_v1",
)
TEXT_ARCHITECTURES = {
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen3_5_text": "Qwen3_5ForCausalLM",
    "qwen3_5_moe_text": "Qwen3_5MoeForCausalLM",
}


def _index(path):
    index = path / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text())
        weights = data.get("weight_map")
        if (
            not isinstance(weights, dict)
            or not weights
            or any(
                not isinstance(name, str)
                or not isinstance(shard, str)
                or Path(shard).name != shard
                for name, shard in weights.items()
            )
        ):
            raise ValueError("decision shard index must use local filenames")
        return data
    weights = {}
    for shard in path.glob("model*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as file:
            weights.update({name: shard.name for name in file.keys()})
    if not weights:
        raise ValueError("checkpoint has no safetensors backbone")
    return {"weight_map": weights}


def _copy(source, output):
    for file in source.iterdir():
        if (
            file.is_file()
            and (
                file.name.startswith(
                    (
                        "model",
                        "tokenizer",
                        "vocab",
                        "merges",
                        "added_tokens",
                        "special_tokens",
                        "chat_template",
                    )
                )
                or file.name in ("config.json", "generation_config.json")
            )
            and file.suffix in (".json", ".safetensors", ".model", ".txt", ".jinja")
        ):
            if file.suffix == ".safetensors":
                try:
                    os.link(file, output / file.name)
                except OSError:
                    shutil.copy2(file, output / file.name)
            else:
                shutil.copy2(file, output / file.name)


def _labels(tokenizer, count=255):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    names = list(alphabet) + [a + b for a in alphabet for b in alphabet]
    labels = []
    for name in names:
        ids = tokenizer.encode(name, add_special_tokens=False)
        if len(ids) == 1:
            labels.append((name, ids[0]))
        if len(labels) == count:
            break
    if len(labels) != count or len({t for _, t in labels}) != count:
        raise ValueError("checkpoint cannot represent the requested option labels")
    return labels


def _vocabulary_weight(source: Path, token_ids):
    index = _index(source)
    candidates = (
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    )
    name = next((k for k in candidates if k in index["weight_map"]), None)
    if name is None:
        raise ValueError("decision vocabulary projection is missing")
    if name.endswith("embed_tokens.weight"):
        config = json.loads((source / "config.json").read_text())
        text = config.get("text_config", config)
        if (
            text.get("tie_word_embeddings", config.get("tie_word_embeddings"))
            is not True
        ):
            raise ValueError("untied decision checkpoint is missing its lm_head weight")
    with safe_open(
        source / index["weight_map"][name], framework="pt", device="cpu"
    ) as file:
        return file.get_tensor(name)[token_ids].clone()


def _text_backbone(output: Path, config: dict):
    """Export a composite Qwen language backbone without its vision/MTP modules."""
    if "text_config" not in config:
        return config
    text = dict(config["text_config"])
    kind = text.get("model_type")
    if kind not in ("qwen3_5_text", "qwen3_5_moe_text"):
        raise ValueError("unsupported composite decision backbone")
    for name in ("tie_word_embeddings", "dtype", "torch_dtype"):
        if name not in text and name in config:
            text[name] = config[name]
    text["architectures"] = [TEXT_ARCHITECTURES[kind]]
    text.pop("mtp_num_hidden_layers", None)
    text.pop("mtp_use_dedicated_embeddings", None)
    index = _index(output)
    renamed = {}
    destinations = set()
    for name in index["weight_map"]:
        if any(part in ("visual", "vision_tower", "mtp") for part in name.split(".")):
            continue
        destination = name
        for prefix, replacement in (
            ("model.language_model.", "model."),
            ("language_model.model.", "model."),
            ("language_model.lm_head.", "lm_head."),
        ):
            if name.startswith(prefix):
                destination = replacement + name[len(prefix) :]
                break
        if destination in destinations:
            raise ValueError("composite decision weights collide after text export")
        renamed[name] = destination
        destinations.add(destination)
    if not renamed:
        raise ValueError("composite decision checkpoint has no language weights")
    weights = {}
    for shard in set(index["weight_map"].values()):
        names = [
            name for name, filename in index["weight_map"].items() if filename == shard
        ]
        kept = [name for name in names if name in renamed]
        if not kept:
            (output / shard).unlink()
            continue
        if len(kept) != len(names) or any(renamed[name] != name for name in kept):
            with safe_open(output / shard, framework="pt", device="cpu") as file:
                tensors = {renamed[name]: file.get_tensor(name) for name in kept}
                # _copy may have hard-linked the source shard. Replace the directory
                # entry with a new file; never write through that source hard link.
                temporary = output / f".{shard}.text"
                try:
                    save_file(tensors, temporary, metadata={"format": "pt"})
                    temporary.replace(output / shard)
                finally:
                    temporary.unlink(missing_ok=True)
        weights.update({renamed[name]: shard for name in kept})
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights}, indent=2) + "\n"
    )
    (output / "config.json").write_text(json.dumps(text, indent=2) + "\n")
    return text


def export_jevk5(source: Path, output: Path, model_id: str, revision: str):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    labels = list("ABCDEFGHIJKLMNOP")
    encoded = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(row) != 1 for row in encoded):
        raise ValueError("JevK5 answer letters must be single tokens")
    calibration = json.loads((source / "jevk5_config.json").read_text())
    ids = [row[0] for row in encoded]
    weight = _vocabulary_weight(source, ids)
    _copy(source, output)
    return _write(
        output,
        model_id,
        revision,
        {
            "prompt_protocol": "jevk5_letters_v1",
            "max_options": 16,
            "max_length": 16384,
            "temperature": calibration.get("temperature", 1.0),
            "labels": labels,
            "label_token_ids": ids,
        },
        {"weight": weight},
    )


def mica_labels(tokenizer):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    names = list(alphabet + alphabet.lower())
    for left in alphabet:
        for right in alphabet:
            label = left + right
            if len(tokenizer.encode(label, add_special_tokens=False)) == 1:
                names.append(label)
            if len(names) == 255:
                return names
    raise ValueError("Mica requires its complete single-token codebook")


def export_mica(source: Path, output: Path, model_id: str, revision: str):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    labels = mica_labels(tokenizer)
    encoded = [
        tokenizer.encode(label, add_special_tokens=False)
        for label in ["No", "Yes", *labels]
    ]
    if any(len(row) != 1 for row in encoded):
        raise ValueError("Mica codebook contains a multi-token label")
    ids = [row[0] for row in encoded]
    if len(set(ids)) != len(ids):
        raise ValueError("Mica codebook labels collide")
    calibration = (
        json.loads((source / "calibration.json").read_text())
        if (source / "calibration.json").is_file()
        else {}
    )
    if calibration.get("noul_logit_bias", 0):
        raise ValueError("this Mica release requires a Noul calibration offset")
    weight = _vocabulary_weight(source, ids)
    _copy(source, output)
    return _write(
        output,
        model_id,
        revision,
        {
            "prompt_protocol": "mica_labels_v1",
            "max_options": 255,
            "temperature": calibration.get("temperature", 1.0),
            "labels": labels,
            "label_token_ids": ids,
        },
        {"weight": weight},
    )


def _write(output, model_id, revision, specification, head):
    config = json.loads((output / "config.json").read_text())
    text = _text_backbone(output, config)
    if text.get("model_type") not in TEXT_ARCHITECTURES:
        raise ValueError("unsupported decision backbone")
    if not all(torch.isfinite(value).all() for value in head.values()):
        raise ValueError("nonfinite decision head")
    save_file(
        {k: v.float().contiguous() for k, v in head.items()},
        output / "decision_head.safetensors",
    )
    index = _index(output)
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    manifest = {
        "format": "vllm-jev-decision-v1",
        "source_repository": model_id,
        "source_revision": revision,
        "hidden_size": text["hidden_size"],
        "max_length": min(32768, text.get("max_position_embeddings", 32768)),
        **specification,
        "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "decision_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    verify_decision(output, full=True)
    return manifest


def export_decider(source: Path, output: Path, model_id: str, revision: str):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    calibration = json.loads((source / "decider_config.json").read_text())
    if calibration.get("layout", "plain") != "plain":
        raise ValueError("this Decider checkpoint needs a chat-layout protocol")
    if "text_config" in config:
        raise ValueError("this checkpoint needs the multimodal Decider protocol")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    labels = _labels(tokenizer)
    weight = _vocabulary_weight(source, [token for _, token in labels])
    if weight.shape != (255, text["hidden_size"]):
        raise ValueError("invalid vocabulary readout")
    _copy(source, output)
    return _write(
        output,
        model_id,
        revision,
        {
            "prompt_protocol": "decider_slot_v1",
            "max_options": min(255, calibration.get("max_options", 255)),
            "temperature": calibration.get("temperature", 1.0),
            "temperatures": calibration.get("temperature_by_type", {}),
            "isolated_levels": calibration.get("isolated_levels", True),
            "labels": [name for name, _ in labels],
            "label_token_ids": [token for _, token in labels],
        },
        {"weight": weight},
    )


def export_thisthat(source: Path, output: Path, model_id: str, revision: str):
    import tempfile

    with tempfile.TemporaryDirectory(
        prefix=".thisthat-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        _copy(source, staging)
        (staging / "decider_config.json").write_text(
            json.dumps({"temperature": 1.0, "isolated_levels": False})
        )
        manifest = export_decider(staging, output, model_id, revision)
        manifest["prompt_protocol"] = "thisthat_slot_v1"
        (output / "decision_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        verify_decision(output, full=True)
        return manifest


def export_task_json(source: Path, output: Path, model_id: str, revision: str):
    from .decision_protocols import task_prompt

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    names = list("ABCDEFGHIJKLMNOPQRSTUVWX")
    prompt = task_prompt(
        tokenizer,
        "sample",
        "Choose one.",
        ["first", "second"],
        ["First option", "Second option"],
    )
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = []
    for name in names:
        tokens = tokenizer.encode(prompt + name, add_special_tokens=False)
        if tokens[:-1] != prefix:
            raise ValueError("task answer is not one additional token")
        ids.append(tokens[-1])
    weight = _vocabulary_weight(source, ids)
    _copy(source, output)
    return _write(
        output,
        model_id,
        revision,
        {
            "prompt_protocol": "task_json_v1",
            "max_options": 24,
            "temperature": 1.0,
            "labels": names,
            "label_token_ids": ids,
        },
        {"weight": weight},
    )


def export_task_adapter(
    base: Path, source: Path, output: Path, model_id: str, revision: str
):
    import tempfile

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    with tempfile.TemporaryDirectory(
        prefix=".decision-merge-", dir=output.parent
    ) as temporary:
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, local_files_only=True
        )
        model = PeftModel.from_pretrained(
            model, source, local_files_only=True
        ).merge_and_unload()
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer_source = source if (source / "tokenizer.json").exists() else base
        AutoTokenizer.from_pretrained(
            tokenizer_source, local_files_only=True
        ).save_pretrained(temporary)
        return export_task_json(Path(temporary), output, model_id, revision)


def export_kev(base: Path, source: Path, output: Path, model_id: str, revision: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    meta = torch.load(source / "head.pt", map_location="cpu", weights_only=True)
    if meta.get("option_isolation", False):
        raise ValueError(
            "option-isolated Kev needs a native attention-mask implementation"
        )
    if meta.get("special_embeddings", False):
        raise ValueError("this Kev release needs its trained delimiter embeddings")
    adapter = json.loads((source / "adapter_config.json").read_text())
    if adapter["base_model_name_or_path"] != meta["base"]:
        raise ValueError("Kev adapter and head refer to different bases")
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, local_files_only=True
    )
    backbone = PeftModel.from_pretrained(
        model.model, source, local_files_only=True
    ).merge_and_unload()
    config = backbone.config.to_dict()
    if config.get("model_type") not in TEXT_ARCHITECTURES:
        raise ValueError("unsupported Kev backbone")
    config["architectures"] = [TEXT_ARCHITECTURES[config["model_type"]]]
    config["tie_word_embeddings"] = True
    state = {
        "model." + k: v.detach().contiguous() for k, v in backbone.state_dict().items()
    }
    save_file(state, output / "model.safetensors")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.save_pretrained(output)
    markers = [
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|fim_suffix|>",
    ]
    marker_ids = [tokenizer.convert_tokens_to_ids(marker) for marker in markers]
    if len(set(marker_ids)) != 5 or any(
        i is None or i >= config["vocab_size"] for i in marker_ids
    ):
        raise ValueError("Kev delimiters are missing from the backbone vocabulary")
    head = meta.get("head", {})
    if set(head) != {"q.weight", "q.bias", "k.weight", "k.bias"}:
        raise ValueError("unsupported Kev pointer head")
    return _write(
        output,
        model_id,
        revision,
        {
            "prompt_protocol": "kev_pointer_v1",
            "max_options": 255,
            "temperature": meta.get("temperature", 1.0),
            "markers": marker_ids,
            "foundation_repository": meta["base"],
            "foundation_revision": meta.get("base_revision"),
        },
        head,
    )


def verify_decision(path: Path, *, full=True):
    path = path.resolve()
    manifest = json.loads((path / "decision_manifest.json").read_text())
    if (
        manifest.get("format") != "vllm-jev-decision-v1"
        or manifest.get("prompt_protocol") not in PROTOCOLS
    ):
        raise ValueError("unrecognized decision checkpoint")
    if not isinstance(manifest.get("hidden_size"), int) or manifest["hidden_size"] <= 0:
        raise ValueError("invalid decision hidden size")
    if (
        not 1 <= manifest.get("max_options", 0) <= 255
        or not 1 <= manifest.get("max_length", 0) <= 32768
    ):
        raise ValueError("invalid decision checkpoint limits")
    temperatures = [
        manifest.get("temperature"),
        *manifest.get("temperatures", {}).values(),
    ]
    if any(
        not isinstance(t, (float, int)) or not math.isfinite(t) or t <= 0
        for t in temperatures
    ):
        raise ValueError("invalid decision calibration")
    head = load_file(str(path / "decision_head.safetensors"))
    h = manifest["hidden_size"]
    if manifest["prompt_protocol"] != "kev_pointer_v1":
        labels = manifest.get("labels")
        ids = manifest.get("label_token_ids")
        extra = 2 if manifest["prompt_protocol"] == "mica_labels_v1" else 0
        if (
            not isinstance(labels, list)
            or len(labels) != manifest["max_options"]
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
            or not isinstance(ids, list)
            or len(ids) != len(labels) + extra
            or any(type(token) is not int or token < 0 for token in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("invalid decision label table")
    if manifest["prompt_protocol"] == "kev_pointer_v1":
        if set(head) != {"q.weight", "q.bias", "k.weight", "k.bias"}:
            raise ValueError("invalid pointer weights")
        d = head["q.weight"].shape[0]
        if (
            head["q.weight"].shape != (d, h)
            or head["k.weight"].shape != (d, h)
            or head["q.bias"].shape != (d,)
            or head["k.bias"].shape != (d,)
        ):
            raise ValueError("invalid pointer dimensions")
    elif set(head) != {"weight"} or head["weight"].shape != (
        len(manifest.get("label_token_ids", [])),
        h,
    ):
        raise ValueError("invalid vocabulary projection dimensions")
    if not all(torch.isfinite(value).all() for value in head.values()):
        raise ValueError("nonfinite decision weights")
    checked = verify_files(
        path,
        manifest,
        _index(path),
        (
            "config.json",
            "model.safetensors.index.json",
            "decision_head.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
        full=full,
    )
    return {
        "format": manifest["format"],
        "protocol": manifest["prompt_protocol"],
        "checked_files": checked,
        "full": full,
    }
