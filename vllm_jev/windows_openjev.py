"""Pinned OpenJev-0.6B release and an experimental native CPU adapter.

The original Qwen backbone, unmerged LoRA adapter, and scalar decision head
run in FP32. Serving uses full candidate sequences without a KV cache and
never imports downloaded Python code or modifies the verified source files.
"""

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

from .decision_template import current_template
from .prompt import branch_token_ids

MODEL_ID = "IamBusy/OpenJev-0.6B"
REVISION = "b4bf78e7c053407b5d54a71bcc682ef5bbe59e72"
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROTOCOL = "openjev_branch_v03"
BUNDLE_HASHES = {
    "MANIFEST.json": "8a0df9df5e30f2e7b980de40460dd81ddf5cbc21ad263baf7ea256100a86584f",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "README.md": "058c4af0d3a2a127705a95ceacc40a08cd855da226c0f73c9772863266ca15b9",
    "RESULTS.md": "b57c93eacd96cb86f6a5ff51f07e158dce6658d592bd009c25c3f39175667363",
    "THIRD_PARTY.md": "2edf4dfbe190a24cf81c9d9fc3b3b5f4e3d7e936c66348732b6d6bc293b16c6c",
    "adapter_config.json": "ebfa81c887c626f1782ba3d5f5303373c47120f8a32ca3a81ef14fdb532c8cf8",
    "adapter_model.safetensors": "112e80f7ccc835b57266f9af6cfdf18eaf3e99c02c91a71941d5d1ac89c5ec07",
    "calibration-v03.json": "114210f173f334fa1b40c3d1b99356111fec708891801800286e7a5b6abe555b",
    "example_request.json": "ae3dc5feca0dd426ceece7c3d191b9969002d0e5e1346cfbc8ef02fb0c01ee93",
    "head.safetensors": "5f9e0780f6c8ace63a43fdbd58cfd2293e81d22d637ed139221a578fa7a9ac8d",
    "openjev_config.json": "a3db71d948690c661fbf9406ad0b96ae54e78a3ee3e282528857d214f66c98c4",
}
BASE_HASHES = {
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
SOURCE_HASHES = {
    **BUNDLE_HASHES,
    **{f"base/{name}": digest for name, digest in BASE_HASHES.items()},
}
SOURCE_FILES = tuple(SOURCE_HASHES)


def _require_model(model_id: str) -> None:
    if model_id != MODEL_ID:
        raise ValueError(f"this Windows CPU adapter supports only {MODEL_ID}")


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    """Require the exact release bundle and its separately pinned Qwen base."""
    _require_model(model_id)
    checkpoint = Path(checkpoint).resolve()
    base = checkpoint / "base"
    if base.is_dir():
        for entry in base.iterdir():
            if entry.name == ".cache" and entry.is_dir():
                continue
            if entry.name in {*BASE_HASHES, "README.md", "LICENSE", ".gitattributes"}:
                if entry.is_file():
                    continue
            # Transformers can discover extra adapters and chat templates even
            # when every pinned file is unchanged. Reject all such overrides.
            raise ValueError(f"unexpected OpenJev base source entry: base/{entry.name}")
    for name, expected in SOURCE_HASHES.items():
        source = checkpoint / name
        if not source.is_file():
            raise ValueError(f"OpenJev source file is missing: {name}")
        with source.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"OpenJev source checksum mismatch: {name}")
    manifest = json.loads((checkpoint / "MANIFEST.json").read_text(encoding="utf-8"))
    if (
        manifest.get("base_model") != BASE_MODEL_ID
        or manifest.get("base_revision") != BASE_REVISION
        or manifest.get("files")
        != {
            name: digest
            for name, digest in BUNDLE_HASHES.items()
            if name != "MANIFEST.json"
        }
    ):
        raise ValueError("unsupported OpenJev release manifest")
    config = json.loads(
        (checkpoint / "openjev_config.json").read_text(encoding="utf-8")
    )
    if (
        config.get("model_name") != BASE_MODEL_ID
        or config.get("model_revision") != BASE_REVISION
        or config.get("architecture") != "SharedStateCandidateBranches"
        or config.get("max_prefix_tokens") != 768
        or config.get("max_branch_tokens") != 192
        or config.get("branch_batch_size") != 16
    ):
        raise ValueError("unsupported OpenJev source architecture")
    return config


def identify_source(checkpoint: Path) -> str:
    verify_source(checkpoint)
    return MODEL_ID


def prepare(model_id: str, workspace: Path) -> Path:
    """Publish both pinned downloads only after all source hashes match."""
    _require_model(model_id)
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".openjev-download-", dir=output.parent
        ) as directory:
            source = Path(directory) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=REVISION,
                token=False,
                local_dir=source,
                allow_patterns=list(BUNDLE_HASHES),
            )
            snapshot_download(
                repo_id=BASE_MODEL_ID,
                revision=BASE_REVISION,
                token=False,
                local_dir=source / "base",
                allow_patterns=list(BASE_HASHES),
            )
            verify_source(source, model_id)
            source.replace(output)
    return output


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must contain 1 to 4096 characters of text")
    return value


def _questions(questions: dict) -> tuple[list, int]:
    """Validate the SDK question schema inside the service's request budget."""
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    prepared = []
    candidate_count = 0
    for identifier, definition in questions.items():
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or not isinstance(definition, dict)
        ):
            raise ValueError("question IDs and definitions must be nonempty")
        if set(definition) - {"type", "instructions", "criteria"}:
            raise ValueError("unsupported OpenJev question fields")
        kind = definition.get("type")
        question = _text(definition.get("instructions"), "instructions")
        criteria = definition.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError("Choice requires 2 to 255 candidates")
            keys, descriptions = list(criteria), list(criteria.values())
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires 2 to 10 descriptive levels")
            keys = [str(i) for i in range(len(criteria))]
            descriptions = list(criteria)
        elif kind == "noul":
            keys = ["false", "true"]
            if criteria is None:
                descriptions = [
                    f"The following statement is false: {question}",
                    f"The following statement is true: {question}",
                ]
            elif isinstance(criteria, dict) and set(criteria) == {"false", "true"}:
                descriptions = [criteria[key] for key in keys]
            else:
                raise ValueError("Noul criteria must define exactly false and true")
        else:
            raise ValueError("question type must be choice, score or noul")
        if any(not isinstance(key, str) or not key.strip() for key in keys):
            raise ValueError("candidate names must be nonempty strings")
        if criteria is not None:
            descriptions = [
                _text(value, "candidate descriptions") for value in descriptions
            ]
        if len(set(descriptions)) != len(descriptions):
            raise ValueError("candidate descriptions must be distinct")
        candidate_count += len(descriptions)
        prepared.append((identifier, kind, question, criteria, keys, descriptions))
    if candidate_count > 256:
        raise ValueError("System One exceeds 256 candidate sequences")
    return prepared, candidate_count


class OpenJevAgent:
    """Synchronous CPU adapter; the server serializes requests and cancellation."""

    device = "cpu"
    protocol = PROTOCOL

    def __init__(self, model, tokenizer, weight, temperatures, *, branch_batch_size=16):
        if set(temperatures) != {"choice", "noul", "score"} or any(
            not math.isfinite(value) or value <= 0 for value in temperatures.values()
        ):
            raise ValueError("all three temperatures must be finite and positive")
        if type(branch_batch_size) is not int or branch_batch_size < 1:
            raise ValueError("branch batch size must be a positive integer")
        self.model = model
        self.tokenizer = tokenizer
        self.weight = weight
        self.temperatures = {kind: float(value) for kind, value in temperatures.items()}
        self.branch_batch_size = branch_batch_size

    def _scores(self, branches: list[list[int]]) -> list[float]:
        """Read the final real token in each complete, right-padded sequence."""
        import torch
        from torch.nn import functional as F

        scores = []
        with torch.inference_mode():
            for start in range(0, len(branches), self.branch_batch_size):
                batch = branches[start : start + self.branch_batch_size]
                lengths = [len(ids) for ids in batch]
                if not lengths or min(lengths) < 1:
                    raise ValueError("candidate token sequences must be nonempty")
                ids = torch.full(
                    (len(batch), max(lengths)),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device="cpu",
                )
                for row, tokens in enumerate(batch):
                    ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                # Match the SDK's uncached causal forward. Padding occurs only
                # AFTER the readout position, so it cannot affect that position.
                hidden = self.model(
                    input_ids=ids, use_cache=False, return_dict=True
                ).last_hidden_state
                selected = hidden[torch.arange(len(batch)), torch.tensor(lengths) - 1]
                scores.extend(F.linear(selected.float(), self.weight)[:, 0].tolist())
        return scores

    def system_one(self, state, questions) -> dict:
        if self.model is None:
            raise RuntimeError("OpenJev service is closed")
        if not isinstance(state, (str, dict, list)):
            raise ValueError("state must be text or JSON")
        try:
            json.dumps([state, questions], ensure_ascii=False, allow_nan=False)
            state_text = (
                state
                if isinstance(state, str)
                else json.dumps(
                    state,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("decision data must be finite JSON") from error
        if not state_text.strip() or len(state_text) > 100000:
            raise ValueError("state must contain 1 to 100000 characters")
        current_template().validate_protocol(PROTOCOL)
        prepared, candidate_count = _questions(questions)
        # Validate every branch before any model forward, including template limits.
        encoded = [
            branch_token_ids(self.tokenizer, state, question, descriptions)
            for _, _, question, _, _, descriptions in prepared
        ]
        branches = [ids for group in encoded for ids in group]
        import numpy as np

        started = time.perf_counter()
        scores = self._scores(branches)
        if len(scores) != candidate_count or any(not math.isfinite(x) for x in scores):
            raise ValueError("OpenJev returned invalid candidate scores")
        answers = {}
        offset = 0
        for identifier, kind, _, _, keys, descriptions in prepared:
            logits = np.asarray(scores[offset : offset + len(keys)], dtype=np.float64)
            logits /= self.temperatures[kind]
            offset += len(keys)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            if kind == "noul":
                answers[identifier] = {"type": kind, "noul": float(probabilities[1])}
                continue
            confidence = float(probabilities.max())
            answer = {
                "type": kind,
                "probabilities": dict(zip(keys, map(float, probabilities))),
                "confidence": confidence,
            }
            if kind == "choice":
                tied = np.flatnonzero(probabilities >= confidence - 1e-7)
                selected = min(
                    tied, key=lambda i: (descriptions[i].casefold(), descriptions[i])
                )
                answer["choice"] = keys[int(selected)]
            else:
                answer["score"] = float(probabilities @ np.arange(len(keys)))
                answer["legend"] = dict(zip(keys, descriptions))
            answers[identifier] = answer
        return {
            "answers": answers,
            "model": MODEL_ID,
            "usage": {"input_tokens": sum(map(len, branches)), "output_tokens": 0},
            "metadata": {
                "method": "lora_decision_head",
                "temperature": dict(self.temperatures),
                "candidate_sequences": candidate_count,
                "cached_tokens": 0,
                "inference_seconds": time.perf_counter() - started,
            },
        }

    def close(self) -> None:
        self.model = self.tokenizer = self.weight = None


def load_agent(
    checkpoint: Path, model_id: str = MODEL_ID, threads: int | None = None
) -> OpenJevAgent:
    """Load verified local FP32 weights; leave the original bundle untouched."""
    _require_model(model_id)
    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("threads must be between 1 and the logical CPU count")
    checkpoint = Path(checkpoint).resolve()
    config = verify_source(checkpoint, model_id)
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    base = checkpoint / "base"
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    lm = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
    ).to("cpu")
    lm = PeftModel.from_pretrained(
        lm, str(checkpoint), is_trainable=False, local_files_only=True
    )
    lm.eval().requires_grad_(False)
    weights = load_file(str(checkpoint / "head.safetensors"), device="cpu")
    if (
        set(weights) != {"weight"}
        or weights["weight"].shape != (1, lm.config.hidden_size)
        or not torch.isfinite(weights["weight"]).all()
    ):
        raise ValueError("invalid OpenJev scalar head")
    calibration = json.loads(
        (checkpoint / "calibration-v03.json").read_text(encoding="utf-8")
    )
    return OpenJevAgent(
        lm.get_base_model().model,
        tokenizer,
        weights["weight"].float().contiguous(),
        {kind: data["temperature"] for kind, data in calibration.items()},
        branch_batch_size=config["branch_batch_size"],
    )
