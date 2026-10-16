"""The pinned Kev pointer checkpoint, evaluated with FP32 on Windows CPU.

Published files stay read-only. No downloaded Python is executed by serving;
the original author implementation is used separately by the reference runner.
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

MODEL_ID = "jaredpalmer/kev-0.8b"
REVISION = "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"
BASE_ID = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
PROTOCOL = "kev_pointer_v1"
MAX_SEQUENCE_TOKENS = 1024
TEMPERATURE = 2.3510958125672174
MARKERS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
# Small-file digests were calculated from the pinned original bytes. Large-file
# digests are the upstream LFS SHA256; every downloaded file is checked locally.
SOURCE_HASHES = {
    "adapter_config.json": "748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854",
    "adapter_model.safetensors": "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a",
    "head.pt": "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6",
    "provenance.json": "fce039fbda7de19451c9b5d3065aee2c5e063e1562a8bc18c72e297fbd15e9d8",
    "training_config.json": "3ec3d92146248265652d4387f8eb65ee8defaed4dd9173b049e7b5193045ece6",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45",
    "base/config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "base/LICENSE": "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
    "base/model.safetensors.index.json": "ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a",
    "base/model.safetensors-00001-of-00001.safetensors": "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c",
    "base/tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "base/tokenizer_config.json": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
}
SOURCE_FILES = tuple(SOURCE_HASHES)
LORA_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
}
MODEL_ID_06 = "jaredpalmer/kev-0.6b"
SOURCE_HASHES_06 = {
    "adapter_config.json": "5bfcf3dd85b1383c802146cfdef179e2e772685e310ac972fb672f22d7079b4f",
    "adapter_model.safetensors": "deaab63b61f95d628503831e8b336e1c3e334b9a42d3530d979a5149b429c492",
    "head.pt": "ce6cd9ffc54db41c179b65a33d60973dc8280c29e886218eb3ec07dc28d6f28b",
    "provenance.json": "321bc9db7427b490ad943fa33576c7e01f5da4c20da1036fcbfee996f4b28f10",
    "training_config.json": "ab20e55cbb832f409de9daf42f868b69be1eb59e51720302b166bc560fd8d240",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "67e5a0a11cd35f9c00ee52e0af4cdc0baa75fea0cb5fce7d1beb251b4621d15c",
    "base/config.json": "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    "base/LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "base/tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    "base/tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "base/model.safetensors": "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba",
}
MODELS = {
    MODEL_ID: {
        "revision": REVISION,
        "base_id": BASE_ID,
        "base_revision": BASE_REVISION,
        "source_hashes": SOURCE_HASHES,
        "temperature": TEMPERATURE,
        "config_type": "qwen3_5",
        "text_type": "qwen3_5_text",
        "layers": 24,
        "lora_targets": LORA_TARGETS,
    },
    MODEL_ID_06: {
        "revision": "dece6dba8d43f0f7ded45e9f5b9df12474d90843",
        "base_id": "Qwen/Qwen3-0.6B-Base",
        "base_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "source_hashes": SOURCE_HASHES_06,
        "temperature": 1.0,
        "config_type": "qwen3",
        "text_type": "qwen3",
        "layers": 28,
        "lora_targets": {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        },
    },
}


def _require_model(model_id):
    if model_id not in MODELS:
        raise ValueError(f"unsupported Windows Kev model: {model_id}")
    return MODELS[model_id]


def verify_source(checkpoint: Path, model_id: str = MODEL_ID) -> dict:
    """Verify the entire allowlisted release before deserializing its weights."""
    spec = _require_model(model_id)
    source_hashes = spec["source_hashes"]
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("Kev source directory is missing")
    for directory, prefix in ((checkpoint, ""), (checkpoint / "base", "base/")):
        if not directory.is_dir():
            raise ValueError("Kev base directory is missing")
        for entry in directory.iterdir():
            if entry.name == ".cache" and entry.is_dir():
                continue
            if not prefix and entry.name == "base" and entry.is_dir():
                continue
            if prefix + entry.name not in source_hashes or not entry.is_file():
                raise ValueError(f"unexpected Kev source entry: {prefix}{entry.name}")
    for name, expected in source_hashes.items():
        path = checkpoint / name
        if not path.is_file():
            raise ValueError(f"Kev source file is missing: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Kev source checksum mismatch: {name}")
    config = json.loads((checkpoint / "base/config.json").read_text(encoding="utf-8"))
    adapter = json.loads(
        (checkpoint / "adapter_config.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (checkpoint / "training_config.json").read_text(encoding="utf-8")
    )
    text = config.get("text_config", config)
    if (
        config.get("model_type") != spec["config_type"]
        or text.get("model_type") != spec["text_type"]
        or text.get("hidden_size") != 1024
        or text.get("num_hidden_layers") != spec["layers"]
        or config.get("auto_map")
        or text.get("auto_map")
        or adapter.get("base_model_name_or_path") != spec["base_id"]
        or adapter.get("r") != 16
        or adapter.get("lora_alpha") != 32
        or adapter.get("lora_dropout") != 0.05
        or adapter.get("task_type") != "FEATURE_EXTRACTION"
        or set(adapter.get("target_modules", [])) != spec["lora_targets"]
        or adapter.get("bias") != "none"
        or adapter.get("modules_to_save")
        or adapter.get("trainable_token_indices")
        or adapter.get("use_dora")
        or adapter.get("use_rslora")
        or training.get("base_revision") != spec["base_revision"]
        or training.get("args", {}).get("base") != spec["base_id"]
    ):
        raise ValueError("unsupported Kev source identity or protocol")
    return adapter


def identify_source(checkpoint: Path) -> str:
    training = json.loads(
        (Path(checkpoint) / "training_config.json").read_text(encoding="utf-8")
    )
    for model_id, spec in MODELS.items():
        if training.get("args", {}).get("base") == spec["base_id"]:
            verify_source(checkpoint, model_id)
            return model_id
    raise ValueError("unsupported Windows Kev source identity")


def prepare(model_id: str, workspace: Path) -> Path:
    spec = _require_model(model_id)
    source_files = tuple(spec["source_hashes"])
    output = Path(workspace).resolve() / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            verify_source(output, model_id)
            return output
        with tempfile.TemporaryDirectory(
            prefix=".kev-download-", dir=output.parent
        ) as temporary:
            source = Path(temporary) / "source"
            snapshot_download(
                repo_id=model_id,
                revision=spec["revision"],
                token=False,
                local_dir=source,
                allow_patterns=[
                    name for name in source_files if not name.startswith("base/")
                ],
            )
            snapshot_download(
                repo_id=spec["base_id"],
                revision=spec["base_revision"],
                token=False,
                local_dir=source / "base",
                allow_patterns=[
                    name.removeprefix("base/")
                    for name in source_files
                    if name.startswith("base/")
                ],
            )
            verify_source(source, model_id)
            source.replace(output)
    return output


def _questions(questions):
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("System One requires 1 to 64 questions")
    if (
        len(json.dumps(questions, ensure_ascii=False, allow_nan=False).encode())
        > 2_000_000
    ):
        raise ValueError("decision questions exceed 2 MiB")
    result = {}
    for identifier, source in questions.items():
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(source, dict)
        ):
            raise ValueError("question IDs and definitions must be nonempty")
        question = dict(source)
        kind, criteria = question.get("type"), question.get("criteria")
        if kind == "choice" and isinstance(criteria, list):
            if any(not isinstance(key, str) or not key for key in criteria) or len(
                set(criteria)
            ) != len(criteria):
                raise ValueError("Choice labels must be distinct nonempty strings")
            question["criteria"] = dict.fromkeys(criteria)
        elif kind == "noul" and criteria is not None:
            if not isinstance(criteria, dict) or not set(criteria) <= {"false", "true"}:
                raise ValueError(
                    "Noul criteria must contain optional true/false descriptions"
                )
        if "isolated" in question:
            raise ValueError(
                "Kev Score uses joint pointer probabilities; isolated is unsupported"
            )
        result[identifier] = question
    return result


class KevAgent:
    device = "cpu"

    def __init__(
        self, model, tokenizer, head, temperature=TEMPERATURE, *, model_id=MODEL_ID
    ):
        from .decision import DecisionCompiler

        self.model_id = model_id
        self.revision = _require_model(model_id)["revision"]
        self.model = model.eval().requires_grad_(False)
        self.head = {key: value.detach().contiguous() for key, value in head.items()}
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("Kev temperature must be finite and positive")
        self.temperature = temperature
        self.compiler = DecisionCompiler.__new__(DecisionCompiler)
        self.compiler.manifest = {
            "max_options": 255,
            "max_length": MAX_SEQUENCE_TOKENS,
            "temperature": temperature,
            "prompt_protocol": PROTOCOL,
        }
        self.compiler.tokenizer = tokenizer
        self.compiler.model_names = [model_id]
        self.compiler.model_id = model_id
        self.compiler.protocol = PROTOCOL
        self.compiler.max_length = MAX_SEQUENCE_TOKENS
        self.compiler.labels = []
        self.compiler.special = [
            tokenizer.convert_tokens_to_ids(marker) for marker in MARKERS
        ]
        vocab = model.get_input_embeddings().weight.shape[0]
        if len(set(self.compiler.special)) != 5 or any(
            type(i) is not int or not 0 <= i < vocab for i in self.compiler.special
        ):
            raise ValueError("Kev delimiters are missing from the backbone vocabulary")
        self._precision()

    def _precision(self):
        import torch

        if self.model is None:
            raise RuntimeError("Kev CPU model is closed")
        tensors = [*self.model.parameters(), *self.model.buffers(), *self.head.values()]
        if any(t.is_meta or t.device.type != "cpu" for t in tensors):
            raise ValueError("Kev parameters must be materialized on CPU")
        embedding = self.model.get_input_embeddings().weight
        if any(p.dtype != torch.float32 for p in self.model.parameters()) or any(
            t.dtype != torch.float32 for t in self.head.values()
        ):
            raise ValueError(
                "Kev requires FP32 backbone, merged adapter and pointer head"
            )
        shapes = {
            "q.weight": (256, embedding.shape[1]),
            "q.bias": (256,),
            "k.weight": (256, embedding.shape[1]),
            "k.bias": (256,),
        }
        if set(self.head) != set(shapes) or any(
            tuple(self.head[key].shape) != shape for key, shape in shapes.items()
        ):
            raise ValueError("invalid Kev pointer head shape")
        if any(not torch.isfinite(value).all() for value in self.head.values()):
            raise ValueError("nonfinite Kev pointer head")
        return {
            "backbone_dtype": str(embedding.dtype).removeprefix("torch."),
            "head_dtype": str(self.head["q.weight"].dtype).removeprefix("torch."),
            "logits_dtype": "float32",
            "row_batch_size": 1,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "attention_implementation": "eager",
            "lora_merged": True,
            "use_cache": True,
            "cache_reuse": False,
        }

    def _probabilities(self, ids, positions, count):
        import torch
        import torch.nn.functional as F

        if len(positions) != count + 1 or positions[-1] != len(ids) - 1:
            raise RuntimeError("invalid Kev pointer readout positions")
        with torch.inference_mode():
            tokens = torch.tensor([ids], dtype=torch.long, device="cpu")
            hidden = self.model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                position_ids=torch.arange(len(ids)).unsqueeze(0),
                use_cache=True,
                return_dict=True,
            ).last_hidden_state
            if (
                hidden.shape != (1, len(ids), self.head["q.weight"].shape[1])
                or hidden.dtype != torch.float32
                or hidden.device.type != "cpu"
            ):
                raise RuntimeError("Kev returned unexpected hidden states")
            selected = hidden[0, positions].float()
            query = F.linear(selected[-1], self.head["q.weight"], self.head["q.bias"])
            keys = F.linear(selected[:-1], self.head["k.weight"], self.head["k.bias"])
            # Preserve the author's FP32 projection, multiply, then calibration.
            logits = (keys @ query) * (1 / math.sqrt(query.numel()))
            logits = logits / self.temperature
            if not torch.isfinite(logits).all():
                raise RuntimeError("Kev returned nonfinite pointer logits")
            return torch.softmax(logits, -1).tolist()

    def system_one(self, state, questions):
        from .decision import _answer
        from .request_models import SystemOneRequest

        precision = self._precision()
        if not isinstance(state, (str, dict, list)):
            raise ValueError("decision state must be text or JSON")
        serialized = json.dumps(state, ensure_ascii=False, allow_nan=False)
        if len(serialized) > 50_000:
            raise ValueError("decision state exceeds 50,000 characters")
        payload = SystemOneRequest(
            state=state, questions=_questions(questions), model=self.model_id
        )
        prepared, state_tokens = self.compiler.compile(payload)
        started = time.perf_counter()
        encoded, answers = {}, {}
        input_tokens = state_tokens
        for identifier, kind, keys, descriptions, isolated, rows in prepared:
            if isolated or len(rows) != 1:
                raise RuntimeError("unexpected Kev candidate compilation")
            ids, positions, count = rows[0]
            key = (tuple(ids), tuple(positions))
            if key not in encoded:
                encoded[key] = self._probabilities(ids, positions, count)
            answers[identifier] = _answer(kind, keys, descriptions, encoded[key])
            input_tokens += len(ids) - state_tokens
        return {
            "model": self.model_id,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "internal_usage": {"compute_tokens": sum(len(key[0]) for key in encoded)},
            "metadata": {
                **precision,
                "temperature": self.temperature,
                "cached_tokens": 0,
                "inference_seconds": time.perf_counter() - started,
            },
        }

    def close(self):
        self.model = self.head = self.compiler = None


def load_agent(checkpoint: Path, model_id: str = MODEL_ID, threads=None):
    spec = _require_model(model_id)
    if threads is None:
        threads = min(4, os.cpu_count() or 1)
    if type(threads) is not int or not 1 <= threads <= (os.cpu_count() or 1):
        raise ValueError("threads must be between 1 and the logical CPU count")
    checkpoint = Path(checkpoint).resolve()
    adapter = verify_source(checkpoint, model_id)
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint / "base", local_files_only=True, trust_remote_code=False
    )
    full, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint / "base",
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    # Transformers explicitly ignores the published visual tower and MTP keys
    # when selecting Qwen3.5's CausalLM compatibility class. All text keys must load.
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("Kev text backbone weights were not loaded completely")
    backbone = full.model
    del full
    if (
        backbone.config.hidden_size != 1024
        or backbone.config.model_type != spec["text_type"]
    ):
        raise ValueError("Kev loaded an unexpected text backbone")
    # The pinned release uses ordinary LoRA only. Construct precisely that
    # configuration rather than interpreting newer PEFT's unrelated null fields.
    configuration = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=adapter["r"],
        lora_alpha=adapter["lora_alpha"],
        lora_dropout=adapter["lora_dropout"],
        target_modules=adapter["target_modules"],
        bias="none",
        inference_mode=True,
    )
    backbone = get_peft_model(backbone, configuration)
    expected = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
    if not expected or set(expected) != set(
        get_peft_model_state_dict(backbone, save_embedding_layers=False)
    ):
        raise ValueError("Kev LoRA keys do not match the pinned backbone")
    loaded = set_peft_model_state_dict(backbone, expected)
    actual = get_peft_model_state_dict(backbone, save_embedding_layers=False)
    if loaded.unexpected_keys or any(
        value.is_meta
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or not torch.isfinite(value).all()
        or not torch.equal(value, expected[key])
        for key, value in actual.items()
    ):
        raise ValueError("Kev LoRA parameters did not load completely")
    del expected, actual
    backbone = backbone.merge_and_unload(safe_merge=True)
    backbone.config.use_cache = True
    meta = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    if (
        meta.get("base") != spec["base_id"]
        or meta.get("base_revision") != spec["base_revision"]
        or meta.get("lora") != 16
        or meta.get("head_dim") != 256
        or meta.get("option_isolation") is not False
        or meta.get("special_embeddings") is not False
        # The older 0.6B artifact omits these fields. Match the original
        # Checkpoint.Meta defaults, exclusively for this hash-pinned release.
        or meta.get("weights_dtype", "fp32" if model_id == MODEL_ID_06 else None)
        != "fp32"
        or meta.get("temperature", 1.0 if model_id == MODEL_ID_06 else None)
        != spec["temperature"]
    ):
        raise ValueError("unsupported Kev head identity or calibration")
    return KevAgent(
        backbone, tokenizer, meta["head"], spec["temperature"], model_id=model_id
    )
