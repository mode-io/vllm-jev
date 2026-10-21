"""Pinned upstream bytes, not a fabricated tokenizer configuration."""

import hashlib
from pathlib import Path

from vllm_jev.laya_export import SOURCE_HASHES


def test_multilingual_tokenizer_hash_matches_fixed_upstream_blob():
    # convaiinnovations/laya-multilingual, e4e9ddf21a7b1903b7acffd8814ad4307bf63a67
    data = (
        Path(__file__).parent / "fixtures/laya_multilingual_tokenizer_config.json"
    ).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    assert blob == "eea1ed61121530d74c1722dc2aab6ec917a75909"
    assert (
        hashlib.sha256(data).hexdigest()
        == SOURCE_HASHES["convaiinnovations/laya-multilingual"][-1]
    )
