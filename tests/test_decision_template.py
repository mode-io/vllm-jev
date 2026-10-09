"""Template validation and exact default prompt regression tests."""

import json

import pytest

from vllm_jev.decision_template import (
    ENVIRONMENT,
    DecisionTemplate,
    current_template,
    load_template,
)
from vllm_jev.prompt import candidate_prompts, noul_prompt, render_value


@pytest.fixture(autouse=True)
def isolated_template(monkeypatch):
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    current_template.cache_clear()
    yield
    current_template.cache_clear()


@pytest.mark.parametrize("state", ["Some text", {"b": 2, "a": "中"}, ["one", {"x": 1}]])
def test_native_prompt_bytes_unchanged(state):
    prefix = f"Context:\n{render_value(state)}\n\nQuestion: Question?\n"
    assert candidate_prompts(state, "Question?", ["A", "B"]) == [
        prefix
        + f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
        for option in ["A", "B"]
    ]
    assert (
        noul_prompt(state, "Question?")
        == prefix + "Is the answer to this question yes? Answer Yes or No."
    )


def test_layout_keeps_json_and_candidate_readout(monkeypatch):
    monkeypatch.setenv(ENVIRONMENT, '{"layout":"instructions-first"}')
    prompts = candidate_prompts({"x": "{candidate}"}, "Rules", ["A", "B"])
    assert (
        prompts[0]
        == 'Question: Rules\nContext:\n{"x": "{candidate}"}\n\nProposed answer: A\nIs this proposed answer correct? Answer Yes or No.'
    )
    assert noul_prompt("evidence", "rules").startswith(
        "Question: rules\nContext:\nevidence"
    )


@pytest.mark.parametrize(
    "config",
    [
        {"layout": "unknown"},
        {"unknown": "x"},
        [],
        {"choice_prompt": "{state} {instructions}"},
        {"choice_prompt": "{state.__class__} {instructions} {candidate}"},
        {"choice_prompt": "{state[0]} {instructions} {candidate}"},
        {"choice_prompt": "{state!r} {instructions} {candidate}"},
        {"choice_prompt": "{state:1000000} {instructions} {candidate}"},
        {"noul_prompt": "{state}"},
        {"instruction_template": "replaced"},
        {"instruction_template": "{"},
        {"instruction_template": 2},
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        DecisionTemplate(config)


def test_custom_file_is_resolved_once(tmp_path, monkeypatch):
    path = tmp_path / "template.json"
    path.write_text(
        json.dumps({"choice_prompt": "{instructions}\n{state}\n{candidate}"})
    )
    template = load_template(str(path))
    monkeypatch.setenv(ENVIRONMENT, json.dumps(template.config))
    path.unlink()
    assert candidate_prompts("state", "rules", ["A", "B"]) == [
        "rules\nstate\nA",
        "rules\nstate\nB",
    ]
    monkeypatch.setenv(ENVIRONMENT, '{"layout":"native"}')
    assert candidate_prompts("state", "rules", ["A", "B"])[0] == "rules\nstate\nA"


def test_other_protocols_allow_layout_but_protect_full_prompts():
    template = DecisionTemplate(
        {"instruction_template": "Shared policy\n{instructions}"}
    )
    for protocol in [
        "clef_joint_v1",
        "rsijev_xattn_v1",
        "laya_markers_v1",
        "kev_pointer_v1",
        "openjev_branch_v03",
        "tiny_jev_marker",
    ]:
        template.validate_protocol(protocol)
        DecisionTemplate({"layout": "instructions-first"}).validate_protocol(protocol)
        with pytest.raises(ValueError, match="require Open-Jev"):
            DecisionTemplate(
                {"choice_prompt": "{state} {instructions} {candidate}"}
            ).validate_protocol(protocol)


def test_size_limit(tmp_path):
    path = tmp_path / "big.json"
    path.write_text(" " * 65537)
    with pytest.raises(ValueError, match="64 KiB"):
        load_template(str(path))


def test_cli_resolves_template_and_does_not_forward_flag(tmp_path, monkeypatch):
    import sys

    from vllm_jev import cli

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "jev_manifest.json").write_text(
        '{"prompt_protocol":"open_jev_choice"}'
    )
    config = tmp_path / "custom.json"
    config.write_text('{"layout":"instructions-first"}')
    calls = {}
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm-jev", "serve", str(checkpoint), "--decision-template", str(config)],
    )
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *a, **kw: calls.update(prepare_env=kw["env"])
    )
    monkeypatch.setattr(
        cli.os, "execvpe", lambda exe, argv, env: calls.update(argv=argv, env=env)
    )
    cli.main()
    assert "--decision-template" not in calls["argv"]
    assert json.loads(calls["env"][ENVIRONMENT]) == {"layout": "instructions-first"}
    assert ENVIRONMENT not in calls["prepare_env"]


def test_cli_rejects_incompatible_full_prompt_before_engine(tmp_path, monkeypatch):
    import sys

    from vllm_jev import cli

    (tmp_path / "jev_manifest.json").write_text('{"prompt_protocol":"tiny_jev_marker"}')
    config = tmp_path / "prompt.json"
    config.write_text('{"choice_prompt":"{state} {instructions} {candidate}"}')
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm-jev",
            "serve",
            str(tmp_path),
            "--decision-template",
            str(config),
        ],
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli.os, "execvpe", lambda *a: pytest.fail("must not start engine")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_state_wrapper_preserves_json_and_rejects_message_envelopes():
    template = DecisionTemplate({"state_template": "Policy\n{state}"})
    assert (
        template.state({"z": 1, "a": "{state}"}) == 'Policy\n{"a": "{state}", "z": 1}'
    )
    with pytest.raises(ValueError, match="state.messages"):
        template.state(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "data:..."}}
                        ],
                    }
                ]
            }
        )
    state = {"messages": []}
    assert DecisionTemplate({"state_template": "{state}"}).state(state) is state
    with pytest.raises(ValueError):
        template.state(123)


def test_cli_keeps_unicode_environment_bounded(tmp_path, monkeypatch):
    import sys

    from vllm_jev import cli

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "jev_manifest.json").write_text(
        '{"prompt_protocol":"open_jev_choice"}'
    )
    spec = {"instruction_template": "😀" * 12000 + "{instructions}"}
    path = tmp_path / "template.json"
    path.write_text(json.dumps(spec, ensure_ascii=False))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        sys,
        "argv",
        ["vllm-jev", "serve", str(checkpoint), "--decision-template", str(path)],
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: None)
    captured = {}
    monkeypatch.setattr(cli.os, "execvpe", lambda exe, argv, env: captured.update(env))
    cli.main()
    assert json.loads(captured[ENVIRONMENT]) == spec
    assert len(captured[ENVIRONMENT].encode("utf-8")) < 65536


def test_custom_prompt_can_include_fixed_full_options(monkeypatch):
    monkeypatch.setenv(
        ENVIRONMENT,
        json.dumps(
            {
                "choice_prompt": "{instructions}\nAll candidates: {options}\n{state}\nSelected: {candidate}"
            }
        ),
    )
    assert candidate_prompts("input", "rules", ["A", "B"]) == [
        'rules\nAll candidates: ["A", "B"]\ninput\nSelected: A',
        'rules\nAll candidates: ["A", "B"]\ninput\nSelected: B',
    ]
