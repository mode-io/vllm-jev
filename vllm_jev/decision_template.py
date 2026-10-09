"""Deployment-owned decision text templates, separate from model readout protocols."""

import json
import os
from functools import lru_cache
from pathlib import Path
from string import Formatter

ENVIRONMENT = "VLLM_JEV_DECISION_TEMPLATE"
MAX_BYTES = 65536
FIELDS = {
    "state_template": {"state"},
    "instruction_template": {"instructions"},
    "choice_prompt": {"state", "instructions", "candidate"},
    "noul_prompt": {"state", "instructions"},
}


class DecisionTemplate:
    def __init__(self, config: dict):
        if not isinstance(config, dict) or set(config) - {
            "layout",
            "context_order",
            *FIELDS,
        }:
            raise ValueError("unknown decision-template fields")
        self.config = dict(config)
        self.fields = {}
        order = config.get("context_order")
        if "context_order" in config:
            if (
                not isinstance(order, list)
                or len(order) != 3
                or any(not isinstance(field, str) for field in order)
                or set(order) != {"state", "instructions", "criteria"}
            ):
                raise ValueError(
                    "context_order must list state, instructions and criteria exactly once"
                )
            if {"layout", "choice_prompt", "noul_prompt"} & config.keys():
                raise ValueError(
                    "context_order cannot be combined with layout or full prompt overrides"
                )
        self.context_order = tuple(order) if order is not None else None
        self.layout = config.get("layout", "native")
        if self.layout not in ("native", "instructions-first"):
            raise ValueError(
                "decision-template layout must be native or instructions-first"
            )
        for key, required in FIELDS.items():
            if key not in config:
                continue
            text = config[key]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{key} must be nonempty text")
            try:
                fields = set()
                for _, field, spec, conversion in Formatter().parse(text):
                    if field is None:
                        continue
                    allowed = required | (
                        {"options"} if key == "choice_prompt" else set()
                    )
                    if field not in allowed or spec or conversion:
                        raise ValueError(f"invalid placeholder in {key}: {field}")
                    fields.add(field)
                if not required.issubset(fields):
                    raise ValueError(
                        f"{key} requires placeholders: {', '.join(sorted(required))}"
                    )
                self.fields[key] = fields
            except ValueError as error:
                raise ValueError(f"invalid {key}: {error}") from error

    @property
    def requires_open_jev(self) -> bool:
        return any(key in self.config for key in ("choice_prompt", "noul_prompt"))

    @property
    def instructions_first(self) -> bool:
        return self.layout == "instructions-first"

    def arrange(self, blocks):
        """Order logical blocks without flattening token or media-message objects."""
        if self.context_order is None:
            raise ValueError("context_order is not configured")
        return [item for field in self.context_order for item in blocks[field]]

    def context(self, state: str, instructions: str, criteria: str) -> str:
        blocks = {
            "state": ["Context:\n" + state + "\n\n"],
            "instructions": ["Question: " + instructions + "\n\n"],
            "criteria": ["Criteria:\n" + criteria + "\n\n"],
        }
        return "".join(self.arrange(blocks))

    def validate_protocol(self, protocol: str) -> None:
        if self.requires_open_jev and protocol != "open_jev_choice":
            raise ValueError(
                "full decision prompts require Open-Jev-2B/9B (open_jev_choice); "
                "use layout, instruction_template or state_template for other models"
            )

    def instructions(self, value):
        template = self.config.get("instruction_template")
        # Leave absent or malformed fields to each adapter's own validation.
        if template is None or not isinstance(value, str) or not value.strip():
            return value
        return template.format_map({"instructions": value})

    def state(self, value):
        template = self.config.get("state_template")
        if template is None or template == "{state}":
            return value
        if not isinstance(value, (str, dict, list)):
            raise ValueError("state_template requires text or a JSON object/list")
        if isinstance(value, dict) and "messages" in value:
            raise ValueError(
                "state_template cannot wrap state.messages; use instruction_template "
                "to preserve chat and media structure"
            )
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
        return template.format_map({"state": text})

    def choice(
        self, state: str, instructions: str, candidate: str, options: str = ""
    ) -> str:
        template = self.config.get("choice_prompt")
        if template is None and self.context_order is not None:
            return self.context(state, instructions, options) + (
                "Proposed answer: " + candidate + "\n"
                "Is this proposed answer correct? Answer Yes or No."
            )
        if template is None:
            prefix = (
                "Question: {instructions}\nContext:\n{state}\n\n"
                if self.layout == "instructions-first"
                else "Context:\n{state}\n\nQuestion: {instructions}\n"
            )
            template = prefix + (
                "Proposed answer: {candidate}\n"
                "Is this proposed answer correct? Answer Yes or No."
            )
        return template.format_map(
            {
                "state": state,
                "instructions": instructions,
                "candidate": candidate,
                "options": options,
            }
        )

    def noul(self, state: str, instructions: str) -> str:
        template = self.config.get("noul_prompt")
        if template is None and self.context_order is not None:
            return self.context(state, instructions, '["false", "true"]') + (
                "Is the answer to this question yes? Answer Yes or No."
            )
        if template is None:
            prefix = (
                "Question: {instructions}\nContext:\n{state}\n\n"
                if self.layout == "instructions-first"
                else "Context:\n{state}\n\nQuestion: {instructions}\n"
            )
            template = prefix + "Is the answer to this question yes? Answer Yes or No."
        return template.format_map({"state": state, "instructions": instructions})


def load_template(value: str) -> DecisionTemplate:
    if value in ("native", "instructions-first"):
        return DecisionTemplate({"layout": value})
    path = Path(value)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("decision-template JSON exceeds 64 KiB")
    return decode_template(path.read_text())


def decode_template(text: str) -> DecisionTemplate:
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("decision-template JSON exceeds 64 KiB")
    try:
        config = json.loads(text)
    except (ValueError, TypeError) as error:
        raise ValueError("decision-template must be a JSON object") from error
    return DecisionTemplate(config)


@lru_cache(maxsize=1)
def current_template() -> DecisionTemplate:
    """Freeze the resolved configuration for the server process."""
    return decode_template(os.environ.get(ENVIRONMENT, '{"layout":"native"}'))
