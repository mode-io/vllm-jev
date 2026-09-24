"""Exact Open-Jev Choice prompt contract and stable categorical postprocessing."""

import json
import math
from collections.abc import Sequence


def render_value(value):
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False
    )


def candidate_prompts(state, question, options: Sequence[str]) -> list[str]:
    if not isinstance(state, (str, dict, list)):
        raise ValueError("state must be text or a JSON-compatible object")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty text")
    if not 2 <= len(options) <= 255:
        raise ValueError("Choice requires 2 to 255 candidates")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("every candidate must be nonempty text")
    if len(set(options)) != len(options):
        raise ValueError("candidates must be distinct")
    prefix = f"Context:\n{render_value(state)}\n\nQuestion: {render_value(question)}\n"
    return [
        prefix
        + f"Proposed answer: {render_value(option)}\n"
        + "Is this proposed answer correct? Answer Yes or No."
        for option in options
    ]


def probabilities(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not scores or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("scores and positive finite temperature are required")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    shifted = [score / temperature for score in scores]
    peak = max(shifted)
    weights = [math.exp(score - peak) for score in shifted]
    total = sum(weights)
    return [weight / total for weight in weights]


def choice_result(options: Sequence[str], scores: Sequence[float], temperature: float = 1.0) -> dict:
    if len(options) != len(scores):
        raise ValueError("candidate and score counts differ")
    probs = probabilities(scores, temperature)
    selected = max(range(len(probs)), key=probs.__getitem__)
    selected_probability = probs[selected]
    uniform = 1.0 / len(probs)
    return {
        "choice": options[selected],
        "probabilities": dict(zip(options, probs)),
        "scores": dict(zip(options, map(float, scores))),
        "selected_probability": selected_probability,
        "confidence": (selected_probability - uniform) / (1.0 - uniform),
    }
