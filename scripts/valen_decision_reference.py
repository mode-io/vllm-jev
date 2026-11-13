"""Compare a served Valen-Decision-1.5 checkpoint with the release's own model.

``--reference-only`` loads the published release with ``trust_remote_code`` in
this process and records ``model.predict`` answers for a request set, in
bfloat16 or float32 (``--dtype``), optionally under a deliberately wrong
causal mask as a control (``--mask causal``). Without it, the script sends
the same requests to a running ``vllm-jev serve`` and records every
difference from a reference recording. ``--summarize`` prints the
distribution of differences between four recordings.

The fixed request set covers text Choice/Noul/Score, one multi-question
shared state, images, and ``bbox`` questions over one and two images.
``--sweep`` uses a wider set of 16 drawings instead, for statistics.

vLLM runs this backbone in bfloat16 only, and the release's own answers move
between bfloat16 and float32 by more than a tight tolerance allows. So a
comparison fails on structural differences (answer types, candidate keys,
legends, input-token counts) and reports numeric differences and selected
answers as data; ``--tolerance`` adds a numeric gate. Judge correctness by
``--summarize``: the served answers should sit as close to the release's
float32 answers as the release's bfloat16 answers do, and far closer than
the causal-mask control.
"""

import argparse
import base64
import io
import json
import math
import platform
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS = (
    "Valen-Team/Valen-Decision-1.5-0.8B",
    "Valen-Team/Valen-Decision-1.5-2B",
    "Valen-Team/Valen-Decision-1.5-4B",
    "Valen-Team/Valen-Decision-1.5-9B",
    "Valen-Team/Valen-Decision-1.5-27B",
)


def scene(width: int, height: int, offset: int = 0):
    """A deterministic drawing, so both sides read identical image bytes."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (width, height), (242, 242, 238))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((40 + offset, 60, 40 + offset + 150, 210), fill=(198, 40, 40))
    draw.ellipse((width - 220, 40, width - 80, 180), fill=(21, 101, 192))
    draw.rectangle((60, height - 120, width - 60, height - 70), fill=(69, 90, 100))
    draw.text((50, height - 60), "shelf 7", fill=(20, 20, 20))
    return canvas


def data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def sweep_scene(index: int):
    """One of a fixed family of drawings with a red box, a blue circle and a green bar."""
    from PIL import Image, ImageDraw

    width, height = ((640, 448), (480, 360), (800, 608), (512, 512))[index % 4]
    canvas = Image.new("RGB", (width, height), (242, 242, 238))
    draw = ImageDraw.Draw(canvas)
    # Positions and sizes move with the index; the shapes never overlap.
    side = 70 + 12 * (index % 5)
    left = 20 + (index * 37) % (width // 2 - side - 20)
    top = 20 + (index * 53) % (height // 2 - side - 20)
    draw.rectangle((left, top, left + side, top + side), fill=(198, 40, 40))
    radius = 35 + 6 * (index % 4)
    cx = width // 2 + radius + (index * 29) % (width // 2 - 2 * radius - 10)
    cy = radius + 10 + (index * 41) % (height // 2 - 2 * radius - 10)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius), fill=(21, 101, 192)
    )
    bar = height // 2 + 30 + (index * 17) % (height // 2 - 90)
    draw.rectangle((40, bar, width - 40, bar + 36), fill=(46, 125, 50))
    return canvas


def sweep_requests(count: int = 16) -> dict:
    """A wider set for statistics: per drawing, three bbox questions and three decisions."""
    cases = {}
    for index in range(count):
        state = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url(sweep_scene(index))},
                        },
                        {"type": "text", "text": "A drawing of three shapes."},
                    ],
                }
            ]
        }
        for target in ("the red square", "the blue circle", "the green bar"):
            name = target.split()[-1]
            cases[f"sweep_{index:02d}_bbox_{name}"] = {
                "state": state,
                "questions": {"where": {"type": "bbox", "instructions": target}},
            }
        cases[f"sweep_{index:02d}_decisions"] = {
            "state": state,
            "questions": {
                "left": {
                    "type": "choice",
                    "instructions": "Which shape is further to the left?",
                    "criteria": {
                        "square": "The red square.",
                        "circle": "The blue circle.",
                    },
                },
                "three": {
                    "type": "noul",
                    "instructions": "The drawing shows exactly three shapes.",
                },
                "size": {
                    "type": "score",
                    "instructions": "How large is the red square compared with the drawing?",
                    "criteria": ["Tiny.", "Small.", "Medium.", "Large."],
                },
            },
        }
    return cases


def requests_for(grounding: bool) -> dict:
    """The fixed request set; ``bbox`` questions need a grounding head."""
    one = data_url(scene(640, 448))
    two = data_url(scene(480, 360, offset=120))
    text = "The checkout page returns a 500 error and no orders are going through."
    chinese = "用户说：昨天下单的手机还没发货，想知道什么时候能到，并且要求退运费。"

    def image_state(urls, caption):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *(
                            {"type": "image_url", "image_url": {"url": url}}
                            for url in urls
                        ),
                        {"type": "text", "text": caption},
                    ],
                }
            ]
        }

    cases = {
        "text_choice": {
            "state": text,
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {
                        "infra": "Infrastructure and outages",
                        "billing": "Payments and invoices",
                        "product": "Product questions",
                    },
                }
            },
        },
        "text_noul": {
            "state": text,
            "questions": {
                "down": {
                    "type": "noul",
                    "instructions": "A production service is currently unavailable.",
                }
            },
        },
        "text_noul_rubric": {
            "state": text,
            "questions": {
                "escalate": {
                    "type": "noul",
                    "instructions": "Should this be escalated tonight?",
                    "criteria": {
                        "true": "Customers are blocked right now.",
                        "false": "It can wait for business hours.",
                    },
                }
            },
        },
        "text_score": {
            "state": text,
            "questions": {
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this ticket?",
                    "criteria": [
                        "Can wait a week",
                        "This week",
                        "Today",
                        "Right now",
                    ],
                }
            },
        },
        "text_shared_state": {
            "state": text,
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this ticket?",
                    "criteria": {
                        "infra": "Infrastructure and outages",
                        "billing": "Payments and invoices",
                    },
                },
                "down": {
                    "type": "noul",
                    "instructions": "A production service is currently unavailable.",
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this ticket?",
                    "criteria": ["Can wait a week", "This week", "Today"],
                },
                "refund": {
                    "type": "noul",
                    "instructions": "The customer is asking for money back.",
                },
            },
        },
        "text_chinese": {
            "state": chinese,
            "questions": {
                "意图": {
                    "type": "choice",
                    "instructions": "用户的主要诉求是什么？",
                    "criteria": {
                        "物流": "询问发货或配送进度",
                        "退款": "要求退款或退运费",
                        "换货": "要求更换商品",
                    },
                },
                "满意": {"type": "noul", "instructions": "用户对当前处理感到满意。"},
            },
        },
        "image_decisions": {
            "state": image_state([one], "A photo of one shelf in the warehouse."),
            "questions": {
                "red": {
                    "type": "noul",
                    "instructions": "A red box is on the shelf.",
                },
                "shape": {
                    "type": "choice",
                    "instructions": "Which shape is blue?",
                    "criteria": {
                        "square": "A square",
                        "circle": "A circle",
                        "triangle": "A triangle",
                    },
                },
                "fill": {
                    "type": "score",
                    "instructions": "How full does the shelf look?",
                    "criteria": ["Empty", "Half full", "Full"],
                },
            },
        },
        "two_images": {
            "state": image_state([one, two], "Two shelves from the same aisle."),
            "questions": {
                "same": {
                    "type": "noul",
                    "instructions": "Both images show the same arrangement.",
                },
                "which": {
                    "type": "choice",
                    "instructions": "Which image has the red box further right?",
                    "criteria": {"first": "The first image", "second": "The second"},
                },
            },
        },
    }
    if grounding:
        cases["image_bbox"] = {
            "state": image_state([one], "A photo of one shelf in the warehouse."),
            "questions": {
                "box": {"type": "bbox", "instructions": "the red box"},
                "circle": {"type": "bbox", "instructions": "the blue circle"},
                "red": {"type": "noul", "instructions": "A red box is on the shelf."},
            },
        }
        cases["two_images_bbox"] = {
            "state": image_state([one, two], "Two shelves from the same aisle."),
            "questions": {
                "first": {
                    "type": "bbox",
                    "instructions": "the red box",
                    "image": 0,
                },
                "second": {
                    "type": "bbox",
                    "instructions": "the red box",
                    "image": 1,
                },
            },
        }
    return cases


def native(
    model_path: Path, requests: dict, dtype: str = "bfloat16", mask: str | None = None
) -> dict:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if mask:
        # A deliberately wrong mask: a control for how far a mask error moves answers.
        config.valen_full_attention_mask = mask
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
    ).eval()
    model.to("cuda")
    answers = {}
    for name, request in requests.items():
        started = time.perf_counter()
        response = model.predict(json.loads(json.dumps(request)))
        answers[name] = {
            "answers": response["answers"],
            "usage": response["usage"],
            "seconds": time.perf_counter() - started,
        }
    return {
        "source": str(model_path),
        "device": torch.cuda.get_device_name(0),
        "dtype": f"{dtype} backbone, fp32 heads",
        "full_attention_mask": config.valen_full_attention_mask,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "cases": answers,
    }


def served(url: str, requests: dict, model_id: str | None, timeout: float) -> dict:
    cases = {}
    for name, request in requests.items():
        body = dict(request)
        if model_id:
            body["model"] = model_id
        data = json.dumps(body).encode()
        http = urllib.request.Request(
            f"{url}/v1/systemone",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http, timeout=timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise SystemExit(f"{name}: HTTP {error.code} {error.read().decode()}")
        cases[name] = {
            "answers": payload["answers"],
            "usage": payload["usage"],
            "seconds": time.perf_counter() - started,
        }
    return cases


def differences(
    expected: dict, actual: dict, tolerance: float = math.inf
) -> tuple[list, list, float]:
    """Problems, disagreeing selections and the largest numeric difference."""
    problems, disagreements, largest = [], [], 0.0
    if set(expected) != set(actual):
        problems.append(f"answer IDs differ: {sorted(expected)} vs {sorted(actual)}")
        return problems, disagreements, largest
    for qid in expected:
        want, have = expected[qid], actual[qid]
        if want.get("type") != have.get("type"):
            problems.append(f"{qid}: type {want.get('type')} vs {have.get('type')}")
            continue
        kind = want["type"]
        if kind == "bbox":
            for index, (a, b) in enumerate(zip(want["bbox"], have["bbox"])):
                largest = max(largest, abs(a - b))
                if abs(a - b) > tolerance:
                    problems.append(f"{qid}: box edge {index} {a} vs {b}")
            largest = max(largest, abs(want["confidence"] - have["confidence"]))
            if abs(want["confidence"] - have["confidence"]) > tolerance:
                problems.append(
                    f"{qid}: confidence {want['confidence']} vs {have['confidence']}"
                )
            continue
        if kind == "noul":
            difference = abs(want["noul"] - have["noul"])
            largest = max(largest, difference)
            if difference > tolerance:
                problems.append(f"{qid}: noul {want['noul']} vs {have['noul']}")
            if (want["noul"] >= 0.5) != (have["noul"] >= 0.5):
                disagreements.append(f"{qid}: noul decision disagrees")
            continue
        if set(want["probabilities"]) != set(have["probabilities"]):
            problems.append(f"{qid}: candidate keys differ")
            continue
        for key, value in want["probabilities"].items():
            difference = abs(value - have["probabilities"][key])
            largest = max(largest, difference)
            if difference > tolerance:
                problems.append(
                    f"{qid}: probability {key} {value} vs {have['probabilities'][key]}"
                )
        if kind == "choice":
            if want["choice"] != have["choice"]:
                disagreements.append(
                    f"{qid}: choice {want['choice']} vs {have['choice']}"
                )
        else:
            largest = max(largest, abs(want["score"] - have["score"]))
            if abs(want["score"] - have["score"]) > tolerance:
                problems.append(f"{qid}: score {want['score']} vs {have['score']}")
            if want["legend"] != have["legend"]:
                problems.append(f"{qid}: legend differs")
            best = max(want["probabilities"], key=want["probabilities"].get)
            theirs = max(have["probabilities"], key=have["probabilities"].get)
            if best != theirs:
                disagreements.append(f"{qid}: most likely level {best} vs {theirs}")
    return problems, disagreements, largest


def _values(answers: dict) -> tuple[dict, dict, dict]:
    """Probabilities, selected answers and boxes of one response, by question."""
    numbers, selected, boxes = {}, {}, {}
    for qid, answer in answers.items():
        kind = answer["type"]
        if kind == "bbox":
            boxes[qid] = answer["bbox"]
        elif kind == "noul":
            numbers[(qid, "noul")] = answer["noul"]
            selected[qid] = answer["noul"] >= 0.5
        else:
            for key, value in answer["probabilities"].items():
                numbers[(qid, key)] = value
            selected[qid] = (
                answer["choice"]
                if kind == "choice"
                else max(answer["probabilities"], key=answer["probabilities"].get)
            )
    return numbers, selected, boxes


def _iou(a: list, b: list) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - width * height
    )
    return width * height / union if union > 0 else 0.0


def _quantile(values: list, fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * len(values)))]


def distribution(first: dict, second: dict) -> dict:
    """How two recordings of the same requests differ."""
    probabilities, overlaps, differing, selections = [], [], 0, 0
    for name in first:
        a, picked_a, boxes_a = _values(first[name])
        b, picked_b, boxes_b = _values(second[name])
        probabilities += [abs(a[key] - b[key]) for key in a]
        selections += len(picked_a)
        differing += sum(picked_a[qid] != picked_b[qid] for qid in picked_a)
        overlaps += [_iou(boxes_a[qid], boxes_b[qid]) for qid in boxes_a]
    result = {
        "probability_median": _quantile(probabilities, 0.5),
        "probability_p90": _quantile(probabilities, 0.9),
        "probability_max": max(probabilities),
        "selections": selections,
        "selections_differing": differing,
    }
    if overlaps:
        result.update(
            boxes=len(overlaps),
            iou_median=_quantile(overlaps, 0.5),
            iou_p10=_quantile(overlaps, 0.1),
            iou_min=min(overlaps),
        )
    return result


def _recorded(path: Path) -> dict:
    """Answers by request from a reference recording or a comparison report."""
    cases = json.loads(path.read_text())["cases"]
    return {
        name: case["served_answers"] if "served_answers" in case else case["answers"]
        for name, case in cases.items()
    }


def summarize(fp32: Path, bf16: Path, control: Path, served: Path) -> dict:
    """Served answers against the release's precision floor and a mask control."""
    truth, release = _recorded(fp32), _recorded(bf16)
    wrong, ours = _recorded(control), _recorded(served)
    if not set(truth) == set(release) == set(wrong) == set(ours):
        raise SystemExit("the four recordings cover different request sets")
    return {
        "served bfloat16 vs release float32": distribution(ours, truth),
        "release bfloat16 vs release float32": distribution(release, truth),
        "causal-mask control vs release float32": distribution(wrong, truth),
        "served bfloat16 vs release bfloat16": distribution(ours, release),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", choices=MODELS)
    parser.add_argument("--url", default="http://127.0.0.1:8795")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--no-bbox", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--mask",
        choices=("causal",),
        help="record a control reference under a deliberately wrong attention mask",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="use the wider 16-drawing set for statistics instead of the fixed set",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        help="also fail when a value differs from the reference by more than this",
    )
    parser.add_argument(
        "--summarize",
        nargs=4,
        type=Path,
        metavar=("FP32", "BF16", "CONTROL", "SERVED"),
        help="print the distribution of differences between four recordings",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(*args.summarize), indent=2))
        return
    if args.output is None:
        parser.error("--output is required")
    requests = sweep_requests() if args.sweep else requests_for(not args.no_bbox)
    if args.reference_only:
        if args.model_path is None:
            parser.error("--reference-only needs --model-path")
        report = native(args.model_path, requests, args.dtype, args.mask)
        report["model_id"] = args.model_id
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("VALEN_REFERENCE_READY", args.output, flush=True)
        return
    if args.reference is None:
        parser.error("comparison needs --reference from a --reference-only run")
    reference = json.loads(args.reference.read_text())
    if sorted(reference["cases"]) != sorted(requests):
        parser.error("the recorded reference covers a different request set")
    tolerance = math.inf if args.tolerance is None else args.tolerance
    cases = served(args.url, requests, args.model_id, args.timeout)
    results, failures, disagreeing, largest = {}, 0, 0, 0.0
    for name in requests:
        problems, disagreements, difference = differences(
            reference["cases"][name]["answers"], cases[name]["answers"], tolerance
        )
        if (
            reference["cases"][name]["usage"]["input_tokens"]
            != cases[name]["usage"]["input_tokens"]
        ):
            problems.append("input token counts differ")
        if args.tolerance is not None:
            problems += disagreements
        largest = max(largest, difference)
        failures += len(problems)
        disagreeing += len(disagreements)
        results[name] = {
            "max_abs_difference": difference,
            "problems": problems,
            "disagreements": disagreements,
            "questions": len(requests[name]["questions"]),
            "reference_input_tokens": reference["cases"][name]["usage"]["input_tokens"],
            "served_input_tokens": cases[name]["usage"]["input_tokens"],
            "reference_seconds": reference["cases"][name]["seconds"],
            "served_seconds": cases[name]["seconds"],
            "served_answers": cases[name]["answers"],
        }
    report = {
        "model_id": args.model_id or reference.get("model_id"),
        "url": args.url,
        "tolerance": args.tolerance,
        "device": reference.get("device"),
        "reference_dtype": reference.get("dtype"),
        "reference_versions": reference.get("versions"),
        "max_abs_difference": largest,
        "disagreeing_selections": disagreeing,
        "problem_count": failures,
        "cases": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "VALEN_COMPARISON",
        "PASS" if failures == 0 and math.isfinite(largest) else "FAIL",
        f"max_abs_difference={largest:.3e}",
        f"disagreeing_selections={disagreeing}",
        flush=True,
    )
    for name, result in results.items():
        for problem in result["problems"]:
            print(f"  {name}: {problem}")
    if failures or not math.isfinite(largest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
