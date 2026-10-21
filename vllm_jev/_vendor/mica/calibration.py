"""One calibration format and logit transform for evaluation and serving."""
import json
import math
from pathlib import Path


def load_calibration(path, *, model_identity=None):
    fitted = json.loads(Path(path).read_text(encoding='utf-8'))
    if fitted.get('schema') == 'mica-v3-temperatures-v1':
        identity = fitted.get('model_identity', {}).get('id')
        if not identity or model_identity != identity:
            raise ValueError('Calibrator model identity mismatch; provide the pinned runtime identity')
        temperatures = fitted.get('temperatures')
        if not isinstance(temperatures, dict) or not temperatures:
            raise ValueError('Calibrator has no fitted temperatures')
    elif 'schema' not in fitted and 'temperature' in fitted:
        temperatures = {kind: fitted['temperature'] for kind in ('noul', 'choice', 'score')}
    else:
        raise ValueError('Unsupported calibration schema')
    for kind, value in temperatures.items():
        if kind not in ('noul', 'choice', 'score') or type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Calibration temperature must be a positive finite number for a supported kind')
    bias = fitted.get('noul_logit_bias', 0.)
    if type(bias) not in (int, float) or not math.isfinite(bias):
        raise ValueError('Noul calibration bias must be finite')
    return fitted, dict(temperatures), float(bias)


def calibrated_logits(values, temperature=1., noul_logit_bias=0.):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if not math.isfinite(noul_logit_bias):
        raise ValueError('Noul calibration bias must be finite')
    if len(values) < 2 or any(math.isnan(v) or v == math.inf for v in values) or max(values) == -math.inf:
        raise ValueError('Invalid runtime logits')
    result = [v / temperature for v in values]
    if noul_logit_bias:
        if len(values) != 2:
            raise ValueError('Noul bias requires two logits')
        result[1] += noul_logit_bias
    return result


def distribution(values, temperature=1., noul_logit_bias=0.):
    scaled = calibrated_logits(values, temperature, noul_logit_bias)
    peak = max(scaled)
    weights = [math.exp(v - peak) for v in scaled]
    total = sum(weights)
    return [v / total for v in weights]
