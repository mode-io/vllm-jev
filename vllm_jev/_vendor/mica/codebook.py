"""The fixed output codebook shared by training, evaluation and deployment.

Every judgment is read out of one forward pass as a distribution over label
tokens, so the labels must each be a single token in the deployed tokenizer and
must not move between training and serving. The first 52 entries are the
original A-Z/a-z labels, so an adapter trained before the codebook grew keeps
its exact output rows; the rest are the uppercase two-letter combinations that
this tokenizer encodes as one token, in alphabetical order, up to the 255
candidates a Choice may carry.

``No``/``Yes`` are the Noul readout and are kept separate from the Choice
labels: they are distinct single tokens, and a Noul never uses a letter label.
"""
from __future__ import annotations

import json
import string
from pathlib import Path

MAX_CANDIDATES = 255
# Derived once from the deployed tokenizer and stored, so training, evaluation,
# export and the local runtime cannot drift apart on which token means which
# candidate. Regenerating it is a deliberate act, not a side effect of an import.
CODEBOOK_PATH = Path(__file__).with_name('codebook_255.json')
_CODEBOOK = json.loads(CODEBOOK_PATH.read_text(encoding='utf-8'))

NOUL_LABELS = tuple(_CODEBOOK['noul_labels'])
OPTION_LABELS = list(_CODEBOOK['option_labels'])
READOUT_LABELS = list(NOUL_LABELS) + OPTION_LABELS
TOKENIZER_REVISION = _CODEBOOK.get('revision')

if len(OPTION_LABELS) != MAX_CANDIDATES or len(set(READOUT_LABELS)) != len(READOUT_LABELS):
    raise ValueError('stored codebook must hold 255 distinct candidate labels')
if OPTION_LABELS[:52] != list(string.ascii_uppercase + string.ascii_lowercase):
    raise ValueError('the first 52 labels must stay A-Z/a-z so earlier adapters keep their rows')


def verify(tokenizer) -> dict[str, int]:
    """Fail loudly if this checkpoint cannot serve the codebook.

    Called wherever the codebook is bound to a real tokenizer, so a mismatch
    surfaces before training or export rather than as silent label collisions.
    """
    encoded = [tokenizer.encode(label, add_special_tokens=False) for label in READOUT_LABELS]
    multi = [label for label, ids in zip(READOUT_LABELS, encoded) if len(ids) != 1]
    if multi:
        raise ValueError(f'{len(multi)} readout labels are not single tokens, first: {multi[:5]}')
    ids = [ids[0] for ids in encoded]
    if len(set(ids)) != len(ids):
        seen, collisions = {}, []
        for label, token in zip(READOUT_LABELS, ids):
            if token in seen:
                collisions.append((seen[token], label))
            seen[token] = label
        raise ValueError(f'readout labels collide on the same token: {collisions[:5]}')
    return {'labels': len(READOUT_LABELS), 'candidates': MAX_CANDIDATES,
            'first_token_id': ids[0], 'last_token_id': ids[-1]}
