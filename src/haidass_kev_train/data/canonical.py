"""Canonical Decision Records and Decision View materialization (``data_format="canonical_choice_v1"``).

A Canonical Decision Record is the frozen training case published by the offline builder:

```json
{
  "source": "ufw-zh",
  "state": "permitted shared context; a string, legitimately empty when the question is self-contained",
  "question": "the one question",
  "gold": "the source-grounded answer",
  "distractors": ["five", "distinct", "incorrect", "alternatives", "exactly five"],
  "_meta": {
    "id": "stable source-row identity (never a traversal index)",
    "group_id": "Source Group identity used by split-integrity checks",
    "source": "must equal the top-level source",
    "source_ref": {"path": "source shard relative path", "line": 0,
                   "sha256": "SHA-256 hex of the raw source text",
                   "question_span": [q0, q1], "answer_span": [a0, a1]},
    "validation": "Candidate Validation Path, e.g. programmatic or llm_adjudicated"
  }
}
```

``source_ref`` is the minimal Source Trace: the shard path and zero-based ``line`` locate the
source row, while ``question_span``/``answer_span`` are zero-based, half-open character spans
into the raw source text whose SHA-256 is pinned in ``sha256``; a changed source hash
invalidates old spans. Sources with an ``ufw`` prefix additionally retain a non-empty ``uid``
and a ``state_span`` locating the original context; when the source question was
multiple-choice and underwent presentation conversion, ``mcq`` is true and ``option_spans``
locates the old options (short-answer sources omit both). Sources with a ``finemath`` prefix
preserve the available ``url`` and/or ``snapshot_type`` identity and may add a
``givens_span`` for the necessary givens. Optional trace fields are structurally validated
whenever present. Slice verification against the raw text happens at build time (the trainer
never sees source material); loading validates the trace structurally.

No candidate set, position label, or soft target is persisted. Training materializes one
dynamic Decision View per record per epoch: K is drawn from the configured K=2..6
distribution (default 0.10/0.20/0.30/0.25/0.15), K-1 distractors are sampled without
replacement, gold is always retained, and the complete candidate set is shuffled.
Development and the Training Probe instead use five Fixed Evaluation Views per record, one
per K, whose membership and order never depend on the epoch. All randomness derives from
seed, epoch-or-K, stable record id and purpose - never global RNG, access counts, or
traversal order - and materialization never mutates the canonical record.

Views are existing Kev choice records: ``state`` is the canonical state, one ``choice``
question named ``decision`` carries the question as ``instructions``, each candidate text as
a criteria key with a null description (insertion order is presentation order), the gold
text as ``label`` and the canonical source as ``src``. The existing encoder maps the gold
key to the current one-hot position; no role names or metadata are rendered to the model.
"""

from __future__ import annotations

import hashlib
import math
import random

from haidass_kev_train.data.packing import encode_record, load_suite

__all__ = [
    "DEFAULT_K_PROBABILITIES",
    "K_VALUES",
    "check_k_probabilities",
    "evaluation_views",
    "load_canonical_suite",
    "preflight",
    "training_view",
    "validate_record",
]

K_VALUES = (2, 3, 4, 5, 6)
DEFAULT_K_PROBABILITIES = (0.10, 0.20, 0.30, 0.25, 0.15)
_DISTRACTOR_COUNT = 5
# Supervision belongs to views, never to the persisted canonical record.
_FORBIDDEN = ("label", "target", "candidates")


def _span_valid(span) -> bool:
    return (
        isinstance(span, (list, tuple))
        and len(span) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in span)
        and 0 <= span[0] < span[1]
    )


def _span_field(source_ref: dict, rid: str, name: str, required: bool) -> None:
    span = source_ref.get(name)
    if span is None:
        if required:
            raise ValueError(f"{rid}: source_ref.{name} must locate the original text with a zero-based half-open span")
        return
    if not _span_valid(span):
        raise ValueError(f"{rid}: source_ref.{name} must be a zero-based half-open [start, end) span")


def _span_list_field(source_ref: dict, rid: str, name: str, required: bool) -> None:
    spans = source_ref.get(name)
    if spans is None:
        if required:
            raise ValueError(f"{rid}: source_ref.{name} must locate the original options")
        return
    if not isinstance(spans, list) or not spans or any(not _span_valid(span) for span in spans):
        raise ValueError(f"{rid}: source_ref.{name} must be a non-empty list of zero-based half-open spans")


def validate_record(record: object) -> None:
    """Reject a Canonical Decision Record violating this module's contract.

    Must run on the raw record, before ``load_suite`` fills identity defaults; a missing
    ``_meta.id`` is an error here even though the loader would otherwise synthesize one.
    """
    if not isinstance(record, dict):
        raise ValueError(f"canonical record must be an object, got {type(record).__name__}")
    meta = record.get("_meta")
    rid = meta.get("id") if isinstance(meta, dict) else None
    rid = rid if isinstance(rid, str) and rid else "<no id>"
    source = record.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{rid}: canonical source must be a non-empty string")
    if not isinstance(record.get("state"), str):
        raise ValueError(f"{rid}: canonical state must be a string (legitimately empty is allowed)")
    if not isinstance(record.get("question"), str) or not record["question"].strip():
        raise ValueError(f"{rid}: canonical question must be a non-empty string")
    gold = record.get("gold")
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError(f"{rid}: canonical gold must be a non-empty string")
    distractors = record.get("distractors")
    if (
        not isinstance(distractors, list)
        or len(distractors) != _DISTRACTOR_COUNT
        or any(not isinstance(item, str) or not item.strip() for item in distractors)
    ):
        raise ValueError(f"{rid}: canonical distractors must be exactly {_DISTRACTOR_COUNT} non-empty strings")
    candidates = [gold.strip(), *(item.strip() for item in distractors)]
    if len(set(candidates)) != 1 + _DISTRACTOR_COUNT:
        raise ValueError(f"{rid}: gold and distractors must be textually distinct")
    for forbidden in _FORBIDDEN:
        if forbidden in record:
            raise ValueError(f"{rid}: canonical records do not persist {forbidden!r}")
    if not isinstance(meta, dict):
        raise ValueError(f"{rid}: canonical record needs a _meta object")
    if rid == "<no id>":
        raise ValueError("canonical _meta.id must be a non-empty stable string")
    if not isinstance(meta.get("group_id"), str) or not meta["group_id"].strip():
        raise ValueError(f"{rid}: canonical _meta.group_id must be a non-empty string")
    if meta.get("source") != source:
        raise ValueError(f"{rid}: canonical _meta.source must match the top-level source")
    source_ref = meta.get("source_ref")
    if not isinstance(source_ref, dict):
        raise ValueError(f"{rid}: canonical _meta.source_ref must locate the source row")
    if not isinstance(source_ref.get("path"), str) or not source_ref["path"].strip():
        raise ValueError(f"{rid}: source_ref.path must name the source shard")
    line = source_ref.get("line")
    if isinstance(line, bool) or not isinstance(line, int) or line < 0:
        raise ValueError(f"{rid}: source_ref.line must be a non-negative row locator")
    sha = source_ref.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64 or set(sha) - set("0123456789abcdef"):
        raise ValueError(f"{rid}: source_ref.sha256 must be the raw source text SHA-256 hex digest")
    _span_field(source_ref, rid, "question_span", True)
    _span_field(source_ref, rid, "answer_span", True)
    # Validated whenever present, for every source; UFW sources must always carry it.
    _span_field(source_ref, rid, "state_span", False)
    lowered = source.lower()
    if lowered.startswith("ufw"):
        uid = source_ref.get("uid")
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError(f"{rid}: UFW source_ref.uid must retain the non-empty source row identity")
        _span_field(source_ref, rid, "state_span", True)
        mcq = source_ref.get("mcq", False)
        if not isinstance(mcq, bool):
            raise ValueError(f"{rid}: source_ref.mcq must be a boolean")
        # Old-option locations apply only to converted multiple-choice questions.
        _span_list_field(source_ref, rid, "option_spans", mcq)
        _span_field(source_ref, rid, "givens_span", False)
    elif lowered.startswith("finemath"):
        for name in ("url", "snapshot_type"):
            value = source_ref.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{rid}: FineMath source_ref.{name} must be a non-empty string")
        if source_ref.get("url") is None and source_ref.get("snapshot_type") is None:
            raise ValueError(f"{rid}: FineMath source_ref must preserve the available url or snapshot identity")
        _span_field(source_ref, rid, "givens_span", False)
        _span_list_field(source_ref, rid, "option_spans", False)
    else:
        _span_field(source_ref, rid, "givens_span", False)
        _span_list_field(source_ref, rid, "option_spans", False)
    if not isinstance(meta.get("validation"), str) or not meta["validation"].strip():
        raise ValueError(f"{rid}: canonical _meta.validation must name the Candidate Validation Path")


def load_canonical_suite(path, split: str = "train") -> list[dict]:
    """One canonical split: manifest SHA-256/count verified via ``load_suite``, contract
    validated on every raw record, and canonical ids unique within the split."""
    records = load_suite(path, split, validate=validate_record)
    ids = [record["_meta"]["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}/{split}.jsonl: duplicate canonical ids")
    return records


def check_k_probabilities(probabilities) -> tuple:
    """Validate the configured K=2..6 sampling distribution; None selects the default."""
    if probabilities is None:
        return DEFAULT_K_PROBABILITIES
    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != len(K_VALUES):
        raise ValueError(f"k_probabilities must be {len(K_VALUES)} numbers for K={list(K_VALUES)}")
    values = tuple(probabilities)
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in values):
        raise ValueError("k_probabilities must be finite non-negative numbers")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError(f"k_probabilities must sum to 1, got {sum(values)}")
    return values


def _rng(record_id: str, seed: int, purpose: str, *, epoch=None, k=None) -> random.Random:
    key = f"{seed}:{purpose}:{record_id}"
    if epoch is not None:
        key += f":epoch{epoch}"
    if k is not None:
        key += f":k{k}"
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def _candidates(record: dict, rng: random.Random, k: int) -> list[str]:
    """Gold plus K-1 distractors sampled without replacement, fully shuffled."""
    candidates = [record["gold"], *rng.sample(record["distractors"], k - 1)]
    rng.shuffle(candidates)
    return candidates


def _view(record: dict, candidates: list[str], view_id: str, k: int) -> dict:
    """One Kev choice record presenting ``candidates`` in order; candidate texts are unique
    by ``validate_record``, so keyed criteria cannot silently drop one. ``k`` is kept in the
    view metadata for per-K reporting."""
    meta = record["_meta"]
    return {
        "state": record["state"],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": record["question"],
                "criteria": {text: None for text in candidates},
                "label": record["gold"],
                "src": record["source"],
            }
        },
        "_meta": {
            "id": view_id,
            "canonical_id": meta["id"],
            "group_id": meta["group_id"],
            "variant": "clean",
            "source": meta["source"],
            "k": k,
        },
    }


def training_view(record: dict, *, seed: int, epoch: int, probabilities=DEFAULT_K_PROBABILITIES) -> dict:
    """One dynamic Decision View of ``record`` for ``epoch``; K follows the validated
    ``probabilities``. Deterministic in seed/epoch/record id; never mutates ``record``."""
    rng = _rng(record["_meta"]["id"], seed, "train", epoch=epoch)
    k = rng.choices(K_VALUES, weights=probabilities, k=1)[0]
    return _view(record, _candidates(record, rng, k), record["_meta"]["id"], k)


def evaluation_views(record: dict, *, seed: int, purpose: str) -> list[dict]:
    """The five Fixed Evaluation Views of ``record``, one per K=2..6, derived from
    seed/record id/K/purpose and never the epoch; ids carry the K suffix."""
    rid = record["_meta"]["id"]
    return [
        _view(record, _candidates(record, _rng(rid, seed, purpose, k=k), k), f"{rid}/k{k}", k)
        for k in K_VALUES
    ]


def preflight(records, tokenizer, *, max_packed: int) -> None:
    """Encode every record with all six candidates through the real tokenizer and encoder.

    Per-field encoding makes packed length invariant to candidate order, so one
    all-candidate view per record bounds every sampled view. Overflow and structural-marker
    collisions are rejected at startup, never truncated and never left to a later K=6 epoch.
    """
    for record in records:
        view = _view(record, [record["gold"], *record["distractors"]], record["_meta"]["id"], 6)
        try:
            encode_record(view, tokenizer, max_packed=max_packed)
        except ValueError as error:
            raise ValueError(f"canonical preflight rejected {record['_meta']['id']}: {error}") from error
