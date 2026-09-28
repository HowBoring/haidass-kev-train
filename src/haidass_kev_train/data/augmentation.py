"""Deterministic, per-record online augmentation for decision training."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import random

__all__ = ["augment_record"]

_NONE_OPTIONS = (
    ("other", "None of the above"),
    ("other", "A reason that fits none of the above"),
    ("none", "None of these"),
    ("other", "Something else"),
    ("not_listed", "Not listed here"),
    ("none_of_the_above", None),
    ("other", "A category that fits none of the above"),
    ("other", "None of the listed options apply"),
    ("unknown", "Cannot be determined from the options given"),
    ("other", "Other"),
    ("none", None),
    ("other", "An answer not covered by the other options"),
    ("no_match", "No option matches"),
)
_DISTRACTORS = {
    "weather": "Bad weather caused it",
    "purple": "The colour purple",
    "pancakes": "A recipe for pancakes",
    "taxes": "Unrelated: quarterly tax filing",
}
_MAX_OPTIONS = 255


def _record_id(record: dict) -> str:
    meta = record.get("_meta") or {}
    if not isinstance(meta, dict):
        raise ValueError("record _meta must be an object")
    if meta.get("id"):
        return str(meta["id"])
    payload = json.dumps(
        {"state": record.get("state"), "questions": record.get("questions")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"record/{hashlib.sha256(payload).hexdigest()[:16]}"


def _rng(record_id: str, seed: int, epoch: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{epoch}:{record_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _target(question: dict, keys: list[str], name: str, record_id: str) -> tuple[str, dict[str, float]]:
    soft = question.get("target")
    if soft is None:
        if "label" not in question or question["label"] not in keys:
            raise ValueError(f"{record_id}: choice question {name!r} needs a label in its criteria")
        return "hard", {key: float(key == question["label"]) for key in keys}
    if isinstance(soft, dict) and set(soft) == set(keys):
        shape = "dict"
        values = {key: float(soft[key]) for key in keys}
    elif isinstance(soft, list) and len(soft) == len(keys):
        shape = "list"
        values = {key: float(value) for key, value in zip(keys, soft)}
    else:
        raise ValueError(f"{record_id}: choice question {name!r} target must cover every option exactly")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()) or not math.isclose(
        sum(values.values()), 1.0, abs_tol=1e-5
    ):
        raise ValueError(f"{record_id}: choice question {name!r} target must be a normalized finite distribution")
    if "label" in question and question["label"] not in keys:
        raise ValueError(f"{record_id}: choice question {name!r} label is not in its criteria")
    return shape, values


def _store_target(question: dict, keys: list[str], shape: str, target: dict[str, float]) -> None:
    if shape == "dict":
        question["target"] = {key: target[key] for key in keys}
    elif shape == "list":
        question["target"] = [target[key] for key in keys]


def _removed_key(question: dict, keys: list[str], shape: str, target: dict[str, float], rng: random.Random) -> str:
    if shape == "hard":
        return question["label"]
    draw = rng.random()
    total = 0.0
    for key in keys:
        total += target[key]
        if draw < total:
            return key
    return keys[-1]


def _none_option(criteria: dict, rng: random.Random, name: str, record_id: str) -> tuple[str, object]:
    available = [option for option in _NONE_OPTIONS if option[0] not in criteria]
    if not available:
        raise ValueError(f"{record_id}: choice question {name!r} collides with every supported none-option key")
    return rng.choice(available)


def _addable(criteria: dict, name: str, record_id: str) -> None:
    if len(criteria) >= _MAX_OPTIONS:
        raise ValueError(f"{record_id}: choice question {name!r} cannot add an option beyond {_MAX_OPTIONS}")


def _augment_choice(
    question: dict,
    *,
    name: str,
    record_id: str,
    rng: random.Random,
    shuffle: bool,
    p_none: float,
    p_none_distract: float,
    p_distract: float,
) -> dict:
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or not criteria or len(criteria) > _MAX_OPTIONS:
        raise ValueError(f"{record_id}: choice question {name!r} criteria must be an object of 1..{_MAX_OPTIONS} options")
    out = deepcopy(question)
    keys = list(criteria)
    shape, target = _target(out, keys, name, record_id)
    result = dict(criteria)
    draw = rng.random()

    if len(result) > 2 and draw < p_none:
        none_key, none_description = _none_option(result, rng, name, record_id)
        removed = _removed_key(out, keys, shape, target, rng)
        del result[removed]
        result[none_key] = none_description
        target[none_key] = target.pop(removed)
        if shape == "hard" or out.get("label") == removed:
            out["label"] = none_key
    elif p_none <= draw < p_none + p_none_distract:
        _addable(result, name, record_id)
        none_key, none_description = _none_option(result, rng, name, record_id)
        result[none_key] = none_description
        target[none_key] = 0.0
    elif p_none + p_none_distract <= draw < p_none + p_none_distract + p_distract:
        _addable(result, name, record_id)
        available = [key for key in _DISTRACTORS if key not in result]
        if not available:
            raise ValueError(f"{record_id}: choice question {name!r} collides with every supported distractor key")
        key = rng.choice(available)
        result[key] = _DISTRACTORS[key]
        target[key] = 0.0

    order = list(result)
    if shuffle:
        rng.shuffle(order)
    out["criteria"] = {key: result[key] for key in order}
    _store_target(out, order, shape, target)
    return out


def _pair(record: dict, record_id: str, rng: random.Random, shuffle: bool) -> tuple[dict, dict] | None:
    candidates = []
    for name, question in record["questions"].items():
        criteria = question.get("criteria")
        if question.get("type") == "choice" and isinstance(criteria, dict) and 3 <= len(criteria) < _MAX_OPTIONS:
            candidates.append((name, question))
    if not candidates:
        return None

    name, source = rng.choice(candidates)
    criteria = source["criteria"]
    keys = list(criteria)
    shape, source_target = _target(source, keys, name, record_id)
    none_key, none_description = _none_option(criteria, rng, name, record_id)
    removed = _removed_key(source, keys, shape, source_target, rng)
    order = keys + [none_key]
    if shuffle:
        rng.shuffle(order)

    present_question = deepcopy(source)
    present_criteria = {key: none_description if key == none_key else criteria[key] for key in order}
    present_target = {**source_target, none_key: 0.0}
    present_question["criteria"] = present_criteria
    _store_target(present_question, order, shape, present_target)

    absent_question = deepcopy(present_question)
    absent_order = [key for key in order if key != removed]
    absent_target = {key: value for key, value in present_target.items() if key != removed}
    absent_target[none_key] += source_target[removed]
    absent_question["criteria"] = {key: present_criteria[key] for key in absent_order}
    _store_target(absent_question, absent_order, shape, absent_target)
    if shape == "hard" or absent_question.get("label") == removed:
        absent_question["label"] = none_key

    return (
        {"state": deepcopy(record["state"]), "questions": {name: present_question}},
        {"state": deepcopy(record["state"]), "questions": {name: absent_question}},
    )


def _metadata(record: dict, record_id: str, seed: int, epoch: int, variant: str) -> dict:
    meta = deepcopy(record.get("_meta") or {})
    group_id = meta.get("group_id") or record_id
    suffix = hashlib.sha256(f"{seed}:{epoch}:{record_id}:{variant}".encode()).hexdigest()[:16]
    meta.update({"id": f"{record_id}/{variant}/{suffix}", "group_id": group_id, "variant": variant})
    return meta


def augment_record(
    record: dict,
    *,
    seed: int,
    epoch: int,
    shuffle: bool = True,
    p_none: float = 0.1,
    p_none_distract: float = 0.12,
    p_distract: float = 0.15,
    p_none_pair: float = 0.25,
) -> list[dict]:
    """Return deterministic augmented siblings without mutating ``record`` or global RNG state."""
    probabilities = (p_none, p_none_distract, p_distract, p_none_pair)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
        raise ValueError("augmentation probabilities must be finite values from zero to one")
    if p_none + p_none_distract + p_distract > 1:
        raise ValueError("p_none, p_none_distract, and p_distract must sum to at most one")
    if not isinstance(record, dict) or "state" not in record:
        raise ValueError("record must be an object with state")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("record needs a non-empty questions object")

    record_id = _record_id(record)
    rng = _rng(record_id, seed, epoch)
    augmented = deepcopy(record)
    augmented["questions"] = {}
    for name, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"{record_id}: question {name!r} must be an object")
        qtype = question.get("type")
        if qtype == "choice":
            augmented["questions"][name] = _augment_choice(
                question,
                name=name,
                record_id=record_id,
                rng=rng,
                shuffle=shuffle,
                p_none=p_none,
                p_none_distract=p_none_distract,
                p_distract=p_distract,
            )
        elif qtype in ("noul", "score"):
            augmented["questions"][name] = deepcopy(question)
        else:
            raise ValueError(f"{record_id}: question {name!r} has unsupported type {qtype!r}")
    if not shuffle and not any(probabilities):
        return [deepcopy(record)]
    augmented["_meta"] = _metadata(record, record_id, seed, epoch, "augmented")
    variants = [augmented]

    if p_none_pair and rng.random() < p_none_pair:
        pair = _pair(record, record_id, rng, shuffle)
        if pair:
            for variant, item in zip(("none_present", "none_absent"), pair):
                item["_meta"] = _metadata(record, record_id, seed, epoch, variant)
                variants.append(item)
    return variants
