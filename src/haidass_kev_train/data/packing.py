"""Encode Kev decision records into padded, block-causal training batches.

The shared state and structured instruction/criterion fields are rendered as text.
Each question becomes `<question> instructions <option> text <option_end> ... <decide>`.
Branch positions restart at the state length, options stay in criteria order inside one causal
sequence, and the additive block-causal mask lets every branch read the whole state while
isolating branches from each other.

Nothing is truncated: a record that does not fit `max_packed` is rejected with its identity.

The tokenizer has no literal `<|kev_*|>` entries, so the five logical markers are inserted as
the existing, rarely used delimiters pinned in `docs/agents/training.md`; vocabulary size and
embedding rows stay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
from pathlib import Path
import re

import torch

__all__ = [
    "MARKER_IDS",
    "MARKER_TOKENS",
    "EncodedRecord",
    "PackedDecisionBatch",
    "check_group_integrity",
    "collate",
    "encode_record",
    "load_suite",
    "marker_collision_counts",
    "option_text",
    "question_keys",
    "render",
    "resolve_marker_ids",
    "user_tokens",
]

# Logical Kev marker -> existing tokenizer spelling it is inserted as.
from haidass_kev_train.model.decision import MARKER_TOKENS as _MODEL_MARKERS, resolve_markers

MARKER_TOKENS = {key[2:-2]: value for key, value in _MODEL_MARKERS.items()}
MARKER_IDS = {key: index for index, key in enumerate(MARKER_TOKENS, 6)}

QTYPES = ("choice", "noul", "score")
MAX_OPTIONS = 255

# Used only by the preflight collision report.
_DELIMITER = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


# ------------------------------------------------------------------------------------- renderer


def render(value, indent: int = 0) -> str:
    """Flatten str | object | array into the text the model sees; field names stay as labels."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {render(item, indent + 1).lstrip()}" for item in value)
    return "\n".join(
        f"{pad}{key}:\n{render(item, indent + 1)}" if isinstance(item, (dict, list)) else f"{pad}{key}: {render(item)}"
        for key, item in value.items()
    )


def option_text(name: str, desc) -> str:
    """One rendered option: the key alone, or `key: description` when a description exists."""
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria) -> list[str]:
    """Option keys in option order: criteria names (choice), ["false", "true"] (noul), level indices (score)."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(index) for index in range(len(criteria))]


def user_tokens(tokenizer, text: str) -> list[int]:
    """Reject literal reused delimiters instead of changing user text."""
    if any(token in text for token in MARKER_TOKENS.values()):
        raise ValueError("Input text collides with a reused structural token")
    return tokenizer(text, add_special_tokens=False).input_ids


def resolve_marker_ids(tokenizer) -> dict[str, int]:
    return {key[2:-2]: value for key, value in resolve_markers(tokenizer).items()}


# ------------------------------------------------------------------------------------ records


@dataclass(frozen=True)
class _Question:
    name: str
    qtype: str
    instructions: str
    options: list[str]
    keys: list[str]
    target: list[float]
    src: str


def _identity(record: dict, fallback: str | None = None) -> dict:
    """Record metadata with the stable identity fields filled in; never None."""
    meta = dict(record.get("_meta") or {})
    if not meta.get("id"):
        meta["id"] = fallback or f"record/{hashlib.sha256(json.dumps(record.get('state'), sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]}"
    meta.setdefault("group_id", meta["id"])
    meta.setdefault("variant", "clean")
    meta.setdefault("source", "unknown")
    return meta


def _options(qtype: str, criteria, name: str, rid: str) -> tuple[list[str], list[str]]:
    """(keys, rendered option texts) for one question, validating the polymorphic criteria shape."""
    if qtype == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise ValueError(f"{rid}: question {name!r} choice criteria must be an object of 1..{MAX_OPTIONS} options")
        return question_keys(qtype, criteria), [option_text(key, desc) for key, desc in criteria.items()]
    if qtype == "noul":
        criteria = {} if criteria is None else criteria
        if not isinstance(criteria, dict):
            raise ValueError(f"{rid}: question {name!r} noul criteria must be an object")
        return question_keys(qtype, criteria), [option_text("no", criteria.get("false")), option_text("yes", criteria.get("true"))]
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_OPTIONS:
        raise ValueError(f"{rid}: question {name!r} score criteria must be a list of 2..{MAX_OPTIONS} ordered levels")
    return question_keys(qtype, criteria), [render(level) for level in criteria]


def _target(question: dict, qtype: str, keys: list[str], name: str, rid: str) -> list[float]:
    """Target distribution over `keys`: a normalized soft `target` when present, else a one-hot label."""
    soft = question.get("target")
    if soft is not None:
        if isinstance(soft, dict) and set(soft) == set(keys):
            mass = [float(soft[key]) for key in keys]
        elif isinstance(soft, list) and len(soft) == len(keys):
            mass = [float(value) for value in soft]
        else:
            raise ValueError(f"{rid}: question {name!r} target must cover every option exactly")
        if any(not 0 <= value <= 1 for value in mass) or abs(sum(mass) - 1.) > 1e-5:
            raise ValueError(f"{rid}: question {name!r} target must be a normalized finite distribution")
        return mass
    if "label" not in question:
        raise ValueError(f"{rid}: question {name!r} has neither a label nor a target")
    label = question["label"]
    if qtype == "choice":
        if label not in keys:
            raise ValueError(f"{rid}: question {name!r} label {label!r} is not one of {keys}")
        index = keys.index(label)
    elif qtype == "noul":
        if label not in (True, False, 0, 1):
            raise ValueError(f"{rid}: question {name!r} noul label must be true/false, got {label!r}")
        index = int(bool(label))
    else:
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(keys):
            raise ValueError(f"{rid}: question {name!r} score label must be a level index in 0..{len(keys) - 1}, got {label!r}")
        index = label
    one_hot = [0.0] * len(keys)
    one_hot[index] = 1.0
    return one_hot


def _materialize(record: dict) -> tuple[str, list[_Question], dict]:
    """Labelled Kev request -> (state text, questions in order, identity metadata)."""
    if not isinstance(record, dict):
        raise ValueError(f"record must be an object, got {type(record).__name__}")
    if "state" not in record:
        raise ValueError("record has no state")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"{_identity(record)['id']}: record needs a non-empty questions object")
    meta = _identity(record)
    questions_out: list[_Question] = []
    for name, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"{meta['id']}: question {name!r} must be an object")
        qtype = question.get("type")
        if qtype not in QTYPES:
            raise ValueError(f"{meta['id']}: question {name!r} has unsupported type {qtype!r}")
        keys, options = _options(qtype, question.get("criteria"), name, meta["id"])
        questions_out.append(
            _Question(
                name=name,
                qtype=qtype,
                instructions=render(question.get("instructions")),
                options=options,
                keys=keys,
                target=_target(question, qtype, keys, name, meta["id"]),
                src=str(question.get("src") or meta.get("source") or "unknown"),
            )
        )
    return render(record["state"]), questions_out, meta


# ------------------------------------------------------------------------------------ encoding


@dataclass(frozen=True)
class EncodedRecord:
    """One packed record: state tokens then one isolated branch per question, positions reset per branch."""

    input_ids: list[int]
    position_ids: list[int]
    segment_ids: list[int]
    decide_positions: list[int]
    option_end_positions: list[list[int]]
    target_probs: list[list[float]]
    metadata: list[dict]


def encode_record(record: dict, tokenizer, max_packed: int = 2048) -> EncodedRecord:
    """Pack one labelled record, rejecting (never truncating) anything longer than `max_packed`."""
    markers = resolve_marker_ids(tokenizer)
    state_text, questions, meta = _materialize(record)
    ids = [markers["kev_state"]] + user_tokens(tokenizer, state_text)
    state_length = len(ids)
    position_ids = list(range(state_length))
    segment_ids = [0] * state_length
    decide_positions: list[int] = []
    option_end_positions: list[list[int]] = []
    target_probs: list[list[float]] = []
    metadata: list[dict] = []
    branch_lengths: list[int] = []
    for index, question in enumerate(questions, start=1):
        branch = [markers["kev_question"]] + user_tokens(tokenizer, question.instructions)
        ends: list[int] = []
        for option in question.options:
            branch += [markers["kev_option"]] + user_tokens(tokenizer, option) + [markers["kev_option_end"]]
            ends.append(len(ids) + len(branch) - 1)
        branch.append(markers["kev_decide"])
        base = len(ids)
        branch_lengths.append(len(branch))
        ids += branch
        position_ids += list(range(state_length, state_length + len(branch)))
        segment_ids += [index] * len(branch)
        decide_positions.append(base + len(branch) - 1)
        option_end_positions.append(ends)
        target_probs.append(question.target)
        metadata.append(
            {
                "record_id": meta["id"],
                "question_name": question.name,
                "question_type": question.qtype,
                "src": question.src,
                "group_id": meta["group_id"],
                "variant": meta["variant"],
                **{key: meta[key] for key in ("canonical_id", "k") if key in meta},
                "option_keys": list(question.keys),
            }
        )
    if len(ids) > max_packed:
        raise ValueError(
            f"record {meta['id']}: packed length {len(ids)} exceeds max_packed={max_packed} "
            f"(state {state_length}, questions {len(questions)}, branches {branch_lengths})"
        )
    return EncodedRecord(ids, position_ids, segment_ids, decide_positions, option_end_positions, target_probs, metadata)


# ------------------------------------------------------------------------------------ batching


@dataclass
class PackedDecisionBatch:
    """Right-padded batch: `[B,L]` ids/positions, additive `[B,1,L,L]` block-causal bias, `[B,Q,K]` readouts."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_bias: torch.Tensor
    decide_positions: torch.Tensor
    option_end_positions: torch.Tensor
    question_mask: torch.Tensor
    option_mask: torch.Tensor
    target_probs: torch.Tensor | None
    metadata: list[list[dict]]

    def to(self, device) -> "PackedDecisionBatch":
        """Move every tensor to `device`; metadata stays on the CPU."""
        moved = {field.name: value.to(device) for field in fields(self) if isinstance(value := getattr(self, field.name), torch.Tensor)}
        return replace(self, **moved)


def _padding_metadata(record: EncodedRecord) -> dict:
    """Metadata for a padded question slot: identity only; `question_mask` keeps it out of the model."""
    base = record.metadata[0]
    return {
        "record_id": base["record_id"],
        "question_name": None,
        "question_type": None,
        "src": None,
        "group_id": base["group_id"],
        "variant": base["variant"],
        "option_keys": [],
    }


def collate(records: list[EncodedRecord], pad_token_id: int = 0) -> PackedDecisionBatch:
    """Pad `EncodedRecord`s into one batch; padded tokens/options/questions are masked, never attended."""
    if not records:
        raise ValueError("collate needs at least one record")
    batch, length = len(records), max(len(record.input_ids) for record in records)
    questions = max(len(record.decide_positions) for record in records)
    options = max(len(ends) for record in records for ends in record.option_end_positions)
    input_ids = torch.full((batch, length), pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((batch, length), dtype=torch.long)
    segment_ids = torch.full((batch, length), -1, dtype=torch.long)
    decide_positions = torch.zeros((batch, questions), dtype=torch.long)
    option_end_positions = torch.zeros((batch, questions, options), dtype=torch.long)
    question_mask = torch.zeros((batch, questions), dtype=torch.bool)
    option_mask = torch.zeros((batch, questions, options), dtype=torch.bool)
    target_probs = torch.zeros((batch, questions, options), dtype=torch.float32)
    metadata: list[list[dict]] = []
    for row, record in enumerate(records):
        count = len(record.input_ids)
        input_ids[row, :count] = torch.tensor(record.input_ids, dtype=torch.long)
        position_ids[row, :count] = torch.tensor(record.position_ids, dtype=torch.long)
        segment_ids[row, :count] = torch.tensor(record.segment_ids, dtype=torch.long)
        asked = len(record.decide_positions)
        decide_positions[row, :asked] = torch.tensor(record.decide_positions, dtype=torch.long)
        question_mask[row, :asked] = True
        for index in range(asked):
            ends = record.option_end_positions[index]
            option_end_positions[row, index, : len(ends)] = torch.tensor(ends, dtype=torch.long)
            option_mask[row, index, : len(ends)] = True
            target_probs[row, index, : len(ends)] = torch.tensor(record.target_probs[index], dtype=torch.float32)
        metadata.append([dict(record.metadata[index]) for index in range(asked)] + [_padding_metadata(record) for _ in range(questions - asked)])
    # attend(i, j) iff j <= i, j is real, and j is state (segment 0) or j shares i's question segment.
    # Padded rows keep their own diagonal so no query row is empty.
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
    same_branch = (segment_ids.unsqueeze(1) == 0) | (segment_ids.unsqueeze(1) == segment_ids.unsqueeze(2))
    allow = causal & same_branch & (segment_ids != -1).unsqueeze(1)
    allow |= torch.eye(length, dtype=torch.bool).unsqueeze(0)
    attention_bias = torch.zeros((batch, 1, length, length), dtype=torch.bfloat16).masked_fill(~allow.unsqueeze(1), float("-inf"))
    return PackedDecisionBatch(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_bias=attention_bias,
        decide_positions=decide_positions,
        option_end_positions=option_end_positions,
        question_mask=question_mask,
        option_mask=option_mask,
        target_probs=target_probs,
        metadata=metadata,
    )


# --------------------------------------------------------------------------------------- suites


def load_suite(path, split: str = "train", *, validate=None) -> list[dict]:
    """One frozen suite split, verified against the suite manifest's SHA-256, with stable identities.

    ``validate``, when given, runs on each raw record before identity defaults are filled.
    """
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found; load_suite takes a suite directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files") or {}
    name = f"{split}.jsonl"
    if name not in entries:
        raise ValueError(f"{path}: manifest has no {name} (has {sorted(entries)})")
    raw = (path / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = (entries[name] or {}).get("sha256")
    if digest != expected:
        raise ValueError(f"{path}/{name}: sha256 {digest} does not match manifest {expected}")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    declared = (entries[name] or {}).get("records")
    if declared is not None and len(records) != declared:
        raise ValueError(f"{path}/{name}: {len(records)} records, manifest declares {declared}")
    for index, record in enumerate(records):
        if validate is not None:
            validate(record)
        record["_meta"] = _identity(record, f"{path.name}/{split}/{index}")
    return records


def check_group_integrity(path, splits=("train", "calibration", "development")) -> dict:
    """Group-split report: the same case must never appear in two splits (data invariant).

    `overlaps` must be 0; `duplicate_groups` counts sibling records of one group inside a split.
    """
    groups: dict[str, dict[str, int]] = {}
    for split in splits:
        counts: dict[str, int] = {}
        for record in load_suite(path, split):
            group = record["_meta"]["group_id"]
            counts[group] = counts.get(group, 0) + 1
        groups[split] = counts
    names = list(groups)
    return {
        "splits": {
            split: {"records": sum(counts.values()), "groups": len(counts), "duplicate_groups": sum(1 for n in counts.values() if n > 1)}
            for split, counts in groups.items()
        },
        "overlaps": {
            f"{a}|{b}": len(set(groups[a]) & set(groups[b]))
            for index, a in enumerate(names)
            for b in names[index + 1 :]
        },
    }


def marker_collision_counts(records) -> dict:
    """How many records contain a marker spelling literally in their text.

    Literal reused delimiters are rejected by `user_tokens`; this scan reports
    their presence before encoding.
    """
    counts = {token: 0 for token in MARKER_TOKENS.values()}
    counts["<|...|>"] = 0
    for record in records:
        text = json.dumps({"state": record.get("state"), "questions": record.get("questions")}, sort_keys=True, ensure_ascii=False)
        found = {match.group(0) for match in _DELIMITER.finditer(text)}
        if found:
            counts["<|...|>"] += 1
        for token in MARKER_TOKENS.values():
            if token in found:
                counts[token] += 1
    return counts
