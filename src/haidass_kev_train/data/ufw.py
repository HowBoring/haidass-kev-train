"""Conservative deterministic extraction of independent UFW en/zh short-answer QA.

Ambiguous markers and original multiple-choice presentations are deferred to ticket #24.
Positions always refer to original Parquet ``content`` character offsets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as parquet

_MARKERS = re.compile(r"(?<![A-Za-z])(?:Question|Answer|问题|答案)\s*[:：]", re.IGNORECASE)
_OLD_OPTIONS = re.compile(r"(?<![A-Za-z])[A-DＡ-Ｄ][).．、:：]\s*\S", re.IGNORECASE)
_DEPENDENT = re.compile(r"(?i)\b(?:which of the following|all of the above|none of the above|option\s*[A-D]|choices?\s*[A-D])\b|(?:以下|下列)(?:选项|四项)|以上(?:皆|都|均)(?:是|不)|[A-DＡ-Ｄ]选项")


class Rejected(ValueError):
    """A source-row rejection, distinct from a system/build failure."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_id(source, uid, raw):
    encoded = json.dumps([source, uid, digest(raw)], ensure_ascii=False, separators=(",", ":"))
    return f"{source}/{digest(encoded)}"


def _strip_span(raw, start, end):
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return [start, end]


def parse(raw, language):
    """Recover a contiguous terminal Question/Answer chain; do not guess partial locations."""
    if not isinstance(raw, str) or not raw.strip():
        raise Rejected("structure")
    matches = list(_MARKERS.finditer(raw))
    if len(matches) < 2 or len(matches) % 2:
        raise Rejected("structure")
    question_label, answer_label = (("Question", "Answer") if language == "en" else ("问题", "答案"))
    for position, marker in enumerate(matches):
        expected = question_label if position % 2 == 0 else answer_label
        if marker.group().split(":")[0].split("：")[0].strip().casefold() != expected.casefold():
            raise Rejected("structure")
    # The first marker starts the QA tail; an embedded label in prose cannot become State.
    first = matches[0].start()
    if first == 0 or not raw[:first].strip() or (not raw[first - 1].isspace() and raw[first - 1] not in ".。!?！？:："):
        raise Rejected("structure")
    state_span = _strip_span(raw, 0, first)
    if state_span[0] == state_span[1]:
        raise Rejected("structure")
    cases = []
    for index in range(0, len(matches), 2):
        q, a = matches[index:index + 2]
        next_start = matches[index + 2].start() if index + 2 < len(matches) else len(raw)
        q_span = _strip_span(raw, q.end(), a.start())
        a_span = _strip_span(raw, a.end(), next_start)
        question, answer = raw[slice(*q_span)], raw[slice(*a_span)]
        if not question or not answer or "\n" in question or "\n" in answer:
            raise Rejected("structure")
        if _OLD_OPTIONS.search(question) or _DEPENDENT.search(question) or _DEPENDENT.search(answer):
            # Old-option mapping/presentation conversion requires ticket #24.
            continue
        cases.append({"question": question, "gold": answer, "question_span": q_span,
                      "answer_span": a_span, "state_span": state_span})
    if not cases:
        raise Rejected("original_options_or_presentation")
    return cases


def select(cases, seed, identity, tokenizer, max_answer_tokens):
    eligible = [case for case in cases if len(tokenizer(case["gold"], add_special_tokens=False).input_ids) <= max_answer_tokens]
    if not eligible:
        raise Rejected("answer_length")
    key = digest(json.dumps([seed, identity], separators=(",", ":")))
    return random.Random(int(key[:16], 16)).choice(eligible)


def iter_rows(sources, finished):
    """Stream local original qa Parquet row batches; line is a zero-based per-shard locator."""
    for source, directory in sorted(sources.items()):
        language = source.split("-")[-1]
        shards = sorted(Path(directory).glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No original QA Parquet shards under {directory}")
        for shard in shards:
            if finished(source):
                break
            for line, batch in enumerate_batches(shard):
                if finished(source):
                    break
                for offset, row in enumerate(batch):
                    if finished(source):
                        break
                    yield source, language, shard.name, line + offset, row


def enumerate_batches(shard):
    reader = parquet.ParquetFile(shard)
    if not {"uid", "content", "style"} <= set(reader.schema.names):
        raise ValueError(f"UFW original QA schema mismatch: {shard}")
    line = 0
    for batch in reader.iter_batches(batch_size=64, columns=["uid", "content", "style"]):
        rows = batch.to_pylist()
        yield line, rows
        line += len(rows)
