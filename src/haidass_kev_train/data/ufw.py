"""Recover original UFW document/QA spans and convert source MCQ presentation.

All spans are zero-based half-open character offsets in the original Parquet content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as parquet

_MARKERS = re.compile(r"(?<![A-Za-z])(?:Question|Answer|问题|答案)\s*[:：]", re.IGNORECASE)
_QUESTION_LABEL = re.compile(r"(?:Question|问题|题目|试题|提问)\s*[:：]\s*$", re.IGNORECASE)
_ANSWER_LABEL = re.compile(r"(?:Answer|答案|答)\s*[:：]\s*$", re.IGNORECASE)
_ANY_QA = re.compile(r"(?<![A-Za-z])(?:Question|Answer|问题|答案|题目|试题|提问|答)\s*[:：]", re.IGNORECASE)
_OPTION = re.compile(r"(?<![A-Za-z0-9])(?:[（(]([A-FＡ-Ｆ])[)）]|([A-FＡ-Ｆ])[).．、:：）])\s*(?=\S)", re.IGNORECASE)
_DEPENDENT = re.compile(
    r"(?i)\b(?:all of the above|none of the above|options?\s*[A-F](?:\s*(?:and|or|/|vs\.?)\s*[A-F])?|"
    r"choices?\s*[A-F]|both\s+[A-F]\s+and\s+[A-F])\b|"
    r"(?:[A-FＡ-Ｆ]\s*(?:选项|项)|选项\s*[A-FＡ-Ｆ]|以上(?:皆|都|均)(?:是|不)|"
    r"(?<![A-Za-z])[A-FＡ-Ｆ]\s*(?:/|vs\.?|versus|and|or|与|和|比|较)\s*[A-FＡ-Ｆ](?![A-Za-z]))")
_PRESENTATION = (("下面四项", "下列"), ("下列四项", "下列"),
                 ("以下四项", "以下"), ("下面六项", "下列"), ("下列六项", "下列"),
                 ("以下六项", "以下"), ("which of the following four options", "which of the following"),
                 ("which of the following six options", "which of the following"))
_ASSISTED_PRESENTATION = (("以下4个选项中，", "下列"), ("下面4个选项中，", "下列"),
                          ("以下四个选项中，", "下列"), ("下面四个选项中，", "下列"),
                          ("among these four choices", "among these choices"))
_REFERENCE = re.compile(r"(?i)^(?:option|choice)\s*([A-F])$|^([A-FＡ-Ｆ])(?:选项|项)$|^[（(]([A-FＡ-Ｆ])[)）]$")
_MULTI_QUESTION = re.compile(r"(?i)\b(?:choose|select)\s+(?:all|every|two|multiple)\b|多选|哪些(?:选项|项)(?:都|均)|选出所有")
_OPTION_COUNT = re.compile(
    r"(?i)\b(?:two|three|four|five|six|[2-6])\s+(?:options?|choices?)\b|"
    r"(?:两|二|三|四|五|六|[2-6])(?:个)?(?:选项|项)")


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


def _slice(raw, span):
    if (not isinstance(span, list) or len(span) != 2 or
            any(type(point) is not int for point in span) or not 0 <= span[0] < span[1] <= len(raw)):
        raise Rejected("assisted_location")
    return raw[span[0]:span[1]]


def _option_parts(raw, question_span):
    """Return the source question without the old list and ordered option text spans."""
    q_start, q_end = question_span
    text = raw[q_start:q_end]
    matches = list(_OPTION.finditer(text))
    if not matches:
        return question_span, []
    labels = [(match[1] or match[2]).translate(str.maketrans("ＡＢＣＤＥＦ", "ABCDEF")).upper() for match in matches]
    if len(matches) < 2 or labels != list("ABCDEF"[:len(matches)]):
        raise Rejected("answer_mapping_ambiguous")
    base = _strip_span(raw, q_start, q_start + matches[0].start())
    if base[0] == base[1]:
        raise Rejected("presentation_conversion")
    options = []
    for index, match in enumerate(matches):
        end = q_start + (matches[index + 1].start() if index + 1 < len(matches) else len(text))
        span = _strip_span(raw, q_start + match.end(), end)
        if span[0] == span[1] or not _slice(raw, span):
            raise Rejected("answer_mapping_ambiguous")
        options.append(span)
    # Text after the last label has no next label to delimit it. A second
    # sentence may be a condition, not part of the option; reject ambiguity.
    last = raw[slice(*options[-1])]
    sentence = re.finditer(r"\b([A-Za-z]+)[.!?;]\s+(?=[A-Za-z])", last)
    if ("\n" in last or re.search(r"[。！？；]\s*\S", last) or
            any(match[1].casefold() not in {"st", "dr", "mr", "mrs", "ms", "jr", "sr"}
                and len(match[1]) > 1 for match in sentence)):
        raise Rejected("presentation_conversion")
    if len({raw[slice(*span)].casefold() for span in options}) != len(options):
        raise Rejected("answer_mapping_ambiguous")
    return base, options


def _case(raw, state_span, question_span, answer_span, supplied_options=None):
    state, question, answer = (_slice(raw, span) for span in (state_span, question_span, answer_span))
    if not state.strip() or not question.strip() or not answer.strip():
        raise Rejected("structure")
    base, options = _option_parts(raw, question_span)
    if supplied_options is not None and supplied_options != options:
        raise Rejected("assisted_location")
    mcq = bool(options)
    gold = answer
    if mcq:
        question = raw[slice(*base)]
        if _MULTI_QUESTION.search(question):
            raise Rejected("answer_mapping_ambiguous")
        values = [raw[slice(*span)] for span in options]
        reference = _REFERENCE.fullmatch(answer.strip())
        label = (next((part for part in reference.groups() if part), None) if reference else None)
        ordinal = re.fullmatch(
            r"第([一二三四五六1-6])项|([1-6])|(?i:(first|second|third|fourth|fifth|sixth))\s+(?i:option|choice)",
            answer)
        if ordinal:
            value = next(part for part in ordinal.groups() if part)
            index = ("一二三四五六".index(value) if value in "一二三四五六" else
                     (int(value) - 1 if value.isdigit() else
                      ("first", "second", "third", "fourth", "fifth", "sixth").index(value.lower())))
            label = "ABCDEF"[index]
        elif label:
            label = label.translate(str.maketrans("ＡＢＣＤＥＦ", "ABCDEF")).upper()
        elif re.fullmatch(r"[A-FＡ-Ｆ]", answer, re.IGNORECASE):
            label = answer.translate(str.maketrans("ＡＢＣＤＥＦ", "ABCDEF")).upper()
        else:
            labelled = re.fullmatch(r"([A-FＡ-Ｆ])[).．、:：]\s*(.+)", answer, re.IGNORECASE)
            if labelled:
                label = labelled[1].translate(str.maketrans("ＡＢＣＤＥＦ", "ABCDEF")).upper()
                if label not in "ABCDEF"[:len(options)] or labelled[2] != values["ABCDEF".index(label)]:
                    raise Rejected("answer_mapping_ambiguous")
        if label and (label not in "ABCDEF"[:len(options)] or
                      any(value.casefold() == answer.casefold() for value in values)):
            raise Rejected("answer_mapping_ambiguous")
        if label:
            gold = values["ABCDEF".index(label)]
        elif answer not in values:
            raise Rejected("answer_mapping_ambiguous")
        if (re.search(r"(?i)\b(?:all|none|both) of (?:the )?(?:above|following)\b|以上(?:皆|都|均)|都(?:不是|是)", gold)
                or _DEPENDENT.search(gold)):
            raise Rejected("answer_mapping_ambiguous")
    elif _DEPENDENT.search(question) or _MULTI_QUESTION.search(question):
        raise Rejected("presentation_conversion")
    return {"question": question, "gold": gold, "question_span": question_span,
            "answer_span": answer_span, "state_span": state_span, "mcq": mcq,
            "option_spans": options}


def _deterministic(raw, language):
    matches = list(_MARKERS.finditer(raw))
    if len(matches) < 2 or len(matches) % 2:
        raise Rejected("structure")
    question_label, answer_label = (("Question", "Answer") if language == "en" else ("问题", "答案"))
    for index, marker in enumerate(matches):
        expected = question_label if index % 2 == 0 else answer_label
        if marker.group().split(":")[0].split("：")[0].strip().casefold() != expected.casefold():
            raise Rejected("structure")
    first = matches[0].start()
    if first == 0 or not raw[:first].strip():
        raise Rejected("structure")
    for marker in matches[::2]:
        before = marker.start() - 1
        while before >= 0 and raw[before].isspace():
            before -= 1
        if before < 0 or (raw[before] not in ".。!?！？" and
                          "\n" not in raw[before + 1:marker.start()]):
            raise Rejected("structure")
    state_span = _strip_span(raw, 0, first)
    if _ANY_QA.search(raw[slice(*state_span)]):
        raise Rejected("structure")
    cases = []
    rejected = None
    for index in range(0, len(matches), 2):
        q, a = matches[index:index + 2]
        end = matches[index + 2].start() if index + 2 < len(matches) else len(raw)
        question_span = _strip_span(raw, q.end(), a.start())
        answer_span = _strip_span(raw, a.end(), end)
        if "\n" in raw[slice(*answer_span)]:
            raise Rejected("structure")
        try:
            cases.append(_case(raw, state_span, question_span, answer_span))
        except Rejected as error:
            if error.reason in ("structure", "assisted_location"):
                raise
            rejected = error
    if not cases:
        raise rejected or Rejected("structure")
    return cases


def _location_schema(value):
    return (set(value) == {"sha256", "state_span", "state_text", "qas"} and
            isinstance(value["qas"], list) and bool(value["qas"]) and
            all(isinstance(qa, dict) and
                set(qa) == {"question_span", "question_text", "answer_span", "answer_text", "option_spans", "option_texts"}
                for qa in value["qas"]))


def _assisted(raw, ask):
    result = ask("ufw_locate", {"content": raw, "sha256": digest(raw),
        "requirement": "Return sha256, state_span, state_text, qas array of question_span/question_text/"
                       "answer_span/answer_text/option_spans/option_texts. Spans are original zero-based "
                       "half-open character positions; state is the document prefix excluding every QA, "
                       "questions include old option lists, option spans contain option TEXT without labels. "
                       "Do not reconstruct text or invent an answer. Return each QA in source order."}, _location_schema)
    if result is None or result["sha256"] != digest(raw):
        raise Rejected("assisted_location")
    state_span = result["state_span"]
    if (_slice(raw, state_span) != result["state_text"] or
            state_span[0] != _strip_span(raw, 0, len(raw))[0] or _ANY_QA.search(result["state_text"])):
        raise Rejected("assisted_location")
    cases = []
    rejected = None
    previous = state_span[1]
    for qa in result["qas"]:
        question_span, answer_span = qa["question_span"], qa["answer_span"]
        question, answer = _slice(raw, question_span), _slice(raw, answer_span)
        if (question != qa["question_text"] or answer != qa["answer_text"] or
                previous > question_span[0] or question_span[1] > answer_span[0] or
                _ANY_QA.search(question) or _ANY_QA.search(answer) or
                not _QUESTION_LABEL.fullmatch(raw[previous:question_span[0]].strip()) or
                not _ANSWER_LABEL.fullmatch(raw[question_span[1]:answer_span[0]].strip())):
            raise Rejected("assisted_location")
        option_spans, option_texts = qa["option_spans"], qa["option_texts"]
        if (not isinstance(option_spans, list) or not isinstance(option_texts, list) or
                len(option_spans) != len(option_texts) or
                any(_slice(raw, span) != text or not question_span[0] <= span[0] < span[1] <= question_span[1]
                    for span, text in zip(option_spans, option_texts))):
            raise Rejected("assisted_location")
        try:
            cases.append(_case(raw, state_span, question_span, answer_span, option_spans))
        except Rejected as error:
            if error.reason in ("structure", "assisted_location"):
                raise Rejected("assisted_location") from error
            rejected = error
        previous = answer_span[1]
    if raw[previous:].strip() or _MARKERS.search(raw[:state_span[1]]):
        raise Rejected("assisted_location")
    if not cases:
        raise rejected or Rejected("assisted_location")
    return cases


def parse(raw, language, ask=None):
    """Deterministic extraction first; ambiguous structure requires verified source locations."""
    if not isinstance(raw, str) or not raw.strip():
        raise Rejected("structure")
    try:
        return _deterministic(raw, language)
    except Rejected as error:
        if error.reason != "structure" or ask is None:
            raise
    return _assisted(raw, ask)


def select(cases, seed, identity, tokenizer, max_answer_tokens):
    eligible = [case for case in cases if len(tokenizer(case["gold"], add_special_tokens=False).input_ids) <= max_answer_tokens]
    if not eligible:
        raise Rejected("answer_length")
    key = digest(json.dumps([seed, identity], separators=(",", ":")))
    return random.Random(int(key[:16], 16)).choice(eligible)


def prepare(chosen, ask):
    """Convert only old option-count wording; never rewrite factual or logical content."""
    if not chosen["mcq"]:
        return chosen
    question = chosen["question"]
    for old, new in _PRESENTATION:
        question = re.sub(re.escape(old), new, question, flags=re.IGNORECASE)
    if _DEPENDENT.search(question):
        raise Rejected("presentation_conversion")
    if _OPTION_COUNT.search(question):
        allowed = {re.sub(re.escape(old), new, question, flags=re.IGNORECASE)
                   for old, new in _ASSISTED_PRESENTATION if re.search(re.escape(old), question, re.IGNORECASE)}
        if not allowed:
            raise Rejected("presentation_conversion")
        result = ask("ufw_cleanup", {"question": question,
            "requirement": "Return {\"question\": string} replacing only old option-count wording; "
                           "preserve every fact, negation, comparison and condition."},
            lambda value: set(value) == {"question"} and isinstance(value["question"], str))
        if result is None or result["question"] not in allowed or result["question"] == question:
            raise Rejected("presentation_conversion")
        question = result["question"]
    if not question.strip() or _DEPENDENT.search(question) or _OPTION_COUNT.search(question):
        raise Rejected("presentation_conversion")
    return {**chosen, "question": question}


def lane_shards(shards, shard_index, shard_count):
    """Partition files first, then partition rows if there are more lanes than files."""
    file_lanes = min(len(shards), shard_count)
    slot = shard_index % file_lanes
    stripe = shard_index // file_lanes
    stripes = (shard_count - 1 - slot) // file_lanes + 1
    return shards[slot::file_lanes], stripe, stripes


def iter_rows(sources, finished, *, shard_index=0, shard_count=1):
    """Stream original QA shards assigned to one disjoint lane; line is per-shard."""
    for source, directory in sorted(sources.items()):
        language = source.split("-")[-1]
        shards = sorted(Path(directory).glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No original QA Parquet shards under {directory}")
        assigned, stripe, stripes = lane_shards(shards, shard_index, shard_count)
        for shard in assigned:
            if finished(source):
                break
            for line, batch in enumerate_batches(shard):
                if finished(source):
                    break
                for offset, row in enumerate(batch):
                    if finished(source):
                        break
                    if (line + offset) % stripes == stripe:
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
