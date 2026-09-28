"""Recover source-grounded single FineMath problems from original webpage Parquet."""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import pyarrow.parquet as parquet

from haidass_kev_train.data.ufw import Rejected, digest

_QUESTION = re.compile(r"(?im)^[^\S\r\n]*(?:problem|question)[^\S\r\n]*[:：][^\S\r\n]*")
_ANSWER = re.compile(r"(?im)^[^\S\r\n]*(?:final[^\S\r\n]+answer|answer)[^\S\r\n]*[:：][^\S\r\n]*")
_SOLUTION = re.compile(r"(?im)^[^\S\r\n]*(?:solution|proof|explanation|worked[^\S\r\n]+solution)[^\S\r\n]*[:：]")
_MISSING = re.compile(r"(?i)\b(?:figure|diagram|pictured|image|shown below|see (?:above|below|previous)|preceding (?:page|section)|as illustrated)\b|<img\b|!\[[^]]*\]\(")
_PROHIBITED = re.compile(r"(?i)\b(?:celsius|fahrenheit|kelvin|exchange rate|currency exchange|usd|eur|months?)\b|°[CFK]")
_ABSOLUTE_TEMPERATURE = re.compile(r"\b\d+(?:\.\d+)?[^\S\r\n]*[KCF]\b")
_QUESTION_END = re.compile(r"(?im)\b(?:calculate|find|evaluate|determine|solve)\b[^\r\n]*?[.!。](?=[^\S\r\n]|$)|[?？]")
_GIVEN_LINE = re.compile(r"(?i)^[^\S\r\n]*(?:given|where|let)\b")
_WORKED_TERMS = re.compile(r"(?i)\b(?:adding|subtracting|multiplying|dividing|computing|calculating|simplifying|gives|yields|therefore|thus|hence|solution|answer|we\s+(?:get|find|obtain))\b")
_WORKED_CONCLUSION = re.compile(
    r"(?i)\b(?:adding|subtracting|multiplying|dividing|computing|calculating|simplifying|solving)\b"
    r"[^.!?。！？\n]{0,120}\b(?:gives|yields|equals|results?\s+in|gets?)\b"
    r"|\b(?:we\s+(?:find|get|obtain)|the\s+(?:final\s+)?answer\s+is|this\s+(?:gives|yields))\b"
)
_EQUATION_STEP = re.compile(r"[+\-*/^().%√²³\w \t]+")
_SEPARATE_GIVENS = re.compile(r"(?i)\b(?:and|or|given|where|let)\b")


def prohibited(*texts):
    return any(_PROHIBITED.search(text) or _ABSOLUTE_TEMPERATURE.search(text) for text in texts)


def identity(source, url, snapshot, raw):
    return f"{source}/{digest(json.dumps([source, url, snapshot, digest(raw)], ensure_ascii=False, separators=(',', ':')))}"


def group_id(source, url, raw):
    if isinstance(url, str) and url.strip():
        try:
            parsed = urlsplit(url.strip())
            if parsed.scheme.lower() in ("http", "https") and parsed.hostname:
                # Query strings distinguish resources; fragments do not reach the server.
                normalized = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))
                return f"{source}/{digest(normalized)}"
        except ValueError:
            pass
    return f"{source}/{digest(raw)}"


def iter_rows(directory, finished):
    shards = sorted(Path(directory).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No original FineMath Parquet shards under {directory}")
    for shard in shards:
        if finished():
            break
        reader = parquet.ParquetFile(shard)
        columns = ("text", "url", "snapshot_type")
        if not set(columns) <= set(reader.schema.names):
            raise ValueError(f"FineMath original Parquet schema mismatch: {shard}")
        line = 0
        for batch in reader.iter_batches(batch_size=64, columns=list(columns)):
            for row in batch.to_pylist():
                if finished():
                    return
                yield shard.name, line, row
                line += 1


def _trim(raw, start, end):
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return [start, end]


def _checked(raw, question_span, answer_span, givens_span=None):
    spans = [question_span, answer_span] + ([givens_span] if givens_span is not None else [])
    if any(not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span)
           or not 0 <= span[0] < span[1] <= len(raw) for span in spans):
        raise Rejected("invalid_source_location")
    question = raw[slice(*question_span)].strip()
    gold = raw[slice(*answer_span)].strip()
    givens = raw[slice(*givens_span)].strip() if givens_span is not None else ""
    if (question != raw[slice(*question_span)] or gold != raw[slice(*answer_span)]
            or (givens_span is not None and givens != raw[slice(*givens_span)])):
        raise Rejected("invalid_source_location")
    if not question or not gold or (givens_span is not None and givens_span[1] > question_span[0]) or answer_span[0] <= question_span[1]:
        raise Rejected("invalid_source_location")
    # Verify labels and nearest field attribution, not just that their values occur somewhere.
    question_labels = list(_QUESTION.finditer(raw[:question_span[0]]))
    answer_labels = list(_ANSWER.finditer(raw[question_span[1]:answer_span[0]]))
    if (not question_labels or not answer_labels
            or raw[question_labels[-1].end():question_span[0]].strip()
            or raw[question_span[1] + answer_labels[-1].end():answer_span[0]].strip()
            or _QUESTION.search(raw[question_span[1]:answer_span[0]])):
        raise Rejected("invalid_source_location")
    prefix = raw[:question_labels[-1].start()].strip()
    if prefix and (givens_span is None or raw[:givens_span[0]].strip()
                   or raw[givens_span[1]:question_labels[-1].start()].strip()):
        raise Rejected("incomplete_problem")
    if raw[answer_span[1]:].split("\n", 1)[0].strip():
        raise Rejected("invalid_source_location")
    if not ("?" in question or "？" in question or re.search(r"(?i)\b(?:find|calculate|determine|evaluate|solve|how many|what is)\b", question)):
        raise Rejected("incomplete_problem")
    if (_MISSING.search(question + " " + givens) or
            not givens and re.fullmatch(r"(?i)(?:what is|find|calculate|determine)\s+(?:the value of\s+)?[a-z]\s*\?", question)):
        raise Rejected("missing_figure_or_conditions")
    if prohibited(question, givens, gold):
        raise Rejected("prohibited_conversion")
    visible = question + "\n" + givens
    if (re.search(r"(?i)\b(?:solution|proof|explanation|final answer|answer)\s*[:：]", visible)
            or re.search(r"(?i)\b(?:therefore|thus|hence|we obtain|answer is|equals the answer)\b", visible)
            or _WORKED_CONCLUSION.search(visible)):
        raise Rejected("solution_leakage")
    if visible.count("=") >= 2 and any(
            0 < len(step) <= 96 and _EQUATION_STEP.fullmatch(step)
            and not _SEPARATE_GIVENS.search(step)
            for step in visible.split("=")[1:-1]):
        raise Rejected("solution_leakage")
    boundary = _QUESTION_END.search(question)
    # Without a completed question sentence, any extra line is ambiguous working
    # unless it explicitly introduces more givens from the original problem.
    trailing = (question[boundary.end():] if boundary else
                question.split("\n", 1)[1] if "\n" in question else "")
    if any(line.strip() and (not _GIVEN_LINE.match(line) or _WORKED_TERMS.search(line))
           for line in trailing.splitlines()):
        raise Rejected("solution_leakage")
    if "\n" in gold or len(gold) > 160 or re.search(r"(?i)\b(?:therefore|thus|hence|because|solution)\b", gold):
        raise Rejected("invalid_source_answer")
    return {"state": givens, "question": question, "gold": gold,
            "question_span": question_span, "answer_span": answer_span,
            **({"givens_span": givens_span} if givens_span is not None else {})}


def extract(raw, generator):
    """Source offsets only: the model may locate existing text, never supply an answer."""
    if not isinstance(raw, str) or not raw.strip():
        raise Rejected("source_schema")
    if _MISSING.search(raw):
        raise Rejected("missing_figure_or_conditions")
    if prohibited(raw):
        raise Rejected("prohibited_conversion")
    questions, answers = list(_QUESTION.finditer(raw)), list(_ANSWER.finditer(raw))
    if not answers:
        raise Rejected("missing_source_answer")
    if len(questions) == len(answers) == 1 and questions[0].end() < answers[0].start():
        question, answer = questions[0], answers[0]
        solution = _SOLUTION.search(raw, question.end(), answer.start())
        end = solution.start() if solution else answer.start()
        qspan = _trim(raw, question.end(), end)
        aspan = _trim(raw, answer.end(), len(raw))
        if "\n" not in raw[slice(*aspan)]:
            prefix = raw[:question.start()]
            givens = _trim(raw, 0, question.start()) if re.fullmatch(
                r"(?is)given[^\S\r\n]*[:：].+", prefix.strip()) else None
            if not prefix.strip() or givens is not None:
                return _checked(raw, qspan, aspan, givens)
    located = generator.ask("finemath_extract", {"text": raw,
        "requirement": "Locate ONE complete existing problem and its existing final answer. Return JSON with "
                       "question_span, answer_span as original-text [start,end) offsets and optional givens_span, "
                       "and complete:boolean. Do not solve; never include worked solution in problem."},
        lambda data: set(data) in ({"question_span", "answer_span", "complete"},
                                    {"question_span", "answer_span", "givens_span", "complete"})
        and type(data["complete"]) is bool, thinking=True)
    if located is None:
        raise Rejected("malformed_response")
    if not located["complete"]:
        raise Rejected("incomplete_problem")
    return _checked(raw, located["question_span"], located["answer_span"], located.get("givens_span"))
