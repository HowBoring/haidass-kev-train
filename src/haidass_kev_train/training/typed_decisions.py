"""Prepare deterministic case-level Stage 2 splits from official typed-decisions TRAIN."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import tomllib

from datasets import Dataset
from transformers import AutoTokenizer

from haidass_kev_train.data.packing import encode_record

WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)
EXPECTED_COLUMNS = {
    "id",
    "workflow",
    "split",
    "state",
    "questions",
    "gold",
    "factors",
    "label_agreement",
    "n_questions",
}
SPLIT_COUNTS = {"train": 240, "development": 30, "calibration": 30}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(value, field: str, case_id: str) -> dict:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{case_id}: {field} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{case_id}: {field} must be a JSON object")
    return parsed


def _convert(row: dict) -> dict:
    case_id, workflow = row["id"], row["workflow"]
    state = _json_object(row["state"], "state", case_id)
    questions = _json_object(row["questions"], "questions", case_id)
    gold = _json_object(row["gold"], "gold", case_id)
    if row["split"] != "train":
        raise ValueError(f"{case_id}: source row is not from official TRAIN")
    if row["n_questions"] != 5 or len(questions) != 5 or set(questions) != set(gold):
        raise ValueError(f"{case_id}: expected the same five questions in questions and gold")
    source = f"typed-decisions/{workflow}"
    converted = {}
    for name, question in questions.items():
        if not isinstance(question, dict) or question.get("type") not in {"choice", "noul", "score"}:
            raise ValueError(f"{case_id}/{name}: invalid question")
        answer = gold[name]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValueError(f"{case_id}/{name}: gold type disagrees with question")
        probabilities = answer.get("probabilities")
        criteria = question.get("criteria")
        if question["type"] == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError(f"{case_id}/{name}: choice criteria must be a non-empty object")
            keys = list(criteria)
        elif question["type"] == "noul":
            if criteria is not None and not isinstance(criteria, dict):
                raise ValueError(f"{case_id}/{name}: noul criteria must be an object when present")
            keys = ["false", "true"]
        else:
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError(f"{case_id}/{name}: score criteria must be ordered levels")
            keys = [str(index) for index in range(len(criteria))]
        if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
            raise ValueError(f"{case_id}/{name}: probabilities must cover every option exactly")
        target = {key: float(probabilities[key]) for key in keys}
        if any(not math.isfinite(value) or value < 0 for value in target.values()) or abs(sum(target.values()) - 1.0) > 1e-5:
            raise ValueError(f"{case_id}/{name}: probabilities are not a normalized finite distribution")
        converted[name] = {**question, "target": target, "src": source}
    return {
        "state": state,
        "questions": converted,
        "_meta": {
            "id": f"typed-decisions/{case_id}",
            "group_id": f"typed-decisions/{case_id}",
            "variant": "clean",
            "source": source,
            "workflow": workflow,
            "upstream_id": case_id,
        },
    }


def split_rows(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Validate the canonical 1,200 TRAIN cases and split each workflow 240/30/30."""
    if len(rows) != 1200:
        raise ValueError(f"canonical typed-decisions TRAIN must contain 1200 cases, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("canonical typed-decisions TRAIN requires 1200 unique non-empty case IDs")
    by_workflow: dict[str, list[dict]] = {workflow: [] for workflow in WORKFLOWS}
    observed = Counter(row.get("workflow") for row in rows)
    expected = Counter({workflow: 300 for workflow in WORKFLOWS})
    if observed != expected:
        raise ValueError(f"canonical workflow counts disagree: {dict(observed)}")
    for row in rows:
        by_workflow[row["workflow"]].append(row)

    result = {name: [] for name in SPLIT_COUNTS}
    for workflow in WORKFLOWS:
        ranked = sorted(
            by_workflow[workflow],
            key=lambda row: (hashlib.sha256(f"{seed}\0{row['id']}".encode()).digest(), row["id"]),
        )
        start = 0
        for split, count in SPLIT_COUNTS.items():
            result[split].extend(_convert(row) for row in ranked[start : start + count])
            start += count
    for records in result.values():
        records.sort(key=lambda record: record["_meta"]["id"])
    return result


def _length_summary(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    percentile = lambda fraction: ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def prepare(
    input_path: str | Path,
    output: str | Path,
    *,
    tokenizer_path: str | Path = "models/base/haidass1.5-143m",
    max_packed: int = 2048,
    seed: int = 42,
    resources_path: str | Path = "configs/resources.toml",
) -> dict:
    input_path, output = Path(input_path), Path(output)
    resource = tomllib.loads(Path(resources_path).read_text())["data"]["typed_decisions"]
    expected_input = Path(resource["local_dir"]) / "all" / "train-00000-of-00001.parquet"
    if input_path.resolve() != expected_input.resolve():
        raise ValueError(f"input must be the official aggregate TRAIN parquet: {expected_input}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite prepared data: {output}")
    dataset = Dataset.from_parquet(str(input_path))
    if set(dataset.column_names) != EXPECTED_COLUMNS:
        raise ValueError(f"typed-decisions schema mismatch: {dataset.column_names}")
    records = split_rows([dataset[index] for index in range(len(dataset))], seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    packed: dict[str, dict] = {}
    blockers: list[str] = []
    for split, split_records in records.items():
        lengths = []
        for record in split_records:
            try:
                lengths.append(len(encode_record(record, tokenizer, max_packed=max_packed).input_ids))
            except ValueError as error:
                blockers.append(str(error))
        if lengths:
            packed[split] = _length_summary(lengths)
    if blockers:
        raise ValueError(f"{len(blockers)} oversized/invalid packed cases; first blocker: {blockers[0]}")

    output.mkdir(parents=True, exist_ok=True)
    files = {}
    split_manifest = {}
    for split, split_records in records.items():
        path = output / f"{split}.jsonl"
        payload = "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in split_records).encode()
        path.write_bytes(payload)
        case_ids = [record["_meta"]["id"] for record in split_records]
        workflows = Counter(record["_meta"]["workflow"] for record in split_records)
        files[path.name] = {"sha256": hashlib.sha256(payload).hexdigest(), "records": len(split_records)}
        split_manifest[split] = {
            "cases": len(split_records),
            "case_ids_sha256": hashlib.sha256(("\n".join(case_ids) + "\n").encode()).hexdigest(),
            "workflows": dict(sorted(workflows.items())),
            "packed_length": packed[split],
        }
    manifest = {
        "format_version": 1,
        "files": files,
        "source": {
            "repo_id": resource["repo_id"],
            "revision": resource["revision"],
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "split": "train",
            "aggregate_only": True,
        },
        "split_strategy": {
            "unit": "case",
            "stratify": "workflow",
            "seed": seed,
            "per_workflow": SPLIT_COUNTS,
        },
        "splits": split_manifest,
        "max_packed": max_packed,
        "tokenizer_path": str(tokenizer_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/raw/typed-decisions/all/train-00000-of-00001.parquet",
        help="Official aggregate TRAIN parquet; workflow shards and TEST are rejected.",
    )
    parser.add_argument("--output", default="data/processed/typed-decisions-stage2")
    parser.add_argument("--tokenizer", default="models/base/haidass1.5-143m")
    parser.add_argument("--max-packed", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.input, args.output, tokenizer_path=args.tokenizer, max_packed=args.max_packed, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
