"""Read-only source-backed review materials and independent human Data Quality Gate.

No model or builder call occurs here. JSONL assessments are operator evidence, not
machine-generated quality judgements; synthetic assessments prove only software behavior.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet

from haidass_kev_train.data.canonical import load_canonical_suite
from haidass_kev_train.data.packing import check_group_integrity
from haidass_kev_train.data.finemath import identity as math_identity
from haidass_kev_train.data.ufw import source_id


SEVERE_CATEGORIES = frozenset({"wrong_gold", "correct_distractor", "ambiguity", "changed_question",
                               "leakage", "shortcut", "source_mismatch", "other"})
COVERAGE = ("ufw-en", "ufw-zh", "finemath", "symbolic", "unit", "finemath_llm_adjudicated",
            "finemath_programmatic", "ufw_model_screened")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(value).hexdigest()

def _parse(line):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON evidence field: {key}")
            result[key] = value
        return result
    return json.loads(line, object_pairs_hook=object_pairs)


def _identity(suite):
    data = (suite / "manifest.json").read_bytes()
    manifest = json.loads(data)
    if manifest.get("format") != "canonical_choice_v1":
        raise ValueError("review requires a canonical_choice_v1 frozen suite")
    build = manifest.get("build")
    policy = build.get("policy_sha256") if isinstance(build, dict) else None
    if not isinstance(policy, str) or len(policy) != 64 or set(policy) - set("0123456789abcdef"):
        raise ValueError("manifest missing valid build.policy_sha256")
    if "policy" not in build or _sha(_json(build["policy"]).encode()) != policy:
        raise ValueError("manifest build.policy_sha256 does not match build.policy")
    strategy = build["policy"]
    config = build.get("config")
    if (not isinstance(strategy, dict) or not isinstance(config, dict) or
            not isinstance(config.get("sources"), dict) or
            strategy.get("sources") != {source: str(Path(directory).resolve())
                                        for source, directory in config["sources"].items()} or
            strategy.get("limits") != {name: config.get(name) for name in (
                "max_packed", "max_answer_tokens", "max_source_tokens", "max_context_tokens", "max_output_tokens")} or
            any(strategy.get(name) != build.get(name) for name in (
                "model", "base_url", "prompt_version", "thinking", "source_location",
                "tokenizer_sha256", "generator_tokenizer_sha256"))):
        raise ValueError("manifest build policy differs from resolved generation strategy")
    return manifest, _sha(data), policy


def _source(record, config):
    ref = record["_meta"]["source_ref"]
    source = record["source"]
    locations = config.get("sources")
    if not isinstance(locations, dict) or source not in locations:
        raise ValueError(f"{source}: missing original source directory in build manifest")
    directory = Path(locations[source]).resolve(strict=True)
    if source == "finemath":
        prefix = f"{directory.name}/"
        fields, text_field = ["text", "url", "snapshot_type"], "text"
    elif source in ("ufw-en", "ufw-zh") and directory.name == "qa" and directory.parent.name == f"ultrafineweb_{source[-2:]}_l3":
        prefix = f"{directory.parent.name}/qa/"
        fields, text_field = ["uid", "content", "style"], "content"
    else:
        raise ValueError(f"{source}: unexpected source directory")
    locator = ref["path"]
    if not locator.startswith(prefix) or "/" in locator[len(prefix):] or locator[len(prefix):] in ("", ".", ".."):
        raise ValueError(f"{source}: invalid source_ref.path {locator!r}")
    shard = (directory / locator[len(prefix):]).resolve(strict=True)
    if shard.parent != directory or shard.suffix != ".parquet":
        raise ValueError(f"{source}: source_ref.path escapes original Parquet directory")
    reader = parquet.ParquetFile(shard)
    if not set(fields) <= set(reader.schema.names):
        raise ValueError(f"{shard}: original source schema mismatch")
    if ref["line"] >= reader.metadata.num_rows:
        raise ValueError(f"{source}: source_ref.line is outside original Parquet")
    position = ref["line"]
    for group in range(reader.metadata.num_row_groups):
        rows = reader.metadata.row_group(group).num_rows
        if position < rows:
            break
        position -= rows
    else:
        raise ValueError(f"{source}: source_ref.line not found in row groups")
    offset = 0
    row = None
    for batch in reader.iter_batches(row_groups=[group], batch_size=64, columns=fields):
        if offset + batch.num_rows > position:
            row = batch.slice(position - offset, 1).to_pylist()[0]
            break
        offset += batch.num_rows
    if row is None:
        raise ValueError(f"{source}: source_ref.line not found")
    raw = row[text_field]
    if not isinstance(raw, str) or _sha(raw.encode("utf-8")) != ref["sha256"]:
        raise ValueError(f"{source}: original source text SHA-256 differs from trace")
    if source == "finemath":
        url = row["url"].strip() or None if isinstance(row["url"], str) else None
        snapshot = row["snapshot_type"].strip() or None if isinstance(row["snapshot_type"], str) else None
        if (url, snapshot) != (ref.get("url"), ref.get("snapshot_type")) or record["_meta"]["id"] != math_identity(source, url, snapshot, raw):
            raise ValueError(f"{source}: source URL/snapshot/identity differs from trace")
    elif row["uid"] != ref["uid"] or row["style"] != "qa" or record["_meta"]["id"] != source_id(source, row["uid"], raw):
        raise ValueError(f"{source}: original uid/style/identity differs from trace")
    spans = {}
    for field in ("state_span", "givens_span", "question_span", "answer_span"):
        if field in ref:
            start, end = ref[field]
            if end > len(raw):
                raise ValueError(f"{source}: {field} exceeds original source text")
            spans[field.removesuffix("_span")] = {"span": [start, end], "text": raw[start:end]}
    if "option_spans" in ref:
        spans["old_options"] = []
        for start, end in ref["option_spans"]:
            if end > len(raw) or not ref["question_span"][0] <= start < end <= ref["question_span"][1]:
                raise ValueError(f"{source}: old option span exceeds original question")
            spans["old_options"].append({"span": [start, end], "text": raw[start:end]})
    return str(shard), raw, spans


def _materials(suite):
    manifest, batch, policy = _identity(suite)
    groups = check_group_integrity(suite, splits=("train", "development"))
    if any(groups["overlaps"].values()):
        raise ValueError("review suite has Source Group overlap across splits")
    records = [(split, record) for split in ("train", "development")
               for record in load_canonical_suite(suite, split)]
    if len(records) > 100:
        raise ValueError("first Generation Trial review exceeds 100 machine-accepted records")
    ids = [record["_meta"]["id"] for _, record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate canonical id across frozen suite")
    materials = []
    for split, record in records:
        ref = record["_meta"]["source_ref"]
        shard, raw, spans = _source(record, manifest["build"]["config"])
        case_id = _sha(_json([batch, record["_meta"]["id"]]).encode())
        material = {"case_id": case_id, "canonical_id": record["_meta"]["id"], "split": split,
                    "batch_manifest_sha256": batch, "policy_sha256": policy, "source": record["source"],
                    "validation_path": record["_meta"]["validation"], "source_ref": ref,
                    "source_parquet": shard, "original_source_text": raw, "original_spans": spans,
                    "state": record["state"], "question": record["question"], "gold": record["gold"],
                    "distractors": record["distractors"]}
        material["trace_sha256"] = _sha(_json(material).encode())
        materials.append(material)
    return sorted(materials, key=lambda item: item["case_id"]), batch, policy


def prepare_review(suite: str | Path, output: str | Path) -> Path:
    """Verify the frozen batch AND each original Parquet row, then write review.jsonl.

    ``output`` is a new directory; never overwrite evidence from an earlier batch.
    """
    suite, output = Path(suite), Path(output)
    if output.exists():
        raise FileExistsError(f"review directory already exists: {output}")
    materials, _, _ = _materials(suite)
    output.mkdir(parents=True)
    path = output / "review.jsonl"
    path.write_text("".join(_json(item) + "\n" for item in materials), encoding="utf-8")
    return path


def quality_gate(suite: str | Path, review: str | Path, assessments: str | Path | None, output: str | Path) -> dict:
    """Consume operator-written JSONL; invalid/spoofed evidence raises, never passes.

    Assessments require case_id, batch_manifest_sha256, policy_sha256, trace_sha256,
    reviewer, source_verified (boolean), serious_error (boolean), category, reason,
    and paths (list of human-confirmed symbolic/unit tags). For serious errors,
    category and reason are mandatory; failed source verification is serious.
    """
    suite, review, output = map(Path, (suite, review, output))
    assessments = Path(assessments) if assessments is not None else None
    if output.exists():
        raise FileExistsError(f"quality report already exists: {output}")
    materials, batch, policy = _materials(suite)
    expected = {item["case_id"]: item for item in materials}
    lines = review.read_text(encoding="utf-8").splitlines()
    if [_parse(line) for line in lines] != materials:
        raise ValueError("review.jsonl differs from verified batch and original source rows")
    reviewed = {}
    for line in (assessments.read_text(encoding="utf-8").splitlines() if assessments is not None else ()):
        if not line.strip():
            continue
        item = _parse(line)
        if not isinstance(item, dict) or set(item) != {"case_id", "batch_manifest_sha256", "policy_sha256",
                                                    "trace_sha256", "reviewer", "source_verified",
                                                    "serious_error", "category", "reason", "paths"}:
            raise ValueError("assessment needs exact case/batch/policy/trace, reviewer, verification and verdict fields")
        case = expected.get(item["case_id"]) if isinstance(item["case_id"], str) else None
        if case is None or item["case_id"] in reviewed:
            raise ValueError("unknown or duplicate assessment case_id")
        if any(item[name] != case[name] for name in ("batch_manifest_sha256", "policy_sha256", "trace_sha256")):
            raise ValueError(f"{item['case_id']}: assessment batch, policy or source trace mismatch")
        if (not isinstance(item["reviewer"], str) or not item["reviewer"].strip() or
                type(item["source_verified"]) is not bool or type(item["serious_error"]) is not bool or
                not isinstance(item["paths"], list) or
                any(type(path) is not str or path not in ("symbolic", "unit") for path in item["paths"]) or
                len(set(item["paths"])) != len(item["paths"])):
            raise ValueError(f"{item['case_id']}: invalid reviewer, source verification, verdict or human path tags")
        if item["paths"] and case["source"] != "finemath":
            raise ValueError(f"{item['case_id']}: symbolic/unit coverage requires FineMath source")
        if (item["serious_error"] or not item["source_verified"]):
            if (not isinstance(item["category"], str) or item["category"] not in SEVERE_CATEGORIES or
                    not isinstance(item["reason"], str) or not item["reason"].strip()):
                raise ValueError(f"{item['case_id']}: serious/source-verification error needs category and reason")
            if not item["source_verified"] and not item["serious_error"]:
                raise ValueError(f"{item['case_id']}: unverified source must be declared a serious error")
        elif item["category"] is not None or not isinstance(item["reason"], str):
            raise ValueError(f"{item['case_id']}: non-serious assessment needs null category and string reason")
        reviewed[item["case_id"]] = item
    severe = [{"case_id": case_id, "canonical_id": expected[case_id]["canonical_id"],
               "source": expected[case_id]["source"], "validation_path": expected[case_id]["validation_path"],
               "source_parquet": expected[case_id]["source_parquet"],
               "source_ref": expected[case_id]["source_ref"], "trace_sha256": expected[case_id]["trace_sha256"],
               "reviewer": item["reviewer"], "category": item["category"], "reason": item["reason"]}
              for case_id, item in reviewed.items() if item["serious_error"]]
    available = Counter()
    covered = Counter()
    for case in materials:
        tags = (case["source"], case["validation_path"])
        available.update(tags)
        if case["case_id"] in reviewed:
            covered.update((*tags, *reviewed[case["case_id"]]["paths"]))
    coverage = {name: {"available": available[name] if name not in ("symbolic", "unit") else None,
                       "reviewed": covered[name], "verified": covered[name] > 0}
                for name in COVERAGE}
    manifest, _, _ = _identity(suite)
    report = {"status": "fail" if severe else "pass" if (
                  manifest.get("complete") is True and len(materials) == len(reviewed) == 100 and
                  all(covered[source] for source in ("ufw-en", "ufw-zh", "finemath"))) else "incomplete",
              "policy_sha256": policy, "batch_manifest_sha256": batch,
              "accepted": len(materials), "audited": len(reviewed), "required_audited": 100,
              "severe_count": len(severe), "severe_examples": severe, "coverage": coverage,
              "unverified_paths": [name for name in COVERAGE if not covered[name]],
              "review_material": str(review), "human_assessments": str(assessments) if assessments is not None else None,
              "report_path": str(output)}
    with output.open("x", encoding="utf-8") as evidence:
        evidence.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Human-evidence consumer for an existing frozen trial; never generates or auto-certifies data")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="verify original Parquet source rows and write review.jsonl")
    prepare.add_argument("--suite", required=True)
    prepare.add_argument("--output", required=True, help="new review directory")
    gate = sub.add_parser("report", help="check operator JSONL assessments against exact batch and source trace")
    gate.add_argument("--suite", required=True)
    gate.add_argument("--review", required=True, help="review.jsonl created by prepare")
    gate.add_argument("--assessments", help="independently written human assessment JSONL; omit for no human evidence")
    gate.add_argument("--output", required=True, help="new quality gate JSON report")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(prepare_review(args.suite, args.output))
    else:
        print(json.dumps(quality_gate(args.suite, args.review, args.assessments, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
