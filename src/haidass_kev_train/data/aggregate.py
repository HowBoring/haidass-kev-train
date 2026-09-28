"""Offline merge of separately authorized, completed builder batches under one audited policy.

Does not generate records, perform human review, start training, or authorize resources.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

from haidass_kev_train.data.build import _split
from haidass_kev_train.data.canonical import load_canonical_suite
from haidass_kev_train.data.finemath import group_id as math_group
from haidass_kev_train.data.ufw import digest
from haidass_kev_train.data.quality import _identity, quality_gate

SOURCES = frozenset(("ufw-en", "ufw-zh", "finemath"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_identity(row):
    meta = row["_meta"]
    ref = meta["source_ref"]
    source = row["source"]
    if source == "finemath":
        raw_key = _json([source, ref.get("url"), ref.get("snapshot_type"), ref["sha256"]])
        group = math_group(source, ref.get("url"), "")
        if group == f"{source}/{digest('')}":
            group = f"{source}/{ref['sha256']}"
    else:
        raw_key = _json([source, ref["uid"], ref["sha256"]])
        group = f"{source}/{digest(row['state'])}"
    return f"{source}/{digest(raw_key)}", group


def _without_locator(row):
    meta = row["_meta"]
    ref = {key: value for key, value in meta["source_ref"].items() if key not in ("path", "line")}
    return {**row, "_meta": {**meta, "source_ref": ref}}


def aggregate(batches: list[str | Path], quality_report: str | Path, audited_suite: str | Path,
              review: str | Path, assessments: str | Path, minimum_train: int,
              output: str | Path) -> dict:
    """Verify the source-backed 100-case audit and merge complete frozen batches.

    `minimum_train` is a predeclared minimum, not a quota that discards extra cases.
    Every source batch keeps its own <=100 admission limit; aggregation does not
    authorize another batch, and insufficient coverage fails without publishing.
    """
    quality_report, audited_suite, review, assessments, output = map(
        Path, (quality_report, audited_suite, review, assessments, output))
    if output.exists():
        raise FileExistsError(f"aggregate suite already exists: {output}")
    if type(minimum_train) is not int or minimum_train < 1:
        raise ValueError("minimum_train must be a positive integer")
    if not batches:
        raise ValueError("at least one completed source batch is required")
    claimed = json.loads(quality_report.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        observed = quality_gate(audited_suite, review, assessments, Path(directory) / "quality.json")
    if ({k: value for k, value in claimed.items() if k != "report_path"} !=
            {k: value for k, value in observed.items() if k != "report_path"} or
            observed["status"] != "pass" or
            not (observed["audited"] == observed["required_audited"] == 100) or
            observed["severe_count"] != 0):
        raise ValueError("source-backed audit of all 100 cases with zero severe errors is required")

    collected = {"train": {}, "development": {}}
    group_split = {}
    source_rows = {}
    scanned_ranges = {source: [] for source in SOURCES}
    provenance = []
    total_input = 0
    seen_batches = set()
    split_seed = None
    build = None
    for path in map(Path, batches):
        path = path.resolve()
        manifest, manifest_sha, policy = _identity(path)
        if manifest.get("aggregation") is not None or manifest.get("derivation") is not None:
            raise ValueError(f"{path}: expected an original builder batch, not an aggregate/subset")
        if manifest_sha in seen_batches:
            raise ValueError(f"{path}: duplicate input batch manifest")
        seen_batches.add(manifest_sha)
        if policy != observed["policy_sha256"]:
            raise ValueError(f"{path}: generation strategy differs from the audited policy")
        if manifest.get("complete") is not True:
            raise ValueError(f"{path}: unfinished builder batch cannot be aggregated")
        config = manifest["build"]["config"]
        seed = config.get("split_seed")
        if type(seed) is not int or split_seed is not None and seed != split_seed:
            raise ValueError(f"{path}: split_seed differs across builder batches")
        split_seed = seed
        if build is None:
            build = manifest["build"]  # Representative source batch; quotas and offsets are NOT aggregate totals.
        current = {split: load_canonical_suite(path, split) for split in collected}
        counts = manifest.get("counts")
        if not isinstance(counts, dict) or any(
                counts.get(split) != {"records": len(rows),
                                      "groups": len({row["_meta"]["group_id"] for row in rows}),
                                      "sources": dict(sorted(Counter(row["source"] for row in rows).items()))}
                for split, rows in current.items()):
            raise ValueError(f"{path}: builder split counts disagree with verified data")
        accepted = sum(len(rows) for rows in current.values())
        if accepted > 100:
            raise ValueError(f"{path}: a builder batch cannot exceed 100 accepted cases")
        summary_path = path / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("accepted") != accepted or summary.get("splits") != counts or
                summary.get("complete") is not True or summary.get("trial_status") != "complete" or
                summary.get("stop_reason") != manifest.get("stop_reason")):
            raise ValueError(f"{path}: completed source batch summary disagrees with verified splits")
        total_input += accepted
        offsets = config.get("source_start_rows")
        scanned = summary.get("scanned_by_source")
        skipped = summary.get("skipped_by_offset")
        if (not isinstance(offsets, dict) or set(offsets) != set(config["sources"]) or
                not isinstance(scanned, dict) or not isinstance(skipped, dict)):
            raise ValueError(f"{path}: missing bounded source offset/scanning evidence")
        for source, start in offsets.items():
            count, actual_skip = scanned.get(source, 0), skipped.get(source, 0)
            if (type(start) is not int or start < 0 or type(count) is not int or count < 0 or
                    type(actual_skip) is not int or not 0 <= actual_skip <= start or
                    count and actual_skip != start):
                raise ValueError(f"{path}: invalid source scan interval for {source}")
            end = start + count
            if count and any(max(start, before) < min(end, after)
                             for before, after in scanned_ranges[source]):
                raise ValueError(f"{path}: overlapping authorized source scan interval for {source}")
            if count:
                scanned_ranges[source].append((start, end))
        for split, rows in current.items():
            for row in rows:
                meta = row["_meta"]
                rid, group = meta["id"], meta["group_id"]
                if (rid, group) != _stable_identity(row):
                    raise ValueError(f"{path}: {rid} changed stable source row/group identity")
                if _split(seed, group) != split:
                    raise ValueError(f"{path}: {rid} violates fixed Source Group split assignment")
                prior_split = group_split.setdefault(group, split)
                if prior_split != split:
                    raise ValueError(f"{group}: Source Group spans train and development")
                ref = meta["source_ref"]
                source_row = (row["source"], ref["path"], ref["line"], ref["sha256"])
                prior_id = source_rows.setdefault(source_row, rid)
                if prior_id != rid:
                    raise ValueError(f"{path}: original source row changed canonical identity")
                existing = collected[split].get(rid)
                if rid in collected["development" if split == "train" else "train"]:
                    raise ValueError(f"{rid}: canonical ID spans train and development")
                if existing is not None:
                    if _without_locator(existing) != _without_locator(row):
                        raise ValueError(f"{rid}: conflicting canonical content across batches")
                else:
                    collected[split][rid] = row  # Keep the first verified source row locator.
        provenance.append({"suite": str(path), "manifest_sha256": manifest_sha,
                           "summary_sha256": _sha(summary_path), "accepted": accepted,
                           "split_seed": seed, "source_start_rows": offsets,
                           "scanned_by_source": scanned, "skipped_by_offset": skipped,
                           "counts": counts})
    if len(collected["train"]) < minimum_train:
        raise ValueError(f"aggregate has {len(collected['train'])} distinct train canonicals, below requested {minimum_train}")
    if {row["source"] for row in collected["train"].values()} != SOURCES or not collected["development"]:
        raise ValueError("aggregate requires all three train sources and nonempty independent development")
    if {row["source"] for row in collected["development"].values()} != SOURCES:
        raise ValueError("aggregate development must cover all three sources")

    counts = {split: {"records": len(rows),
                      "groups": len({row["_meta"]["group_id"] for row in rows.values()}),
                      "sources": dict(sorted(Counter(row["source"] for row in rows.values()).items()))}
              for split, rows in collected.items()}
    distinct_accepted = len(collected["train"]) + len(collected["development"])
    aggregation = {"batch_manifests": provenance,
                   "audited_batch_manifest_sha256": observed["batch_manifest_sha256"],
                   "quality_report": str(quality_report.resolve()),
                   "quality_report_sha256": _sha(quality_report),
                   "review_sha256": _sha(review), "assessments_sha256": _sha(assessments),
                   "policy_sha256": observed["policy_sha256"], "split_seed": split_seed,
                   "minimum_train": minimum_train, "input_accepted": total_input,
                   "distinct_accepted": distinct_accepted,
                   "duplicate_canonicals_collapsed": total_input - distinct_accepted,
                   "counts": counts}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as directory:
        staging = Path(directory)
        files = {}
        for split, rows in collected.items():
            name = f"{split}.jsonl"
            payload = "".join(_json(row) + "\n" for _, row in sorted(rows.items())).encode()
            (staging / name).write_bytes(payload)
            files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "records": len(rows)}
        manifest = {"format": "canonical_choice_v1", "complete": True,
                    "files": files, "counts": counts, "build": build, "aggregation": aggregation}
        (staging / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        os.rename(staging, output)
    return aggregation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="append", required=True,
                        help="separately authorized completed original builder suite; repeat per batch")
    for key in ("quality", "audited-suite", "review", "assessments", "minimum-train", "output"):
        parser.add_argument(f"--{key}", required=True, type=int if key == "minimum-train" else str)
    args = parser.parse_args(argv)
    print(json.dumps(aggregate(args.batch, args.quality, args.audited_suite, args.review,
                               args.assessments, args.minimum_train, args.output), indent=2))


if __name__ == "__main__":
    main()
