"""Prepare an audited, immutable 128-case suite for the existing Decision SFT trainer.

Preparation is offline; it never starts generation, an audit, or a training run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib

from transformers import AutoTokenizer

from haidass_kev_train.data.canonical import load_canonical_suite, preflight
from haidass_kev_train.data.packing import check_group_integrity
from haidass_kev_train.data.quality import _identity, quality_gate
from haidass_kev_train.training.sft import _config

SOURCES = frozenset(("ufw-en", "ufw-zh", "finemath"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _toml(config: dict) -> str:
    """Serialize the flat SFT configuration and its existing augmentation/canonical tables."""
    lines = []
    for key, value in config.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_json(value)}")
    for key, table in config.items():
        if isinstance(table, dict):
            lines.extend(("", f"[{key}]"))
            lines.extend(f"{name} = {_json(value)}" for name, value in table.items())
    return "\n".join(lines) + "\n"


def prepare_overfit(suite: str | Path, quality_report: str | Path, audited_suite: str | Path,
                    review: str | Path, assessments: str | Path, base_config: str | Path,
                    output: str | Path) -> dict:
    """Verify genuine audit evidence and builder provenance, then freeze a runnable recipe.

    `output` must not exist. Different audited and later batch hashes are expected; the
    verified build *policy* must match. Select 128 different train canonicals;
    `probe_groups=128` selects all of their groups (at most 128), including siblings.
    The source suite's development records and group assignments are never moved.
    """
    suite, quality_report, audited_suite, review, assessments, base_config, output = map(
        Path, (suite, quality_report, audited_suite, review, assessments, base_config, output))
    if output.exists():
        raise FileExistsError(f"overfit recipe already exists: {output}")
    claimed = json.loads(quality_report.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        observed = quality_gate(audited_suite, review, assessments, Path(temporary) / "quality.json")
    if {key: value for key, value in claimed.items() if key != "report_path"} != {
            key: value for key, value in observed.items() if key != "report_path"}:
        raise ValueError("quality report differs from verified human assessments and audited suite")
    if (observed["status"] != "pass" or observed["audited"] != 100 or
            observed["severe_count"] != 0):
        raise ValueError("100 reviewed cases with zero serious errors are required before overfit")

    source_manifest, _, policy = _identity(suite)
    if (source_manifest.get("complete") is not True or
            source_manifest.get("derivation") is not None or
            policy != observed["policy_sha256"]):
        raise ValueError("original complete builder suite and audited generation strategy must match")
    build = source_manifest["build"]
    integrity = check_group_integrity(suite, splits=("train", "development"))
    if any(integrity["overlaps"].values()):
        raise ValueError("later suite train/development Source Groups overlap")
    train = load_canonical_suite(suite, "train")
    development = load_canonical_suite(suite, "development")
    ids = [record["_meta"]["id"] for record in train + development]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical identities overlap across train/development")
    if not development:
        raise ValueError("later suite has empty development split")
    if {record["source"] for record in train} != SOURCES:
        raise ValueError("later train split must cover ufw-en, ufw-zh and finemath only")
    config = tomllib.loads(base_config.read_text(encoding="utf-8"))
    budget = config.get("max_steps")
    if type(budget) is not int or not 1 <= budget <= 500:
        raise ValueError("overfit max_steps must be 1..500 optimizer updates; never extend a run")
    # Freeze the caller's update budget/seed/optimizer and pin all probe groups.
    if len(train) < 128:
        raise ValueError(f"overfit requires 128 distinct train canonicals; found {len(train)}")
    seed = config.get("seed")
    if type(seed) is not int:
        raise ValueError("overfit requires a fixed integer seed")
    ranked = sorted(train, key=lambda row: (
        hashlib.sha256(f"{seed}\0{row['_meta']['id']}".encode()).digest(), row["_meta"]["id"]))
    chosen = ranked[:128]
    for source in sorted(SOURCES - {row["source"] for row in chosen}):
        replacement = next(row for row in ranked[128:] if row["source"] == source)
        replace_index = next(index for index in range(len(chosen) - 1, -1, -1)
                             if sum(row["source"] == chosen[index]["source"] for row in chosen) > 1)
        chosen[replace_index] = replacement
    chosen.sort(key=lambda row: row["_meta"]["id"])
    config.update(suite_path=str((output / "suite").resolve()), data_format="canonical_choice_v1",
                  probe_groups=128, train_sources=sorted(SOURCES), development_selection="clean")
    config["augmentation"] = {"shuffle": False, "p_none": 0.0, "p_none_distract": 0.0,
                              "p_distract": 0.0, "p_none_pair": 0.0}
    # Preserve caller-provided canonical sampling probabilities, if any.
    config_path = output / "config.toml"
    # Check config against the actual public trainer before publishing anything.
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "config.toml"
        candidate.write_text(_toml(config), encoding="utf-8")
        config = _config(candidate)
    config["canonical"]["k_probabilities"] = list(config["canonical"]["k_probabilities"])
    if (config["eval_interval"] > budget or config["checkpoint_interval"] > budget or
            config["checkpoint_interval"] != config["eval_interval"]):
        raise ValueError("evaluation and checkpoint cadence must match within the frozen update budget")
    # Same tokenizer and six-candidate preflight as the public canonical SFT loader.
    tokenizer = AutoTokenizer.from_pretrained(config["base_path"], local_files_only=True)
    preflight([*chosen, *development], tokenizer, max_packed=config["max_packed"])
    selected = [{"id": row["_meta"]["id"], "group_id": row["_meta"]["group_id"],
                 "source": row["source"]} for row in chosen]
    provenance = {"source_suite": str(suite.resolve()),
                  "source_manifest_sha256": _sha(suite / "manifest.json"),
                  "source_train_sha256": _sha(suite / "train.jsonl"),
                  "source_development_sha256": _sha(suite / "development.jsonl"),
                  "audited_suite": str(audited_suite.resolve()),
                  "audited_batch_manifest_sha256": observed["batch_manifest_sha256"],
                  "policy_sha256": observed["policy_sha256"],
                  "base_config": str(base_config.resolve()), "base_config_sha256": _sha(base_config),
                  "quality_report": str(quality_report.resolve()),
                  "quality_report_sha256": _sha(quality_report),
                  "review_sha256": _sha(review), "assessments_sha256": _sha(assessments),
                  "selected": selected}
    counts = {split: {"records": len(records),
                      "groups": len({row["_meta"]["group_id"] for row in records}),
                      "sources": dict(sorted(Counter(row["source"] for row in records).items()))}
              for split, records in (("train", chosen), ("development", development))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        derived = staging / "suite"
        derived.mkdir()
        files = {}
        for name, payload in (
            ("train.jsonl", "".join(_json(record) + "\n" for record in chosen).encode()),
            ("development.jsonl", (suite / "development.jsonl").read_bytes()),
        ):
            (derived / name).write_bytes(payload)
            files[name] = {"sha256": hashlib.sha256(payload).hexdigest(),
                           "records": 128 if name == "train.jsonl" else len(development)}
        manifest = {"format": "canonical_choice_v1", "complete": True,
                    "files": files, "counts": counts, "build": build, "derivation": provenance}
        (derived / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        (staging / "config.toml").write_text(_toml(config), encoding="utf-8")
        plan = {"overfit": {"suite": str((output / "suite").resolve()),
                            "config": str((output / "config.toml").resolve()),
                            "updates": budget, "train_canonicals": 128, "probe_views": 640,
                            "development_views": 5 * len(development),
                            "eval_interval": config["eval_interval"],
                            "checkpoint_interval": config["checkpoint_interval"],
                            "config_sha256": _sha(staging / "config.toml"),
                            "manifest_sha256": _sha(derived / "manifest.json")},
                "pilot": {"target_canonicals": [1000, 5000], "automatic_run": False},
                "full": {"target_canonicals": 30000, "automatic_run": False},
                "evaluation": {"temperature": 1, "nll": "natural logarithm, equal canonical and K weight",
                               "probe": "all 128 train canonicals, K=2..6",
                               "selection": "development source-macro NLL; best differs from final",
                               "success": "same checkpoint <=500 updates: every K accuracy >=0.95 and mean NLL <=0.15"},
                "authorization": "Review quality, engineering and overfit evidence before separately authorizing pilot/full; freeze their own suite, seed, initialization, update budget and fixed development before launch.",
                "quality": {"report": str(quality_report.resolve()), "status": observed["status"],
                            "policy_sha256": observed["policy_sha256"],
                            "batch_manifest_sha256": observed["batch_manifest_sha256"]}}
        (staging / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(staging, output)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("suite", "quality", "audited-suite", "review", "assessments", "base-config", "output"):
        parser.add_argument(f"--{key}", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_overfit(args.suite, args.quality, args.audited_suite,
                                     args.review, args.assessments, args.base_config,
                                     args.output), indent=2))


if __name__ == "__main__":
    main()
