"""Read-only, conservative report for the authorized canonical data progression stages.

Local training artifacts are trusted files, not cryptographic proof of model execution.
This command never generates data, starts training, or authorizes spending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import tomllib

from haidass_kev_train.data.canonical import load_canonical_suite
from haidass_kev_train.data.packing import check_group_integrity
from haidass_kev_train.data.quality import quality_gate, _identity
from haidass_kev_train.training.sft import _config

SOURCES = ("ufw-en", "ufw-zh", "finemath")
KS = tuple(map(str, range(2, 7)))
EVALUATION = {"temperature": 1.0, "model_precision": "model_forward",
              "log_probability_dtype": "float32", "aggregation_dtype": "float64",
              "nll": "natural_log", "reduction": "equal_case_equal_k"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _gate(status, evidence, reason, **details):
    return {"status": status, "evidence": {name: str(path) for name, path in evidence.items()},
            "reason": reason, **details}


class EvidenceIncomplete(ValueError):
    """Required coverage or completed evidence has not yet been supplied."""


def _missing(error):
    return isinstance(error, (FileNotFoundError, NotADirectoryError))


def _stage(fn, evidence):
    try:
        return fn()
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        return _gate("incomplete" if _missing(error) or isinstance(error, EvidenceIncomplete)
                     else "fail", evidence, str(error))


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _near(a, b):
    return isinstance(a, (float, int)) and not isinstance(a, bool) and math.isfinite(a) and abs(a - b) < 1e-7


def _stats(report, expected, *, permutation=False, seed=None):
    canonical = report["canonical"]
    _require(canonical["evaluation"] == EVALUATION and _near(canonical["chance_accuracy"], .29),
             "diagnostics are not raw natural-log equal-case/equal-K T=1")
    source_counts = {src: sum(record["_meta"]["source"] == src for record in expected) for src in SOURCES}
    _require(all(source_counts.values()) and set(canonical["by_source"]) == set(SOURCES),
             "fixed diagnostic lacks one of the three sources")
    _require(canonical["views"] == 5 * len(expected) and canonical["canonicals"] == len(expected) and
             canonical["groups"] == len({r["_meta"]["group_id"] for r in expected}),
             "fixed diagnostic case/view/group denominators disagree with suite")
    _require(set(canonical["by_k"]) == set(KS), "missing K diagnostic")
    for k in KS:
        bucket = canonical["by_k"][k]
        _require(bucket["views"] == bucket["canonicals"] == len(expected) and
                 bucket["groups"] == canonical["groups"], f"K={k} coverage does not include every case")
        _require(_near(bucket["accuracy"], bucket["accuracy"]) and
                 _near(bucket["nll"], bucket["nll"]) and 0 <= bucket["accuracy"] <= 1 and bucket["nll"] >= 0,
                 f"K={k} has nonfinite/out-of-range scores")
    for source in SOURCES:
        bucket = canonical["by_source"][source]
        count = source_counts[source]
        groups = len({r["_meta"]["group_id"] for r in expected if r["_meta"]["source"] == source})
        _require(bucket["views"] == 5 * count and bucket["canonicals"] == count and bucket["groups"] == groups
                 and set(bucket["by_k"]) == set(KS), f"{source}: coverage mismatch")
        for k in KS:
            sub = bucket["by_k"][k]
            _require(sub["views"] == sub["canonicals"] == count and sub["groups"] == groups and
                     _near(sub["nll"], sub["nll"]) and sub["nll"] >= 0 and
                     _near(sub["accuracy"], sub["accuracy"]) and 0 <= sub["accuracy"] <= 1,
                     f"{source}/K={k}: incomplete or invalid scores")
        for metric in ("nll", "accuracy"):
            _require(_near(bucket[metric], sum(bucket["by_k"][k][metric] for k in KS) / 5),
                     f"{source}: {metric} is not equal-K")
    for k in KS:
        _require(all(_near(canonical["by_k"][k][metric],
                           sum(source_counts[src] * canonical["by_source"][src]["by_k"][k][metric]
                               for src in SOURCES) / len(expected))
                     for metric in ("accuracy", "nll")), f"K={k}: source scores disagree with case weights")
    _require(all(_near(canonical[metric], sum(canonical["by_k"][k][metric] for k in KS) / 5)
                 for metric in ("accuracy", "nll")), "overall scores are not equal-K")
    if permutation:
        perm = canonical["permutation"]
        selected = min(200, len(expected))
        _require(seed is not None, "permutation requires frozen training seed")
        ranked = sorted(expected, key=lambda row: (
            hashlib.sha256(f"{seed}\0permutation\0{row['_meta']['id']}".encode()).digest(),
            row["_meta"]["id"]))[:selected]
        chosen_counts = {src: sum(row["_meta"]["source"] == src for row in ranked) for src in SOURCES}
        present = {src for src in SOURCES if chosen_counts[src]}
        _require(perm["views"] == 5 * selected and perm["canonicals"] == selected and
                 perm["groups"] == len({row["_meta"]["group_id"] for row in ranked}) and
                 set(perm["source_counts"]) == set(perm["by_source"]) == present and
                 all(perm["source_counts"][src]["views"] == 5 * chosen_counts[src] and
                     perm["source_counts"][src]["canonicals"] == chosen_counts[src] and
                     perm["source_counts"][src]["groups"] ==
                     len({row["_meta"]["group_id"] for row in ranked if row["_meta"]["source"] == src})
                     for src in present), "permutation sample disagrees with deterministic canonical selection")
        _require(set(perm["by_k"]) == set(KS), "missing permutation K")
        for k in KS:
            item = perm["by_k"][k]
            _require(item["orders"] == (2 if k == "2" else 3) and item["count"] == selected and
                     type(item["flips"]) is int and 0 <= item["flips"] <= selected and
                     _near(item["rate"], item["flips"] / selected), f"K={k}: invalid permutation summary")
            _require(sum(perm["by_source"][src][k]["flips"] for src in present) == item["flips"],
                     f"K={k}: source flips disagree")
            for src in present:
                source_item = perm["by_source"][src][k]
                size = chosen_counts[src]
                _require(source_item["count"] == size and type(source_item["flips"]) is int and
                         0 <= source_item["flips"] <= size and
                         _near(source_item["rate"], source_item["flips"] / size),
                         f"{src}/K={k}: invalid permutation summary")
    return canonical


def _suite(path):
    path = Path(path)
    manifest, batch, policy = _identity(path)
    records = {split: load_canonical_suite(path, split) for split in ("train", "development")}
    _require(not any(check_group_integrity(path, ("train", "development"))["overlaps"].values()),
             "train and development groups overlap")
    ids = [r["_meta"]["id"] for records_in_split in records.values() for r in records_in_split]
    _require(len(ids) == len(set(ids)), "canonical IDs repeat across splits")
    if manifest.get("complete") is not True:
        raise EvidenceIncomplete("frozen suite is incomplete")
    return manifest, batch, policy, records


def _quality(report, suite, review, assessments):
    evidence = {"quality_report": report, "audited_suite": suite, "review": review, "assessments": assessments}
    def check():
        original = _load(report)
        with tempfile.TemporaryDirectory() as tmp:
            rebuilt = quality_gate(suite, review, assessments, Path(tmp) / "rechecked.json")
        _require({k: v for k, v in original.items() if k != "report_path"} ==
                 {k: v for k, v in rebuilt.items() if k != "report_path"} and
                 original.get("report_path") == str(report),
                 "quality report differs from source-backed human assessment evidence")
        _require(original["status"] in ("pass", "fail", "incomplete"), "invalid quality result")
        if original["status"] != "pass":
            return _gate(original["status"], evidence, "source-backed audit did not pass", audit=original)
        _require(original["audited"] == original["required_audited"] == 100 and
                 original["severe_count"] == 0 and original["accepted"] == 100,
                 "quality audit did not verify all 100 with zero severe errors")
        return _gate("pass", evidence, "100 original-source human assessments verified", audit=original)
    return _stage(check, evidence)


def _run(config_path, run, suite, records):
    run = Path(run)
    config = tomllib.loads(Path(config_path).read_text())
    resolved_config = json.loads(json.dumps(_config(config_path)))
    _require(config.get("data_format") == "canonical_choice_v1" and
             Path(config["suite_path"]).resolve() == Path(suite).resolve() and
             (run / "config.toml").read_bytes() == Path(config_path).read_bytes(),
             "run configuration is not frozen canonical suite/config")
    _require(config["development_selection"] in ("clean", "all") and
             config["augmentation"].get("shuffle") is False and
             not any(config["augmentation"].get(key, 0) for key in ("p_none", "p_none_distract", "p_distract", "p_none_pair")),
             "run does not preserve canonical sampling and source-macro checkpoint selection")
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    ready = [event for event in events if event.get("event") == "ready"]
    _require(bool(ready) and all(event.get("config") == resolved_config and
                            event["train_records"] == len(records["train"]) and
                            event["development_records"] == 5 * len(records["development"])
                            for event in ready), "ready event disagrees with frozen config/suite")
    init = _load(run / "initialization.json")
    initialization_events = [event for event in events if event.get("event") == "initialization"]
    _require(initialization_events and all(event.get("identity_sha256") == init["identity_sha256"] and
                event.get("report") == init["report"] for event in initialization_events),
             "initialization baseline differs from run metrics")
    # Rebuild the exact SFT resume identity, not merely a report's declared suite path.
    import torch
    normalized = ready[-1]["config"]
    identity = {"config": normalized, "resources_sha256": _sha("configs/resources.toml"),
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "deterministic": True,
                            "cublas_workspace": ":4096:8", "tf32": True},
                "train_sha256": _sha(Path(suite) / "train.jsonl"),
                "development_sha256": _sha(Path(suite) / "development.jsonl"),
                "manifest_sha256": _sha(Path(suite) / "manifest.json")}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    _require(init["identity_sha256"] == expected,
             "initialization identity differs from SFT config/resources/runtime/frozen suite (use matching runtime)")
    _stats(init["report"], records["development"], permutation=True, seed=config["seed"])
    return config, events, init


def _overfit(suite, config, run, quality, audited_suite, review, assessments):
    evidence = {"suite": suite, "config": config, "run": run, "metrics": Path(run) / "metrics.jsonl",
                "best": Path(run) / "best.json"}
    def check():
        manifest, _, policy, records = _suite(suite)
        derivation = manifest["derivation"]
        parent = Path(derivation["source_suite"])
        parent_manifest, parent_sha, parent_policy, parent_records = _suite(parent)
        audit = _load(quality)
        _require(policy == parent_policy == audit["policy_sha256"] == derivation["policy_sha256"] and
                 derivation["audited_suite"] == str(Path(audited_suite).resolve()) and
                 derivation["quality_report"] == str(Path(quality).resolve()) and
                 derivation["quality_report_sha256"] == _sha(quality) and
                 derivation["review_sha256"] == _sha(review) and
                 derivation["assessments_sha256"] == _sha(assessments) and
                 derivation["audited_batch_manifest_sha256"] == audit["batch_manifest_sha256"],
                 "overfit recipe is not bound to the verified source-backed audit/policy")
        _require(derivation["source_manifest_sha256"] == parent_sha and
                 derivation["source_train_sha256"] == _sha(parent / "train.jsonl") and
                 derivation["source_development_sha256"] == _sha(parent / "development.jsonl") and
                 manifest["files"]["development.jsonl"]["sha256"] ==
                 parent_manifest["files"]["development.jsonl"]["sha256"] and
                 (Path(suite) / "development.jsonl").read_bytes() == (parent / "development.jsonl").read_bytes(),
                 "overfit suite does not preserve parent frozen development split")
        selected = [{"id": r["_meta"]["id"], "group_id": r["_meta"]["group_id"],
                     "source": r["_meta"]["source"]} for r in records["train"]]
        parent_by_id = {r["_meta"]["id"]: r for r in parent_records["train"]}
        _require(derivation["selected"] == selected and
                 all(parent_by_id.get(row["_meta"]["id"]) == row for row in records["train"]),
                 "derived overfit train cases differ from original parent records")
        if len(records["train"]) != 128 or {r["_meta"]["source"] for r in records["train"]} != set(SOURCES):
            raise EvidenceIncomplete("overfit needs 128 distinct train canonicals across all three sources")
        cfg, events, _ = _run(config, run, suite, records)
        _require(type(cfg["max_steps"]) is int and 1 <= cfg["max_steps"] <= 500 and
                 cfg["probe_groups"] == 128 and cfg["development_selection"] == "clean" and
                 cfg.get("train_sources") == sorted(SOURCES) and
                 cfg["eval_interval"] == cfg["checkpoint_interval"] <= cfg["max_steps"],
                 "overfit update budget, full probe or checkpoint cadence invalid")
        view_ids = {f"{r['_meta']['id']}/k{k}" for r in records["train"] for k in range(2, 7)}
        _require(len(view_ids) == 640 and
                 all(set(event["train_probe_record_ids"]) == view_ids and
                     len(event["train_probe_record_ids"]) == 640
                     for event in events if event.get("event") == "ready"),
                 "ready event does not select all 640 fixed five-K train views")
        probes = [event for event in events if event.get("event") == "train_probe"]
        updates = {e["step"] for e in events if e.get("event") == "train"}
        checkpoints = {e["step"] for e in events if e.get("event") == "checkpoint"
                       and type(e.get("step")) is int
                       and Path(e["path"]).resolve() ==
                           (Path(run) / f"step-{e['step']:06d}").resolve()
                       and (Path(run) / f"step-{e['step']:06d}" / "training_state.pt").is_file()}
        scores = []
        for probe in probes:
            step = probe["step"]
            _require(type(step) is int and 1 <= step <= cfg["max_steps"] and step in updates and
                     step in checkpoints and set(probe["record_ids"]) == view_ids and
                     len(probe["record_ids"]) == 640,
                     "probe lacks same-update saved checkpoint/optimizer update or full five-K train views")
            canonical = _stats(probe["report"], records["train"])
            scores.append({"step": step, "checkpoint": f"step-{step:06d}",
                           "accuracy_by_k": {k: canonical["by_k"][k]["accuracy"] for k in KS},
                           "nll": canonical["nll"], "by_source": canonical["by_source"],
                           "pass": canonical["nll"] <= .15 and
                           all(canonical["by_k"][k]["accuracy"] >= .95 for k in KS)})
        _require(len(scores) == len({score["step"] for score in scores}), "duplicate probe optimizer step")
        finished_events = [e for e in events if e.get("event") == "finished" and e.get("step") == cfg["max_steps"]]
        finished = bool(finished_events)
        gradients = [e["gradient_groups"] for e in events if e.get("event") == "train"]
        _require(not finished or (updates == set(range(1, cfg["max_steps"] + 1)) and
                 all(any(name in groups and _near(groups[name]["grad_norm"], groups[name]["grad_norm"])
                         and groups[name]["grad_norm"] > 0 for groups in gradients)
                     for name in ("backbone", "head"))),
                 "engineering evidence lacks continuous optimizer updates or nonzero backbone/head gradients")
        best_checkpoint = None
        if finished:
            best = _load(Path(run) / "best.json")
            best_checkpoint = best["checkpoint"]
            best_step = int(best_checkpoint.removeprefix("step-"))
            development = [e for e in events if e.get("event") == "development"]
            _require(best_checkpoint == f"step-{best_step:06d}" and best_step in checkpoints and
                     finished_events[-1].get("best_checkpoint") == best_checkpoint and
                     Path(finished_events[-1]["checkpoint"]).name == f"step-{cfg['max_steps']:06d}" and
                     cfg["max_steps"] in checkpoints and
                     best["selection"]["split"] == "development" and
                     best["selection"]["subset"] == "clean" and best["selection"]["metric"] == "macro_nll" and
                     len([e for e in development if e.get("step") == best_step and
                          e.get("selection") == best["selection"]]) == 1 and
                     all(e["selection"]["value"] >= best["selection"]["value"] - 1e-7 and
                         _near(e["selection"]["value"], e["report"]["clean"]["macro_nll"])
                         for e in development), "overfit best/final checkpoint or development selection inconsistent")
        passed = next((score for score in scores if score["pass"]), None)
        status = "pass" if passed and finished else "fail" if finished and scores else "incomplete"
        return _gate(status, evidence, "same-update 640-view overfit threshold" if passed and finished else
                     "completed budget did not demonstrate the overfit threshold" if finished else
                     "no completed overfit run with full probe evidence", selected=passed, probes=scores,
                     best_checkpoint=best_checkpoint, final_checkpoint=f"step-{cfg['max_steps']:06d}" if finished else None)
    return _stage(check, evidence)


def _scaling(suite, config, run, *, size, policy=None):
    evidence = {"suite": suite, "config": config, "run": run,
                "metrics": Path(run) / "metrics.jsonl", "initialization": Path(run) / "initialization.json",
                "best": Path(run) / "best.json"}
    def check():
        _, _, suite_policy, records = _suite(suite)
        if policy is not None:
            _require(suite_policy == policy, "later data does not use the source-backed audited builder policy")
        count = len(records["train"])
        _require((1000 <= count <= 5000 if size == "pilot" else 25000 <= count <= 35000),
                 f"{size} train size {count} outside frozen stage target")
        groups = {src: len({r["_meta"]["group_id"] for r in records["development"]
                            if r["_meta"]["source"] == src}) for src in SOURCES}
        if any(value < 50 for value in groups.values()):
            return _gate("incomplete", evidence, "each development source needs 50 independent groups", groups=groups)
        cfg, events, init = _run(config, run, suite, records)
        _require(type(cfg["max_steps"]) is int and cfg["max_steps"] > 0,
                 "finite positive optimizer update budget required")
        finished = [event for event in events if event.get("event") == "finished" and event.get("step") == cfg["max_steps"]]
        if not finished:
            return _gate("incomplete", evidence, "training has not exhausted its frozen finite update budget", groups=groups)
        training = {event["step"]: event for event in events if event.get("event") == "train"}
        _require(set(training) == set(range(1, cfg["max_steps"] + 1)) and
                 all(_near(event["loss"], event["loss"]) and event["loss"] >= 0
                     for event in training.values()), "missing or nonfinite optimizer-update training loss")
        best = _load(Path(run) / "best.json")
        _require(best["selection"]["split"] == "development" and
                 best["selection"]["subset"] == cfg["development_selection"] and
                 best["selection"]["metric"] == "macro_nll", "best checkpoint not selected by development macro NLL")
        selected_step = int(best["checkpoint"].removeprefix("step-"))
        _require(best["checkpoint"] == f"step-{selected_step:06d}" and
                 (Path(run) / best["checkpoint"] / "training_state.pt").is_file(),
                 "selected best checkpoint absent")
        last_path = Path(finished[-1]["checkpoint"])
        _require(last_path.name == f"step-{cfg['max_steps']:06d}" and
                 (Path(run) / last_path.name / "training_state.pt").is_file() and
                 finished[-1].get("best_checkpoint") == best["checkpoint"],
                 "selected and final checkpoint artifacts disagree")
        evaluations = [event for event in events if event.get("event") == "development"]
        selection = [event for event in evaluations if event.get("step") == selected_step]
        _require(len(selection) == 1 and selected_step in {e.get("step") for e in events if e.get("event") == "train"},
                 "best checkpoint has no optimizer-update development report")
        for event in evaluations:
            _require(event["selection"]["split"] == "development" and
                     event["selection"]["subset"] == cfg["development_selection"] and
                     event["selection"]["metric"] == "macro_nll", "development selection contract changed")
            current = _stats(event["report"], records["development"])
            source_macro = sum(current["by_source"][src]["nll"] for src in SOURCES) / 3
            _require(_near(event["selection"]["value"], source_macro) and
                     _near(event["selection"]["value"],
                           event["report"][cfg["development_selection"]]["macro_nll"]),
                     "development selection score disagrees with per-source fixed five-K NLL")
        _require(_near(best["selection"]["value"], selection[0]["selection"]["value"]) and
                 all(best["selection"]["value"] <= e["selection"]["value"] + 1e-7 for e in evaluations),
                 "best.json is not the run's macro-NLL minimum")
        canonical = _stats(selection[0]["report"], records["development"], permutation=True, seed=cfg["seed"])
        baseline = _stats(init["report"], records["development"], permutation=True, seed=cfg["seed"])
        source = {src: {"accuracy": canonical["by_source"][src]["accuracy"],
                        "nll": canonical["by_source"][src]["nll"],
                        "initial_nll": baseline["by_source"][src]["nll"],
                        "groups": groups[src], "by_k": canonical["by_source"][src]["by_k"],
                        "permutation_by_k": canonical["permutation"]["by_source"].get(src)}
                  for src in SOURCES}
        _require(_near(selection[0]["report"][cfg["development_selection"]]["macro_nll"],
                       sum(canonical["by_source"][src]["nll"] for src in SOURCES) / 3),
                 "selected macro score is not three-source equal-K macro NLL")
        passed = all(stats["accuracy"] >= .34 and stats["nll"] < stats["initial_nll"]
                     for stats in source.values())
        return _gate("pass" if passed else "fail", evidence,
                     "all three sources meet raw accuracy and same-initialization NLL" if passed else
                     "at least one source misses accuracy or initialization NLL", best_checkpoint=best,
                     final_checkpoint=last_path.name, source=source, by_k=canonical["by_k"],
                     accuracy=canonical["accuracy"], nll=canonical["nll"],
                     training_loss={"selected_step": training[selected_step]["loss"],
                                    "final_step": training[cfg["max_steps"]]["loss"]},
                     permutation=canonical["permutation"], selection_step=selected_step)
    return _stage(check, evidence)


def report(*, output, quality=None, audited_suite=None, review=None, assessments=None,
           overfit_suite=None, overfit_config=None, overfit_run=None, pilot_suite=None,
           pilot_config=None, pilot_run=None, full_suite=None, full_config=None, full_run=None):
    """Write a decision report; optional stages never imply permission to execute them."""
    quality_args = (quality, audited_suite, review, assessments)
    quality_gate_result = (_quality(*quality_args) if all(quality_args) else
                           _gate("incomplete", {}, "quality report, audited suite, review and assessments required"))
    overfit_args = (overfit_suite, overfit_config, overfit_run)
    overfit = (_overfit(*overfit_args, *quality_args)
               if quality_gate_result["status"] == "pass" and all(overfit_args) else
               _gate("fail" if quality_gate_result["status"] == "fail" else "incomplete",
                     {name: path for name, path in zip(("suite", "config", "run"), overfit_args) if path},
                     "verified passing quality and complete overfit suite/config/run required"))
    policy = quality_gate_result.get("audit", {}).get("policy_sha256") if quality_gate_result["status"] == "pass" else None
    pilot_args = (pilot_suite, pilot_config, pilot_run)
    pilot = (_scaling(*pilot_args, size="pilot", policy=policy) if all(pilot_args) else
             _gate("incomplete", {}, "pilot suite, config and run required"))
    gates = {"quality": quality_gate_result, "overfit": overfit, "pilot_to_full": pilot}
    if any(path is not None for path in (full_suite, full_config, full_run)):
        gates["full_review"] = (_scaling(full_suite, full_config, full_run, size="full", policy=policy)
                                if all(path is not None for path in (full_suite, full_config, full_run)) else
                                _gate("incomplete", {}, "full stage requires suite, config and run"))
    # No stage may be promoted solely by a report it did not verify.
    if quality_gate_result["status"] != "pass" or overfit["status"] != "pass":
        gates["pilot_to_full"] = {**pilot, "status": "fail" if "fail" in
                                  (quality_gate_result["status"], overfit["status"], pilot["status"])
                                  else "incomplete", "reason": "quality/engineering/overfit prerequisite: " +
                                  ", ".join((quality_gate_result["status"], overfit["status"], pilot["status"]))}
    if "full_review" in gates and not all(gates[name]["status"] == "pass"
                                         for name in ("quality", "overfit", "pilot_to_full")):
        stage = gates["full_review"]
        gates["full_review"] = {**stage, "status": "fail" if "fail" in
                                (stage["status"], gates["pilot_to_full"]["status"]) else "incomplete",
                                "reason": "pilot-to-full prerequisite not satisfied"}
    decision = {"gates": gates, "recommend_full": all(gates[name]["status"] == "pass"
                  for name in ("quality", "overfit", "pilot_to_full")),
                "interpretation": "Development learning signal only; not spending authorization, SFT Complete, OOD or significance."}
    encoded = json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(output)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f".{path.name}.",
                                     dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(encoded)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary_path, path)  # atomic no-clobber publication on the same filesystem
    finally:
        temporary_path.unlink(missing_ok=True)
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quality", "audited-suite", "review", "assessments", "overfit-suite",
                 "overfit-config", "overfit-run", "pilot-suite", "pilot-config", "pilot-run",
                 "full-suite", "full-config", "full-run"):
        parser.add_argument("--" + name)
    parser.add_argument("--output", required=True)
    args = vars(parser.parse_args(argv))
    print(json.dumps(report(**{key.replace("-", "_"): value for key, value in args.items()}),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
