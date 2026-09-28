"""Software-only synthetic report artifacts; these are NOT CUDA training or human audit evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from haidass_kev_train.evaluation.gates import _overfit, _scaling, report
from haidass_kev_train.training.sft import _config

SOURCES = ("ufw-en", "ufw-zh", "finemath")


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(src, index, split, group=None):
    name = f"{split}-{src}-{index}"
    ref = {"path": f"{src}.parquet", "line": index, "sha256": "a" * 64,
           "question_span": [0, 1], "answer_span": [1, 2]}
    if src != "finemath":
        ref.update(uid=name, state_span=[0, 1])
    return {"source": src, "state": "state", "question": "question", "gold": "answer",
            "distractors": [f"wrong {n}" for n in range(5)],
            "_meta": {"id": name, "group_id": f"{split}-{src}-group-{group if group is not None else index}",
                      "source": src, "source_ref": ref, "validation": "synthetic-software-only"}}


def suite(path, *, short_groups=False):
    path.mkdir()
    files = {}
    for split, amounts in (("train", (334, 333, 333)), ("development", (50, 50, 50))):
        records = [canonical(src, i, split, 48 if short_groups and split == "development" and src == "ufw-en" and i == 49 else None)
                   for src, amount in zip(SOURCES, amounts) for i in range(amount)]
        target = path / f"{split}.jsonl"
        target.write_text("".join(json.dumps(row) + "\n" for row in records))
        files[target.name] = {"sha256": sha(target), "records": len(records)}
    build = {"config": {"sources": {src: str(path.resolve()) for src in SOURCES}},
             "model": "synthetic-only", "base_url": "synthetic-only", "prompt_version": "synthetic-only",
             "thinking": {}, "source_location": "synthetic-only", "tokenizer_sha256": "synthetic-only",
             "generator_tokenizer_sha256": "synthetic-only"}
    build["policy"] = {"sources": build["config"]["sources"],
                       "limits": {name: None for name in ("max_packed", "max_answer_tokens", "max_source_tokens",
                                                          "max_context_tokens", "max_output_tokens")},
                       **{name: build[name] for name in ("model", "base_url", "prompt_version", "thinking",
                                                        "source_location", "tokenizer_sha256", "generator_tokenizer_sha256")}}
    encoded = json.dumps(build["policy"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    build["policy_sha256"] = hashlib.sha256(encoded).hexdigest()
    dump(path / "manifest.json", {"format": "canonical_choice_v1", "complete": True,
                                   "files": files, "build": build})
    return [canonical(src, i, "development", 48 if short_groups and src == "ufw-en" and i == 49 else None)
            for src in SOURCES for i in range(50)]


def diagnostics(records, accuracies, nlls, *, seed=17):
    by_source = {}
    by_k = {}
    for src in SOURCES:
        rows = [row for row in records if row["source"] == src]
        groups = len({row["_meta"]["group_id"] for row in rows})
        by_source[src] = {"accuracy": accuracies[src], "nll": nlls[src], "views": len(rows) * 5,
                          "canonicals": len(rows), "groups": groups,
                          "by_k": {str(k): {"accuracy": accuracies[src], "nll": nlls[src],
                                              "views": len(rows), "canonicals": len(rows), "groups": groups}
                                   for k in range(2, 7)}}
    for k in range(2, 7):
        by_k[str(k)] = {"accuracy": sum(accuracies.values()) / 3,
                         "nll": sum(nlls.values()) / 3, "views": len(records),
                         "canonicals": len(records), "groups": len({r["_meta"]["group_id"] for r in records})}
    selected = sorted(records, key=lambda row: (hashlib.sha256(
        f"{seed}\0permutation\0{row['_meta']['id']}".encode()).digest(), row["_meta"]["id"]))[:200]
    source_counts = {src: sum(row["source"] == src for row in selected) for src in SOURCES}
    permutation = {"views": len(selected) * 5, "canonicals": len(selected),
                   "groups": len({r["_meta"]["group_id"] for r in selected}),
                   "source_counts": {}, "by_source": {}, "by_k": {}}
    for src, count in source_counts.items():
        permutation["source_counts"][src] = {"views": count * 5, "canonicals": count,
                                                 "groups": len({r["_meta"]["group_id"] for r in selected
                                                                if r["source"] == src})}
        permutation["by_source"][src] = {str(k): {"count": count, "flips": 0, "rate": 0.0}
                                          for k in range(2, 7)}
    permutation["by_k"] = {str(k): {"orders": 2 if k == 2 else 3,
                                     "count": len(selected), "flips": 0, "rate": 0.0}
                            for k in range(2, 7)}
    canonical_report = {"views": len(records) * 5, "canonicals": len(records),
                        "groups": len({r["_meta"]["group_id"] for r in records}),
                        "accuracy": sum(accuracies.values()) / 3, "nll": sum(nlls.values()) / 3,
                        "chance_accuracy": .29,
                        "evaluation": {"temperature": 1.0, "model_precision": "model_forward",
                                       "log_probability_dtype": "float32", "aggregation_dtype": "float64",
                                       "nll": "natural_log", "reduction": "equal_case_equal_k"},
                        "by_source": by_source, "by_k": by_k, "permutation": permutation}
    return {"canonical": canonical_report, "clean": {"macro_nll": sum(nlls.values()) / 3}}


class SyntheticConsumerBoundaries(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.suite = self.root / "suite"
        self.records = suite(self.suite)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.config = self.root / "config.toml"
        self.config.write_text('''data_format = "canonical_choice_v1"
base_path = "synthetic-only"
suite_path = "''' + str(self.suite) + '''"
seed = 17
batch_size = 1
gradient_accumulation = 1
learning_rate = 0.0001
weight_decay = 0.0
max_steps = 2
warmup_steps = 0
eval_interval = 1
checkpoint_interval = 1
max_packed = 1024
eval_batch_size = 4
max_grad_norm = 1.0
training_mode = "lora"
probe_groups = 4
development_selection = "clean"
[augmentation]
shuffle = false
''')
        (self.run_dir / "config.toml").write_bytes(self.config.read_bytes())

    def artifacts(self, accuracy=None, nll=None, baseline=None):
        accuracy = accuracy or {src: .34 for src in SOURCES}
        nll = nll or {src: .69 for src in SOURCES}
        baseline = baseline or {src: .70 for src in SOURCES}
        cfg = json.loads(json.dumps(_config(self.config)))
        identity = {"config": cfg, "resources_sha256": sha(Path("configs/resources.toml")),
                    "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                "deterministic": True, "cublas_workspace": ":4096:8", "tf32": True},
                    "train_sha256": sha(self.suite / "train.jsonl"),
                    "development_sha256": sha(self.suite / "development.jsonl"),
                    "manifest_sha256": sha(self.suite / "manifest.json")}
        identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        init = {"identity_sha256": identity_sha,
                "report": diagnostics(self.records, {src: .29 for src in SOURCES}, baseline)}
        dump(self.run_dir / "initialization.json", init)
        selected = diagnostics(self.records, accuracy, nll)
        value = sum(nll.values()) / 3
        for step in (1, 2):
            checkpoint = self.run_dir / f"step-{step:06d}"
            checkpoint.mkdir(exist_ok=True)
            (checkpoint / "training_state.pt").write_text("synthetic-only, not a checkpoint")
        dump(self.run_dir / "best.json", {"checkpoint": "step-000001",
                                      "selection": {"split": "development", "subset": "clean",
                                                    "metric": "macro_nll", "value": value}})
        events = [{"event": "initialization", "step": 0, "identity_sha256": identity_sha,
                   "report": init["report"]},
                  {"event": "ready", "step": 0, "config": cfg, "train_records": 1000,
                   "development_records": 750},
                  {"event": "train", "step": 1, "loss": .8},
                  {"event": "development", "step": 1,
                   "selection": {"split": "development", "subset": "clean",
                                 "metric": "macro_nll", "value": value}, "report": selected},
                  {"event": "train", "step": 2, "loss": .7},
                  {"event": "development", "step": 2,
                   "selection": {"split": "development", "subset": "clean",
                                 "metric": "macro_nll", "value": value + .01},
                   "report": diagnostics(self.records, accuracy, {src: nll[src] + .01 for src in SOURCES})},
                  {"event": "finished", "step": 2, "checkpoint": str(self.run_dir / "step-000002"),
                   "best_checkpoint": "step-000001"}]
        (self.run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))

    def test_raw_boundaries_are_per_source_and_equal_initialization_is_failure(self):
        self.artifacts()
        result = _scaling(self.suite, self.config, self.run_dir, size="pilot")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["best_checkpoint"]["checkpoint"], "step-000001")
        self.assertEqual(result["final_checkpoint"], "step-000002")
        self.assertIn("permutation", result)
        self.artifacts(accuracy={**{src: .34 for src in SOURCES}, "finemath": .32})
        self.assertEqual(_scaling(self.suite, self.config, self.run_dir, size="pilot")["status"], "fail")
        self.artifacts(nll={**{src: .69 for src in SOURCES}, "finemath": .70})
        self.assertEqual(_scaling(self.suite, self.config, self.run_dir, size="pilot")["status"], "fail")

    def test_competing_development_event_cannot_inflate_macro_to_select_wrong_best(self):
        self.artifacts()
        metrics = self.run_dir / "metrics.jsonl"
        events = [json.loads(line) for line in metrics.read_text().splitlines()]
        # Step 2 actually has lower source-macro NLL, but its copied selection
        # scalar falsely claims worse performance to favor step 1.
        events[5]["report"]["canonical"] = diagnostics(
            self.records, {src: .36 for src in SOURCES}, {src: .68 for src in SOURCES})["canonical"]
        metrics.write_text("".join(json.dumps(row) + "\n" for row in events))
        result = _scaling(self.suite, self.config, self.run_dir, size="pilot")
        self.assertEqual(result["status"], "fail")
        self.assertIn("per-source fixed five-K NLL", result["reason"])

    def test_invalid_nested_number_never_overwrites_or_publishes_partial_report(self):
        self.artifacts()
        metrics = self.run_dir / "metrics.jsonl"
        events = [json.loads(line) for line in metrics.read_text().splitlines()]
        events[3]["report"]["canonical"]["by_source"]["ufw-en"]["by_k"]["2"]["brier"] = float("nan")
        metrics.write_text("".join(json.dumps(row) + "\n" for row in events))
        destination = self.root / "existing-decision.json"
        destination.write_text("operator evidence stays unchanged\n")
        for path in (destination, self.root / "new-decision.json"):
            with self.assertRaises(ValueError):
                report(pilot_suite=self.suite, pilot_config=self.config, pilot_run=self.run_dir,
                       output=path)
        self.assertEqual(destination.read_text(), "operator evidence stays unchanged\n")
        self.assertFalse((self.root / "new-decision.json").exists())

    def test_valid_report_does_not_clobber_existing_evidence(self):
        self.artifacts()
        destination = self.root / "existing-decision.json"
        destination.write_text("operator evidence stays unchanged\n")
        with self.assertRaises(FileExistsError):
            report(pilot_suite=self.suite, pilot_config=self.config, pilot_run=self.run_dir,
                   output=destination)
        self.assertEqual(destination.read_text(), "operator evidence stays unchanged\n")

    def test_group_shortfall_is_incomplete_not_fivefold_views(self):
        self.suite.rename(self.root / "unused")
        self.records = suite(self.suite, short_groups=True)
        self.artifacts()
        result = _scaling(self.suite, self.config, self.run_dir, size="pilot")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["groups"]["ufw-en"], 49)

    def test_forged_quality_json_and_missing_audit_never_promote(self):
        self.artifacts()
        forged = self.root / "quality.json"
        dump(forged, {"status": "pass", "audited": 100, "severe_count": 0})
        result = report(quality=forged, audited_suite=self.root / "absent-audit-suite",
                        review=self.root / "absent-review", assessments=self.root / "absent-assessments",
                        pilot_suite=self.suite, pilot_config=self.config, pilot_run=self.run_dir,
                        output=self.root / "decision.json")
        self.assertEqual(result["gates"]["quality"]["status"], "incomplete")
        self.assertEqual(result["gates"]["pilot_to_full"]["status"], "incomplete")
        self.assertFalse(result["recommend_full"])

    def test_overfit_thresholds_must_coincide_at_one_saved_optimizer_step(self):
        # Construct ONLY report-consumer inputs; no mock model or claimed training run.
        derived = self.root / "derived"
        derived.mkdir()
        parents = [json.loads(line) for line in (self.suite / "train.jsonl").read_text().splitlines()]
        selected = [*parents[:50], *parents[334:374], *parents[667:705]]
        self.assertEqual(len(selected), 128)
        train = derived / "train.jsonl"
        train.write_text("".join(json.dumps(row) + "\n" for row in selected))
        (derived / "development.jsonl").write_bytes((self.suite / "development.jsonl").read_bytes())
        quality = self.root / "synthetic-audit.json"
        dump(quality, {"policy_sha256": json.loads((self.suite / "manifest.json").read_text())
                       ["build"]["policy_sha256"], "batch_manifest_sha256": "1" * 64})
        review, assessments = self.root / "review.jsonl", self.root / "assessments.jsonl"
        review.write_text("synthetic only")
        assessments.write_text("synthetic only")
        manifest = json.loads((self.suite / "manifest.json").read_text())
        manifest["files"]["train.jsonl"] = {"sha256": sha(train), "records": 128}
        manifest["derivation"] = {
            "source_suite": str(self.suite.resolve()),
            "source_manifest_sha256": sha(self.suite / "manifest.json"),
            "source_train_sha256": sha(self.suite / "train.jsonl"),
            "source_development_sha256": sha(self.suite / "development.jsonl"),
            "audited_suite": str(self.suite.resolve()),
            "audited_batch_manifest_sha256": "1" * 64,
            "policy_sha256": manifest["build"]["policy_sha256"],
            "quality_report": str(quality.resolve()), "quality_report_sha256": sha(quality),
            "review_sha256": sha(review), "assessments_sha256": sha(assessments),
            "selected": [{"id": row["_meta"]["id"], "group_id": row["_meta"]["group_id"],
                          "source": row["source"]} for row in selected]}
        dump(derived / "manifest.json", manifest)
        config = self.root / "overfit.toml"
        config.write_text(self.config.read_text().replace(str(self.suite), str(derived))
                          .replace("probe_groups = 4", "probe_groups = 128")
                          .replace("seed = 17", 'train_sources = ["finemath", "ufw-en", "ufw-zh"]\nseed = 17'))
        run = self.root / "overfit-run"
        run.mkdir()
        (run / "config.toml").write_bytes(config.read_bytes())
        cfg = json.loads(json.dumps(_config(config)))
        identity = {"config": cfg, "resources_sha256": sha(Path("configs/resources.toml")),
                    "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                "deterministic": True, "cublas_workspace": ":4096:8", "tf32": True},
                    "train_sha256": sha(train), "development_sha256": sha(derived / "development.jsonl"),
                    "manifest_sha256": sha(derived / "manifest.json")}
        baseline = diagnostics(self.records, {src: .29 for src in SOURCES},
                               {src: .70 for src in SOURCES})
        initialization = {"identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                          "report": baseline}
        dump(run / "initialization.json", initialization)
        for step in (1, 2):
            checkpoint = run / f"step-{step:06d}"
            checkpoint.mkdir()
            (checkpoint / "training_state.pt").write_text("synthetic-only, not a checkpoint")
        ids = [f"{row['_meta']['id']}/k{k}" for row in selected for k in range(2, 7)]
        events = [{"event": "initialization", "step": 0, **initialization},
                  {"event": "ready", "step": 0, "config": cfg, "train_records": 128,
                   "development_records": 750, "train_probe_record_ids": ids}]

        def probe(step, accuracy, nll):
            events.extend((
                {"event": "train", "step": step, "gradient_groups": {
                    "backbone": {"grad_norm": .4}, "head": {"grad_norm": .5}}},
                {"event": "train_probe", "step": step, "record_ids": ids,
                 "report": diagnostics(selected, {src: accuracy for src in SOURCES},
                                       {src: nll for src in SOURCES})},
                {"event": "checkpoint", "step": step, "path": str(run / f"step-{step:06d}")}))

        probe(1, .96, .151)
        probe(2, .94, .15)
        selection = {"split": "development", "subset": "clean", "metric": "macro_nll", "value": .70}
        dump(run / "best.json", {"checkpoint": "step-000001", "selection": selection})
        events.extend([
            {"event": "development", "step": 1, "selection": selection, "report": baseline},
            {"event": "development", "step": 2,
             "selection": {**selection, "value": .71},
             "report": diagnostics(self.records, {src: .29 for src in SOURCES},
                                   {src: .71 for src in SOURCES})},
            {"event": "finished", "step": 2, "checkpoint": str(run / "step-000002"),
             "best_checkpoint": "step-000001"}])

        def consume():
            (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
            return _overfit(derived, config, run, quality, self.suite, review, assessments)

        self.assertEqual(consume()["status"], "fail")
        events[6]["report"] = diagnostics(selected, {src: .96 for src in SOURCES},
                                           {src: .149 for src in SOURCES})
        passed = consume()
        self.assertEqual(passed["status"], "pass")
        self.assertEqual(passed["selected"]["step"], 2)
        self.assertEqual(passed["selected"]["checkpoint"], "step-000002")
        events[6]["record_ids"] = ids[:-1]
        missing = consume()
        self.assertEqual(missing["status"], "fail")
        self.assertIn("five-K train views", missing["reason"])
        events[6]["record_ids"] = ids
        events[7]["path"] = str(run / "step-000001")  # claims update 2 with update-1 checkpoint
        wrong_checkpoint = consume()
        self.assertEqual(wrong_checkpoint["status"], "fail")
        self.assertIn("same-update saved checkpoint", wrong_checkpoint["reason"])


if __name__ == "__main__":
    unittest.main()
