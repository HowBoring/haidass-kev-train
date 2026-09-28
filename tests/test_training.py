"""Behavior tests for the public Decision SFT entrypoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

import numpy as np
from safetensors.torch import load_file
import torch


_BASE_CONFIG = """\
base_path = "models/base"
suite_path = "data/suite"
train_sources = ["public", "compositional"]
seed = 42
batch_size = 2
gradient_accumulation = 2
learning_rate = 0.0002
head_learning_rate = 0.0002
weight_decay = 0.01
max_steps = 4
warmup_steps = 1
scheduler = "cosine"
eval_interval = 2
checkpoint_interval = 2
max_packed = 2048
eval_batch_size = 4
max_grad_norm = 1.0
training_mode = "lora"
probe_groups = 2
development_selection = "clean"

[augmentation]
shuffle = true
p_none = 0.1
p_none_distract = 0.12
p_distract = 0.15
p_none_pair = 0.25
"""


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _write_suite(path):
    record = {
        "state": "A parcel arrived with a cracked screen.",
        "questions": {
            "condition": {
                "type": "choice",
                "instructions": "Choose the condition.",
                "criteria": {"broken": "cracked", "working": "intact", "unknown": "not enough evidence"},
                "label": "broken",
                "src": "tiny",
            },
            "usable": {
                "type": "noul",
                "instructions": "Is the screen intact?",
                "criteria": {"false": "damaged", "true": "intact"},
                "label": False,
                "src": "tiny",
            },
        },
        "_meta": {"id": "tiny/one", "group_id": "tiny/one", "variant": "clean", "source": "tiny"},
    }
    path.mkdir()
    files = {}
    for split in ("train", "development"):
        payload = (json.dumps(record) + "\n").encode()
        (path / f"{split}.jsonl").write_bytes(payload)
        files[f"{split}.jsonl"] = {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}
    (path / "manifest.json").write_text(json.dumps({"files": files}))


def _run_training(config, output, *extra):
    command = [sys.executable, "-m", "haidass_kev_train.training.sft",
               "--config", str(config), "--output", str(output), *map(str, extra)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(f"{command} failed:\n{result.stdout}\n{result.stderr}")


def _training_config(suite, *, scheduler="onecycle", max_steps=20, warmup_steps=0):
    return f"""\
base_path = "models/base/haidass1.5-143m"
suite_path = "{suite}"
train_sources = ["tiny"]
seed = 17
batch_size = 1
gradient_accumulation = 1
learning_rate = 0.0002
head_learning_rate = 0.0001
weight_decay = 0.01
max_steps = {max_steps}
warmup_steps = {warmup_steps}
scheduler = "{scheduler}"
eval_interval = {max_steps}
checkpoint_interval = {max_steps}
max_packed = 256
eval_batch_size = 1
max_grad_norm = 1.0
training_mode = "lora"
probe_groups = 1
development_selection = "clean"

[augmentation]
shuffle = true
p_none = 0.0
p_none_distract = 0.0
p_distract = 0.0
p_none_pair = 0.0
"""


class TrainingEntrypointTests(unittest.TestCase):
    def test_plan_only_resolves_distinct_equal_budget_comparison_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.toml"
            output = root / "comparison"
            base.write_text(_BASE_CONFIG)
            command = [
                sys.executable,
                "-m",
                "haidass_kev_train.training.experiments",
                "--config",
                str(base),
                "--output",
                str(output),
                "--budgets",
                "8",
                "--learning-rates",
                "0.0002",
                "0.00005",
                "--modes",
                "lora",
                "full",
                "--seeds",
                "7",
                "--plan-only",
            ]

            subprocess.run(command, check=True, capture_output=True, text=True)

            plan = json.loads((output / "comparison.json").read_text())
            self.assertTrue(plan["plan_only"])
            self.assertEqual(len(plan["runs"]), 4)
            resolved = [tomllib.loads((output / run["name"] / "config.toml").read_text()) for run in plan["runs"]]
            self.assertEqual({config["max_steps"] for config in resolved}, {8})
            self.assertEqual({config["seed"] for config in resolved}, {7})
            self.assertEqual({config["training_mode"] for config in resolved}, {"lora", "full"})
            self.assertEqual({config["learning_rate"] for config in resolved}, {0.0002, 0.00005})
            self.assertEqual({config["scheduler"] for config in resolved}, {"cosine"})
            self.assertTrue(all(config["train_sources"] == ["public", "compositional"] for config in resolved))
            self.assertTrue(all(config["augmentation"]["p_none_pair"] == 0.25 for config in resolved))
            self.assertTrue(all(run["status"] == "planned" for run in plan["runs"]))
            self.assertTrue(all("config_sha256" in run for run in plan["runs"]))

            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Refusing to overwrite", repeated.stderr)


    def test_comparison_rejects_unrunnable_axes_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.toml"
            base.write_text(_BASE_CONFIG)
            cases = ((["--budgets", "1", "--learning-rates", "0.0002"], "warmup_steps"),
                     (["--budgets", "4", "--learning-rates", "nan"], "learning rate"))
            for index, (axes, message) in enumerate(cases):
                with self.subTest(message=message):
                    output = root / f"invalid-{index}"
                    command = [sys.executable, "-m", "haidass_kev_train.training.experiments",
                               "--config", str(base), "--output", str(output), *axes,
                               "--modes", "lora", "--seeds", "7", "--plan-only"]
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(output.exists())

            base.write_text(_BASE_CONFIG.replace('scheduler = "cosine"', 'scheduler = "onecycle"'))
            output = root / "invalid-onecycle-warmup"
            command = [sys.executable, "-m", "haidass_kev_train.training.experiments",
                       "--config", str(base), "--output", str(output), "--budgets", "4",
                       "--learning-rates", "0.0002", "--modes", "lora", "--seeds", "7",
                       "--plan-only"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("onecycle requires warmup_steps = 0", result.stderr)
            self.assertFalse(output.exists())


    def test_legacy_cosine_checkpoint_resume_is_exact(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("requires BF16 CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            modern_config, legacy_config = root / "modern.toml", root / "legacy.toml"
            resumed, control = root / "resumed", root / "control"
            _write_suite(suite)
            modern_text = _training_config(
                suite, scheduler="cosine", max_steps=3, warmup_steps=1)
            modern_config.write_text(modern_text)

            _run_training(modern_config, resumed, "--stop-after", 1)
            checkpoint_path = resumed / "step-000001" / "training_state.pt"
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state["identity"]["config"].pop("scheduler")
            torch.save(state, checkpoint_path)
            legacy_text = modern_text.replace('scheduler = "cosine"\n', "")
            legacy_config.write_text(legacy_text)
            (resumed / "config.toml").write_text(legacy_text)

            _run_training(
                legacy_config, resumed, "--resume", resumed / "step-000001")
            _run_training(modern_config, control)

            left, right = resumed / "step-000003", control / "step-000003"
            left_weights = load_file(left / "adapter_model.safetensors")
            right_weights = load_file(right / "adapter_model.safetensors")
            self.assertEqual(left_weights.keys(), right_weights.keys())
            self.assertTrue(all(
                torch.equal(left_weights[key], right_weights[key])
                for key in left_weights
            ))
            left_state = torch.load(
                left / "training_state.pt", map_location="cpu", weights_only=False)
            right_state = torch.load(
                right / "training_state.pt", map_location="cpu", weights_only=False)
            self.assertTrue(_equal(left_state, right_state))
            ready = [
                json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()
                if json.loads(line)["event"] == "ready"
            ]
            self.assertEqual(ready[-1]["config"]["scheduler"], "cosine")


    def test_onecycle_augmented_epoch_rollover_resume_is_exact_and_reports_diagnostics(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("requires BF16 CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, config = root / "suite", root / "training.toml"
            resumed, control = root / "resumed", root / "control"
            _write_suite(suite)
            config.write_text(_training_config(suite))

            _run_training(config, resumed, "--stop-after", 4)
            _run_training(config, resumed, "--resume", resumed / "step-000004")
            _run_training(config, control)

            left, right = resumed / "step-000020", control / "step-000020"
            left_weights = load_file(left / "adapter_model.safetensors")
            right_weights = load_file(right / "adapter_model.safetensors")
            self.assertEqual(left_weights.keys(), right_weights.keys())
            self.assertTrue(all(torch.equal(left_weights[key], right_weights[key]) for key in left_weights))
            left_state = torch.load(left / "training_state.pt", map_location="cpu", weights_only=False)
            right_state = torch.load(right / "training_state.pt", map_location="cpu", weights_only=False)
            self.assertTrue(_equal(left_state, right_state))
            self.assertGreaterEqual(left_state["epoch"], 19)

            events = [json.loads(line) for line in (control / "metrics.jsonl").read_text().splitlines()]
            ready = next(row for row in events if row["event"] == "ready")
            self.assertEqual(ready["train_probe_record_ids"], ["tiny/one"])
            self.assertEqual(ready["config"]["scheduler"], "onecycle")
            trains = [row for row in events if row["event"] == "train"]
            self.assertEqual(set(trains[0]["gradient_groups"]), {"backbone", "head"})
            backbone_lrs = [row["gradient_groups"]["backbone"]["lr"] for row in trains]
            head_lrs = [row["gradient_groups"]["head"]["lr"] for row in trains]
            self.assertAlmostEqual(backbone_lrs[0], 0.0002 / 25, places=15)
            self.assertAlmostEqual(head_lrs[0], 0.0001 / 25, places=15)
            self.assertAlmostEqual(max(backbone_lrs), 0.0002, places=15)
            self.assertAlmostEqual(max(head_lrs), 0.0001, places=15)
            self.assertAlmostEqual(backbone_lrs[-1], 0.0002 / 25 / 10_000, places=15)
            self.assertAlmostEqual(head_lrs[-1], 0.0001 / 25 / 10_000, places=15)
            diagnostics = [row for row in events if row["event"] in {"train_probe", "development"}]
            self.assertTrue(any(row["event"] == "train_probe" for row in diagnostics))
            development = next(row for row in diagnostics if row["event"] == "development")
            self.assertEqual(development["selection"]["subset"], "clean")
            self.assertEqual(development["selection"]["metric"], "macro_nll")
            self.assertIn("choice_reorder", development["report"])

if __name__ == "__main__":
    unittest.main()
