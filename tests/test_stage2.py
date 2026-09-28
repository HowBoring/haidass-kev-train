"""Focused behavioral checks for Stage 2 objectives, data, and resumable streams."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from safetensors.torch import load_file
import torch

from haidass_kev_train.training.objectives import proper_loss, rlcd_loss
from haidass_kev_train.training.stage2 import _ramp, _take
from haidass_kev_train.training.typed_decisions import SPLIT_COUNTS, WORKFLOWS, split_rows


def _batch(option_mask, question_mask, targets, question_types):
    metadata = [
        [{"question_type": question_types[row][column]} for column in range(len(question_types[row]))]
        for row in range(len(question_types))
    ]
    return SimpleNamespace(
        option_mask=torch.tensor(option_mask, dtype=torch.bool),
        question_mask=torch.tensor(question_mask, dtype=torch.bool),
        target_probs=torch.tensor(targets, dtype=torch.float32),
        metadata=metadata,
    )


def _one_question_batch():
    return _batch([[[1, 1, 1]]], [[1]], [[[1.0, 0.0, 0.0]]], [["choice"]])


def _typed_rows():
    questions = {
        "choice_a": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}},
        "choice_b": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}},
        "noul": {"type": "noul", "instructions": "yes?", "criteria": {"false": "no", "true": "yes"}},
        "score_a": {"type": "score", "instructions": "rate", "criteria": ["low", "high"]},
        "score_b": {"type": "score", "instructions": "rate", "criteria": ["low", "high"]},
    }
    gold = {
        name: {
            "type": question["type"],
            "probabilities": ({"a": 0.25, "b": 0.75} if question["type"] == "choice" else
                              {"false": 0.4, "true": 0.6} if question["type"] == "noul" else
                              {"0": 0.3, "1": 0.7}),
        }
        for name, question in questions.items()
    }
    return [
        {
            "id": f"tr_{workflow}_{index:06d}",
            "workflow": workflow,
            "split": "train",
            "state": json.dumps({"case": index}),
            "questions": json.dumps(questions),
            "gold": json.dumps(gold),
            "factors": "{}",
            "label_agreement": "{}",
            "n_questions": 5,
        }
        for workflow in WORKFLOWS
        for index in range(300)
    ]


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


def _decision_record(record_id: str, *, singleton: bool = False) -> dict:
    criteria = {"only": "the only valid outcome"} if singleton else {"left": "choose left", "right": "choose right"}
    target = {"only": 1.0} if singleton else {"left": 0.75, "right": 0.25}
    return {
        "state": {"request": record_id},
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "Choose an outcome.",
                "criteria": criteria,
                "target": target,
                "src": "tiny",
            },
        },
        "_meta": {
            "id": record_id,
            "group_id": record_id,
            "variant": "clean",
            "source": "tiny",
        },
    }


def _write_decision_suite(path: Path, *, singleton: bool = False) -> None:
    path.mkdir()
    files = {}
    for split in ("train", "development"):
        record = _decision_record(f"tiny/{split}", singleton=singleton)
        payload = (json.dumps(record) + "\n").encode()
        (path / f"{split}.jsonl").write_bytes(payload)
        files[f"{split}.jsonl"] = {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}
    (path / "manifest.json").write_text(json.dumps({"files": files}))

def _sft_config(suite: Path) -> str:
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
max_steps = 20
warmup_steps = 0
scheduler = "onecycle"
eval_interval = 10
checkpoint_interval = 1
max_packed = 256
eval_batch_size = 1
max_grad_norm = 1.0
training_mode = "lora"
probe_groups = 1
development_selection = "clean"

[augmentation]
shuffle = false
p_none = 0.0
p_none_distract = 0.0
p_distract = 0.0
p_none_pair = 0.0
"""


def _stage2_config(typed_suite: Path, replay_suite: Path, mode: str) -> str:
    return f"""\
base_path = "models/base/haidass1.5-143m"
typed_suite_path = "{typed_suite}"
replay_suite_path = "{replay_suite}"
replay_sources = ["tiny"]
seed = 17
rl_seed = 23
batch_size = 1
replay_batch_size = 1
gradient_accumulation = 1
learning_rate = 0.00001
head_learning_rate = 0.00002
weight_decay = 0.01
max_steps = 2
warmup_steps = 0
eval_interval = 2
checkpoint_interval = 1
max_packed = 256
eval_batch_size = 1
max_grad_norm = 1.0
training_mode = "inherit"
mode = "{mode}"
ce_weight = 1.0
objective_weight = 0.1
objective_ramp_steps = 0
replay_weight = 0.25
group_size = 4
sigma = 0.1
normalize_advantage = true
spherical_weight = 0.5
rps_weight = 1.0
log_floor = -9.21
"""


def _run_module(module: str, *arguments) -> None:
    command = [sys.executable, "-m", module, *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(f"{command} failed:\n{result.stdout}\n{result.stderr}")


class ObjectiveTests(unittest.TestCase):
    def test_ordinal_rps_is_normalized_and_only_applied_to_scores(self):
        batch = _batch(
            [[[1, 1, 1], [1, 1, 1]]],
            [[1, 1]],
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
            [["choice", "score"]],
        )
        logits = torch.tensor([[[2.0, 0.0, -1.0], [2.0, 0.0, -1.0]]], requires_grad=True)
        losses, valid = proper_loss(logits, batch, spherical_weight=0.0, rps_weight=1.0, log_floor=-100.0)
        probabilities = logits[0, 0].softmax(0)
        target = batch.target_probs[0, 0]
        expected_rps = (probabilities.cumsum(0)[:-1] - target.cumsum(0)[:-1]).square().mean()
        self.assertTrue(torch.equal(valid, torch.tensor([[True, True]])))
        self.assertTrue(torch.allclose(losses[0, 1] - losses[0, 0], expected_rps))

    def test_rlcd_masks_padding_and_singletons_and_is_shift_invariant(self):
        batch = _batch(
            [
                [[1, 1, 1], [1, 0, 0]],
                [[0, 0, 0], [1, 1, 0]],
            ],
            [[1, 1], [0, 1]],
            [
                [[0.7, 0.2, 0.1], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.25, 0.75, 0.0]],
            ],
            [["choice", "choice"], [None, "score"]],
        )
        logits = torch.tensor(
            [[[0.3, -0.1, 0.8], [2.0, float("-inf"), float("-inf")]],
             [[0.0, 0.0, 0.0], [-0.4, 0.2, float("-inf")]]],
            requires_grad=True,
        )
        shifted = logits.detach().clone()
        shifted[0, 0, :3] += 17.0
        shifted[0, 1, 0] -= 9.0
        shifted[1, 1, :2] += 3.5
        shifted.requires_grad_()
        first = torch.Generator().manual_seed(8)
        second = torch.Generator().manual_seed(8)
        values, valid = rlcd_loss(logits, batch, group_size=32, sigma=0.2, generator=first)
        shifted_values, shifted_valid = rlcd_loss(shifted, batch, group_size=32, sigma=0.2, generator=second)
        values.sum().backward()
        shifted_values.sum().backward()
        expected = torch.tensor([[True, False], [False, True]])
        self.assertTrue(torch.equal(valid, expected))
        self.assertTrue(torch.equal(shifted_valid, expected))
        self.assertTrue(torch.equal(values[~valid], torch.zeros_like(values[~valid])))
        self.assertTrue(torch.allclose(values, shifted_values, atol=2e-5, rtol=2e-5))
        finite = batch.option_mask & batch.question_mask.unsqueeze(-1)
        self.assertTrue(torch.isfinite(logits.grad[finite]).all())
        self.assertTrue(torch.allclose(logits.grad[finite], shifted.grad[finite], atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.equal(logits.grad[~finite], torch.zeros_like(logits.grad[~finite])))

    def test_rlcd_detached_samples_move_toward_rewarded_option(self):
        batch = _one_question_batch()
        logits = torch.zeros((1, 1, 3), requires_grad=True)
        generator = torch.Generator().manual_seed(19)
        values, valid = rlcd_loss(
            logits,
            batch,
            group_size=8192,
            sigma=0.3,
            spherical_weight=0.0,
            rps_weight=0.0,
            log_floor=-100.0,
            normalize_advantage=False,
            generator=generator,
        )
        values[valid].mean().backward()
        self.assertLess(float(logits.grad[0, 0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 0, 1]), 0.0)
        self.assertGreater(float(logits.grad[0, 0, 2]), 0.0)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_rl_noise_and_optimizer_state_resume_exactly(self):
        batch = _one_question_batch()

        def update(parameter, optimizer, generator):
            optimizer.zero_grad(set_to_none=True)
            values, valid = rlcd_loss(parameter, batch, group_size=64, sigma=0.2, generator=generator)
            values[valid].mean().backward()
            optimizer.step()

        control = torch.nn.Parameter(torch.zeros((1, 1, 3)))
        control_optimizer = torch.optim.AdamW([control], lr=0.01)
        control_generator = torch.Generator().manual_seed(5)
        update(control, control_optimizer, control_generator)

        resumed = torch.nn.Parameter(control.detach().clone())
        resumed_optimizer = torch.optim.AdamW([resumed], lr=0.01)
        resumed_optimizer.load_state_dict(copy.deepcopy(control_optimizer.state_dict()))
        resumed_generator = torch.Generator()
        resumed_generator.set_state(control_generator.get_state())

        update(control, control_optimizer, control_generator)
        update(resumed, resumed_optimizer, resumed_generator)
        self.assertTrue(torch.equal(control, resumed))
        self.assertTrue(torch.equal(control_generator.get_state(), resumed_generator.get_state()))
        control_state = next(iter(control_optimizer.state.values()))
        resumed_state = next(iter(resumed_optimizer.state.values()))
        self.assertTrue(all(torch.equal(control_state[key], resumed_state[key]) for key in control_state))


class Stage2DataTests(unittest.TestCase):
    def test_canonical_workflow_stratified_case_split_preserves_soft_targets(self):
        splits = split_rows(_typed_rows(), seed=42)
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 960, "development": 120, "calibration": 120})
        identities = {
            name: {record["_meta"]["group_id"] for record in records}
            for name, records in splits.items()
        }
        self.assertFalse(identities["train"] & identities["development"])
        self.assertFalse(identities["train"] & identities["calibration"])
        self.assertFalse(identities["development"] & identities["calibration"])
        for name, records in splits.items():
            counts = {workflow: 0 for workflow in WORKFLOWS}
            for record in records:
                counts[record["_meta"]["workflow"]] += 1
                self.assertEqual(len(record["questions"]), 5)
                self.assertTrue(all("target" in question for question in record["questions"].values()))
            self.assertEqual(set(counts.values()), {SPLIT_COUNTS[name]})
        repeated = split_rows(list(reversed(_typed_rows())), seed=42)
        self.assertEqual(
            [[record["_meta"]["id"] for record in splits[name]] for name in splits],
            [[record["_meta"]["id"] for record in repeated[name]] for name in repeated],
        )

    def test_data_stream_cursor_resume_matches_uninterrupted_order(self):
        data = list(range(7))
        state = {"epoch": 0, "cursor": 0}
        prefix = _take(data, state, 5, seed=42, stream=1)
        saved = dict(state)
        suffix = _take(data, state, 11, seed=42, stream=1)
        resumed = _take(data, saved, 11, seed=42, stream=1)
        control_state = {"epoch": 0, "cursor": 0}
        self.assertEqual(prefix + suffix, _take(data, control_state, 16, seed=42, stream=1))
        self.assertEqual(suffix, resumed)
        self.assertEqual(state, saved)
        self.assertEqual((_ramp(0, 0.1, 50), _ramp(25, 0.1, 50), _ramp(50, 0.1, 50)), (0.0, 0.05, 0.1))


class Stage2CudaIntegrationTests(unittest.TestCase):
    def test_real_handoff_singleton_logging_and_exact_resume(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("requires BF16 CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_suite = root / "replay-suite"
            typed_suite = root / "typed-suite"
            singleton_suite = root / "singleton-suite"
            _write_decision_suite(replay_suite)
            _write_decision_suite(typed_suite)
            _write_decision_suite(singleton_suite, singleton=True)

            sft_config = root / "sft.toml"
            sft_output = root / "sft"
            sft_config.write_text(_sft_config(replay_suite))
            _run_module(
                "haidass_kev_train.training.sft",
                "--config", sft_config,
                "--output", sft_output,
                "--stop-after", 1,
            )
            parent = sft_output / "step-000001"
            parent_state = torch.load(parent / "training_state.pt", map_location="cpu", weights_only=False)
            parent_betas = tuple(parent_state["optimizer"]["param_groups"][0]["betas"])
            self.assertNotEqual(parent_betas, (0.9, 0.999))

            b_config = root / "b.toml"
            b_output = root / "b"
            b_config.write_text(_stage2_config(typed_suite, replay_suite, "B"))
            _run_module(
                "haidass_kev_train.training.stage2",
                "--config", b_config,
                "--output", b_output,
                "--parent", parent,
                "--stop-after", 1,
            )
            b_state = torch.load(b_output / "step-000001" / "training_state.pt", map_location="cpu", weights_only=False)
            self.assertEqual(tuple(b_state["optimizer"]["param_groups"][0]["betas"]), (0.9, 0.999))
            b_train = next(
                row for row in map(json.loads, (b_output / "metrics.jsonl").read_text().splitlines())
                if row["event"] == "train"
            )
            self.assertEqual(b_train["weights"], {"ce": 1.0, "objective": 0.0, "replay_ce": 0.25})

            singleton_config = root / "singleton-d.toml"
            singleton_output = root / "singleton-d"
            singleton_config.write_text(_stage2_config(singleton_suite, replay_suite, "D"))
            _run_module(
                "haidass_kev_train.training.stage2",
                "--config", singleton_config,
                "--output", singleton_output,
                "--parent", parent,
            )
            singleton_train = [
                row for row in map(json.loads, (singleton_output / "metrics.jsonl").read_text().splitlines())
                if row["event"] == "train"
            ]
            self.assertEqual([row["step"] for row in singleton_train], [1, 2])
            self.assertEqual(singleton_train[1]["questions"]["objective"], 0)
            self.assertIsNone(singleton_train[1]["objective"])

            d_config = root / "d.toml"
            control_output = root / "d-control"
            resumed_output = root / "d-resumed"
            d_config.write_text(_stage2_config(typed_suite, replay_suite, "D"))
            _run_module(
                "haidass_kev_train.training.stage2",
                "--config", d_config,
                "--output", control_output,
                "--parent", parent,
            )
            _run_module(
                "haidass_kev_train.training.stage2",
                "--config", d_config,
                "--output", resumed_output,
                "--parent", parent,
                "--stop-after", 1,
            )
            _run_module(
                "haidass_kev_train.training.stage2",
                "--config", d_config,
                "--output", resumed_output,
                "--resume", resumed_output / "step-000001",
            )

            control_checkpoint = control_output / "step-000002"
            resumed_checkpoint = resumed_output / "step-000002"
            control_weights = load_file(control_checkpoint / "adapter_model.safetensors")
            resumed_weights = load_file(resumed_checkpoint / "adapter_model.safetensors")
            self.assertEqual(control_weights.keys(), resumed_weights.keys())
            self.assertTrue(all(
                torch.equal(control_weights[name], resumed_weights[name])
                for name in control_weights
            ))
            control_state = torch.load(
                control_checkpoint / "training_state.pt", map_location="cpu", weights_only=False
            )
            resumed_state = torch.load(
                resumed_checkpoint / "training_state.pt", map_location="cpu", weights_only=False
            )
            for key in (
                "optimizer",
                "scheduler",
                "global_step",
                "current_stream",
                "replay_stream",
                "rng",
                "rl_generator_state",
            ):
                self.assertTrue(_equal(control_state[key], resumed_state[key]), key)


if __name__ == "__main__":
    unittest.main()
