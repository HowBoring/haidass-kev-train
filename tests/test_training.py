from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest

import numpy as np
from safetensors.torch import load_file
import torch

from haidass_kev_train.data.canonical import (
    DEFAULT_K_PROBABILITIES,
    K_VALUES,
    check_k_probabilities,
    evaluation_views,
    load_canonical_suite,
    preflight,
    training_view,
    validate_record,
)
from haidass_kev_train.data.packing import check_group_integrity, encode_record
from haidass_kev_train.evaluation.metrics import predict, summarize
from haidass_kev_train.model.decision import build_model, load_artifact
from haidass_kev_train.training.sft import configure_runtime


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

class _FakeTokenizer:
    """Marker-aware tokenizer: one token per character, so packed length tracks text length."""

    _markers = {
        "<|object_ref_start|>": 6,
        "<|object_ref_end|>": 7,
        "<|box_start|>": 8,
        "<|box_end|>": 9,
        "<|quad_start|>": 10,
    }

    def __len__(self):
        return 64000

    def encode(self, text, add_special_tokens=False):
        if text in self._markers:
            return [self._markers[text]]
        return self(text, add_special_tokens=add_special_tokens).input_ids

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[ord(char) for char in text])


def _raw_source(record_id, source, question, gold, state="", options=()):
    lines = [f"{source}|{record_id}"]
    if state:
        lines.append(f"C: {state}")
    lines.append(f"Q: {question}")
    lines.append(f"A: {gold}")
    lines.extend(f"O: {option}" for option in options)
    return "\n".join(lines) + "\n"


def _span(raw, text, marker, start=0):
    index = raw.index(text, raw.index(marker, start))
    return [index, index + len(text)]


def _canonical_record(record_id, group_id, source, question, gold, distractors, *, state="",
                      validation="programmatic", uid=None, mcq=False, url=None, snapshot_type=None):
    old_options = [gold, *list(distractors)[:3]] if mcq else []
    raw = _raw_source(record_id, source, question, gold, state=state, options=old_options)
    source_ref = {
        "path": f"{source}/shard-00.jsonl",
        "line": 7,
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "question_span": _span(raw, question, "Q: "),
        "answer_span": _span(raw, gold, "A: "),
    }
    if state:
        source_ref["state_span"] = _span(raw, state, "C: ")
    if uid is not None:
        source_ref["uid"] = uid
    if mcq:
        source_ref["mcq"] = True
        start = raw.index("O: ")
        source_ref["option_spans"] = []
        for option in old_options:
            span = _span(raw, option, "O: ", start)
            source_ref["option_spans"].append(span)
            start = span[1]
    if url is not None:
        source_ref["url"] = url
    if snapshot_type is not None:
        source_ref["snapshot_type"] = snapshot_type
    return {
        "source": source,
        "state": state,
        "question": question,
        "gold": gold,
        "distractors": list(distractors),
        "_meta": {
            "id": record_id,
            "group_id": group_id,
            "source": source,
            "source_ref": source_ref,
            "validation": validation,
        },
    }

def _trace_patch(record, **changes):
    ref = copy.deepcopy(record["_meta"]["source_ref"])
    for key, value in changes.items():
        if value is ...:
            del ref[key]
        else:
            ref[key] = value
    return ref


def _write_canonical_suite(path, train_records, development_records):
    path.mkdir(exist_ok=True)
    files = {}
    for split, records in (("train", train_records), ("development", development_records)):
        payload = b"".join((json.dumps(record, ensure_ascii=False) + "\n").encode() for record in records)
        (path / f"{split}.jsonl").write_bytes(payload)
        files[f"{split}.jsonl"] = {"sha256": hashlib.sha256(payload).hexdigest(), "records": len(records)}
    (path / "manifest.json").write_text(json.dumps({"files": files}))


def _canonical_config(suite, *, max_steps=7, eval_interval=7, checkpoint_interval=4, data_format='"canonical_choice_v1"',
                      shuffle="false", k_probabilities="[0.10, 0.20, 0.30, 0.25, 0.15]"):
    return f"""\
base_path = "models/base/haidass1.5-143m"
suite_path = "{suite}"
seed = 17
batch_size = 1
gradient_accumulation = 1
learning_rate = 0.0002
head_learning_rate = 0.0001
weight_decay = 0.01
max_steps = {max_steps}
warmup_steps = 0
scheduler = "onecycle"
eval_interval = {eval_interval}
checkpoint_interval = {checkpoint_interval}
max_packed = 1024
eval_batch_size = 8
max_grad_norm = 1.0
training_mode = "lora"
probe_groups = 1
development_selection = "clean"
data_format = {data_format}

[augmentation]
shuffle = {shuffle}
p_none = 0.0
p_none_distract = 0.0
p_distract = 0.0
p_none_pair = 0.0

[canonical]
k_probabilities = {k_probabilities}
"""


def _canonical_suite_records():
    train = [
        _canonical_record("tiny/t1", "tiny/doc1", "tiny", "Which colour is the car?", "red",
                          ["blue", "green", "yellow", "black", "white"], state="A car is parked outside."),
        _canonical_record("tiny/t2", "tiny/doc2", "tiny", "How many days are in a week?", "seven",
                          ["five", "six", "eight", "ten", "nine"], state=""),
        _canonical_record("tiny/t3", "tiny/doc3", "tiny", "What is the capital of France?", "Paris",
                          ["London", "Berlin", "Madrid", "Rome", "Vienna"], state=""),
    ]
    development = [
        _canonical_record("tiny/d1", "tiny/doc4", "tiny", "Which animal barks?", "dog",
                          ["cat", "cow", "duck", "sheep", "goat"], state=""),
        _canonical_record("tiny/d2", "tiny/doc5", "tiny", "What is 2 plus 2?", "4",
                          ["3", "5", "6", "22", "0"], state=""),
    ]
    return train, development


class CanonicalRecordTests(unittest.TestCase):
    def setUp(self):
        self.record = _canonical_record("ufw-zh/abc", "ufw-zh/doc1", "ufw-zh", "下列哪项正确?", "答案甲",
                                        ["错一", "错二", "错三", "错四", "错五"], state="原文背景内容。",
                                        uid="ufw-uid-123")

    def test_valid_record_passes_with_legitimately_empty_state(self):
        record = _canonical_record("tiny/a", "tiny/g", "tiny", "Self-contained question?", "yes",
                                   ["no", "maybe", "never", "always", "sometimes"], state="")
        validate_record(record)
        self.assertEqual(record["state"], "")

    def test_fixture_source_trace_is_verifiable(self):
        raw = _raw_source("ufw-zh/abc", "ufw-zh", "下列哪项正确?", "答案甲", state="原文背景内容。")
        ref = self.record["_meta"]["source_ref"]
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(), ref["sha256"])
        q0, q1 = ref["question_span"]
        a0, a1 = ref["answer_span"]
        s0, s1 = ref["state_span"]
        self.assertEqual(raw[q0:q1], self.record["question"])
        self.assertEqual(raw[a0:a1], self.record["gold"])
        self.assertEqual(raw[s0:s1], self.record["state"])
        validate_record(self.record)

    def test_ufw_source_trace_requires_uid_state_and_mcq_option_spans(self):
        validate_record(self.record)
        for patch in ({"uid": ...}, {"uid": " "}, {"state_span": ...}, {"mcq": "yes"}):
            bad = copy.deepcopy(self.record)
            bad["_meta"]["source_ref"] = _trace_patch(self.record, **patch)
            with self.subTest(patch):
                with self.assertRaises(ValueError):
                    validate_record(bad)

        mcq = _canonical_record("ufw-en/xyz", "ufw-en/doc9", "ufw-en", "Which is right?", "alpha",
                                ["beta", "gamma", "delta", "epsilon", "zeta"], state="Context.",
                                uid="ufw-uid-789", mcq=True)
        validate_record(mcq)
        raw = _raw_source("ufw-en/xyz", "ufw-en", "Which is right?", "alpha", state="Context.",
                          options=["alpha", "beta", "gamma", "delta"])
        ref = mcq["_meta"]["source_ref"]
        self.assertEqual([raw[o0:o1] for o0, o1 in ref["option_spans"]], ["alpha", "beta", "gamma", "delta"])
        bad = copy.deepcopy(mcq)
        del bad["_meta"]["source_ref"]["option_spans"]
        with self.assertRaises(ValueError):
            validate_record(bad)
        bad = copy.deepcopy(mcq)
        bad["_meta"]["source_ref"]["option_spans"] = [[0, 3], [2, 2]]
        with self.assertRaises(ValueError):
            validate_record(bad)

    def test_finemath_source_trace_preserves_url_or_snapshot_identity(self):
        record = _canonical_record("finemath/1", "finemath/g1", "finemath", "What is 1+1?", "2",
                                   ["1", "3", "4", "0", "11"], state="",
                                   url="https://example.org/problem/1", snapshot_type="latest")
        validate_record(record)
        record["_meta"]["source_ref"]["givens_span"] = [0, 5]
        validate_record(record)
        for patch in ({"url": " "}, {"snapshot_type": 7},
                      {"givens_span": [5, 5]}, {"state_span": [9, 2]}, {"state_span": "0-5"}):
            bad = copy.deepcopy(record)
            bad["_meta"]["source_ref"] = _trace_patch(record, **patch)
            with self.subTest(patch):
                with self.assertRaises(ValueError):
                    validate_record(bad)

        # The public loader rejects a malformed present state_span on any source.
        bad = copy.deepcopy(record)
        bad["_meta"]["source_ref"]["state_span"] = [9, 2]
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite"
            _write_canonical_suite(suite, [bad], [record])
            with self.assertRaisesRegex(ValueError, "state_span"):
                load_canonical_suite(suite, "train")

    def test_rejects_invalid_records(self):
        cases = []

        def mutated(description, **changes):
            record = copy.deepcopy(self.record)
            for dotted, value in changes.items():
                target, key = record, dotted
                if "." in dotted:
                    head, key = dotted.rsplit(".", 1)
                    target = record[head]
                if value is ...:
                    del target[key]
                else:
                    target[key] = value
            cases.append((description, record))

        mutated("missing gold", gold=...)
        mutated("blank gold", gold="  ")
        mutated("blank question", question="")
        mutated("non-string question", question=3)
        mutated("missing state", state=...)
        mutated("non-string state", state=["not", "text"])
        mutated("three distractors", distractors=["a", "b", "c"])
        mutated("six distractors", distractors=["a", "b", "c", "d", "e", "f"])
        mutated("duplicate distractors", distractors=["a", "b", "c", "d", "d"])
        mutated("gold duplicated as distractor", distractors=["答案甲", "b", "c", "d", "e"])
        mutated("whitespace-hidden duplicate", distractors=[" 答案甲 ", "b", "c", "d", "e"])
        mutated("persisted position label", label=2)
        mutated("persisted soft target", target=[1.0, 0.0, 0.0, 0.0, 0.0])
        mutated("persisted candidate set", candidates=["答案甲", "错一"])
        mutated("missing _meta", _meta=...)
        mutated("missing id", **{"_meta.id": ...})
        mutated("missing group_id", **{"_meta.group_id": ...})
        mutated("conflicting source", **{"_meta.source": "other"})
        mutated("missing source_ref", **{"_meta.source_ref": ...})
        mutated("source_ref without sha256", **{"_meta.source_ref": _trace_patch(self.record, sha256=...)})
        mutated("source_ref sha256 not hex", **{"_meta.source_ref": _trace_patch(self.record, sha256="x" * 64)})
        mutated("source_ref sha256 wrong length", **{"_meta.source_ref": _trace_patch(self.record, sha256="0" * 63)})
        mutated("source_ref missing path", **{"_meta.source_ref": _trace_patch(self.record, path=...)})
        mutated("source_ref blank path", **{"_meta.source_ref": _trace_patch(self.record, path=" ")})
        mutated("source_ref missing line", **{"_meta.source_ref": _trace_patch(self.record, line=...)})
        mutated("source_ref negative line", **{"_meta.source_ref": _trace_patch(self.record, line=-1)})
        mutated("source_ref missing question_span", **{"_meta.source_ref": _trace_patch(self.record, question_span=...)})
        mutated("source_ref empty span", **{"_meta.source_ref": _trace_patch(self.record, question_span=[4, 4])})
        mutated("source_ref inverted span", **{"_meta.source_ref": _trace_patch(self.record, answer_span=[9, 2])})
        mutated("source_ref non-integer span", **{"_meta.source_ref": _trace_patch(self.record, question_span=[0.0, 5])})
        mutated("source_ref missing answer_span", **{"_meta.source_ref": _trace_patch(self.record, answer_span=...)})
        mutated("missing validation path", **{"_meta.validation": ...})
        for description, record in cases:
            with self.subTest(description):
                with self.assertRaises(ValueError):
                    validate_record(record)
        with self.assertRaises(ValueError):
            validate_record(["not", "a", "record"])

    def test_default_k_distribution_is_valid_and_bad_distributions_rejected(self):
        self.assertEqual(check_k_probabilities(None), (0.10, 0.20, 0.30, 0.25, 0.15))
        self.assertEqual(check_k_probabilities([0, 0, 0, 0, 1]), (0, 0, 0, 0, 1))
        for bad in ([0.2] * 4, [0.5] * 5, [1.1, -0.1, 0.0, 0.0, 0.0], [float("nan")] * 5, "0.1,0.2"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    check_k_probabilities(bad)


class CanonicalViewTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _FakeTokenizer()
        self.record = _canonical_record("ufw-zh/abc", "ufw-zh/doc1", "ufw-zh", "Which option is right?", "gold answer",
                                        ["wrong one", "wrong two", "wrong three", "wrong four", "wrong five"],
                                        state="Some shared context.", uid="ufw-uid-456")

    def test_training_view_samples_k1_distractors_without_replacement_and_aligns_label(self):
        snapshot = copy.deepcopy(self.record)
        for k in K_VALUES:
            pinned = [1.0 if value == k else 0.0 for value in K_VALUES]
            view = training_view(self.record, seed=3, epoch=2, probabilities=pinned)
            question = view["questions"]["decision"]
            keys = list(question["criteria"])
            self.assertEqual(len(keys), k)
            self.assertEqual(len(set(keys)), k)
            self.assertEqual(question["label"], self.record["gold"])
            self.assertEqual(question["instructions"], self.record["question"])
            self.assertEqual(question["src"], self.record["source"])
            self.assertEqual(view["state"], self.record["state"])
            self.assertTrue(all(description is None for description in question["criteria"].values()))
            chosen = [key for key in keys if key != self.record["gold"]]
            self.assertEqual(len(chosen), k - 1)
            self.assertTrue(set(chosen) <= set(self.record["distractors"]))
            encoded = encode_record(view, self.tokenizer, max_packed=4096)
            target = encoded.target_probs[0]
            self.assertEqual(encoded.metadata[0]["option_keys"], keys)
            self.assertEqual(sum(target), 1.0)
            self.assertEqual(target[keys.index(self.record["gold"])], 1.0)
        self.assertEqual(self.record, snapshot)

    def test_training_view_ignores_global_rng_and_traversal_order(self):
        other = _canonical_record("ufw-zh/xyz", "ufw-zh/doc2", "ufw-zh", "Pick one.", "yes",
                                  ["no", "maybe", "never", "always", "sometimes"])
        first = {record["_meta"]["id"]: training_view(record, seed=11, epoch=2) for record in (self.record, other)}
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        second = {record["_meta"]["id"]: training_view(record, seed=11, epoch=2) for record in (other, self.record)}
        self.assertEqual(first, second)
        for view in first.values():
            self.assertEqual(set(view["questions"]["decision"]["criteria"]) & {view["questions"]["decision"]["label"]},
                             {view["questions"]["decision"]["label"]})

    def test_evaluation_views_are_five_fixed_k_views_with_unique_ids(self):
        views = evaluation_views(self.record, seed=5, purpose="development")
        self.assertEqual(len(views), len(K_VALUES))
        self.assertEqual([view["_meta"]["id"] for view in views],
                         [f"ufw-zh/abc/k{k}" for k in K_VALUES])
        for view, k in zip(views, K_VALUES):
            meta = view["_meta"]
            self.assertEqual((meta["canonical_id"], meta["group_id"], meta["source"], meta["variant"]),
                             ("ufw-zh/abc", "ufw-zh/doc1", "ufw-zh", "clean"))
            self.assertEqual(meta["k"], k)
            question = view["questions"]["decision"]
            keys = list(question["criteria"])
            self.assertEqual(len(keys), k)
            self.assertIn(self.record["gold"], keys)
            encoded = encode_record(view, self.tokenizer, max_packed=4096)
            self.assertEqual(encoded.target_probs[0][keys.index(self.record["gold"])], 1.0)
        self.assertEqual(views, evaluation_views(self.record, seed=5, purpose="development"))

    def test_preflight_rejects_overlength_and_marker_collision_without_truncation(self):
        long_record = _canonical_record("tiny/long", "tiny/doc9", "tiny", "q" * 300, "gold",
                                        ["d1", "d2", "d3", "d4", "d5"])
        with self.assertRaisesRegex(ValueError, "tiny/long"):
            preflight([long_record], self.tokenizer, max_packed=64)
        colliding = _canonical_record("tiny/mark", "tiny/doc9", "tiny", "question", "gold",
                                      ["<|box_start|>", "d2", "d3", "d4", "d5"])
        with self.assertRaisesRegex(ValueError, "tiny/mark"):
            preflight([colliding], self.tokenizer, max_packed=1024)
        preflight([self.record], self.tokenizer, max_packed=1024)

    def test_load_canonical_suite_verifies_manifest_contract_and_group_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite"
            train, development = _canonical_suite_records()
            _write_canonical_suite(path, train, development)
            loaded = load_canonical_suite(path, "train")
            self.assertEqual(len(loaded), 3)
            self.assertEqual([record["_meta"]["id"] for record in loaded], ["tiny/t1", "tiny/t2", "tiny/t3"])

            report = check_group_integrity(path, splits=("train", "development"))
            self.assertEqual(report["overlaps"], {"train|development": 0})

            broken = copy.deepcopy(train[0])
            broken["_meta"]["group_id"] = "tiny/doc4"
            _write_canonical_suite(path, [broken, *train[1:]], development)
            report = check_group_integrity(path, splits=("train", "development"))
            self.assertEqual(report["overlaps"]["train|development"], 1)

            _write_canonical_suite(path, train, development)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["files"]["train.jsonl"]["sha256"] = "0" * 64
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_canonical_suite(path, "train")

            _write_canonical_suite(path, [train[0], train[0]], development)
            with self.assertRaisesRegex(ValueError, "duplicate canonical ids"):
                load_canonical_suite(path, "train")

            kev_record = {"state": "legacy", "questions": {"q": {"type": "choice", "instructions": "i",
                                                                 "criteria": {"a": None}, "label": "a"}}}
            _write_canonical_suite(path, [kev_record], development)
            with self.assertRaises(ValueError):
                load_canonical_suite(path, "train")

            no_meta = {key: value for key, value in train[0].items() if key != "_meta"}
            _write_canonical_suite(path, [no_meta], development)
            with self.assertRaisesRegex(ValueError, "_meta"):
                load_canonical_suite(path, "train")


class CanonicalConfigTests(unittest.TestCase):
    """Config-level rejection happens before any CUDA requirement, so it runs anywhere."""

    def _reject(self, config_text, message):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "canonical.toml"
            config.write_text(config_text)
            command = [sys.executable, "-m", "haidass_kev_train.training.sft",
                       "--config", str(config), "--output", str(root / "out")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)

    def test_unknown_data_format_rejected(self):
        self._reject(_canonical_config("unused", data_format='"kev"'), "data_format")

    def test_legacy_augmentation_conflicts_with_canonical_format(self):
        self._reject(_canonical_config("unused", shuffle="true"), "augmentation")

    def test_invalid_k_distribution_rejected(self):
        self._reject(_canonical_config("unused", k_probabilities="[0.5, 0.5, 0.5, 0.5, 0.5]"), "k_probabilities")


class CanonicalSftCudaTests(unittest.TestCase):
    def test_canonical_epoch_rollover_resume_is_exact(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("requires BF16 CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, config = root / "suite", root / "training.toml"
            resumed, control = root / "resumed", root / "control"
            train_records, development_records = _canonical_suite_records()
            _write_canonical_suite(suite, train_records, development_records)
            config.write_text(_canonical_config(suite))

            _run_training(config, resumed, "--stop-after", 4)
            _run_training(config, resumed, "--resume", resumed / "step-000004")
            _run_training(config, control)

            left, right = resumed / "step-000007", control / "step-000007"
            left_weights = load_file(left / "adapter_model.safetensors")
            right_weights = load_file(right / "adapter_model.safetensors")
            self.assertEqual(left_weights.keys(), right_weights.keys())
            self.assertTrue(all(torch.equal(left_weights[key], right_weights[key]) for key in left_weights))
            left_state = torch.load(left / "training_state.pt", map_location="cpu", weights_only=False)
            right_state = torch.load(right / "training_state.pt", map_location="cpu", weights_only=False)
            self.assertTrue(_equal(left_state, right_state))
            self.assertEqual(left_state["epoch"], 2)

            events = [json.loads(line) for line in (control / "metrics.jsonl").read_text().splitlines()]
            ready = next(row for row in events if row["event"] == "ready")
            self.assertEqual(ready["train_records"], 3)
            self.assertEqual(ready["augmented_records"], 3)
            probe_ids = ready["train_probe_record_ids"]
            self.assertEqual(len(probe_ids), 5)
            self.assertEqual({view_id.rsplit("/k", 1)[1] for view_id in probe_ids}, {"2", "3", "4", "5", "6"})
            canonical = {view_id.rsplit("/k", 1)[0] for view_id in probe_ids}
            self.assertEqual(len(canonical), 1)
            self.assertTrue(canonical.pop() in {"tiny/t1", "tiny/t2", "tiny/t3"})
            trains = [row for row in events if row["event"] == "train"]
            self.assertTrue(all(math.isfinite(row["loss"]) for row in trains))
            for row in trains:
                groups = row["gradient_groups"]
                self.assertGreater(groups["backbone"]["grad_norm"], 0)
                self.assertGreater(groups["head"]["grad_norm"], 0)
            # LoRA B-factors are zero at initialization: nonzero values prove real updates.
            self.assertTrue(any(
                "lora_B" in key and bool(tensor.abs().sum() > 0) for key, tensor in left_weights.items()))
            self.assertTrue(any(row["event"] == "train_probe" for row in events))
            development = next(row for row in events if row["event"] == "development")
            self.assertEqual(development["selection"]["subset"], "clean")
            self.assertEqual(development["report"]["clean"]["count"], 2 * len(K_VALUES))

            # The evaluation logged right before the final checkpoint must reproduce from an
            # independent artifact reload on the same fixed views, device, and precision.
            # configure_runtime mirrors the trainer's seeding so build_model reproduces the
            # exact initialized parameters the run started from.
            configure_runtime(17)
            initial_model, _ = build_model("models/base/haidass1.5-143m", training_mode="lora")
            left_model, tokenizer = load_artifact(str(left), "models/base/haidass1.5-143m")
            right_model, _ = load_artifact(str(right), "models/base/haidass1.5-143m")
            views = [view for record in development_records
                     for view in evaluation_views(record, seed=17, purpose="development")]
            encoded = [encode_record(view, tokenizer, max_packed=1024) for view in views]
            left_rows = predict(left_model, encoded, batch_size=8, device="cuda")
            right_rows = predict(right_model, encoded, batch_size=8, device="cuda")
            self.assertEqual(left_rows, right_rows)
            reloaded = summarize(left_rows)
            for key in ("count", "accuracy", "nll", "macro_nll"):
                self.assertEqual(reloaded[key], development["report"]["clean"][key])

            # The pointer head really moved from its seeded initialization.
            initial_head = [parameter.detach().cpu() for parameter in initial_model.pointer_head.parameters()]
            trained_head = [parameter.detach().cpu() for parameter in left_model.pointer_head.parameters()]
            self.assertEqual(len(initial_head), len(trained_head))
            self.assertTrue(any(
                not torch.equal(initial, trained) for initial, trained in zip(initial_head, trained_head)))

    def test_canonical_resume_rejects_changed_suite(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("requires BF16 CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, config = root / "suite", root / "training.toml"
            resumed = root / "resumed"
            train_records, development_records = _canonical_suite_records()
            _write_canonical_suite(suite, train_records, development_records)
            config.write_text(_canonical_config(suite))
            _run_training(config, resumed, "--stop-after", 2)

            changed = copy.deepcopy(train_records)
            changed[0]["distractors"][0] = "violet"
            _write_canonical_suite(suite, changed, development_records)
            command = [sys.executable, "-m", "haidass_kev_train.training.sft",
                       "--config", str(config), "--output", str(resumed),
                       "--resume", str(resumed / "step-000002")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatch", result.stderr)

    def test_canonical_entry_rejects_invalid_suite(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            self.skipTest("requires BF16 CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_records, development_records = _canonical_suite_records()
            duplicate = copy.deepcopy(train_records[0])
            duplicate["distractors"][0] = duplicate["gold"]
            suite = root / "duplicate"
            _write_canonical_suite(suite, [duplicate, *train_records[1:]], development_records)
            config = root / "duplicate.toml"
            config.write_text(_canonical_config(suite))
            result = subprocess.run([sys.executable, "-m", "haidass_kev_train.training.sft",
                                     "--config", str(config), "--output", str(root / "out-duplicate")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("textually distinct", result.stderr)

            overlapping = copy.deepcopy(train_records[0])
            overlapping["_meta"]["group_id"] = "tiny/doc4"
            suite = root / "overlap"
            _write_canonical_suite(suite, [overlapping, *train_records[1:]], development_records)
            config = root / "overlap.toml"
            config.write_text(_canonical_config(suite))
            result = subprocess.run([sys.executable, "-m", "haidass_kev_train.training.sft",
                                     "--config", str(config), "--output", str(root / "out-overlap")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("both train and development", result.stderr)

            suite = root / "empty-dev"
            _write_canonical_suite(suite, train_records, [])
            config = root / "empty-dev.toml"
            config.write_text(_canonical_config(suite))
            result = subprocess.run([sys.executable, "-m", "haidass_kev_train.training.sft",
                                     "--config", str(config), "--output", str(root / "out-empty-dev")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Empty development", result.stderr)

if __name__ == "__main__":
    unittest.main()
