"""Behavioral tests for fixed training diagnostics."""

import json
import math
import random
import types
import unittest

import numpy as np
import torch

from haidass_kev_train.evaluation.diagnostics import select_probe, training_diagnostics


class _Tokenizer:
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
        return [self._markers[text]] if text in self._markers else self(text, add_special_tokens=add_special_tokens).input_ids

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[100 + sum(text.encode()) % 100])


class _TinyDecisionModel(torch.nn.Module):
    """A real Module implementing DecisionModel's public batch interface."""

    probabilities = {
        "rel/a": {"accept": 0.1, "reject": 0.9},
        "rel/b": {"accept": 0.8, "reject": 0.2},
        "inv/a": {"accept": 0.1, "reject": 0.9},
        "inv/b": {"accept": 0.8, "reject": 0.2},
        "partial/a": {"accept": 0.3, "reject": 0.7},
        "score": {"0": 0.6, "1": 0.4},
        "solo": {"accept": 0.25, "reject": 0.75},
        "legacy/a": {"false": 0.2, "true": 0.8},
        "legacy/b": {"false": 0.9, "true": 0.1},
    }

    def forward(self, batch):
        logits = torch.zeros(batch.option_mask.shape, device=batch.option_mask.device)
        for i, questions in enumerate(batch.metadata):
            for j, meta in enumerate(questions):
                for k, key in enumerate(meta["option_keys"]):
                    logits[i, j, k] = math.log(self.probabilities[meta["record_id"]][key])
        return logits.masked_fill(~batch.option_mask, float("-inf"))


def _choice(record_id, label, pair_id=None, sibling=None, pair_kind=None, *, variant="clean", reverse=False):
    criteria = {"reject": None, "accept": None} if reverse else {"accept": None, "reject": None}
    meta = {"id": record_id, "group_id": record_id.rsplit("/", 1)[0], "variant": variant, "source": "fixture"}
    if pair_id is not None:
        meta.update(pair_id=pair_id, sibling=sibling, pair_kind=pair_kind)
    return {
        "state": record_id,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "decide",
                "criteria": criteria,
                "label": label,
                "src": "policy",
            }
        },
        "_meta": meta,
    }


class DiagnosticsContracts(unittest.TestCase):
    def test_probe_selection_is_group_aware_and_order_independent(self):
        records = [
            {"state": "b1", "_meta": {"id": "b/1", "group_id": "b"}},
            {"state": "a1", "_meta": {"id": "a/1", "group_id": "a"}},
            {"state": "b2", "_meta": {"id": "b/2", "group_id": "b"}},
            {"state": "c1", "_meta": {"id": "c/1", "group_id": "c"}},
        ]

        selected = select_probe(records, count=2, seed=19)
        selected_reversed = select_probe(list(reversed(records)), count=2, seed=19)

        self.assertEqual([row["_meta"]["id"] for row in selected], [row["_meta"]["id"] for row in selected_reversed])
        groups = {row["_meta"]["group_id"] for row in selected}
        self.assertEqual(len(groups), 2)
        self.assertTrue(all((row in selected) == (row["_meta"]["group_id"] in groups) for row in records))

    def test_public_report_uses_official_pairs_and_semantic_option_keys(self):
        records = [
            _choice("rel/a", "reject", "relevant", "a", "relevant"),
            _choice("rel/b", "accept", "relevant", "b", "relevant", reverse=True),
            _choice("inv/a", "reject", "invariant", "a", "irrelevant"),
            _choice("inv/b", "reject", "invariant", "b", "irrelevant", reverse=True),
            _choice("partial/a", "reject", "partial", "a", "relevant"),
            {
                "state": "score",
                "questions": {
                    "rating": {
                        "type": "score",
                        "instructions": "rate",
                        "criteria": ["low", "high"],
                        "label": 0,
                        "src": "ordinal",
                    }
                },
                "_meta": {"id": "score", "group_id": "score", "variant": "augmented", "source": "fixture"},
            },
        ]
        model = _TinyDecisionModel().train()
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        report = training_diagnostics(model, _Tokenizer(), records, batch_size=2, device="cpu", max_packed=128)

        self.assertEqual(report["all"]["count"], 6)
        self.assertEqual(report["clean"]["count"], 5)
        self.assertEqual(report["tasks"]["policy"]["count"], 5)
        self.assertEqual(
            report["paired_flip"],
            {
                "relevant": {"n": 1, "incomplete": 1, "flip_rate": 1.0, "both_correct_rate": 1.0},
                "invariant": {"n": 1, "incomplete": 0, "invariance_rate": 0.0, "both_correct_rate": 0.0},
                "untyped_records": 0,
            },
        )
        self.assertEqual(report["choice_reorder"], {"n": 5, "flip_rate": 0.0, "mean_max_delta": 0.0})
        json.dumps(report)
        self.assertTrue(model.training)
        self.assertEqual(random.getstate(), python_state)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        self.assertEqual(after_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_report_distinguishes_no_pairs_and_rejects_duplicate_record_ids(self):
        record = _choice("solo", "reject")
        report = training_diagnostics(
            _TinyDecisionModel(), _Tokenizer(), [record], batch_size=1, device="cpu", max_packed=128
        )

        self.assertEqual(
            report["paired_flip"],
            {
                "relevant": {"n": 0, "incomplete": 0, "flip_rate": None, "both_correct_rate": None},
                "invariant": {"n": 0, "incomplete": 0, "invariance_rate": None, "both_correct_rate": None},
                "untyped_records": 0,
            },
        )
        with self.assertRaisesRegex(ValueError, "duplicate diagnostic record identity 'solo'"):
            training_diagnostics(
                _TinyDecisionModel(), _Tokenizer(), [record, record], batch_size=1, device="cpu", max_packed=128
            )


    def test_explicit_legacy_pairs_without_pair_kind_use_target_keys(self):
        records = [
            {
                "state": "a",
                "questions": {"decision": {"type": "noul", "instructions": "decide", "label": True, "src": "policy"}},
                "_meta": {
                    "id": "legacy/a",
                    "group_id": "legacy",
                    "variant": "clean",
                    "pair_id": "legacy",
                    "sibling": "a",
                },
            },
            {
                "state": "b",
                "questions": {"decision": {"type": "noul", "instructions": "decide", "label": False, "src": "policy"}},
                "_meta": {
                    "id": "legacy/b",
                    "group_id": "legacy",
                    "variant": "clean",
                    "pair_id": "legacy",
                    "sibling": "b",
                },
            },
        ]

        paired = training_diagnostics(
            _TinyDecisionModel(), _Tokenizer(), records, batch_size=2, device="cpu", max_packed=128
        )["paired_flip"]

        self.assertEqual(
            paired,
            {
                "relevant": {"n": 1, "incomplete": 0, "flip_rate": 1.0, "both_correct_rate": 1.0},
                "invariant": {"n": 0, "incomplete": 0, "invariance_rate": None, "both_correct_rate": None},
                "untyped_records": 0,
            },
        )

if __name__ == "__main__":
    unittest.main()
