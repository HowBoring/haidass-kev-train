"""Constructed evidence exercises software gates; it is not a real human audit or GPU run."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from transformers import AutoTokenizer

import test_quality_gate as fixture
from haidass_kev_train.data.quality import quality_gate
from haidass_kev_train.data.canonical import load_canonical_suite
from haidass_kev_train.data.packing import encode_record
from haidass_kev_train.training.overfit import prepare_overfit
from haidass_kev_train.training.sft import _canonical_data, _config


class OverfitRecipeTests(unittest.TestCase):
    root: Path
    tokenizer: Path
    setUp = fixture.HumanGateSoftwareTests.setUp
    trial = fixture.HumanGateSoftwareTests.trial
    assessments = fixture.HumanGateSoftwareTests.assessments

    def later_suite(self, audited, *, total=135, absent=None, overlapping=False,
                    shared_group=False, changed_policy=False):
        original = json.loads((audited / "manifest.json").read_text())
        train = load_canonical_suite(audited, "train")
        development = copy.deepcopy(load_canonical_suite(audited, "development"))
        for index, row in enumerate(development):
            row["_meta"]["id"] = f"later/dev/{index:04d}"
            row["_meta"]["group_id"] = f"later/dev/group/{index:04d}"
            row["_meta"]["source_ref"]["line"] += 20000 + index
        templates = {row["source"]: row for row in train}
        if absent:
            templates.pop(absent)
        names = sorted(templates)
        rows = []
        for index in range(total):
            row = copy.deepcopy(templates[names[index % len(names)]])
            row["_meta"]["id"] = f"later/{index:04d}"
            row["_meta"]["group_id"] = f"later/group/{index:04d}"
            row["_meta"]["source_ref"]["line"] += 10000 + index
            rows.append(row)
        if shared_group:
            rows[3]["_meta"]["group_id"] = rows[0]["_meta"]["group_id"]
        if overlapping:
            rows[0]["_meta"]["group_id"] = development[0]["_meta"]["group_id"]
        suite = self.root / "later"
        suite.mkdir(exist_ok=True)
        for split, records in (("train", rows), ("development", development)):
            payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in records).encode()
            (suite / f"{split}.jsonl").write_bytes(payload)
            original["files"][f"{split}.jsonl"] = {"sha256": hashlib.sha256(payload).hexdigest(), "records": len(records)}
        if changed_policy:
            original["build"]["policy"]["model"] = "different-policy"
            original["build"]["model"] = "different-policy"
            original["build"]["policy_sha256"] = hashlib.sha256(json.dumps(
                original["build"]["policy"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()).hexdigest()
        (suite / "manifest.json").write_text(json.dumps(original, sort_keys=True))
        return suite

    def base_config(self):
        path = self.root / "base.toml"
        path.write_text(f'''base_path = "{self.tokenizer}"
        suite_path = "unused-superseded-by-recipe"
        seed = 7
        batch_size = 2
        gradient_accumulation = 2
        learning_rate = 0.0002
        weight_decay = 0.01
        max_steps = 12
        warmup_steps = 1
        scheduler = "cosine"
        eval_interval = 3
        checkpoint_interval = 3
        max_packed = 1024
        eval_batch_size = 4
        max_grad_norm = 1.0
        training_mode = "lora"
        probe_groups = 10
        development_selection = "clean"
        [augmentation]
        shuffle = true
        p_none = 0.0
        p_none_distract = 0.0
        p_distract = 0.0
        p_none_pair = 0.0
        ''')
        return path

    def evidence(self):
        audited, review, cases = self.trial(100)
        assessments = self.root / "assessments.jsonl"
        assessments.write_text("".join(json.dumps(row) + "\n" for row in self.assessments(cases)))
        report = quality_gate(audited, review, assessments, self.root / "gate-recipe.json")
        self.assertEqual(report["status"], "pass")
        return audited, review, assessments, self.root / "gate-recipe.json"

    def test_prepared_suite_exercises_public_sft_data_path_and_preserves_development(self):
        audited, review, assessments, quality = self.evidence()
        later = self.later_suite(audited, total=128, shared_group=True)
        plan = prepare_overfit(later, quality, audited, review, assessments,
                               self.base_config(), self.root / "recipe")
        config = _config(plan["overfit"]["config"])
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        train, dev, probe = _canonical_data(config, tokenizer)
        self.assertEqual((len(train), len(probe), len(dev)),
                         (128, 640, 5 * len(load_canonical_suite(later, "development"))))
        encoded = [encode_record(view, tokenizer, max_packed=config["max_packed"]) for view in probe]
        self.assertEqual({record.metadata[0]["k"] for record in encoded}, {2, 3, 4, 5, 6})
        self.assertEqual(len({record.metadata[0]["canonical_id"] for record in encoded}), 128)
        gold_by_id = {view["_meta"]["id"]: view["questions"]["decision"]["label"] for view in probe}
        for record in encoded:
            gold = gold_by_id[record.metadata[0]["record_id"]]
            keys = record.metadata[0]["option_keys"]
            self.assertEqual(record.target_probs[0], [float(key == gold) for key in keys])
        self.assertEqual(len({record["_meta"]["group_id"] for record in train}), 127)
        self.assertEqual(len({record["_meta"]["group_id"] for record in probe}), 127)
        self.assertEqual({record["source"] for record in train}, {"ufw-en", "ufw-zh", "finemath"})
        self.assertEqual((config["probe_groups"], config["data_format"], config["max_steps"]),
                         (128, "canonical_choice_v1", 12))
        self.assertFalse(config["augmentation"]["shuffle"])
        self.assertEqual((later / "development.jsonl").read_bytes(),
                         (self.root / "recipe/suite/development.jsonl").read_bytes())
        self.assertNotEqual(plan["quality"]["batch_manifest_sha256"],
                            hashlib.sha256((later / "manifest.json").read_bytes()).hexdigest())

    def test_missing_evidence_underflow_sources_overlap_and_policy_are_blocking(self):
        audited, review, assessments, quality = self.evidence()
        config = self.base_config()
        for change in ({"total": 127}, {"absent": "finemath"},
                       {"overlapping": True}, {"changed_policy": True}):
            with self.subTest(change=change):
                later = self.later_suite(audited, **change)
                with self.assertRaises(ValueError):
                    prepare_overfit(later, quality, audited, review, assessments,
                                    config, self.root / "blocked")
                self.assertFalse((self.root / "blocked").exists())
        config.write_text(config.read_text().replace("max_steps = 12", "max_steps = 501"))
        later = self.later_suite(audited)
        with self.assertRaises(ValueError):
            prepare_overfit(later, quality, audited, review, assessments, config, self.root / "blocked")
        config.write_text(config.read_text().replace("max_steps = 501", "max_steps = 12"))
        later = self.later_suite(audited)
        forged = self.root / "forged.json"
        forged.write_text(json.dumps({**json.loads(quality.read_text()), "audited": 99}))
        with self.assertRaises(ValueError):
            prepare_overfit(later, forged, audited, review, assessments, config, self.root / "blocked")
        assessments.write_text(assessments.read_text().splitlines()[0] + "\n")
        with self.assertRaises(ValueError):
            prepare_overfit(later, quality, audited, review, assessments, config, self.root / "blocked")


if __name__ == "__main__":
    unittest.main()
