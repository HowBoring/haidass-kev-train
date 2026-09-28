"""Controlled builder HTTP and constructed assessment fixtures test software, not human audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import test_builder as builder_fixture
import test_quality_gate as fixture
import test_overfit_recipe as recipe_fixture
from transformers import AutoTokenizer
from haidass_kev_train.data.aggregate import aggregate
from haidass_kev_train.data.build import _split, build
from haidass_kev_train.data.canonical import load_canonical_suite
from haidass_kev_train.data.finemath import group_id as math_group
from haidass_kev_train.data.packing import check_group_integrity
from haidass_kev_train.data.quality import quality_gate
from haidass_kev_train.data.ufw import digest
from haidass_kev_train.training.overfit import prepare_overfit
from haidass_kev_train.training.sft import _canonical_data, _config


class AggregateTests(unittest.TestCase):
    root: Path
    tokenizer: Path
    setUp = fixture.HumanGateSoftwareTests.setUp
    trial = fixture.HumanGateSoftwareTests.trial
    assessments = fixture.HumanGateSoftwareTests.assessments

    def audited(self):
        suite, review, cases = self.trial(100)
        assessments = self.root / "assessments.jsonl"
        assessments.write_text("".join(json.dumps(row) + "\n" for row in self.assessments(cases)))
        quality = self.root / "quality.json"
        self.assertEqual(quality_gate(suite, review, assessments, quality)["status"], "pass")
        return suite, review, assessments, quality

    def later_batch(self, audited, *, duplicate=False):
        """Build another capped batch over offset rows, optionally repeating one original input."""
        manifest = json.loads((audited / "manifest.json").read_text())
        config = manifest["build"]["config"]
        summary = json.loads((audited / "summary.json").read_text())
        offsets = {source: summary["scanned_by_source"].get(source, 0) for source in config["sources"]}
        config["source_start_rows"] = offsets
        config["seed"] = 18  # selection seed differs; policy and split_seed remain fixed.
        math_numbers = list(range(40, 80))
        if not any(_split(config["split_seed"], math_group("finemath", f"https://example.org/math/{n}", "")) == "development"
                   for n in math_numbers):
            math_numbers[-1] = next(n for n in range(80, 1000) if _split(
                config["split_seed"], math_group("finemath", f"https://example.org/math/{n}", "")) == "development")
        duplicate_en_index = None
        for source, language in (("ufw-en", "en"), ("ufw-zh", "zh")):
            numbers = list(range(30, 60))
            def text(n):
                return (f"Atlas volume {n} says Italy's capital is Rome.\nQuestion: What is Italy's capital? Answer: Rome"
                        if language == "en" else
                        f"第{n}册图册记载意大利的首都是罗马。\n问题：意大利的首都是什么？ 答案：罗马")
            def group(n):
                state = text(n).split("\n")[0]
                return f"{source}/{digest(state)}"
            if not any(_split(config["split_seed"], group(n)) == "development" for n in numbers):
                numbers[-1] = next(n for n in range(60, 1000)
                                   if _split(config["split_seed"], group(n)) == "development")
            if duplicate and language == "en":
                duplicate_en_index = next(index for index, n in enumerate(numbers)
                                          if _split(config["split_seed"], group(n)) == "train")
                numbers[duplicate_en_index] = 0
            source_file = Path(config["sources"][source]) / "part.parquet"
            old = pq.read_table(source_file)
            new = pa.Table.from_pylist([{"uid": f"{language[0]}{n}", "style": "qa", "content": text(n)}
                                        for n in numbers], schema=old.schema)
            pq.write_table(pa.concat_tables([old, new]), source_file)
        math_file = Path(config["sources"]["finemath"]) / "part.parquet"
        old = pq.read_table(math_file)
        new = pa.Table.from_pylist([{"text": f"Question: What is {n}+1?\nAnswer: {n+1}",
                                     "url": f"https://example.org/math/{n}", "snapshot_type": "latest"}
                                    for n in math_numbers], schema=old.schema)
        pq.write_table(pa.concat_tables([old, new]), math_file, row_group_size=16)

        calls = 0
        def http(request, timeout):
            nonlocal calls
            index = calls
            calls += 1
            if index < 40:
                number = math_numbers[index]
                return builder_fixture.response({"distractors": [str(number + step) for step in (2, 3, 4, 5, 6)]})
            english = (index - 40) < 60
            if (index - 40) % 2 == 0:
                options = (["Milan", "Venice", "Naples", "Turin", "Genoa"] if english else
                           ["米兰", "威尼斯", "那不勒斯", "都灵", "热那亚"])
                return builder_fixture.response({"distractors": options})
            return builder_fixture.response({"supported": True, "unique": True,
                                             "all_wrong": True, "same_format": True})

        later = self.root / "later"
        with patch("urllib.request.urlopen", side_effect=http):
            report = build(config, later)
        self.assertEqual(report["accepted"], 100)
        self.assertEqual(calls, 160)
        return later

    def test_authorized_offset_batches_merge_without_trimming_or_moving_groups(self):
        audited, review, assessments, quality = self.audited()
        later = self.later_batch(audited)
        output = self.root / "combined"
        report = aggregate([audited, later], quality, audited, review, assessments, 128, output)
        self.assertGreaterEqual(report["counts"]["train"]["records"], 128)
        self.assertEqual(report["input_accepted"], 200)
        self.assertEqual(report["duplicate_canonicals_collapsed"], 0)
        self.assertEqual(len(report["batch_manifests"]), 2)
        self.assertNotEqual(report["batch_manifests"][0]["manifest_sha256"],
                            report["batch_manifests"][1]["manifest_sha256"])
        self.assertFalse(any(check_group_integrity(output, ("train", "development"))["overlaps"].values()))
        for split in ("train", "development"):
            expected = {row["_meta"]["id"] for source in (audited, later)
                        for row in load_canonical_suite(source, split)}
            self.assertEqual({row["_meta"]["id"] for row in load_canonical_suite(output, split)}, expected)
        self.assertTrue(json.loads((output / "manifest.json").read_text())["complete"])
        plan = prepare_overfit(output, quality, audited, review, assessments,
                               recipe_fixture.base_config(self.root, self.tokenizer), self.root / "recipe")
        config = _config(plan["overfit"]["config"])
        train, development, probe = _canonical_data(config, AutoTokenizer.from_pretrained(self.tokenizer))
        self.assertEqual((len(train), len(probe), len(development)),
                         (128, 640, 5 * len(load_canonical_suite(output, "development"))))

    def test_repeated_original_row_collapses_with_first_locator_but_conflicting_supervision_fails(self):
        audited, review, assessments, quality = self.audited()
        later = self.later_batch(audited, duplicate=True)
        original_case = next((split, row) for split in ("train", "development")
                             for row in load_canonical_suite(audited, split)
                             if row["source"] == "ufw-en" and row["_meta"]["source_ref"]["uid"] == "e0")
        split, original_row = original_case
        later_case = next(row for row in load_canonical_suite(later, split)
                          if row["_meta"]["id"] == original_row["_meta"]["id"])
        self.assertNotEqual(original_row["_meta"]["source_ref"]["line"],
                            later_case["_meta"]["source_ref"]["line"])
        output = self.root / "combined"
        report = aggregate([audited, later], quality, audited, review, assessments, 128, output)
        self.assertEqual((report["input_accepted"], report["distinct_accepted"],
                          report["duplicate_canonicals_collapsed"]), (200, 199, 1))
        combined_case = next(row for row in load_canonical_suite(output, split)
                             if row["_meta"]["id"] == original_row["_meta"]["id"])
        self.assertEqual(combined_case["_meta"]["source_ref"], original_row["_meta"]["source_ref"])
        path = later / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(row for row in rows if row["_meta"]["id"] == original_row["_meta"]["id"])["distractors"][0] = "Florence"
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
        path.write_bytes(payload)
        manifest_path = later / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][path.name]["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        blocked = self.root / "conflict"
        with self.assertRaises(ValueError):
            aggregate([audited, later], quality, audited, review, assessments, 128, blocked)
        self.assertFalse(blocked.exists())

    def test_audit_target_batch_uniqueness_and_integrity_block_publication(self):
        audited, review, assessments, quality = self.audited()
        later = self.later_batch(audited)
        output = self.root / "blocked"
        for batches, minimum in (([audited], 128), ([audited, audited], 1),
                                 ([audited, later], 300)):
            with self.subTest(minimum=minimum, count=len(batches)):
                with self.assertRaises(ValueError):
                    aggregate(batches, quality, audited, review, assessments, minimum, output)
                self.assertFalse(output.exists())
        forged = self.root / "forged.json"
        forged.write_text(json.dumps({**json.loads(quality.read_text()), "audited": 99}))
        with self.assertRaises(ValueError):
            aggregate([audited, later], forged, audited, review, assessments, 128, output)
        self.assertFalse(output.exists())
        original = (later / "train.jsonl").read_bytes()
        (later / "train.jsonl").write_bytes(original + b"{}\n")
        with self.assertRaises(ValueError):
            aggregate([audited, later], quality, audited, review, assessments, 128, output)
        self.assertFalse(output.exists())
        (later / "train.jsonl").write_bytes(original)
        manifest_path, summary_path = later / "manifest.json", later / "summary.json"
        original_manifest, original_summary = manifest_path.read_bytes(), summary_path.read_bytes()
        manifest = json.loads(original_manifest)
        summary = json.loads(original_summary)
        manifest["build"]["config"]["source_start_rows"]["finemath"] = 0
        summary["skipped_by_offset"]["finemath"] = 0
        manifest_path.write_text(json.dumps(manifest))
        summary_path.write_text(json.dumps(summary))
        with self.assertRaises(ValueError):
            aggregate([audited, later], quality, audited, review, assessments, 128, output)
        self.assertFalse(output.exists())
        manifest_path.write_bytes(original_manifest)
        summary_path.write_bytes(original_summary)
        manifest["complete"] = False
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            aggregate([audited, later], quality, audited, review, assessments, 128, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
