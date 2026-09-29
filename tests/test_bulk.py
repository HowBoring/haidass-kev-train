"""Bulk lane runner: offset continuation, unreviewed aggregation and frozen failure resume.

Only the external HTTP response is controlled; every reply is content-addressed by generation
task so concurrent lanes stay deterministic.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from haidass_kev_train.data.bulk import run
from haidass_kev_train.data.canonical import load_canonical_suite

# Per-lane state texts keep each lane's Source Groups distinct; the two rows of a lane share
# one group, so both chained batches of a lane split identically.
EN_STATES = ["The atlas lists the capital of Italy as Rome.",
             "The guide records the capital of Italy as Rome."]
ZH_STATES = ["图册记载意大利的首都是罗马。", "指南记录意大利的首都是罗马。"]
MATH_1 = "Question: What length is 5 cm in meters?\nAnswer: 0.05 m"
MATH_2 = "Question: How fast is 36 km/h in m/s?\nAnswer: 10 m/s"
DISTRACTORS = ["Milan", "Venice", "Naples", "Turin", "Genoa"]
# Pinned offline against the real grouping/split code: every source has one train and one
# development lane-group and every lane batch keeps a nonempty development split.
SPLIT_SEED = 2134
def _qa(state, question, answer, language):
    if language == "en":
        return f"{state}\n\nQuestion: {question} Answer: {answer}"
    return f"{state}\n问题：{question} 答案：{answer}"



def _math_distractors(gold):
    match = re.fullmatch(r"(-?[\d.]+)(.*)", gold)
    if not match:
        return ["41", "42", "43", "44", "45"]
    number, unit = match.groups()
    return [f"{float(number) + step:g}{unit}" for step in range(1, 6)]


def serve_http(requests):
    """Reply to every generation task from its own payload; safe for concurrent lanes."""

    def http(request, timeout):
        payload = json.loads(request.data)
        task = payload["messages"][0]["content"].split(" / ")[1].split(".")[0]
        material = json.loads(payload["messages"][1]["content"])
        with threading.Lock():
            requests.append(task)
        if task == "ufw_screen":
            reply = {"supported": True, "unique": True, "all_wrong": True, "same_format": True}
        elif task == "ufw_generate":
            reply = {"distractors": DISTRACTORS}
        elif task == "finemath_generate":
            reply = {"distractors": _math_distractors(material["source_answer"])}
        else:
            raise AssertionError(f"unexpected generation task: {task}")
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(reply)},
                                                   "finish_reason": "stop"}],
                                      "usage": {"prompt_tokens": 3, "completion_tokens": 5}}).encode())

    return http


class BulkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        markers = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
                   "<|box_end|>", "<|quad_start|>"]
        vocab = {f"unused_{index}": index for index in range(64000)}
        vocab["[UNK]"] = 0
        for index, marker in enumerate(markers, 6):
            del vocab[f"unused_{index}"]
            vocab[marker] = index
        del vocab["unused_0"]
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = self.root / "tokenizer"
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                additional_special_tokens=markers).save_pretrained(self.tokenizer)
        generator = Tokenizer(models.WordLevel(vocab={
            "[UNK]": 0, "<|im_start|>": 1, "<|im_end|>": 2}, unk_token="[UNK]"))
        generator.pre_tokenizer = pre_tokenizers.Whitespace()
        chat = PreTrainedTokenizerFast(tokenizer_object=generator, unk_token="[UNK]",
                                       additional_special_tokens=["<|im_start|>", "<|im_end|>"])
        chat.chat_template = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                              "{{ message['content'] }}<|im_end|>{% endfor %}"
                              "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        self.generator_tokenizer = self.root / "generator-tokenizer"
        chat.save_pretrained(self.generator_tokenizer)

    def source(self, name, lane_rows):
        """Two Parquet shards per source; sorted-glob shard i belongs to lane i % shard_count."""
        folder = (self.root / f"ultrafineweb_{name[-2:]}_l3" / "qa") if name != "finemath" \
            else self.root / "finemath-4plus"
        folder.mkdir(parents=True, exist_ok=True)
        for lane, rows in enumerate(lane_rows):
            pq.write_table(pa.Table.from_pylist(rows),
                           folder / (f"part-0000{lane}.parquet" if name != "finemath"
                                     else f"train-0000{lane}-of-00002.parquet"))
        return str(folder)

    def config(self, shard_count=2, target=3):
        def ufw(language, states):
            return [[{"uid": f"{language[1]}{lane}a", "style": "qa",
                      "content": _qa(states[lane], "What is the capital of Italy?" if language == "en"
                                     else "意大利的首都是什么？", "罗马" if language == "zh" else "Rome", language)},
                     {"uid": f"{language[1]}{lane}b", "style": "qa",
                      "content": _qa(states[lane], "Which city is the capital of Italy?" if language == "en"
                                     else "罗马是哪个国家的首都？", "罗马" if language == "zh" else "Rome", language)}]
                    for lane in range(2)]

        sources = {"finemath": self.source("finemath", [
                       [{"text": MATH_1, "url": f"https://example.org/m/{lane}", "snapshot_type": "latest"},
                        {"text": MATH_2, "url": f"https://example.org/m/{lane}", "snapshot_type": "longest"}]
                       for lane in range(2)]),
                   "ufw-en": self.source("ufw-en", ufw("en", EN_STATES)),
                   "ufw-zh": self.source("ufw-zh", ufw("zh", ZH_STATES))}
        lines = [f"shard_count = {shard_count}"] * (shard_count > 1) + [
            "seed = 17", f"split_seed = {SPLIT_SEED}", f"target = {target}",
            "max_attempts = 60", "max_seconds = 60", "timeout = 2", "max_packed = 1024",
            "max_answer_tokens = 32", "max_source_tokens = 4000", "max_context_tokens = 8192",
            "max_output_tokens = 256", f'tokenizer_path = "{self.tokenizer}"',
            f'generator_tokenizer_path = "{self.generator_tokenizer}"', "",
            "[sources]"] + [f'{name} = "{path}"' for name, path in sources.items()] + [
            "", "[source_targets]"] + [f"{name} = 1" for name in sources]
        path = self.root / "base.toml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_offset_continuation_and_unreviewed_manifest(self):
        config = self.config()
        requests = []
        with patch("urllib.request.urlopen", side_effect=serve_http(requests)):
            first = run(config, self.root / "out", minimum_train=4)
        out = self.root / "out"
        for lane in (0, 1):
            first_manifest = json.loads((out / f"lane-0{lane}" / "batch-0000" / "manifest.json").read_text())
            second_manifest = json.loads((out / f"lane-0{lane}" / "batch-0001" / "manifest.json").read_text())
            self.assertTrue(first_manifest["complete"] and second_manifest["complete"])
            start = first_manifest["build"]["config"]["source_start_rows"]
            scanned = json.loads((out / f"lane-0{lane}" / "batch-0000" / "summary.json").read_text())["scanned_by_source"]
            continued = {source: start[source] + scanned.get(source, 0) for source in start}
            self.assertEqual(continued, {"finemath": 1, "ufw-en": 1, "ufw-zh": 1})
            self.assertEqual(second_manifest["build"]["config"]["source_start_rows"], continued)
            self.assertEqual(second_manifest["build"]["config"]["shard_index"], lane)
        self.assertEqual(first["status"], "aggregated")
        self.assertEqual(first["quality_status"], "not_reviewed")
        suite = json.loads((out / "suite" / "manifest.json").read_text())
        aggregation = suite["aggregation"]
        self.assertEqual(aggregation["quality_status"], "not_reviewed")
        self.assertEqual(aggregation["minimum_train"], 4)
        self.assertEqual(len(aggregation["batch_manifests"]), 4)
        for key in ("quality_report", "quality_report_sha256", "audited_batch_manifest_sha256",
                    "review_sha256", "assessments_sha256"):
            self.assertNotIn(key, aggregation)
        self.assertEqual(aggregation["counts"]["train"]["sources"], {"finemath": 2, "ufw-en": 2, "ufw-zh": 2})
        self.assertEqual(aggregation["counts"]["development"]["sources"], {"finemath": 2, "ufw-en": 2, "ufw-zh": 2})
        self.assertEqual(len(load_canonical_suite(out / "suite", "train")), 6)
        # Resuming the same base config and frozen chain never rebuilds or re-aggregates.
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            again = run(config, self.root / "out", minimum_train=4)
        self.assertEqual(again["status"], "already_aggregated")
        self.assertEqual(again["distinct_train"], 6)

    def test_more_lanes_than_files_aggregate_disjoint_row_stripes(self):
        config = self.config(shard_count=4)
        with patch("urllib.request.urlopen", side_effect=serve_http([])):
            report = run(config, self.root / "striped", minimum_train=4)
        self.assertEqual(report["status"], "aggregated")
        locators = set()
        for batch in (self.root / "striped").glob("lane-*/batch-*"):
            lane = int(batch.parent.name.split("-")[1])
            for split in ("train", "development"):
                for row in load_canonical_suite(batch, split):
                    ref = row["_meta"]["source_ref"]
                    self.assertEqual(ref["line"] % 2, lane // 2)
                    locator = row["source"], ref["path"], ref["line"]
                    self.assertNotIn(locator, locators)
                    locators.add(locator)
        self.assertGreaterEqual(len(load_canonical_suite(self.root / "striped" / "suite", "train")), 4)

    def test_failed_batch_stops_frozen_and_resume_refuses_to_continue(self):
        config = self.config(shard_count=1, target=1)
        output = self.root / "out"
        with patch("urllib.request.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(ValueError, "service_error"):
                run(config, output, minimum_train=1)
        batch = output / "lane-00" / "batch-0000"
        self.assertFalse(json.loads((batch / "manifest.json").read_text())["complete"])
        frozen = {path.relative_to(batch): path.read_bytes() for path in sorted(batch.rglob("*")) if path.is_file()}
        with patch("urllib.request.urlopen", side_effect=serve_http([])):
            with self.assertRaisesRegex(ValueError, "service_error"):
                run(config, output, minimum_train=1)
        self.assertEqual(frozen, {path.relative_to(batch): path.read_bytes()
                                  for path in sorted(batch.rglob("*")) if path.is_file()})
        self.assertFalse((output / "lane-00" / "batch-0001").exists())
        self.assertFalse((output / "suite").exists())

    def test_non_contiguous_batch_numbering_is_rejected(self):
        config = self.config(shard_count=1, target=1)
        (self.root / "out" / "lane-00" / "batch-0001").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            run(config, self.root / "out", minimum_train=1)


if __name__ == "__main__":
    unittest.main()
