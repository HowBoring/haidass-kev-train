"""Three-source trial reporting and strategy identity via public offline builder."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from haidass_kev_train.data.build import build
from haidass_kev_train.data.canonical import load_canonical_suite


SCREEN = {"supported": True, "unique": True, "all_wrong": True, "same_format": True}
MATH = "Question: Evaluate √2+√3?\nAnswer: √2+√3"
MATH_OPTIONS = ["√2+√5", "√2+√7", "√3+√5", "√3+√7", "√5+√7"]
APPROVE = {"decision": "approve", "answer_type_valid": True, "relationships": [
    {"left": left, "right": right, "relation": "distinct"}
    for left in range(6) for right in range(left + 1, 6)]}
EN_STATE = "The atlas lists Italy's capital as Rome."
EN = EN_STATE + "\nQuestion: Which city is Italy's capital? Answer: Rome"
ZH_STATE = "图册记载意大利的首都是罗马。"
ZH = ZH_STATE + "\n题目：意大利首都是什么？ 答：罗马"


def located(raw, state, question, answer):
    start = raw.index(question, len(state))
    end = raw.index(answer, start + len(question))
    return {"sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "state_span": [0, len(state)], "state_text": state,
            "qas": [{"question_span": [start, start + len(question)], "question_text": question,
                     "answer_span": [end, end + len(answer)], "answer_text": answer,
                     "option_spans": [], "option_texts": []}]}


class TrialReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        markers = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
                   "<|box_end|>", "<|quad_start|>"]
        vocab = {f"unused_{n}": n for n in range(64000)}
        vocab["[UNK]"] = 0
        del vocab["unused_0"]
        for n, marker in enumerate(markers, 6):
            del vocab[f"unused_{n}"]
            vocab[marker] = n
        training = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        training.pre_tokenizer = pre_tokenizers.Whitespace()
        self.training = self.root / "tokenizer"
        PreTrainedTokenizerFast(tokenizer_object=training, unk_token="[UNK]",
                                additional_special_tokens=markers).save_pretrained(self.training)
        qwen = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "<|im_start|>": 1, "<|im_end|>": 2},
                                          unk_token="[UNK]"))
        qwen.pre_tokenizer = pre_tokenizers.Whitespace()
        chat = PreTrainedTokenizerFast(tokenizer_object=qwen, unk_token="[UNK]",
                                       additional_special_tokens=["<|im_start|>", "<|im_end|>"])
        chat.chat_template = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                              "{{ message['content'] }}<|im_end|>{% endfor %}"
                              "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        self.qwen = self.root / "generator-tokenizer"
        chat.save_pretrained(self.qwen)
        self.sources = {}
        for language, rows in (("en", [("e1", EN)]), ("zh", [("z1", ZH)])):
            folder = self.root / f"ultrafineweb_{language}_l3" / "qa"
            folder.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([
                {"uid": uid, "content": raw, "style": "qa"} for uid, raw in rows]), folder / "source.parquet")
            self.sources[f"ufw-{language}"] = str(folder)
        folder = self.root / "finemath"
        folder.mkdir()
        self.math_shard = folder / "source.parquet"
        self.math_rows([(MATH, "https://example.org/math", "latest")])
        self.sources["finemath"] = str(folder)

    def math_rows(self, rows):
        pq.write_table(pa.Table.from_pylist([
            {"text": raw, "url": url, "snapshot_type": snapshot}
            for raw, url, snapshot in rows]), self.math_shard)

    def run_trial(self, replies, *, output="suite", clock=None, default_targets=False, **changes):
        config = {"sources": self.sources, "tokenizer_path": str(self.training),
                  "generator_tokenizer_path": str(self.qwen), "seed": 7, "split_seed": 2,
                  "target": 3, "source_targets": {source: 1 for source in self.sources},
                  "max_attempts": 24, "max_seconds": 30, "timeout": 2,
                  "max_packed": 1024, "max_answer_tokens": 32, "max_source_tokens": 4000,
                  "max_context_tokens": 8192, "max_output_tokens": 256, **changes}
        if default_targets:
            del config["source_targets"]
        calls = []
        incoming = iter(replies)

        def http(request, timeout):
            payload = json.loads(request.data)
            calls.append(payload)
            if clock is not None:
                clock[0] += .06
            item = next(incoming)
            body = json.dumps({"choices": [{"message": {"content": json.dumps(item, ensure_ascii=False)},
                                             "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 3, "completion_tokens": 5}}).encode()
            return io.BytesIO(body)

        with patch("urllib.request.urlopen", side_effect=http):
            if clock is None:
                report = build(config, self.root / output)
            else:
                with patch("haidass_kev_train.data.generation.time.monotonic", side_effect=lambda: clock[0]):
                    report = build(config, self.root / output)
        manifest_bytes = (self.root / output / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        return report, calls, manifest, hashlib.sha256(manifest_bytes).hexdigest()

    def test_one_budget_through_math_adjudication_retries_ufw_and_assisted_location(self):
        # Sorted source order: FineMath, English, Chinese; no source gets a fresh budget.
        report, calls, manifest, _ = self.run_trial([
            {"distractors": MATH_OPTIONS}, {"decision": "uncertain"},
            {"distractors": ["Milan", "Venice", "Naples", "Turin", "Genoa"]}, SCREEN,
            located(ZH, ZH_STATE, "意大利首都是什么？", "罗马"),
        ], max_attempts=5)
        self.assertEqual([call["messages"][0]["content"].split(" / ")[1].split(".")[0] for call in calls],
                         ["finemath_generate", "finemath_adjudicate", "ufw_generate", "ufw_screen", "ufw_locate"])
        self.assertEqual((report["attempts"], report["stop_reason"], report["accepted"]), (5, "attempt_limit", 1))
        self.assertEqual(report["rejected"]["equivalence_unknown"], 1)
        self.assertEqual(report["scanned_by_source"], {"finemath": 1, "ufw-en": 1, "ufw-zh": 1})
        self.assertEqual(report["unfilled_targets"], {"finemath": 1, "ufw-en": 0, "ufw-zh": 1})
        self.assertEqual((report["trial_status"], manifest["complete"]), ("incomplete", False))
        self.assertIn("target_underfilled", report["incomplete_reasons"])
        rows = (load_canonical_suite(self.root / "suite", "train") +
                load_canonical_suite(self.root / "suite", "development"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "ufw-en")
        self.assertNotIn("quality_pass", report)
        if report["empty_development"]:
            self.assertIn("empty_development", report["incomplete_reasons"])
        self.assertEqual(report, json.loads((self.root / "suite" / "summary.json").read_text()))

    def test_retry_and_cleanup_share_budget_without_shifting_underfilled_source(self):
        state = "图册记载罗马是意大利首都，米兰不是意大利首都。"
        question = "以下4个选项中，哪一个城市被图册明确记载为非意大利首都？ A) 罗马 B) 米兰 C) 巴黎 D) 都灵"
        raw = state + "\n问题：" + question + " 答案：B"
        pq.write_table(pa.Table.from_pylist([{"uid": "z1", "content": raw, "style": "qa"}]),
                       Path(self.sources["ufw-zh"]) / "source.parquet")
        report, calls, _, _ = self.run_trial([
            {"distractors": MATH_OPTIONS}, "bad json", "bad json", APPROVE,
            {"distractors": ["Milan", "Venice", "Naples", "Turin", "Genoa"]}, SCREEN,
            {"question": "下列哪一个城市被图册明确记载为非意大利首都？"},
        ], max_attempts=7)
        self.assertEqual((report["attempts"], report["retries"], report["accepted"]), (7, 2, 2))
        self.assertEqual(report["failures"]["malformed_response"], 2)
        self.assertEqual(report["validation_paths"], {"finemath_llm_adjudicated": 1, "ufw_model_screened": 1})
        self.assertEqual(report["unfilled_targets"]["ufw-zh"], 1)
        self.assertEqual(report["stop_reason"], "attempt_limit")
        self.assertEqual(calls[-1]["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(report["usage"]["prompt_tokens"], 21)

    def test_clock_stops_at_first_limit_and_no_new_source_receives_budget(self):
        report, calls, _, _ = self.run_trial([{"distractors": MATH_OPTIONS}],
                                             max_seconds=.05, clock=[0.0])
        self.assertEqual((report["stop_reason"], report["attempts"], len(calls)), ("time_limit", 1, 1))
        self.assertEqual(report["scanned_by_source"], {"finemath": 1})
        self.assertEqual(report["in_flight_requests"], 0)
        self.assertEqual(report["trial_status"], "incomplete")

    def test_source_groups_split_before_admission_and_empty_dev_is_not_complete(self):
        self.math_rows([("Question: Convert 5 Celsius to Fahrenheit?\nAnswer: 41 Fahrenheit",
                         "https://example.org/math", "latest")])
        pq.write_table(pa.Table.from_pylist([
            {"uid": "reject", "content": EN, "style": "qa"},
            {"uid": "accept", "content": EN, "style": "qa"}]),
            Path(self.sources["ufw-en"]) / "source.parquet")
        groups = [f"ufw-{lang}/{hashlib.sha256(state.encode()).hexdigest()}"
                  for lang, state in (("en", EN_STATE), ("zh", ZH_STATE))]
        split_seed = next(seed for seed in range(1000) if all(
            int.from_bytes(hashlib.sha256(json.dumps([seed, group], separators=(",", ":")).encode()
                                          ).digest()[:8], "big") >= (1 << 64) // 20
            for group in groups))
        report, _, _, _ = self.run_trial([
            {"distractors": ["Rome", "Venice", "Naples", "Turin", "Genoa"]},
            {"distractors": ["Milan", "Venice", "Naples", "Turin", "Genoa"]}, SCREEN,
            located(ZH, ZH_STATE, "意大利首都是什么？", "罗马"),
            {"distractors": ["米兰", "威尼斯", "那不勒斯", "都灵", "热那亚"]}, SCREEN,
        ], split_seed=split_seed)
        self.assertEqual(report["groups_seen"]["ufw-en"], 1)
        self.assertEqual(sum(report["groups_assigned"]["ufw-en"].values()), 1)
        self.assertEqual(report["rejected"]["invalid_distractors"], 1)
        self.assertEqual(report["accepted_by_source"], {"ufw-en": 1, "ufw-zh": 1})
        self.assertEqual(report["group_overlap"], 0)
        for source in self.sources:
            counts = report["split_distribution"][source]
            self.assertEqual(counts["train"] + counts["development"],
                             report["accepted_by_source"].get(source, 0))
        self.assertEqual(report["split_distribution"]["overall"]["development_fraction"],
                         report["splits"]["development"]["records"] / report["accepted"])
        self.assertTrue(report["empty_development"])
        self.assertIn("empty_development", report["incomplete_reasons"])
        self.assertEqual(report["trial_status"], "incomplete")
        self.assertIn("finemath_programmatic", report["unverified_validation_paths"])

    def test_default_three_source_targets_preserve_shortfall(self):
        self.math_rows([("Question: Evaluate 2+2?\nAnswer: 4",
                         "https://example.org/math", "latest")])
        report, _, manifest, _ = self.run_trial(
            [{"distractors": ["1", "2", "3", "5", "7"]}], target=1, default_targets=True)
        self.assertEqual(manifest["build"]["config"]["source_targets"],
                         {"ufw-en": 30, "ufw-zh": 30, "finemath": 40})
        self.assertEqual(report["unfilled_targets"],
                         {"ufw-en": 30, "ufw-zh": 30, "finemath": 39})
        self.assertEqual(report["trial_status"], "incomplete")

    def test_three_source_suite_can_complete_machine_admission_without_quality_pass(self):
        self.math_rows([("Question: Evaluate 2+2?\nAnswer: 4",
                         "https://example.org/math", "latest")])
        math_group = "finemath/" + hashlib.sha256(b"https://example.org/math").hexdigest()
        source_groups = [f"ufw-{lang}/{hashlib.sha256(state.encode()).hexdigest()}"
                         for lang, state in (("en", EN_STATE), ("zh", ZH_STATE))]

        def development(seed, group):
            key = json.dumps([seed, group], separators=(",", ":")).encode()
            return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") < (1 << 64) // 20

        seed = next(n for n in range(10000) if development(n, math_group)
                    and all(not development(n, group) for group in source_groups))
        report, calls, manifest, _ = self.run_trial([
            {"distractors": ["1", "2", "3", "5", "7"]},
            {"distractors": ["Milan", "Venice", "Naples", "Turin", "Genoa"]}, SCREEN,
            located(ZH, ZH_STATE, "意大利首都是什么？", "罗马"),
            {"distractors": ["米兰", "威尼斯", "那不勒斯", "都灵", "热那亚"]}, SCREEN,
        ], split_seed=seed)
        self.assertEqual((report["accepted"], report["attempts"], report["stop_reason"]), (3, 6, "accepted_target"))
        self.assertEqual(set(report["accepted_by_source"]), set(self.sources))
        self.assertEqual((report["trial_status"], report["incomplete_reasons"], manifest["complete"]),
                         ("complete", [], True))
        self.assertEqual((report["splits"]["train"]["records"], report["splits"]["development"]["records"]), (2, 1))
        self.assertEqual({row["source"] for split in ("train", "development")
                          for row in load_canonical_suite(self.root / "suite", split)}, set(self.sources))
        self.assertNotIn("quality_pass", report)
        self.assertEqual(len(calls), 6)

    def test_policy_identity_stays_across_authorized_quantities_and_changes_with_strategy(self):
        first_math = "Question: Evaluate 2+2?\nAnswer: 4"
        second_math = "Question: Evaluate 3+3?\nAnswer: 6"
        replies = [{"distractors": ["1", "2", "3", "5", "7"]}]
        self.math_rows([(first_math, "https://example.org/math", "latest")])
        first, _, one, one_hash = self.run_trial(replies, target=1, max_attempts=2)
        self.math_rows([(second_math, "https://example.org/math", "latest")])
        second, _, two, two_hash = self.run_trial(replies, output="larger", target=1,
                                                 source_targets={source: 2 for source in self.sources},
                                                 max_attempts=3, max_seconds=40, seed=19, split_seed=3)
        self.assertNotEqual(one_hash, two_hash)
        self.assertEqual(one["build"]["policy_sha256"], two["build"]["policy_sha256"])
        self.assertNotEqual(one["build"]["config"]["seed"], two["build"]["config"]["seed"])
        self.assertNotEqual([one["files"][split]["sha256"] for split in ("train.jsonl", "development.jsonl")],
                            [two["files"][split]["sha256"] for split in ("train.jsonl", "development.jsonl")])
        self.assertEqual(one["build"]["policy_sha256"], hashlib.sha256(json.dumps(
            one["build"]["policy"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest())
        (self.training / "model.safetensors").write_bytes(b"unrelated model weights")
        _, _, weight_change, _ = self.run_trial(replies, output="weight-change", target=1)
        self.assertEqual(one["build"]["policy_sha256"], weight_change["build"]["policy_sha256"])
        tokenizer_config = self.training / "tokenizer_config.json"
        tokenizer_config.write_text(tokenizer_config.read_text() + "\n")
        _, _, tokenizer_change, _ = self.run_trial(replies, output="tokenizer-change", target=1)
        self.assertNotEqual(one["build"]["policy_sha256"], tokenizer_change["build"]["policy_sha256"])
        _, _, changed, _ = self.run_trial(replies, output="strategy-change", target=1,
                                          max_answer_tokens=8)
        self.assertNotEqual(one["build"]["policy_sha256"], changed["build"]["policy_sha256"])
        self.assertEqual(set(one["build"]["config"]["source_targets"]), set(self.sources))
        self.assertEqual((first["trial_status"], second["trial_status"]), ("incomplete", "incomplete"))


    def test_selected_versioned_fast_tokenizer_changes_policy_identity(self):
        self.math_rows([("Question: Evaluate 2+2?\nAnswer: 4",
                         "https://example.org/math", "latest")])
        versioned = self.training / "tokenizer.0.0.1.json"
        versioned.write_bytes((self.training / "tokenizer.json").read_bytes())
        config_path = self.training / "tokenizer_config.json"
        config = json.loads(config_path.read_text())
        config["fast_tokenizer_files"] = [versioned.name]
        config_path.write_text(json.dumps(config))
        replies = [{"distractors": ["1", "2", "3", "5", "7"]}]
        baseline, _, original, _ = self.run_trial(replies, target=1)
        tokenizer = json.loads(versioned.read_text())
        tokenizer["model"]["vocab"].pop("unused_63999")
        tokenizer["model"]["vocab"]["new_token"] = 63999
        versioned.write_text(json.dumps(tokenizer))
        changed, _, updated, _ = self.run_trial(replies, output="new-tokenizer", target=1)
        self.assertEqual((baseline["accepted"], changed["accepted"]), (1, 1))
        self.assertIn(versioned.name, original["build"]["tokenizer_sha256"])
        self.assertNotEqual(original["build"]["policy_sha256"], updated["build"]["policy_sha256"])

if __name__ == "__main__":
    unittest.main()
