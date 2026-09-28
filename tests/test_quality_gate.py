"""Quality gate exercises the public builder with controlled external HTTP, not a real audit."""
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
from haidass_kev_train.data.quality import prepare_review, quality_gate


class HumanGateSoftwareTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        markers = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
                   "<|box_end|>", "<|quad_start|>"]
        vocab = {f"unused_{i}": i for i in range(64000)}
        vocab["[UNK]"] = 0
        for i, marker in enumerate(markers, 6):
            del vocab[f"unused_{i}"]
            vocab[marker] = i
        del vocab["unused_0"]
        tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = self.root / "tokenizer"
        PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
                                additional_special_tokens=markers).save_pretrained(self.tokenizer)
        qwen = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "<|im_start|>": 1, "<|im_end|>": 2}, unk_token="[UNK]"))
        qwen.pre_tokenizer = pre_tokenizers.Whitespace()
        chat = PreTrainedTokenizerFast(tokenizer_object=qwen, unk_token="[UNK]",
                                       additional_special_tokens=["<|im_start|>", "<|im_end|>"])
        chat.chat_template = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                              "{{ message['content'] }}<|im_end|>{% endfor %}"
                              "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        self.qwen = self.root / "qwen"
        chat.save_pretrained(self.qwen)

    def trial(self, count, *, only_math=False):
        en = self.root / "ultrafineweb_en_l3" / "qa"
        zh = self.root / "ultrafineweb_zh_l3" / "qa"
        math = self.root / "finemath-4plus"
        for directory in (en, zh, math):
            directory.mkdir(parents=True)
        size_en, size_zh, size_math = ((0, 0, 100) if only_math else
                                       (30, 30, 40) if count == 100 else (1, 1, 1))
        if size_en:
            pq.write_table(pa.Table.from_pylist([
                {"uid": f"e{i}", "style": "qa", "content": f"Atlas volume {i} says Italy's capital is Rome.\nQuestion: What is Italy's capital? Answer: Rome"}
                for i in range(size_en)]), en / "part.parquet")
        if size_zh:
            pq.write_table(pa.Table.from_pylist([
                {"uid": f"z{i}", "style": "qa", "content": f"第{i}册图册记载意大利的首都是罗马。\n问题：意大利的首都是什么？ 答案：罗马"}
                for i in range(size_zh)]), zh / "part.parquet")
        pq.write_table(pa.Table.from_pylist([
            {"text": ("Question: Evaluate √2+√3?\nAnswer: √2+√3" if i == 0 else
                      "Question: What length is 5 cm in meters?\nAnswer: 0.05 m" if i == 1 else
                      f"Question: What is {i}+1?\nAnswer: {i+1}"),
             "url": f"https://example.org/math/{i}", "snapshot_type": "latest"}
            for i in range(size_math)]), math / "part.parquet", row_group_size=16)
        sources = {"finemath": str(math)} if only_math else {
            "ufw-en": str(en), "ufw-zh": str(zh), "finemath": str(math)}
        config = {"sources": sources,
                  "tokenizer_path": str(self.tokenizer), "generator_tokenizer_path": str(self.qwen),
                  "seed": 17, "split_seed": 2, "target": count,
                  "source_targets": {source: {"ufw-en": size_en, "ufw-zh": size_zh,
                                               "finemath": size_math}[source] for source in sources},
                  "max_attempts": 250, "max_seconds": 900, "timeout": 2,
                  "max_packed": 1024, "max_answer_tokens": 32,
                  "max_source_tokens": 4000, "max_context_tokens": 8192, "max_output_tokens": 256}
        calls = []

        def http(request, timeout):
            data = json.loads(request.data)
            calls.append(data)
            # FineMath is streamed first, including one independent unknown-path adjudication.
            index = len(calls) - 1
            if index == 0:
                value = {"distractors": ["√2+√5", "√2+√7", "√3+√5", "√3+√7", "√5+√7"]}
            elif index == 1:
                value = {"decision": "approve", "answer_type_valid": True,
                         "relationships": [{"left": left, "right": right, "relation": "distinct"}
                                           for left in range(6) for right in range(left + 1, 6)]}
            elif index <= size_math:
                n = index - 1
                value = ({"distractors": ["0.06 m", "0.07 m", "0.08 m", "0.09 m", "0.10 m"]}
                         if n == 1 else {"distractors": [str(n + offset) for offset in (2, 3, 4, 5, 6)]})
            else:
                remaining = index - size_math - 1
                if remaining < size_en * 2:
                    value = ({"distractors": ["Milan", "Venice", "Naples", "Turin", "Genoa"]}
                             if remaining % 2 == 0 else
                             {"supported": True, "unique": True, "all_wrong": True, "same_format": True})
                else:
                    value = ({"distractors": ["米兰", "威尼斯", "那不勒斯", "都灵", "热那亚"]}
                             if remaining % 2 == 0 else
                             {"supported": True, "unique": True, "all_wrong": True, "same_format": True})
            body = {"choices": [{"message": {"content": json.dumps(value, ensure_ascii=False)},
                                  "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 13}}
            return io.BytesIO(json.dumps(body).encode())

        suite = self.root / "suite"
        with patch("urllib.request.urlopen", side_effect=http):
            report = build(config, suite)
        self.assertEqual(report["accepted"], count)
        self.assertEqual(len(calls), size_math + 1 + 2 * (size_en + size_zh))
        review = prepare_review(suite, self.root / "review")
        cases = [json.loads(line) for line in review.read_text().splitlines()]
        self.assertEqual(len(cases), count)
        return suite, review, cases

    def assessments(self, cases):
        return [{"case_id": case["case_id"], "policy_sha256": case["policy_sha256"],
                 "batch_manifest_sha256": case["batch_manifest_sha256"],
                 "trace_sha256": case["trace_sha256"], "reviewer": "constructed-test-fixture",
                 "source_verified": True, "serious_error": False, "category": None, "reason": "",
                 "paths": []}
                for case in cases]

    def report(self, suite, review, assessments, index):
        path = self.root / f"assessments-{index}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in assessments))
        return quality_gate(suite, review, path, self.root / f"gate-{index}.json")

    def test_100_case_batch_incomplete_fail_pass_and_coverage_are_human_evidence_only(self):
        suite, review, cases = self.trial(100)
        original = cases[0]
        self.assertEqual(len(original["distractors"]), 5)
        self.assertEqual(original["original_source_text"][slice(*original["original_spans"]["answer"]["span"])],
                         original["original_spans"]["answer"]["text"])
        self.assertEqual(original["source_ref"]["sha256"], hashlib.sha256(original["original_source_text"].encode()).hexdigest())
        assessments = self.assessments(cases)
        self.assertEqual(self.report(suite, review, [], 0)["status"], "incomplete")
        absent = quality_gate(suite, review, None, self.root / "gate-no-evidence.json")
        self.assertEqual((absent["status"], absent["audited"], absent["human_assessments"]), ("incomplete", 0, None))
        partial = self.report(suite, review, assessments[:-1], 1)
        self.assertEqual((partial["status"], partial["audited"]), ("incomplete", 99))
        math_cases = [case for case in cases if case["source"] == "finemath"]
        for case, tag in ((next(case for case in math_cases if "√2" in case["original_source_text"]), "symbolic"),
                          (next(case for case in math_cases if "5 cm" in case["original_source_text"]), "unit")):
            assessments[next(i for i, item in enumerate(assessments) if item["case_id"] == case["case_id"])]["paths"] = [tag]
        complete = self.report(suite, review, assessments, 2)
        self.assertEqual((complete["status"], complete["severe_count"], complete["audited"]), ("pass", 0, 100))
        self.assertEqual(complete["policy_sha256"], cases[0]["policy_sha256"])
        self.assertEqual(complete["batch_manifest_sha256"], hashlib.sha256((suite / "manifest.json").read_bytes()).hexdigest())
        for name in ("ufw-en", "ufw-zh", "finemath", "symbolic", "unit",
                     "finemath_programmatic", "finemath_llm_adjudicated"):
            self.assertTrue(complete["coverage"][name]["verified"])
        bad = self.assessments(cases)
        bad[0].update(serious_error=True, category="correct_distractor", reason="constructed example")
        failed = self.report(suite, review, bad, 3)
        self.assertEqual((failed["status"], failed["severe_count"]), ("fail", 1))
        self.assertEqual(failed["severe_examples"][0]["case_id"], bad[0]["case_id"])

    def test_completed_100_math_only_batch_lacks_required_human_source_coverage(self):
        suite, review, cases = self.trial(100, only_math=True)
        late = next(case for case in cases if case["source_ref"]["line"] == 99)
        self.assertEqual(late["original_spans"]["question"]["text"], "What is 99+1?")
        self.assertEqual(late["original_spans"]["answer"]["text"], "100")
        self.assertTrue(json.loads((suite / "manifest.json").read_text())["complete"])
        gate = self.report(suite, review, self.assessments(cases), 0)
        self.assertEqual((gate["status"], gate["audited"], gate["severe_count"]), ("incomplete", 100, 0))
        self.assertEqual(gate["coverage"]["finemath"]["reviewed"], 100)
        self.assertEqual((gate["coverage"]["ufw-en"]["available"], gate["coverage"]["ufw-zh"]["available"]), (0, 0))
        self.assertIn("ufw-en", gate["unverified_paths"])
        self.assertIn("ufw-zh", gate["unverified_paths"])

    def test_underfilled_and_spoofed_evidence_never_passes(self):
        suite, review, cases = self.trial(3)
        assessments = self.assessments(cases)
        self.assertEqual(self.report(suite, review, assessments, 0)["status"], "incomplete")
        bad = self.assessments(cases)
        bad[0].update(serious_error=True, category="wrong_gold", reason="constructed finding")
        self.assertEqual(self.report(suite, review, bad, 8)["status"], "fail")
        failed_path = self.root / "gate-8.json"
        failed_bytes = failed_path.read_bytes()
        with self.assertRaises(FileExistsError):
            quality_gate(suite, review, self.root / "assessments-0.jsonl", failed_path)
        self.assertEqual(failed_path.read_bytes(), failed_bytes)
        dangling = self.root / "gate-dangling.json"
        dangling.symlink_to(self.root / "absent-report.json")
        with self.assertRaises(FileExistsError):
            quality_gate(suite, review, self.root / "assessments-0.jsonl", dangling)
        self.assertTrue(dangling.is_symlink())
        self.assertFalse((self.root / "absent-report.json").exists())
        for index, field, value in ((1, "policy_sha256", "0" * 64),
                                    (2, "batch_manifest_sha256", "0" * 64),
                                    (3, "trace_sha256", "0" * 64),
                                    (4, "case_id", "absent")):
            forged = self.assessments(cases)
            forged[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.report(suite, review, forged, index)
        with self.assertRaises(ValueError):
            self.report(suite, review, assessments + assessments[:1], 5)
        tampered = [dict(case) for case in cases]
        tampered[0]["gold"] = "tampered"
        review.write_text("".join(json.dumps(case) + "\n" for case in tampered))
        with self.assertRaises(ValueError):
            self.report(suite, review, assessments, 6)
        review.write_text("".join(json.dumps(case) + "\n" for case in cases))
        manifest_path = suite / "manifest.json"
        original_manifest = manifest_path.read_text()
        forged_manifest = json.loads(original_manifest)
        forged_manifest["build"]["policy_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(forged_manifest))
        with self.assertRaisesRegex(ValueError, "policy_sha256"):
            self.report(suite, review, assessments, 9)
        manifest_path.write_text(original_manifest)
        source = Path(cases[0]["source_parquet"])
        rows = pq.read_table(source).to_pylist()
        field = "text" if cases[0]["source"] == "finemath" else "content"
        rows[cases[0]["source_ref"]["line"]][field] += " tampered"
        pq.write_table(pa.Table.from_pylist(rows), source)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.report(suite, review, assessments, 7)
