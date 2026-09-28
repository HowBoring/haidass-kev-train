"""FineMath equivalence through the offline public builder and controlled HTTP only."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from haidass_kev_train.data.build import build
from haidass_kev_train.data.canonical import load_canonical_suite


PAIRS = [{"left": left, "right": right, "relation": "distinct"}
         for left in range(6) for right in range(left + 1, 6)]
APPROVE = {"decision": "approve", "answer_type_valid": True, "relationships": PAIRS}


class AdjudicationBuilderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        markers = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
                   "<|box_end|>", "<|quad_start|>"]
        vocab = {f"unused_{i}": i for i in range(64000)}
        vocab["[UNK]"] = 0
        del vocab["unused_0"]
        for i, marker in enumerate(markers, 6):
            del vocab[f"unused_{i}"]
            vocab[marker] = i
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = self.root / "tokenizer"
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                additional_special_tokens=markers).save_pretrained(self.tokenizer)
        qwen = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "<|im_start|>": 1, "<|im_end|>": 2},
                                          unk_token="[UNK]"))
        qwen.pre_tokenizer = pre_tokenizers.Whitespace()
        chat = PreTrainedTokenizerFast(tokenizer_object=qwen, unk_token="[UNK]",
                                       additional_special_tokens=["<|im_start|>", "<|im_end|>"])
        chat.chat_template = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                              "{{ message['content'] }}<|im_end|>{% endfor %}"
                              "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        self.qwen = self.root / "qwen"
        chat.save_pretrained(self.qwen)
        self.directory = self.root / "finemath"
        self.directory.mkdir()

    def source(self, *rows):
        pq.write_table(pa.Table.from_pylist([
            {"text": text, "url": f"https://example.org/math/{index}", "snapshot_type": "latest"}
            for index, text in enumerate(rows)]), self.directory / "source.parquet")

    def run_builder(self, replies, **changes):
        requests = []
        replies = iter(replies)

        def http(request, timeout):
            payload = json.loads(request.data)
            requests.append(payload)
            item = next(replies)
            if isinstance(item, Exception):
                raise item
            content = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                                                       "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 2, "completion_tokens": 4}}).encode())

        config = {"sources": {"finemath": str(self.directory)}, "tokenizer_path": str(self.tokenizer),
                  "generator_tokenizer_path": str(self.qwen), "seed": 17, "split_seed": 2,
                  "target": 10, "source_targets": {"finemath": 10}, "max_attempts": 25,
                  "max_seconds": 30, "timeout": 2, "max_packed": 1024,
                  "max_answer_tokens": 32, "max_source_tokens": 4000,
                  "max_context_tokens": 8192, "max_output_tokens": 256, **changes}
        with patch("urllib.request.urlopen", side_effect=http):
            report = build(config, self.root / "suite")
        cases = (load_canonical_suite(self.root / "suite", "train") +
                 load_canonical_suite(self.root / "suite", "development"))
        return report, requests, cases

    def test_finite_set_is_one_answer_and_conflicting_set_is_never_adjudicated(self):
        self.source("Question: Solve x²=4 for x?\nAnswer: x=±2",
                    "Question: Solve x²=4 for x?\nAnswer: x=±2")
        options = ["{-2,2}", "{0,2}", "{-2,0}", "{-1,1}", "{1,2}"]
        report, requests, cases = self.run_builder([{"distractors": options},
                                                     {"distractors": ["{0,1}", "{0,2}", "{1,2}", "{-1,1}", "{-2,0}"]},
                                                     APPROVE])
        self.assertEqual(report["rejected"]["equivalent_candidates"], 1)
        self.assertEqual(report["program_rejected"], 1)
        self.assertEqual(len(requests), 2)  # offered approval is never requested after the hard conflict
        self.assertEqual(cases[0]["gold"], "x=±2")
        self.assertEqual(cases[0]["_meta"]["validation"], "finemath_programmatic")
        self.assertEqual(len(cases), 1)

    def test_scalar_and_finite_set_are_incompatible_even_when_values_are_distinct(self):
        self.source("Question: Evaluate 1+1?\nAnswer: 2",
                    "Question: Solve x²=4 for x?\nAnswer: {-2,2}")
        report, requests, cases = self.run_builder([
            {"distractors": ["{1,2}", "3", "4", "5", "6"]},
            {"distractors": ["1", "{0,2}", "{0,1}", "{-1,1}", "{1,2}"]},
            APPROVE,
        ])
        self.assertEqual(report["rejected"]["answer_type_mismatch"], 2)
        self.assertEqual(report["program_rejected"], 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(report["llm_accepted"], 0)
        self.assertEqual(cases, [])

    def test_known_conflict_after_an_unknown_pair_blocks_approving_llm(self):
        self.source("Question: Evaluate √2+√3?\nAnswer: √2+√3")
        report, requests, cases = self.run_builder([
            {"distractors": ["√3+√5", "1", "2", "1.0", "3"]}, APPROVE])
        self.assertEqual(report["rejected"]["equivalent_candidates"], 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(cases, [])

    def test_condition_dependent_square_root_uses_source_conditions(self):
        self.source("Question: For x>0, simplify √(x²)/x?\nAnswer: 1",
                    "Question: Simplify √(x²)/x for x≠0?\nAnswer: √(x²)/x")
        report, requests, cases = self.run_builder([
            {"distractors": ["√(x²)/x", "-1", "0", "2", "3"]},
            {"distractors": ["1", "-1", "0", "2", "3"]},
            {"decision": "uncertain"},
        ])
        self.assertEqual(report["rejected"]["equivalent_candidates"], 1)
        self.assertEqual(report["unknown_cases"], 1)
        self.assertEqual(report["llm_rejected"], 1)
        self.assertEqual(report["rejected"]["equivalence_unknown"], 1)
        self.assertEqual(len(requests), 3)
        self.assertEqual(cases, [])

    def test_in_scope_unknown_approved_once_with_full_context_and_manifest_policy(self):
        self.source("Question: Evaluate √2+√3?\nAnswer: √2+√3")
        options = ["√2+√5", "√2+√7", "√3+√5", "√3+√7", "√5+√7"]
        report, requests, cases = self.run_builder([{"distractors": options}, APPROVE])
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["unknown_cases"], 1)
        self.assertEqual(report["llm_accepted"], 1)
        self.assertEqual(report["validation_paths"], {"finemath_llm_adjudicated": 1})
        self.assertEqual(cases[0]["gold"], "√2+√3")
        self.assertEqual(cases[0]["distractors"], options)
        self.assertEqual(requests[1]["chat_template_kwargs"]["enable_thinking"], True)
        material = json.loads(requests[1]["messages"][1]["content"])
        self.assertEqual(material["question"], cases[0]["question"])
        self.assertEqual(material["source_answer"], cases[0]["gold"])
        self.assertEqual(material["candidates"], [cases[0]["gold"], *options])
        self.assertNotIn("self_assessment", material)
        manifest = json.loads((self.root / "suite" / "manifest.json").read_text())
        self.assertTrue(manifest["build"]["thinking"]["finemath_adjudicate"])
        self.assertIn("prompt_version", manifest["build"])
        self.assertNotIn("reasoning", cases[0])

    def test_unknown_uncertainty_and_bad_pair_schema_retries_with_one_shared_budget(self):
        self.source("Question: Evaluate √2+√3?\nAnswer: √2+√3")
        options = {"distractors": ["√2+√5", "√2+√7", "√3+√5", "√3+√7", "√5+√7"]}
        invalid = {"decision": "approve", "answer_type_valid": True, "relationships": PAIRS[:-1]}
        report, requests, cases = self.run_builder([options, invalid, invalid, APPROVE], max_attempts=3)
        self.assertEqual(report["stop_reason"], "attempt_limit")
        self.assertEqual(report["attempts"], 3)
        self.assertEqual(report["retries"], 2)
        self.assertEqual(report["failures"]["malformed_response"], 2)
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(cases, [])
        self.assertEqual(len(requests), 3)

    def test_three_malformed_adjudication_replies_reject_without_sample(self):
        self.source("Question: Evaluate √2+√3?\nAnswer: √2+√3")
        options = {"distractors": ["√2+√5", "√2+√7", "√3+√5", "√3+√7", "√5+√7"]}
        report, requests, cases = self.run_builder([options, "not json", "not json", "not json"])
        self.assertEqual(report["rejected"]["malformed_response"], 1)
        self.assertEqual(report["llm_rejected"], 1)
        self.assertEqual(report["attempts"], 4)
        self.assertEqual(len(requests), 4)
        self.assertEqual(cases, [])

    def test_adjudicator_service_outage_stops_run_not_sample(self):
        self.source("Question: Evaluate √2+√3?\nAnswer: √2+√3")
        options = {"distractors": ["√2+√5", "√2+√7", "√3+√5", "√3+√7", "√5+√7"]}
        report, requests, cases = self.run_builder([options, URLError("unavailable"),
                                                     URLError("unavailable"), URLError("unavailable")])
        self.assertEqual(report["stop_reason"], "service_error")
        self.assertEqual(report["attempts"], 4)
        self.assertEqual(report["failures"]["service_error"], 3)
        self.assertEqual(report["llm_rejected"], 0)
        self.assertEqual(report["rejected"], {})
        self.assertEqual(len(requests), 4)
        self.assertEqual(cases, [])

    def test_hard_rejections_block_llm_for_dimensions_prohibited_and_hostile_input(self):
        self.source("Question: Convert 5 cm to meters?\nAnswer: 0.05 m",
                    "Question: Convert 5 Celsius to Fahrenheit?\nAnswer: 41 Fahrenheit",
                    "Question: Evaluate 2+2?\nAnswer: __import__('os').system('touch /tmp/no')",
                    "Question: Evaluate 2+2?\nAnswer: 4",
                    "Question: Evaluate 2+2?\nAnswer: 4")
        report, requests, cases = self.run_builder([
            {"distractors": ["5 g", "4 cm", "6 cm", "7 cm", "8 cm"]},
            {"distractors": ["2**999999", "1", "2", "3", "5"]},
            {"distractors": ["2" * 161, "1", "3", "5", "6"]},
        ])
        self.assertEqual(report["rejected"]["dimension_mismatch"], 1)
        self.assertEqual(report["rejected"]["prohibited_conversion"], 1)
        self.assertEqual(report["rejected"]["unsupported_math"], 2)
        self.assertEqual(report["rejected"]["answer_length"], 1)
        self.assertEqual(len(requests), 3)
        self.assertEqual(cases, [])


if __name__ == "__main__":
    unittest.main()
