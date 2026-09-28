"""Conditional symbolic proof and exact polynomial conflicts at the public builder seam."""
from __future__ import annotations

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


class SymbolicBuilderTests(unittest.TestCase):
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
        chat = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "<|im_start|>": 1, "<|im_end|>": 2},
                                          unk_token="[UNK]"))
        chat.pre_tokenizer = pre_tokenizers.Whitespace()
        generator = PreTrainedTokenizerFast(tokenizer_object=chat, unk_token="[UNK]",
                                            additional_special_tokens=["<|im_start|>", "<|im_end|>"])
        generator.chat_template = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                                   "{{ message['content'] }}<|im_end|>{% endfor %}"
                                   "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        self.generator_tokenizer = self.root / "generator"
        generator.save_pretrained(self.generator_tokenizer)
        self.directory = self.root / "finemath"
        self.directory.mkdir()

    def build_with(self, rows, replies):
        pq.write_table(pa.Table.from_pylist([
            {"text": text, "url": f"https://example.org/{i}", "snapshot_type": "latest"}
            for i, text in enumerate(rows)]), self.directory / "source.parquet")
        requests = []
        replies = iter(replies)

        def http(request, timeout):
            requests.append(json.loads(request.data))
            content = json.dumps(next(replies), ensure_ascii=False)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                                                       "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 2, "completion_tokens": 4}}).encode())

        config = {"sources": {"finemath": str(self.directory)}, "tokenizer_path": str(self.tokenizer),
                  "generator_tokenizer_path": str(self.generator_tokenizer), "seed": 17, "split_seed": 2,
                  "target": 4, "source_targets": {"finemath": 4}, "max_attempts": 10,
                  "max_seconds": 30, "timeout": 2, "max_packed": 1024,
                  "max_answer_tokens": 32, "max_source_tokens": 4000,
                  "max_context_tokens": 8192, "max_output_tokens": 256}
        with patch("urllib.request.urlopen", side_effect=http):
            report = build(config, self.root / "suite")
        cases = (load_canonical_suite(self.root / "suite", "train") +
                 load_canonical_suite(self.root / "suite", "development"))
        return report, requests, cases

    def test_negative_branch_and_implicit_coefficient_conflicts_do_not_reach_llm(self):
        report, requests, cases = self.build_with([
            "Question: For x<0, simplify √(x²)/x?\nAnswer: -1",
            "Question: Expand 2x+1?\nAnswer: 2x+1",
        ], [
            {"distractors": ["√(x²)/x", "0", "1", "2", "3"]},
            {"distractors": ["x+x+1", "x+1", "x+2", "x+3", "x+4"]},
        ])
        self.assertEqual(report["rejected"]["equivalent_candidates"], 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(cases, [])

    def test_unconditioned_root_cannot_be_equated_by_probe_or_case_folding(self):
        report, requests, cases = self.build_with([
            "Question: For x≠0, simplify √(x²)/x?\nAnswer: √(x²)/x",
        ], [
            {"distractors": ["1", "X", "-1", "2", "3"]},
            {"decision": "uncertain"},
        ])
        self.assertEqual(report["unknown_cases"], 1)
        self.assertEqual(report["llm_rejected"], 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(cases, [])

    def test_safe_latex_radical_outside_program_checker_requires_adjudication(self):
        gold = r"\sqrt{2}"
        options = [rf"\sqrt{{{n}}}" for n in (3, 5, 6, 7, 10)]
        pairs = [{"left": left, "right": right, "relation": "distinct"}
                 for left in range(6) for right in range(left + 1, 6)]
        report, requests, cases = self.build_with([
            "Question: Evaluate the square root of two?\nAnswer: " + gold,
        ], [
            {"distractors": options},
            {"decision": "approve", "answer_type_valid": True, "relationships": pairs},
        ])
        self.assertEqual(report["unknown_cases"], 1)
        self.assertEqual(report["validation_paths"], {"finemath_llm_adjudicated": 1})
        self.assertEqual(len(requests), 2)
        self.assertEqual(cases[0]["gold"], gold)
        self.assertEqual(cases[0]["distractors"], options)

    def test_ambiguous_exponent_and_pm_scope_reject_before_adjudication(self):
        report, requests, cases = self.build_with([
            "Question: Calculate 2 to the power of 3 squared?\nAnswer: 512",
            "Question: Calculate 2+3?\nAnswer: 5",
        ], [
            {"distractors": ["2^3^2", "1", "2", "3", "4"]},
            {"distractors": ["±2+3", "1", "2", "3", "4"]},
        ])
        self.assertEqual(report["rejected"]["unsupported_math"], 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(cases, [])

    def test_negated_domain_assumption_remains_unknown(self):
        report, requests, cases = self.build_with([
            "Question: Without assuming x>0, simplify √(x²)/x for x≠0?\nAnswer: √(x²)/x",
        ], [
            {"distractors": ["1", "-1", "0", "2", "3"]},
            {"decision": "uncertain"},
        ])
        self.assertEqual(report["unknown_cases"], 1)
        self.assertEqual(report["llm_rejected"], 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(cases, [])

    def test_bounded_polynomial_ceiling_delegates_instead_of_rejecting_scope(self):
        report, requests, cases = self.build_with([
            "Question: Express (a+b+c+d)^8?\nAnswer: (a+b+c+d)^8",
        ], [
            {"distractors": ["1", "2", "3", "4", "5"]},
            {"decision": "uncertain"},
        ])
        self.assertEqual(report["unknown_cases"], 1)
        self.assertEqual(report["llm_rejected"], 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(cases, [])


if __name__ == "__main__":
    unittest.main()
