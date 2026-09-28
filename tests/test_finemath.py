"""Original FineMath Parquet through public builder and manifest-verified canonical suite."""
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
from haidass_kev_train.data.canonical import load_canonical_suite, preflight
from haidass_kev_train.data.packing import check_group_integrity


NUMBERS = ["1", "2", "3", "4", "6"]


class FineMathBuilderTests(unittest.TestCase):
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
        self.directory = self.root / "finemath-4plus"
        self.directory.mkdir()

    def source(self, entries):
        pq.write_table(pa.Table.from_pylist([
            {"text": text, "url": url, "snapshot_type": snap}
            for text, url, snap in entries]), self.directory / "train-00000-of-00001.parquet")

    def run_builder(self, replies, **changes):
        requests = []
        iterator = iter(replies)

        def http(request, timeout):
            payload = json.loads(request.data)
            requests.append(payload)
            answer = next(iterator)
            if callable(answer):
                answer = answer(payload)
            content = json.dumps(answer)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                                                       "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 2, "completion_tokens": 4}}).encode())

        config = {"sources": {"finemath": str(self.directory)}, "tokenizer_path": str(self.tokenizer),
                  "generator_tokenizer_path": str(self.qwen), "seed": 17, "split_seed": 2,
                  "target": 20, "source_targets": {"finemath": 20}, "max_attempts": 80,
                  "max_seconds": 30, "timeout": 2, "max_packed": 1024,
                  "max_answer_tokens": 32, "max_source_tokens": 4000,
                  "max_context_tokens": 8192, "max_output_tokens": 256, **changes}
        with patch("urllib.request.urlopen", side_effect=http):
            report = build(config, self.root / "suite")
        return report, requests

    def cases(self):
        return load_canonical_suite(self.root / "suite", "train") + load_canonical_suite(self.root / "suite", "development")

    def test_frozen_numeric_and_unit_suite_groups_snapshots_before_filtering(self):
        equation = "Given: 3x+5=20\nQuestion: Find x?\nSolution: Subtract 5 and divide by 3.\nFinal answer: 5"
        self.source([(equation, "https://example.org/item/3?v=1", "latest"),
                     (equation, "https://example.org/item/3?v=1", "longest"),
                     ("Question: What length is 5 cm in meters?\nAnswer: 0.05 m",
                      "https://example.org/item/4?v=1", "latest"),
                     ("Question: How fast is 36 km/h in m/s?\nAnswer: 10 m/s",
                      "https://example.org/item/5?v=1", "latest"),
                     ("Question: Evaluate negative one quarter?\nAnswer: -2.5e-1",
                      "https://example.org/item/6?v=1", "latest")])
        replies = [{"distractors": NUMBERS}, {"distractors": NUMBERS},
                   {"distractors": ["0.04 m", "0.06 m", "0.07 m", "0.08 m", "0.09 m"]},
                   {"distractors": ["8 m/s", "9 m/s", "11 m/s", "12 m/s", "13 m/s"]},
                   {"distractors": ["-0.2", "-0.3", "0.25", "-0.4", "-0.5"]}]
        report, requests = self.run_builder(replies)
        cases = self.cases()
        self.assertEqual(len(cases), 5)
        self.assertEqual(report["validation_paths"], {"finemath_programmatic": 5})
        self.assertEqual(report["attempts"], 5)
        self.assertTrue(all(req["chat_template_kwargs"]["enable_thinking"] for req in requests))
        snapshots = [case for case in cases if case["_meta"]["source_ref"]["url"].endswith("/3?v=1")]
        self.assertEqual(len({case["_meta"]["group_id"] for case in snapshots}), 1)
        self.assertEqual(len({case["_meta"]["id"] for case in snapshots}), 2)
        self.assertTrue(all("Subtract 5" not in case["state"] + case["question"] for case in cases))
        self.assertEqual(next(case for case in cases if case["gold"] == "5")["question"], "Find x?")
        self.assertTrue(all("3x+5=20" in case["state"] for case in snapshots))
        for case in cases:
            source = next(row["text"] for row in pq.read_table(self.directory / "train-00000-of-00001.parquet").to_pylist()
                          if hashlib.sha256(row["text"].encode()).hexdigest() == case["_meta"]["source_ref"]["sha256"])
            ref = case["_meta"]["source_ref"]
            self.assertEqual(source[slice(*ref["question_span"])], case["question"])
            self.assertEqual(source[slice(*ref["answer_span"])], case["gold"])
            if "givens_span" in ref:
                self.assertEqual(source[slice(*ref["givens_span"])], case["state"])
        self.assertEqual(check_group_integrity(self.root / "suite", splits=("train", "development"))["overlaps"]["train|development"], 0)
        self.assertGreater(len(load_canonical_suite(self.root / "suite", "development")), 0)
        preflight(cases, PreTrainedTokenizerFast.from_pretrained(self.tokenizer), max_packed=1024)

    def test_url_queries_remain_distinct_and_missing_urls_group_by_raw_text(self):
        text = "Question: Given 3x+5=20, find x?\nAnswer: 5"
        self.source([(text, "https://example.org/item?v=1", "latest"),
                     (text, "https://example.org/item?v=2", "latest"),
                     (text, None, "latest"), (text, None, "longest"),
                     (text, None, None)])
        report, _ = self.run_builder([{"distractors": NUMBERS} for _ in range(5)])
        cases = self.cases()
        self.assertEqual(len(cases), 5)
        urls = {case["_meta"]["source_ref"]["url"]: case["_meta"]["group_id"] for case in cases
                if case["_meta"]["source_ref"]["url"]}
        self.assertNotEqual(urls["https://example.org/item?v=1"], urls["https://example.org/item?v=2"])
        fallback = [case for case in cases if case["_meta"]["source_ref"]["url"] is None]
        self.assertEqual(len({case["_meta"]["group_id"] for case in fallback}), 1)
        self.assertEqual(report["groups_seen"]["finemath"], 3)

    def test_exact_equivalent_gold_and_pairwise_distractors_reject_before_adjudication(self):
        self.source([("Question: What is one half?\nAnswer: 0.5", "https://example.org/1", "latest"),
                     ("Question: What is a meter squared?\nAnswer: 1 m²", "https://example.org/2", "latest"),
                     ("Question: What speed is 36 km/h?\nAnswer: 36 km/h", "https://example.org/3", "latest"),
                     ("Question: Convert 5 cm to m?\nAnswer: 0.05 m", "https://example.org/4", "latest"),
                     ("Question: Express one half as a percentage?\nAnswer: 50%", "https://example.org/5", "latest")])
        report, requests = self.run_builder([
            {"distractors": ["1/2", "1", "2", "3", "4"]},
            {"distractors": ["10000 cm²", "2 m²", "3 m²", "4 m²", "5 m²"]},
            {"distractors": ["10 m/s", "20 m/s", "30 m/s", "40 m/s", "50 m/s"]},
            {"distractors": ["4 cm", "0.04 m", "6 cm", "7 cm", "8 cm"]},
            {"distractors": ["0.5", "10%", "20%", "30%", "40%"]},
        ])
        self.assertEqual(report["rejected"]["equivalent_candidates"], 5)
        self.assertEqual(report["attempts"], 5)
        self.assertEqual(len(requests), 5)
        self.assertEqual(self.cases(), [])

    def test_dimension_mismatch_and_unit_case_unknown_reject_without_adjudication(self):
        self.source([("Question: What volume is 5 mL?\nAnswer: 5 mL",
                      "https://example.org/volume", "latest"),
                     ("Question: What is 5 cm in meters?\nAnswer: 0.05 m",
                      "https://example.org/length", "latest")])
        report, requests = self.run_builder([
            {"distractors": ["4 mL", "6 mL", "7 mL", "8 mL", "5 ML"]},
            {"distractors": ["5 g", "4 cm", "6 cm", "7 cm", "8 cm"]},
        ])
        self.assertEqual(report["rejected"]["equivalence_unknown"], 1)
        self.assertEqual(report["rejected"]["dimension_mismatch"], 1)
        self.assertEqual(report["validation_paths"], {})
        self.assertEqual(report["unknown_cases"], 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(self.cases(), [])

    def test_symbolic_unknown_is_not_programmatic_proof_or_hidden_llm_approval(self):
        self.source([("Question: What is the value of √2?\nAnswer: √2", "https://example.org/1", "latest"),
                     ("Question: What is √3+√5?\nAnswer: √3+√5", "https://example.org/2", "latest"),
                     ("Question: What is the value of x?\nAnswer: x", "https://example.org/2-missing", "latest"),
                     ("Question: Convert 5 Celsius to Fahrenheit?\nAnswer: 41 Fahrenheit", "https://example.org/3", "latest"),
                     ("Question: What is shown in the diagram?\nAnswer: 2", "https://example.org/4", "latest"),
                     ("Question: Find x?\nSolution: x=2", "https://example.org/5", "latest")])
        report, requests = self.run_builder([])
        self.assertEqual(report["validation_paths"], {})
        self.assertEqual(report["unknown_cases"], 2)
        self.assertEqual(report["rejected"]["equivalence_unknown"], 2)
        self.assertEqual(report["rejected"]["prohibited_conversion"], 1)
        self.assertEqual(report["rejected"]["missing_figure_or_conditions"], 2)
        self.assertEqual(report["rejected"]["missing_source_answer"], 1)
        self.assertEqual(len(requests), 0)
        self.assertEqual(self.cases(), [])

    def test_assisted_location_is_original_and_thinking_enabled(self):
        text = "Question: Find the result of 1+1?\nAnswer: 2\nAdditional unrelated footer"
        self.source([(text, "https://example.org/3?v=2", "latest")])
        q = text.index("Find")
        a = text.index("2\nAdditional")
        answer = {"question_span": [q, text.index("?", q) + 1], "answer_span": [a, a + 1], "complete": True}
        report, requests = self.run_builder([answer, {"distractors": ["1", "3", "4", "5", "6"]}])
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(req["chat_template_kwargs"]["enable_thinking"] for req in requests))
        self.assertEqual(self.cases()[0]["gold"], "2")

    def test_assisted_offsets_cannot_promote_solution_to_source_answer_or_question(self):
        text = "Question: Find 1+1?\nSolution: 2\nAnswer: 2\nFooter"
        self.source([(text, "https://example.org/first", "latest"),
                     (text, "https://example.org/second", "latest")])
        q = text.index("Find")
        solution = text.index("2\nAnswer")
        final = text.index("2\nFooter")
        replies = [
            {"question_span": [q, text.index("?", q) + 1],
             "answer_span": [solution, solution + 1], "complete": True},
            {"question_span": [q, text.index("\nAnswer")],
             "answer_span": [final, final + 1], "complete": True},
        ]
        report, requests = self.run_builder(replies)
        self.assertEqual(report["rejected"]["invalid_source_location"], 1)
        self.assertEqual(report["rejected"]["solution_leakage"], 1)
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(self.cases(), [])

    def test_oversized_or_unsupported_math_cannot_be_admitted_by_text_difference(self):
        self.source([("Question: What is 5?\nAnswer: 5", "https://example.org/large", "latest"),
                     ("Question: What is 5?\nAnswer: 5", "https://example.org/exponent", "latest"),
                     ("Question: What is 5?\nAnswer: __import__('os').system('touch /tmp/never')",
                      "https://example.org/code", "latest")])
        report, requests = self.run_builder([
            {"distractors": ["1" * 161, "2", "3", "4", "6"]},
            {"distractors": ["1e999999", "2", "3", "4", "6"]},
        ])
        self.assertEqual(report["rejected"]["answer_length"], 1)
        self.assertEqual(report["rejected"]["equivalence_unknown"], 2)
        self.assertEqual(report["unknown_cases"], 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(self.cases(), [])

    def test_unlabelled_prefix_requires_verified_givens_and_is_never_dropped(self):
        text = ("A rectangle has length 5 cm and width 4 cm.\n"
                "Question: What is its area?\nAnswer: 20 cm²")
        self.source([(text, "https://example.org/rectangle", "latest"),
                     (text, "https://example.org/ignored-prefix", "latest")])
        question_start = text.index("What is")
        answer_start = text.index("20 cm²")
        located = {"question_span": [question_start, text.index("?", question_start) + 1],
                   "answer_span": [answer_start, answer_start + len("20 cm²")],
                   "givens_span": [0, text.index("\nQuestion")], "complete": True}
        report, requests = self.run_builder([
            located,
            {"distractors": ["10 cm²", "12 cm²", "15 cm²", "18 cm²", "24 cm²"]},
            {key: value for key, value in located.items() if key != "givens_span"},
        ])
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["rejected"]["incomplete_problem"], 1)
        self.assertEqual(report["attempts"], 3)
        self.assertTrue(all(request["chat_template_kwargs"]["enable_thinking"] for request in requests))
        case = self.cases()[0]
        self.assertEqual(case["state"], "A rectangle has length 5 cm and width 4 cm.")
        self.assertEqual(case["question"], "What is its area?")
        self.assertEqual(text[slice(*case["_meta"]["source_ref"]["givens_span"])], case["state"])

    def test_assisted_answer_span_must_include_entire_existing_answer(self):
        text = "Question: What is 5×5?\nAnswer: 25\nFooter"
        self.source([(text, "https://example.org/partial", "latest")])
        start = text.index("What is")
        gold = text.index("25")
        report, requests = self.run_builder([{
            "question_span": [start, text.index("?", start) + 1],
            "answer_span": [gold, gold + 1], "complete": True,
        }])
        self.assertEqual(report["rejected"]["invalid_source_location"], 1)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.cases(), [])

    def test_unlabelled_worked_solution_rejects_but_source_given_keeps_answer_digit(self):
        self.source([("Question: What is 2+3?\nAdding 2 and 3 gives 5.\nAnswer: 5",
                      "https://example.org/leak", "latest"),
                     ("Given: x=5\nQuestion: Find x?\nAnswer: 5",
                      "https://example.org/given", "latest")])
        report, requests = self.run_builder([{"distractors": NUMBERS}])
        self.assertEqual(report["rejected"]["solution_leakage"], 1)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.cases()[0]["state"], "Given: x=5")

    def test_imperative_worked_line_rejects_without_banning_source_givens(self):
        self.source([("Question: Calculate 2+3.\nAdding 2 and 3 gives 5.\nAnswer: 5",
                      "https://example.org/imperative-leak", "latest"),
                     ("Question: Calculate x.\nGiven: x=5\nAnswer: 5",
                      "https://example.org/imperative-given", "latest")])
        report, requests = self.run_builder([{"distractors": NUMBERS}])
        self.assertEqual(report["rejected"]["solution_leakage"], 1)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.cases()[0]["question"], "Calculate x.\nGiven: x=5")

    def test_completed_imperative_rejects_same_line_working_and_unlabelled_equation(self):
        self.source([("Question: Calculate 2+3. Adding 2 and 3 gives 5.\nAnswer: 5",
                      "https://example.org/same-line-working", "latest"),
                     ("Question: Solve 2x=10.\nx=5\nAnswer: 5",
                      "https://example.org/unlabelled-equation", "latest"),
                     ("Question: Solve x.\nGiven: x=5\nAnswer: 5",
                      "https://example.org/explicit-given", "latest")])
        report, requests = self.run_builder([{"distractors": NUMBERS}])
        self.assertEqual(report["rejected"]["solution_leakage"], 2)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.cases()[0]["question"], "Solve x.\nGiven: x=5")

    def test_unpunctuated_imperative_requires_labelled_continuation(self):
        self.source([("Question: Calculate 2+3\nAdding 2 and 3 gives 5.\nAnswer: 5",
                      "https://example.org/no-boundary-leak", "latest"),
                     ("Question: Calculate 2+3\nAnswer: 5",
                      "https://example.org/single-line-imperative", "latest"),
                     ("Question: Calculate x\nGiven: x=5\nAnswer: 5",
                      "https://example.org/labelled-continuation", "latest")])
        report, requests = self.run_builder([{"distractors": NUMBERS},
                                             {"distractors": NUMBERS}])
        self.assertEqual(report["rejected"]["solution_leakage"], 1)
        self.assertEqual(report["accepted"], 2)
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual({case["question"] for case in self.cases()},
                         {"Calculate 2+3", "Calculate x\nGiven: x=5"})

    def test_worked_conclusion_in_question_sentence_rejects_without_gold_substring_rule(self):
        self.source([("Question: Calculate 2+3; adding 2 and 3 gives 5.\nAnswer: 5",
                      "https://example.org/embedded-working", "latest"),
                     ("Given: x=5\nQuestion: Calculate x?\nAnswer: 5",
                      "https://example.org/legitimate-given", "latest")])
        report, requests = self.run_builder([{"distractors": NUMBERS}])
        self.assertEqual(report["rejected"]["solution_leakage"], 1)
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.cases()[0]["state"], "Given: x=5")

    def test_equation_chain_working_rejects_without_rejecting_independent_givens(self):
        self.source([("Question: Solve 2x=10; x=10/2=5.\nAnswer: 5",
                      "https://example.org/chain", "latest"),
                     ("Question: Solve 2x=10?\nAnswer: 5",
                      "https://example.org/ordinary-equation", "latest"),
                     ("Given: x=5\nQuestion: Find x?\nAnswer: 5",
                      "https://example.org/single-given", "latest")])
        report, requests = self.run_builder([{"distractors": NUMBERS},
                                             {"distractors": NUMBERS}])
        self.assertEqual(report["rejected"]["solution_leakage"], 1)
        self.assertEqual(report["accepted"], 2)
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual({case["question"] for case in self.cases()},
                         {"Solve 2x=10?", "Find x?"})

    def test_absolute_temperature_abbreviations_reject_before_generation(self):
        self.source([("Question: Convert 273.15 K to C?\nAnswer: 0 C",
                      "https://example.org/temperature", "latest")])
        report, requests = self.run_builder([])
        self.assertEqual(report["rejected"]["prohibited_conversion"], 1)
        self.assertEqual(report["attempts"], 0)
        self.assertEqual(requests, [])

    def test_missing_optional_url_and_snapshot_normalize_to_raw_identity(self):
        self.source([("Question: What is 2+3?\nAnswer: 5", "   ", "  ")])
        report, _ = self.run_builder([{"distractors": NUMBERS}])
        self.assertEqual(report["accepted"], 1)
        case = self.cases()[0]
        self.assertIsNone(case["_meta"]["source_ref"]["url"])
        self.assertIsNone(case["_meta"]["source_ref"]["snapshot_type"])
        self.assertEqual(report["groups_seen"]["finemath"], 1)

    def test_invalid_optional_identity_type_rejects_row_without_generation(self):
        self.source([("Question: What is 2+3?\nAnswer: 5", 42, None)])
        report, requests = self.run_builder([])
        self.assertEqual(report["rejected"]["source_schema"], 1)
        self.assertEqual(report["attempts"], 0)
        self.assertEqual(requests, [])
        self.assertEqual(self.cases(), [])

    def test_blank_line_prefix_does_not_confuse_label_boundaries(self):
        text = "\n" * 5000 + "Question: What is 2+3?\nAnswer: 5"
        self.source([(text, "https://example.org/blank", "latest")])
        report, _ = self.run_builder([{"distractors": NUMBERS}])
        self.assertEqual(report["accepted"], 1)
        self.assertEqual(self.cases()[0]["question"], "What is 2+3?")
