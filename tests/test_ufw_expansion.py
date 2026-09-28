"""UFW recovery and conversion through the public builder and controlled external HTTP."""
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
from haidass_kev_train.data.canonical import evaluation_views, load_canonical_suite
from haidass_kev_train.data.packing import encode_record


DISTRACTORS = ["Milan", "Venice", "Naples", "Turin", "Genoa"]
SCREEN = {"supported": True, "unique": True, "all_wrong": True, "same_format": True}


def located(raw, state, question, answer, *, option_texts=()):
    q_start = raw.index(question, len(state))
    a_start = raw.index(answer, q_start + len(question))
    options = [[raw.index(text, q_start, a_start), raw.index(text, q_start, a_start) + len(text)]
               for text in option_texts]
    return {"sha256": hashlib.sha256(raw.encode()).hexdigest(), "state_span": [0, len(state)],
            "state_text": state, "qas": [{"question_span": [q_start, q_start + len(question)],
                                         "question_text": question, "answer_span": [a_start, a_start + len(answer)],
                                         "answer_text": answer, "option_spans": options,
                                         "option_texts": list(option_texts)}]}


class UfwExpansionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        markers = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
                   "<|box_end|>", "<|quad_start|>"]
        vocab = {f"unused_{i}": i for i in range(64000)}
        vocab["[UNK]"] = 0
        for i, marker in enumerate(markers, 6):
            del vocab[f"unused_{i}"]
            vocab[marker] = i
        del vocab["unused_0"]
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.train_tokenizer = self.root / "train-tokenizer"
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                additional_special_tokens=markers).save_pretrained(self.train_tokenizer)
        generator = Tokenizer(models.WordLevel(vocab={"[UNK]": 0, "<|im_start|>": 1,
                                                "<|im_end|>": 2}, unk_token="[UNK]"))
        generator.pre_tokenizer = pre_tokenizers.Whitespace()
        chat = PreTrainedTokenizerFast(tokenizer_object=generator, unk_token="[UNK]",
                                       additional_special_tokens=["<|im_start|>", "<|im_end|>"])
        chat.chat_template = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                              "{{ message['content'] }}<|im_end|>{% endfor %}"
                              "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        self.generator_tokenizer = self.root / "generator-tokenizer"
        chat.save_pretrained(self.generator_tokenizer)

    def source(self, rows, language="en"):
        folder = self.root / f"ultrafineweb_{language}_l3" / "qa"
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([{"uid": uid, "content": text, "style": "qa"}
                                              for uid, text in rows]), folder / "part.parquet")
        return {f"ufw-{language}": str(folder)}

    def run_build(self, sources, replies, *, output="suite", **overrides):
        calls = []
        incoming = iter(replies)

        def http(request, timeout):
            payload = json.loads(request.data)
            calls.append(payload)
            reply = next(incoming)
            content = json.dumps(reply, ensure_ascii=False)
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                                                       "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 3, "completion_tokens": 5}}).encode())

        config = {"sources": sources, "source_targets": {name: 2 for name in sources},
                  "tokenizer_path": str(self.train_tokenizer),
                  "generator_tokenizer_path": str(self.generator_tokenizer),
                  "seed": 17, "split_seed": 2, "target": 2, "max_attempts": 24,
                  "max_seconds": 30, "timeout": 2, "max_packed": 1024,
                  "max_answer_tokens": 32, "max_source_tokens": 4000,
                  "max_context_tokens": 8192, "max_output_tokens": 256, **overrides}
        with patch("urllib.request.urlopen", side_effect=http):
            report = build(config, self.root / output)
        records = load_canonical_suite(self.root / output, "train") + load_canonical_suite(
            self.root / output, "development")
        return report, records, calls

    def test_assisted_chinese_locations_create_consumable_suite(self):
        state = "图册记载意大利的首都是罗马。"
        raw = state + "\n题目：意大利首都是什么？ 答：罗马"
        location = located(raw, state, "意大利首都是什么？", "罗马")
        # Source also contains 罗马 in the document: only the original answer-field offset is valid.
        sources = self.source([("zh-1", raw)], "zh")
        report, records, calls = self.run_build(sources, [location, {"distractors": ["米兰", "威尼斯", "那不勒斯", "都灵", "热那亚"]}, SCREEN], target=1)
        self.assertEqual((report["accepted"], report["attempts"]), (1, 3))
        self.assertEqual(len(records), 1)
        record = records[0]
        ref = record["_meta"]["source_ref"]
        self.assertEqual((record["state"], record["question"], record["gold"]),
                         (state, "意大利首都是什么？", "罗马"))
        self.assertEqual(ref["sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(raw[slice(*ref["question_span"])], record["question"])
        self.assertEqual(raw[slice(*ref["answer_span"])], record["gold"])
        self.assertTrue(all(call["chat_template_kwargs"]["enable_thinking"] is False for call in calls))
        view = evaluation_views(record, seed=17, purpose="test")[0]
        encoded = encode_record(view, PreTrainedTokenizerFast.from_pretrained(self.train_tokenizer), max_packed=1024)
        self.assertGreater(len(encoded.input_ids), 0)

    def test_assisted_wrong_hash_text_bounds_order_and_qa_leak_reject(self):
        state = "图册记载意大利的首都是罗马。"
        raw = state + "\n题目：意大利首都是什么？ 答：罗马"
        for index, corruption in enumerate(("hash", "text", "bounds", "order", "leak")):
            with self.subTest(corruption=corruption):
                answer = located(raw, state, "意大利首都是什么？", "罗马")
                qa = answer["qas"][0]
                if corruption == "hash":
                    answer["sha256"] = "0" * 64
                elif corruption == "text":
                    qa["answer_text"] = "米兰"
                elif corruption == "bounds":
                    qa["answer_span"] = [len(raw), len(raw) + 1]
                elif corruption == "order":
                    qa["answer_span"] = [0, 2]
                    qa["answer_text"] = raw[:2]
                else:
                    answer["state_span"] = [0, len(raw)]
                    answer["state_text"] = raw
                sources = self.source([("zh-1", raw)], "zh")
                report, records, calls = self.run_build(sources, [answer], output=f"bad-{index}", target=1)
                self.assertEqual(report["rejected"].get("assisted_location"), 1)
                self.assertEqual((records, len(calls)), ([], 1))

    def test_assisted_state_cannot_contain_prior_nonstandard_qa(self):
        document = "图册记载意大利的首都是罗马。"
        first = "\n题目：意大利首都是什么？ 答：罗马"
        raw = document + first + "\n题目：意大利首都叫什么？ 答：罗马"
        location = located(raw, document + first, "意大利首都叫什么？", "罗马")
        report, records, calls = self.run_build(self.source([("leak", raw)], "zh"),
                                                [location], target=1)
        self.assertEqual(report["rejected"].get("assisted_location"), 1)
        self.assertEqual((records, len(calls)), ([], 1))

    def test_assisted_cannot_merge_or_shift_two_identical_qa_fields(self):
        state = "图册记载意大利的首都是罗马。"
        raw = (state + "\n题目：意大利首都是什么？ 答：罗马"
               "\n题目：意大利首都叫什么？ 答：罗马")
        for name in ("question_swallow", "answer_swallow"):
            with self.subTest(name=name):
                location = located(raw, state, "意大利首都是什么？", "罗马")
                qa = location["qas"][0]
                if name == "question_swallow":
                    qa["question_span"][1] = raw.index(" 答：罗马", raw.index("题目：意大利首都叫什么？"))
                    qa["question_text"] = raw[slice(*qa["question_span"])]
                    qa["answer_span"] = [raw.rindex("罗马"), len(raw)]
                    qa["answer_text"] = "罗马"
                else:
                    qa["answer_span"] = [qa["answer_span"][0], len(raw)]
                    qa["answer_text"] = raw[slice(*qa["answer_span"])]
                report, records, calls = self.run_build(self.source([("two", raw)], "zh"),
                                                        [location], output=name, target=1)
                self.assertEqual(report["rejected"].get("assisted_location"), 1)
                self.assertEqual((records, len(calls)), ([], 1))

    def test_source_mcq_maps_unique_B_to_rome_and_preserves_option_trace(self):
        state = "The atlas lists Italy's capital as Rome."
        mcq = (state + "\nQuestion: Which city is Italy's capital? "
               "A) Paris B) Rome C) Milan D) Turin Answer: B")
        short = state + "\nQuestion: Which city is Italy's capital? Answer: Rome"
        sources = self.source([("mcq", mcq), ("short", short)])
        replies = [{"distractors": DISTRACTORS}, SCREEN] * 2
        report, records, calls = self.run_build(sources, replies)
        self.assertEqual((report["accepted"], len(calls)), (2, 4))
        self.assertEqual(len({record["_meta"]["group_id"] for record in records}), 1)
        self.assertEqual({record["gold"] for record in records}, {"Rome"})
        converted = next(record for record in records if record["_meta"]["source_ref"].get("mcq", False))
        ref = converted["_meta"]["source_ref"]
        self.assertEqual(converted["question"], "Which city is Italy's capital?")
        self.assertEqual(mcq[slice(*ref["question_span"])].count("B) Rome"), 1)
        self.assertEqual([mcq[slice(*span)] for span in ref["option_spans"]], ["Paris", "Rome", "Milan", "Turin"])
        self.assertEqual(mcq[slice(*ref["answer_span"])], "B")
        self.assertEqual(mcq[slice(*ref["state_span"])], state)

    def test_literal_B_remains_short_answer_but_ambiguous_mcq_B_rejects(self):
        short = "A diagram contains the letter B.\nQuestion: Which letter is printed? Answer: B"
        mcq = ("A diagram contains Rome and Paris.\nQuestion: Which city is named? "
               "A) B B) Rome C) Paris D) Milan Answer: B")
        sources = self.source([("short", short), ("ambiguous", mcq)])
        report, records, calls = self.run_build(sources, [{"distractors": ["A", "C", "D", "E", "F"]}, SCREEN])
        self.assertEqual(report["rejected"].get("answer_mapping_ambiguous"), 1)
        self.assertEqual((len(records), records[0]["gold"], len(calls)), (1, "B", 2))
        self.assertFalse(records[0]["_meta"]["source_ref"].get("mcq", False))

    def test_literal_conjunctions_inside_verified_option_text_remain_gold(self):
        cases = (
            ("en", "The map names Trinidad and Tobago.",
             "Which country is on the map? A) Canada B) Trinidad and Tobago C) Chile D) Peru",
             "B) Trinidad and Tobago", "Trinidad and Tobago",
             ["Mexico", "Brazil", "Argentina", "Uruguay", "Ecuador"]),
            ("zh", "地图标注成都。",
             "地图标注哪座城市？ A) 北京 B) 成都 C) 南京 D) 西安",
             "B", "成都", ["武汉", "深圳", "广州", "天津", "杭州"]))
        for index, (language, state, question, answer, gold, distractors) in enumerate(cases):
            with self.subTest(language=language):
                labels = ("Question: ", " Answer: ") if language == "en" else ("问题：", " 答案：")
                raw = state + "\n" + labels[0] + question + labels[1] + answer
                report, records, calls = self.run_build(
                    self.source([("literal", raw)], language),
                    [{"distractors": distractors}, SCREEN], output=f"literal-{index}", target=1)
                self.assertEqual((report["accepted"], len(calls)), (1, 2))
                self.assertEqual(records[0]["gold"], gold)
                self.assertTrue(records[0]["_meta"]["source_ref"]["mcq"])

    def test_positional_answer_maps_only_with_source_options(self):
        state = "The atlas lists Italy's capital as Rome."
        raw = (state + "\nQuestion: Which city is Italy's capital? "
               "A) Paris B) Rome C) Milan D) Turin Answer: 2")
        report, records, calls = self.run_build(self.source([("position", raw)]),
                                                [{"distractors": DISTRACTORS}, SCREEN], target=1)
        self.assertEqual((report["accepted"], len(calls)), (1, 2))
        self.assertEqual(records[0]["gold"], "Rome")
        self.assertEqual(raw[slice(*records[0]["_meta"]["source_ref"]["answer_span"])], "2")

    def test_chinese_fullwidth_options_and_second_position(self):
        state = "图册记载意大利的首都是罗马。"
        raw = (state + "\n问题：意大利首都是哪里？ "
               "（Ａ）巴黎 （Ｂ）罗马 （Ｃ）米兰 答案：第二项")
        report, records, calls = self.run_build(
            self.source([("fullwidth", raw)], "zh"),
            [{"distractors": ["巴黎", "威尼斯", "都灵", "热那亚", "那不勒斯"]}, SCREEN], target=1)
        self.assertEqual((report["accepted"], len(calls)), (1, 2))
        self.assertEqual(records[0]["gold"], "罗马")
        ref = records[0]["_meta"]["source_ref"]
        self.assertEqual(raw[slice(*ref["answer_span"])], "第二项")
        self.assertEqual([raw[slice(*span)] for span in ref["option_spans"]], ["巴黎", "罗马", "米兰"])

    def test_cheap_ineligible_qa_is_removed_before_single_selection(self):
        state = "The atlas lists Italy's capital as Rome."
        raw = (state + "\nQuestion: Which city? A) B B) Rome C) Paris D) Milan Answer: B"
               "\nQuestion: What is Italy's capital? Answer: Rome")
        report, records, calls = self.run_build(self.source([("two", raw)]),
                                                [{"distractors": DISTRACTORS}, SCREEN], target=1)
        self.assertEqual((report["accepted"], len(calls)), (1, 2))
        self.assertEqual(records[0]["question"], "What is Italy's capital?")

    def test_deterministic_count_cleanup_and_assisted_cleanup_preserve_negation(self):
        state = "图册记载罗马是意大利首都，米兰不是意大利首都。"
        base = "哪一个城市被图册明确记载为非意大利首都？ A) 罗马 B) 米兰 C) 巴黎 D) 都灵"
        easy = state + "\n问题：下面四项描述，" + base + " 答案：B"
        assisted = state + "\n问题：以下4个选项中，" + base + " 答案：B"
        sources = self.source([("easy", easy), ("assisted", assisted)], "zh")
        choices = ["威尼斯", "热那亚", "那不勒斯", "佛罗伦萨", "博洛尼亚"]
        report, records, calls = self.run_build(sources, [
            {"distractors": choices}, SCREEN,
            {"question": "下列哪一个城市被图册明确记载为非意大利首都？"}, {"distractors": choices}, SCREEN])
        self.assertEqual((report["accepted"], len(calls)), (2, 5))
        self.assertEqual({record["question"] for record in records},
                         {"下列描述，哪一个城市被图册明确记载为非意大利首都？",
                          "下列哪一个城市被图册明确记载为非意大利首都？"})
        self.assertEqual({record["gold"] for record in records}, {"米兰"})
        self.assertTrue(all("A)" not in record["question"] for record in records))

    def test_semantic_changes_dependencies_and_multi_correct_are_rejected(self):
        state = "图册记载罗马是意大利首都，米兰不是意大利首都。"
        source = state + "\n问题：以下4个选项中，哪一个城市被图册明确记载为非意大利首都？ A) 罗马 B) 米兰 C) 巴黎 D) 都灵 答案：B"
        dependent = state + "\n问题：A比B更靠北的是谁？ A) 罗马 B) 米兰 C) 巴黎 D) 都灵 答案：B"
        multiple = state + "\n问题：哪些选项都不是意大利首都？ A) 罗马 B) 米兰 C) 巴黎 D) 都灵 答案：B"
        for i, (raw, replies, reason) in enumerate((
            (source, [{"question": "下列哪一个城市被图册明确记载为意大利首都？"}], "presentation_conversion"),
            (dependent, [], "presentation_conversion"),
            (multiple, [], "answer_mapping_ambiguous"))):
            with self.subTest(reason=reason, index=i):
                sources = self.source([("one", raw)], "zh")
                report, records, calls = self.run_build(sources, replies, output=f"invalid-{i}", target=1)
                self.assertEqual(report["rejected"].get(reason), 1)
                self.assertEqual((records, len(calls)), ([], len(replies)))

    def test_fixed_three_options_cannot_survive_dynamic_candidate_conversion(self):
        raw = ("The atlas says Rome is Italy's capital.\n"
               "Question: Which of these three options is Italy's capital? "
               "A) Paris B) Rome C) Milan Answer: B")
        report, records, calls = self.run_build(self.source([("three", raw)]),
                                                [], output="three", target=1)
        self.assertEqual(report["rejected"].get("presentation_conversion"), 1)
        self.assertEqual((records, len(calls)), ([], 0))

    def test_last_source_option_cannot_swallow_following_condition(self):
        prefix = ("The atlas lists Rome as Italy's capital and Paris as France's capital.\n"
                  "Question: Which city is a capital? A) Paris B) Rome C) Milan D) Turin")
        for index, separator in enumerate(("\n", ". ")):
            with self.subTest(separator=repr(separator)):
                raw = prefix + separator + "Choose Italy's capital, not France's. Answer: B"
                report, records, calls = self.run_build(self.source([("condition", raw)]),
                                                        [], output=f"condition-{index}", target=1)
                self.assertEqual(report["rejected"].get("presentation_conversion"), 1)
                self.assertEqual((records, len(calls)), ([], 0))

    def test_punctuation_within_original_option_text_is_not_an_extra_condition(self):
        raw = ("The atlas lists Rome as Italy's capital.\n"
               "Question: Which city is Italy's capital? "
               "A) Paris B) Rome C) Milan D) St. John's Answer: B")
        report, records, calls = self.run_build(
            self.source([("punctuation", raw)]),
            [{"distractors": DISTRACTORS}, SCREEN], output="punctuation", target=1)
        self.assertEqual((report["accepted"], len(calls)), (1, 2))
        self.assertEqual(records[0]["gold"], "Rome")
        self.assertEqual([raw[slice(*span)] for span in records[0]["_meta"]["source_ref"]["option_spans"]][-1],
                         "St. John's")

    def test_oversized_source_is_rejected_before_parsing_or_assisted_http(self):
        raw = "Atlas: " + "A" * (512 * 32) + "\nQuestion: Which city? Answer: Rome"
        report, records, calls = self.run_build(self.source([("large", raw)]),
                                                [], output="large", target=1,
                                                max_context_tokens=512, max_output_tokens=256)
        self.assertEqual(report["rejected"].get("source_length"), 1)
        self.assertEqual((records, len(calls)), ([], 0))

    def test_selected_qa_failure_never_switches_to_other_qa(self):
        state = "The atlas lists Rome as Italy's capital and Paris as France's capital."
        raw = state + ("\nQuestion: What is Italy's capital? Answer: Rome"
                       "\nQuestion: What is France's capital? Answer: Paris")
        sources = self.source([("two-qa", raw)])
        report, records, calls = self.run_build(sources, [{"distractors": DISTRACTORS},
            {"supported": False, "unique": False, "all_wrong": True, "same_format": True}], target=1)
        self.assertEqual((report["accepted"], report["attempts"]), (0, 2))
        self.assertEqual(report["rejected"].get("unsupported_or_ambiguous"), 1)
        self.assertEqual((records, len(calls)), ([], 2))
