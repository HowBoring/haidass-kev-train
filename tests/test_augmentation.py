import unittest
from copy import deepcopy
import random

from haidass_kev_train.data.augmentation import augment_record
from haidass_kev_train.data.packing import encode_record
from haidass_kev_train.model.decision import MARKER_TOKENS


class _Tokenizer:
    def __len__(self):
        return 64_000

    def encode(self, text, add_special_tokens=False):
        markers = list(MARKER_TOKENS.values())
        return [markers.index(text) + 6] if text in markers else [100 + index for index, _ in enumerate(text.split())]

    def __call__(self, text, add_special_tokens=False):
        return type("Tokens", (), {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)})()


class AugmentationContracts(unittest.TestCase):
    def test_variants_preserve_semantics_and_are_resume_deterministic(self):
        record = {
            "state": {"case": "The parcel was delivered."},
            "questions": {
                "hard": {
                    "type": "choice",
                    "instructions": "What happened?",
                    "criteria": {"lost": None, "delivered": None, "returned": None},
                    "label": "delivered",
                    "src": "worked",
                },
                "soft_dict": {
                    "type": "choice",
                    "instructions": "Likely route?",
                    "criteria": {"air": None, "road": None, "rail": None},
                    "target": {"air": 0.6, "road": 0.3, "rail": 0.1},
                },
                "soft_list": {
                    "type": "choice",
                    "instructions": "Likely time?",
                    "criteria": {"morning": None, "noon": None, "night": None},
                    "target": [0.2, 0.5, 0.3],
                },
                "verified": {"type": "noul", "instructions": "Verified?", "label": True},
                "urgency": {
                    "type": "score",
                    "instructions": "Urgency?",
                    "criteria": ["low", "medium", "high"],
                    "label": 1,
                },
            },
            "_meta": {"id": "record-17", "group_id": "case-2", "variant": "clean", "source": "worked"},
        }
        original = deepcopy(record)

        variants = augment_record(record, seed=41, epoch=3, p_none=1, p_none_distract=0, p_distract=0, p_none_pair=1)

        self.assertEqual(record, original)
        self.assertEqual(variants, augment_record(record, seed=41, epoch=3, p_none=1, p_none_distract=0, p_distract=0, p_none_pair=1))
        self.assertEqual(len(variants), 3)
        self.assertEqual({item["_meta"]["variant"] for item in variants}, {"augmented", "none_present", "none_absent"})
        self.assertEqual({item["_meta"]["group_id"] for item in variants}, {"case-2"})
        self.assertEqual(len({item["_meta"]["id"] for item in variants}), 3)
        other_epochs = [
            augment_record(record, seed=41, epoch=epoch, p_none=1, p_none_distract=0, p_distract=0, p_none_pair=1)
            for epoch in range(4, 8)
        ]
        self.assertTrue(any(other != variants for other in other_epochs))

        augmented = next(item for item in variants if item["_meta"]["variant"] == "augmented")
        self.assertEqual(augmented["questions"]["verified"], record["questions"]["verified"])
        self.assertEqual(augmented["questions"]["urgency"], record["questions"]["urgency"])
        expected_source_targets = {
            "hard": {"lost": 0.0, "delivered": 1.0, "returned": 0.0},
            "soft_dict": {"air": 0.6, "road": 0.3, "rail": 0.1},
            "soft_list": {"morning": 0.2, "noon": 0.5, "night": 0.3},
        }
        for name in ("hard", "soft_dict", "soft_list"):
            before = record["questions"][name]
            after = augmented["questions"][name]
            removed = set(before["criteria"]) - set(after["criteria"])
            added = set(after["criteria"]) - set(before["criteria"])
            self.assertEqual(len(removed), 1)
            self.assertEqual(len(added), 1)
            old_target = expected_source_targets[name]
            if after.get("target") is None:
                new_target = {key: float(key == after["label"]) for key in after["criteria"]}
            elif isinstance(after["target"], dict):
                new_target = after["target"]
            else:
                new_target = dict(zip(after["criteria"], after["target"]))
            removed_key = removed.pop()
            none_key = added.pop()
            self.assertEqual(new_target[none_key], old_target[removed_key])
            self.assertEqual({key: new_target[key] for key in set(before["criteria"]) & set(after["criteria"])}, {key: old_target[key] for key in set(before["criteria"]) & set(after["criteria"])})

        present = next(item for item in variants if item["_meta"]["variant"] == "none_present")
        absent = next(item for item in variants if item["_meta"]["variant"] == "none_absent")
        self.assertEqual(list(present["questions"]), list(absent["questions"]))
        pair_name = next(iter(present["questions"]))
        source_question = record["questions"][pair_name]
        present_question = present["questions"][pair_name]
        absent_question = absent["questions"][pair_name]
        none_key = (set(present_question["criteria"]) - set(source_question["criteria"])).pop()
        removed_key = (set(present_question["criteria"]) - set(absent_question["criteria"])).pop()
        source_target = source_question.get("target")
        if source_target is None:
            source_target = {key: float(key == source_question["label"]) for key in source_question["criteria"]}
        elif isinstance(source_target, list):
            source_target = dict(zip(source_question["criteria"], source_target))
        for question, expected_none in ((present_question, 0.0), (absent_question, source_target[removed_key])):
            target = question.get("target")
            if target is None:
                target = {key: float(key == question["label"]) for key in question["criteria"]}
            elif isinstance(target, list):
                target = dict(zip(question["criteria"], target))
            self.assertEqual(target[none_key], expected_none)

        tokenizer = _Tokenizer()
        for variant in variants:
            encoded = encode_record(variant, tokenizer)
            for question, target in zip(variant["questions"].values(), encoded.target_probs):
                if question.get("target") is not None:
                    expected = question["target"]
                    expected = [expected[key] for key in question["criteria"]] if isinstance(expected, dict) else expected
                    self.assertEqual(target, expected)

    def test_disabled_and_invalid_transformations_are_explicit(self):
        record = {
            "state": "case",
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": "Answer?",
                    "criteria": {"a": None, "b": None, "c": None},
                    "target": [0.7, 0.2, 0.1],
                }
            },
            "_meta": {"id": "baseline", "group_id": "group", "variant": "clean"},
        }
        global_state = random.getstate()
        self.assertEqual(
            augment_record(
                record,
                seed=7,
                epoch=2,
                shuffle=False,
                p_none=0,
                p_none_distract=0,
                p_distract=0,
                p_none_pair=0,
            ),
            [record],
        )
        self.assertEqual(random.getstate(), global_state)

        for probability in ("p_none_distract", "p_distract"):
            kwargs = {"p_none": 0, "p_none_distract": 0, "p_distract": 0, "p_none_pair": 0}
            kwargs[probability] = 1
            question = augment_record(record, seed=7, epoch=2, shuffle=False, **kwargs)[0]["questions"]["answer"]
            self.assertEqual(question["target"][:3], [0.7, 0.2, 0.1])
            self.assertEqual(question["target"][-1], 0.0)

        with self.assertRaisesRegex(ValueError, "sum to at most one"):
            augment_record(record, seed=0, epoch=0, p_none=0.6, p_none_distract=0.5)
        with self.assertRaisesRegex(ValueError, "normalized finite distribution"):
            bad = deepcopy(record)
            bad["questions"]["answer"]["target"] = [0.8, 0.2, 0.2]
            augment_record(bad, seed=0, epoch=0)
        with self.assertRaisesRegex(ValueError, "collides with every supported distractor key"):
            collisions = deepcopy(record)
            collisions["questions"]["answer"]["criteria"] = {
                "weather": None,
                "purple": None,
                "pancakes": None,
                "taxes": None,
            }
            collisions["questions"]["answer"]["target"] = [0.25] * 4
            augment_record(collisions, seed=0, epoch=0, p_none=0, p_none_distract=0, p_distract=1)

    def test_binary_choice_skips_inapplicable_correct_none(self):
        record = {
            "state": "case",
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Which branch?",
                    "criteria": {"left": None, "right": None},
                    "label": "right",
                }
            },
            "_meta": {"id": "binary", "group_id": "binary"},
        }

        variants = augment_record(
            record,
            seed=9,
            epoch=1,
            p_none=1,
            p_none_distract=0,
            p_distract=0,
            p_none_pair=1,
        )

        self.assertEqual(len(variants), 1)
        question = variants[0]["questions"]["decision"]
        self.assertEqual(set(question["criteria"]), {"left", "right"})
        self.assertEqual(question["label"], "right")
        encoded = encode_record(variants[0], _Tokenizer())
        self.assertEqual(encoded.target_probs[0], [float(key == "right") for key in question["criteria"]])


if __name__ == "__main__":
    unittest.main()
