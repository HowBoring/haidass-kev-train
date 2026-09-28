import unittest
from copy import deepcopy

import torch
from transformers import AutoTokenizer

from haidass_kev_train.data.packing import collate, encode_record
from haidass_kev_train.evaluation.metrics import per_question_ce
from haidass_kev_train.evaluation.run import reorder_record


class DecisionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained('models/base/haidass1.5-143m', local_files_only=True)
        cls.record = {'state': {'facts': ['Paid', 'Delivered']}, 'questions': {
            'status': {'type': 'choice', 'instructions': {'question': 'Which status?'},
                       'criteria': {'paid': {'meaning': 'Payment received'}, 'unpaid': None, 'unknown': 'Not stated'},
                       'target': {'paid': 0.7, 'unpaid': 0.2, 'unknown': 0.1}},
            'delivered': {'type': 'noul', 'instructions': 'Delivered?', 'label': True}}}

    def test_isolated_branches_and_padded_loss_gradients(self):
        first = encode_record(self.record, self.tokenizer)
        second = encode_record({**self.record, 'questions': {'delivered': self.record['questions']['delivered']}}, self.tokenizer)
        batch = collate([first, second])
        self.assertEqual(batch.attention_bias.shape[:2], (2, 1))
        for query, segment in enumerate(first.segment_ids):
            for key, source in enumerate(first.segment_ids):
                expected = key <= query and (source == 0 or source == segment)
                self.assertEqual(bool(torch.isfinite(batch.attention_bias[0, 0, query, key])), expected)
        raw = torch.randn(2, 2, 3, requires_grad=True)
        logits = raw.masked_fill(~batch.option_mask, -torch.inf)
        loss, valid = per_question_ce(logits, batch)
        expected = -(torch.tensor([0.7, 0.2, 0.1]) * raw[0, 0].log_softmax(-1)).sum()
        torch.testing.assert_close(loss[0, 0], expected)
        loss[valid].mean().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue((raw.grad[~batch.option_mask] == 0).all())
        bad = logits.detach().clone()
        bad[0, 0, 0] = torch.nan
        with self.assertRaises(ValueError):
            per_question_ce(bad, batch)

    def test_targets_follow_choice_reordering_and_overflow_rejects(self):
        record = deepcopy(self.record)
        record['questions']['status']['target'] = [0.7, 0.2, 0.1]
        forward = encode_record(record, self.tokenizer)
        reverse = encode_record(reorder_record(record), self.tokenizer)
        self.assertEqual(forward.target_probs[0], list(reversed(reverse.target_probs[0])))
        self.assertEqual(forward.target_probs[1], reverse.target_probs[1])
        with self.assertRaises(ValueError):
            encode_record(record, self.tokenizer, max_packed=5)
        record['state'] = 'A literal <|box_start|> in text'
        with self.assertRaises(ValueError):
            encode_record(record, self.tokenizer)


if __name__ == '__main__':
    unittest.main()
