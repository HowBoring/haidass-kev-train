import tempfile
import unittest
from pathlib import Path

import torch

from haidass_kev_train.data.packing import collate, encode_record
from haidass_kev_train.evaluation.metrics import per_question_ce
from haidass_kev_train.model.decision import build_model, load_artifact, save_artifact


class ModelModeContracts(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA required")
    def test_full_model_updates_and_round_trips_for_evaluation_and_resume(self):
        torch.manual_seed(42)
        base = "models/base/haidass1.5-143m"
        model, tokenizer = build_model(base, training_mode="full")
        model.cuda().train()
        record = {"state": "The parcel arrived with a broken screen.", "questions": {
            "condition": {"type": "choice", "instructions": "What condition is the screen in?",
                          "criteria": {"broken": None, "intact": None}, "label": "broken"}}}
        batch = collate([encode_record(record, tokenizer)]).to("cuda")
        token = int(batch.input_ids[0, 0])
        backbone_before = model.backbone.get_input_embeddings().weight[token].detach().clone()
        head_before = model.pointer_head.query.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        logits = model(batch)
        losses, valid = per_question_ce(logits, batch)
        losses[valid].mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(backbone_before, model.backbone.get_input_embeddings().weight[token]))
        self.assertFalse(torch.equal(head_before, model.pointer_head.query.weight))
        optimizer_state = optimizer.state_dict()
        model.eval()
        with torch.no_grad():
            expected = model(batch).cpu()
        expected_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "full"
            save_artifact(model, artifact)
            self.assertTrue((artifact / "model.safetensors").is_file())
            self.assertFalse((artifact / "adapter_model.safetensors").exists())
            frozen, _ = load_artifact(artifact, base, trainable=False)
            self.assertFalse(any(parameter.requires_grad for parameter in frozen.parameters()))
            frozen.cuda().eval()
            with torch.no_grad():
                torch.testing.assert_close(frozen(batch).cpu(), expected, rtol=0, atol=0)
            self.assertEqual(expected_state.keys(), frozen.state_dict().keys())
            for name, tensor in frozen.state_dict().items():
                torch.testing.assert_close(tensor.cpu(), expected_state[name], rtol=0, atol=0)
            del frozen

            resumed, _ = load_artifact(artifact, base, trainable=True)
            self.assertTrue(all(parameter.requires_grad for parameter in resumed.parameters()))
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
            resumed_optimizer.load_state_dict(optimizer_state)
            self.assertEqual(len(resumed_optimizer.state), len(optimizer.state))


if __name__ == "__main__":
    unittest.main()
