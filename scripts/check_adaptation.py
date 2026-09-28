"""Run the Model Adaptation Gate against the pinned real model on CUDA."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile

import torch

from haidass_kev_train.data.packing import collate, encode_record
from haidass_kev_train.evaluation.metrics import per_question_ce
from haidass_kev_train.model.decision import build_model, load_artifact, save_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="models/base/haidass1.5-143m")
    args = parser.parse_args()
    torch.manual_seed(42)
    model, tokenizer = build_model(args.base)
    model.cuda()
    record = {"state": "The parcel arrived yesterday, but its screen was broken.", "questions": {
        "arrived": {"type": "noul", "instructions": "Has the parcel arrived?", "label": True},
        "condition": {"type": "choice", "instructions": "What condition is the screen in?",
                      "criteria": {"broken": None, "intact": None}, "label": "broken"}}}
    encoded = encode_record(record, tokenizer)
    together = collate([encoded]).to("cuda")
    alone = collate([encode_record({**record, "questions": {"condition": record["questions"]["condition"]}}, tokenizer)]).to("cuda")
    model.eval()
    mutated = replace(together, input_ids=together.input_ids.clone())
    sibling_text = torch.tensor([segment == 1 and token > 32 for segment, token in zip(encoded.segment_ids, encoded.input_ids)], device="cuda")
    mutated.input_ids[0, sibling_text] = 123
    with torch.no_grad():
        full_logits, mutated_logits = model(together), model(mutated)
    isolation_error = (full_logits[0, 1] - mutated_logits[0, 1]).abs().max().item()
    assert isolation_error == 0, isolation_error
    # Changing packed length changes BF16 GEMM rounding. Check separate/together
    # equivalence in FP32, independently from the exact same-shape leakage check.
    def reference(batch):
        hidden = model.backbone(input_ids=batch.input_ids, position_ids=batch.position_ids,
                                attention_mask=batch.attention_bias.float(), use_cache=False).last_hidden_state
        rows = torch.arange(len(hidden), device="cuda")
        return model.pointer_head(
            hidden[rows[:, None], batch.decide_positions], hidden[rows[:, None, None], batch.option_end_positions])
    with torch.no_grad():
        reference_error = (reference(together)[0, 1] - reference(alone)[0, 0]).abs().max().item()
    assert reference_error < 1e-4, reference_error
    groups = {"lora": [], "pointer": [], "tokens": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.dtype == torch.float32, (name, parameter.dtype)
            group = "lora" if "lora_" in name else "pointer" if "pointer_head" in name else "tokens"
            groups[group].append((name, parameter, parameter.detach().clone()))
    assert all(groups.values()), {k: len(v) for k, v in groups.items()}
    frozen = [(name, parameter) for name, parameter in model.named_parameters()
              if not parameter.requires_grad and parameter.ndim == 2 and parameter.shape[0] == len(tokenizer)]
    assert frozen, "Expected unchanged frozen embedding table"
    def fingerprints():
        return {name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for name, p in frozen}
    before = fingerprints()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(together)
    values, mask = per_question_ce(logits, together)
    loss = values[mask].mean()
    loss.backward()
    grad_norms = {}
    for group, entries in groups.items():
        grads = [p.grad for _, p, _ in entries if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), group
        grad_norms[group] = sum(float(g.float().norm()) for g in grads)
        assert grad_norms[group] > 0, group
    optimizer.step()
    for group, entries in groups.items():
        assert any(not torch.equal(p.detach(), old) for _, p, old in entries), group
    assert before == fingerprints(), "Frozen embedding rows changed"
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(together).float().cpu()
    with tempfile.TemporaryDirectory() as temp:
        artifact = Path(temp) / "adapter"
        save_artifact(model, artifact)
        del optimizer, model
        torch.cuda.empty_cache()
        restored, restored_tokenizer = load_artifact(artifact, args.base, trainable=True)
        restored.cuda().eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = restored(together).float().cpu()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert len(restored_tokenizer) == len(tokenizer) == 64000
        size = sum(p.stat().st_size for p in artifact.iterdir() if p.is_file())
    print(json.dumps({"gate": "passed", "loss": float(loss.detach()), "vocabulary": len(tokenizer),
                      "question_isolation_maxdiff": isolation_error, "reload_maxdiff": 0,
                      "fp32_separate_question_maxdiff": reference_error,
                      "gradient_norms": grad_norms, "artifact_bytes": size}), flush=True)


if __name__ == "__main__":
    main()
