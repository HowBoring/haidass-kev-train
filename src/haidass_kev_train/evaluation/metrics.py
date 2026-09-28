"""Decision metrics for evaluation and checkpoint selection.

Everything here consumes `PackedDecisionBatch` tensors as produced by
`haidass_kev_train.data.packing` and returns plain Python scalars, so a report
dumps straight to JSON. No objective, no framework: masked CE plus aggregates.
"""

from __future__ import annotations

import torch

from haidass_kev_train.data.packing import collate

__all__ = [
    "iter_logits",
    "option_distribution",
    "per_question_ce",
    "predict",
    "summarize",
    "evaluate",
]

NEG_INF = float("-inf")
_TINY = 1e-12
# Per-question metadata copied into prediction rows; the rest stays in the batch.
_META_FIELDS = (
    "record_id",
    "question_name",
    "question_type",
    "src",
    "group_id",
    "variant",
    "option_keys",
)


def _valid_options(batch) -> torch.Tensor:
    """[B, Q, K] bool: a real option of a real question."""
    return batch.question_mask.bool().unsqueeze(-1) & batch.option_mask.bool()


def _targets(batch) -> torch.Tensor:
    if batch.target_probs is None:
        raise ValueError("evaluation requires target_probs on the batch")
    return batch.target_probs


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def option_distribution(logits: torch.Tensor, option_mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """FP32 probabilities, with zero probability on padded rows/options."""
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    valid = option_mask.bool()
    _check_finite(logits, valid)
    z = (logits.float() / temperature).masked_fill(~valid, NEG_INF)
    z = z.masked_fill(~valid.any(-1, keepdim=True), 0.)
    return z.softmax(-1).masked_fill(~valid, 0.)


def per_question_ce(logits: torch.Tensor, batch, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-target CE; padding contributes no loss or gradient, invalid real logits fail."""
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    valid, rows = _valid_options(batch), batch.question_mask.bool()
    _check_finite(logits, valid)
    if bool((rows & ~valid.any(-1)).any()):
        raise ValueError("Real question has no options")
    target = _targets(batch).float()
    if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise ValueError("Invalid target probabilities")
    if bool((target.masked_select(~valid) != 0).any()) or not torch.allclose(target.sum(-1)[rows], torch.ones_like(target.sum(-1)[rows]), atol=1e-5):
        raise ValueError("Targets must put unit mass on valid options")
    z = (logits.float() / temperature).masked_fill(~valid, NEG_INF)
    z = z.masked_fill(~rows.unsqueeze(-1), 0.)
    log_p = z.log_softmax(-1).masked_fill(~valid, 0.)
    return -(target * log_p).sum(-1), rows


def iter_logits(model, encoded_records, batch_size: int = 4, device: str = "cuda"):
    """Yield `(logits, batch)` over locally collated chunks, model temporarily in eval mode.

    The model is moved to `device` here so every caller does not have to. Non-finite
    logits on a valid option are a hard error: they silently poison softmax and NLL.
    """
    was_training = model.training
    model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(encoded_records), batch_size):
                batch = collate(list(encoded_records[start : start + batch_size])).to(device)
                logits = model(batch)
                _check_finite(logits, _valid_options(batch))
                yield logits, batch
    finally:
        model.train(was_training)


def _check_finite(logits: torch.Tensor, valid: torch.Tensor) -> None:
    bad = valid & ~torch.isfinite(logits.float())
    if bool(bad.any()):
        raise ValueError(f"{int(bad.sum())} valid option(s) carry non-finite logits")


def predict(model, encoded_records, batch_size: int = 4, device: str = "cuda", temperature: float = 1.0) -> list[dict]:
    """Flat per-question rows: metadata, valid-option probabilities, normalized target."""
    rows: list[dict] = []
    for logits, batch in iter_logits(model, encoded_records, batch_size=batch_size, device=device):
        probs = option_distribution(logits, batch.option_mask, temperature).double().cpu()
        log_probs = (logits.float() / temperature).masked_fill(~batch.option_mask, NEG_INF).log_softmax(-1).cpu()
        target = _targets(batch).double().cpu()
        valid = _valid_options(batch).cpu()
        for i, questions in enumerate(batch.metadata):
            for j, meta in enumerate(questions):
                mask = valid[i, j]
                if not bool(mask.any()):
                    continue
                gold = target[i, j][mask]
                mass = float(gold.sum())
                if mass <= 0:
                    raise ValueError(
                        f"question {meta.get('record_id')}/{meta.get('question_name')} has no target mass"
                    )
                rows.append(
                    {
                        **{key: meta.get(key) for key in _META_FIELDS},
                        "probs": probs[i, j][mask].tolist(),
                        "log_probs": log_probs[i, j][mask].tolist(),
                        "target": (gold / mass).tolist(),
                    }
                )
    return rows


def _row_values(row: dict) -> dict:
    p = torch.tensor(row["probs"], dtype=torch.float64)
    t = torch.tensor(row["target"], dtype=torch.float64)
    values = {
        "nll": float(-(t * torch.tensor(row["log_probs"], dtype=torch.float64)).sum()),
        "brier": float((p - t).pow(2).sum()),
        "accuracy": float(torch.argmax(p) == torch.argmax(t)),
        "rps": None,
        "mae": None,
    }
    if row["question_type"] == "score" and p.numel() > 1:
        cdf = torch.arange(p.numel(), dtype=torch.float64)
        values["rps"] = float(((p.cumsum(0)[:-1] - t.cumsum(0)[:-1]).pow(2).sum()) / (p.numel() - 1))
        values["mae"] = float(abs((p * cdf).sum() - (t * cdf).sum()))
    return values


def _stats(rows: list[dict]) -> dict:
    values = [_row_values(row) for row in rows]
    return {
        "count": len(values),
        "nll": _mean([v["nll"] for v in values]),
        "accuracy": _mean([v["accuracy"] for v in values]),
        "brier": _mean([v["brier"] for v in values]),
        "rps": _mean([v["rps"] for v in values if v["rps"] is not None]),
        "mae": _mean([v["mae"] for v in values if v["mae"] is not None]),
        "ordinal_count": sum(1 for v in values if v["rps"] is not None),
    }


def summarize(rows: list[dict], temperature: float = 1.0) -> dict:
    """Aggregate prediction rows into a JSON-serializable report.

    `rps`/`mae` cover only ordinal `score` questions and are `None` when a group
    has none. `macro_nll` weights every source equally, which is what checkpoint
    selection uses; `nll` weights every question equally.
    """
    groups: dict[str, list[dict]] = {}
    types: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["src"], []).append(row)
        types.setdefault(row["question_type"], []).append(row)
    by_source = {src: _stats(group) for src, group in groups.items()}
    return {
        **_stats(rows),
        "temperature": float(temperature),
        "macro_nll": _mean([stats["nll"] for stats in by_source.values() if stats["nll"] is not None]),
        "by_source": by_source,
        "by_question_type": {name: _stats(group) for name, group in types.items()},
    }


def evaluate(
    model,
    encoded_records,
    batch_size: int = 4,
    device: str = "cuda",
    temperature: float = 1.0,
) -> dict:
    """Score `EncodedRecord`s: collates locally, restores training mode, reports scalars."""
    rows = predict(model, encoded_records, batch_size=batch_size, device=device, temperature=temperature)
    return summarize(rows, temperature=temperature)
