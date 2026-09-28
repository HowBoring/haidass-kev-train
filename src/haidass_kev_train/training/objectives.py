"""Per-question proper-scoring and logits-space RLCD objectives."""
from __future__ import annotations

import math

import torch




def _inputs(logits: torch.Tensor, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 3 or logits.shape != batch.option_mask.shape:
        raise ValueError("logits and option_mask must have shape [B,Q,K]")
    options = batch.option_mask.bool()
    questions = batch.question_mask.bool()
    valid = questions.unsqueeze(-1) & options
    if bool((questions & ~options.any(-1)).any()):
        raise ValueError("real question has no options")
    if bool((valid & ~torch.isfinite(logits.float())).any()):
        raise ValueError("valid options require finite logits")
    if batch.target_probs is None:
        raise ValueError("objective requires target_probs")
    target = batch.target_probs.float()
    if target.shape != logits.shape or not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise ValueError("invalid target probabilities")
    if bool((target.masked_select(~valid) != 0).any()):
        raise ValueError("target mass appears on padding")
    if not torch.allclose(target.sum(-1)[questions], torch.ones_like(target.sum(-1)[questions]), atol=1e-5):
        raise ValueError("targets must put unit mass on valid options")
    ordinal = torch.tensor(
        [[meta.get("question_type") == "score" for meta in row] for row in batch.metadata],
        dtype=torch.bool,
        device=logits.device,
    ) & questions
    return options, questions, target, ordinal


def _check_reward_config(spherical_weight: float, rps_weight: float, log_floor: float) -> None:
    if not all(math.isfinite(value) for value in (spherical_weight, rps_weight, log_floor)):
        raise ValueError("reward weights and log_floor must be finite")
    if spherical_weight < 0 or rps_weight < 0 or log_floor > 0:
        raise ValueError("reward weights must be non-negative and log_floor must be non-positive")


def _reward(
    logits: torch.Tensor,
    options: torch.Tensor,
    target: torch.Tensor,
    ordinal: torch.Tensor,
    *,
    spherical_weight: float,
    rps_weight: float,
    log_floor: float,
) -> torch.Tensor:
    """Reward for [B,Q,K] or [G,B,Q,K] logits using identical clipping in every mode."""
    leading = logits.ndim - options.ndim
    for _ in range(leading):
        options, target, ordinal = options.unsqueeze(0), target.unsqueeze(0), ordinal.unsqueeze(0)
    masked = logits.float().masked_fill(~options, float("-inf"))
    probabilities = masked.softmax(-1).masked_fill(~options, 0.0)
    log_probabilities = masked.log_softmax(-1).masked_fill(~options, 0.0).clamp_min(log_floor)
    log_score = (target * log_probabilities).sum(-1)
    spherical = (target * probabilities).sum(-1) / probabilities.square().sum(-1).sqrt().clamp_min(1e-12)

    counts = options.sum(-1)
    thresholds = options[..., :-1] & options[..., 1:]
    rps_numerator = (
        (probabilities.cumsum(-1)[..., :-1] - target.cumsum(-1)[..., :-1]).square() * thresholds
    ).sum(-1)
    rps = rps_numerator / (counts - 1).clamp_min(1)
    return log_score + spherical_weight * spherical - rps_weight * rps * (ordinal & (counts > 1))


def proper_loss(
    logits: torch.Tensor,
    batch,
    *,
    spherical_weight: float = 0.5,
    rps_weight: float = 1.0,
    log_floor: float = -9.21,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directly differentiated negative proper reward, one FP32 value per question."""
    _check_reward_config(spherical_weight, rps_weight, log_floor)
    options, questions, target, ordinal = _inputs(logits, batch)
    reward = _reward(
        logits,
        options,
        target,
        ordinal,
        spherical_weight=spherical_weight,
        rps_weight=rps_weight,
        log_floor=log_floor,
    )
    return -reward.masked_fill(~questions, 0.0), questions


def rlcd_loss(
    logits: torch.Tensor,
    batch,
    *,
    group_size: int = 4,
    sigma: float = 0.1,
    spherical_weight: float = 0.5,
    rps_weight: float = 1.0,
    log_floor: float = -9.21,
    normalize_advantage: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian score-function loss in each valid-option centered logits subspace.

    Samples and rewards are detached. The differentiable log density is evaluated
    around the live, centered logits. The self-inclusive group-mean baseline gives
    the usual ``(G-1)/G`` gradient scale when advantage normalization is disabled.
    """
    _check_reward_config(spherical_weight, rps_weight, log_floor)
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer >= 2")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not isinstance(normalize_advantage, bool):
        raise ValueError("normalize_advantage must be boolean")

    options, questions, target, ordinal = _inputs(logits, batch)
    counts = options.sum(-1, keepdim=True)
    eligible = questions & (counts.squeeze(-1) > 1)
    if not bool(eligible.any()):
        return logits.float().masked_fill(~options, 0.0).sum(-1) * 0.0, eligible
    live = logits.float().masked_fill(~options, 0.0)
    live = (live - live.sum(-1, keepdim=True) / counts.clamp_min(1)).masked_fill(~options, 0.0)

    noise = torch.randn(
        (group_size, *live.shape), dtype=live.dtype, device=live.device, generator=generator
    ).masked_fill(~options.unsqueeze(0), 0.0)
    noise = (noise - noise.sum(-1, keepdim=True) / counts.unsqueeze(0).clamp_min(1)).masked_fill(
        ~options.unsqueeze(0), 0.0
    )
    sampled = live.detach().unsqueeze(0) + sigma * noise
    rewards = _reward(
        sampled,
        options,
        target,
        ordinal,
        spherical_weight=spherical_weight,
        rps_weight=rps_weight,
        log_floor=log_floor,
    ).detach()
    advantages = rewards - rewards.mean(0, keepdim=True)
    if normalize_advantage:
        scale = advantages[:, eligible].square().mean().sqrt().clamp_min(1e-8)
        advantages = advantages / scale

    # The centered Gaussian lives in a K-1 subspace: sum, never average, its valid squared distance.
    squared_distance = ((sampled - live.unsqueeze(0)).square() * options.unsqueeze(0)).sum(-1)
    log_density = -squared_distance / (2.0 * sigma * sigma)
    values = -(advantages * log_density).mean(0)
    return values.masked_fill(~eligible, 0.0), eligible
