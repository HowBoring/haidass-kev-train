"""Numerical edge regression tests for the decision metrics.

Run: python tests/test_metrics.py   (also collectable by pytest)

Covers what silently corrupts a report: option padding, `0 * -inf` in a
full-target CE, padded questions, non-finite model output, temperature scaling,
and the ordinal RPS/MAE normalization.
"""

import json
import math

import torch

from haidass_kev_train.evaluation.metrics import option_distribution, per_question_ce, summarize


class _Batch:
    """`per_question_ce` only needs the question/option masks and the targets."""

    def __init__(self, target_probs, question_mask, option_mask):
        self.target_probs = target_probs
        self.question_mask = question_mask
        self.option_mask = option_mask


def _batch(target, question_mask, option_mask):
    return _Batch(
        torch.tensor(target, dtype=torch.float32),
        torch.tensor(question_mask),
        torch.tensor(option_mask),
    )


def _close(actual, expected, tol=1e-5):
    assert actual is not None and math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), f"{actual} != {expected}"


def _raises(exc, fn, needle):
    try:
        fn()
    except exc as error:
        assert needle in str(error), f"{needle!r} not in {error!r}"
        return
    raise AssertionError(f"expected {exc.__name__} mentioning {needle!r}")


def test_full_target_ce_and_padding():
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0, 5.0],  # option 3 is padding, its 5.0 must not be scored
                [1.0, -200.0, 0.0, 0.0],  # zero-mass option with a huge negative logit
                [1.0, 2.0, 3.0, 4.0],  # padded question slot
            ]
        ],
        dtype=torch.bfloat16,
    )
    batch = _batch(
        target=[[[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        question_mask=[[True, True, False]],
        option_mask=[[[True, True, True, False], [True, True, True, False], [False, False, False, False]]],
    )

    values, valid = per_question_ce(logits, batch)
    assert values.shape == (1, 3)
    assert valid.tolist() == [[True, True, False]], "the valid mask is the question mask"
    assert torch.isfinite(values).all(), "padded questions must not produce NaN"
    assert float(values[0, 2]) == 0.0

    # Hard one-hot over the real options only.
    _close(float(values[0, 0]), float(torch.logsumexp(torch.tensor([2.0, 1.0, 0.0]), dim=-1) - 2.0))

    # Full soft target: every option keeps its mass, a zero-mass option costs nothing.
    log_p = torch.log_softmax(torch.tensor([1.0, -200.0, 0.0]), dim=-1)
    _close(float(values[0, 1]), float(-(0.5 * log_p[0] + 0.5 * log_p[2])))

    # Temperature rescales logits before the log-normalizer.
    warm, _ = per_question_ce(logits, batch, temperature=2.0)
    _close(float(warm[0, 0]), float(torch.logsumexp(torch.tensor([1.0, 0.5, 0.0]), dim=-1) - 1.0))

    # `-inf` on an invalid option is the documented model output, not an error.
    masked, valid = per_question_ce(
        torch.tensor([[[1.0, float("-inf")]]]),
        _batch([[[1.0, 0.0]]], [[True]], [[[True, False]]]),
    )
    assert valid.tolist() == [[True]] and float(masked[0, 0]) == 0.0


def test_malformed_batches_fail_loudly():
    zero = torch.zeros(1, 1, 2)
    # A real question with no real option.
    _raises(
        ValueError,
        lambda: per_question_ce(zero, _batch([[[0.0, 0.0]]], [[True]], [[[False, False]]])),
        "no options",
    )
    # Non-finite logits on a valid option.
    _raises(
        ValueError,
        lambda: per_question_ce(torch.tensor([[[float("nan"), 0.0]]]), _batch([[[1.0, 0.0]]], [[True]], [[[True, True]]])),
        "non-finite",
    )
    # Target mass parked on a non-existent option.
    _raises(
        ValueError,
        lambda: per_question_ce(zero, _batch([[[0.5, 0.5]]], [[True]], [[[True, False]]])),
        "unit mass",
    )
    # Target that is not a distribution at all.
    _raises(
        ValueError,
        lambda: per_question_ce(zero, _batch([[[0.4, 0.4]]], [[True]], [[[True, True]]])),
        "unit mass",
    )


def test_option_distribution_masks_and_temperature():
    logits = torch.tensor([[[2.0, 1.0, 5.0], [1.0, 2.0, 3.0]]])
    option_mask = torch.tensor([[[True, True, False], [False, False, False]]])

    probs = option_distribution(logits, option_mask)
    assert torch.isfinite(probs).all(), "a fully padded option row must not produce NaN"
    assert probs[0, 1].tolist() == [0.0, 0.0, 0.0]
    assert probs[0, 0, 2] == 0.0, "invalid options get no probability"
    _close(float(probs[0, 0].sum()), 1.0)

    sharp = option_distribution(logits, option_mask, temperature=0.5)
    assert float(sharp[0, 0].max()) > float(probs[0, 0].max())
    _raises(ValueError, lambda: option_distribution(logits, option_mask, temperature=0.0), "temperature")


def test_summarize_aggregates():
    rows = [
        {"src": "a", "question_type": "choice", "probs": [1.0, 0.0], "log_probs": [0.0, -50.0], "target": [1.0, 0.0]},
        {
            "src": "a",
            "question_type": "choice",
            "probs": [0.4, 0.6],
            "log_probs": [math.log(0.4), math.log(0.6)],
            "target": [1.0, 0.0],
        },
        {
            "src": "b",
            "question_type": "score",
            "probs": [0.1, 0.8, 0.1],
            "log_probs": [math.log(0.1), math.log(0.8), math.log(0.1)],
            "target": [0.0, 0.0, 1.0],
        },
    ]
    report = summarize(rows, temperature=1.5)
    _close(report["nll"], (0.0 + math.log(2.5) + math.log(10)) / 3)
    _close(report["accuracy"], 1 / 3)
    _close(report["brier"], (0.0 + 0.72 + 1.46) / 3)
    _close(report["rps"], 0.41)
    _close(report["mae"], 1.0)
    assert report["count"] == 3 and report["ordinal_count"] == 1
    _close(report["by_source"]["a"]["nll"], math.log(2.5) / 2)
    _close(report["by_source"]["b"]["nll"], math.log(10))
    _close(report["macro_nll"], (math.log(2.5) / 2 + math.log(10)) / 2)
    assert report["by_source"]["a"]["count"] == 2 and report["by_source"]["a"]["rps"] is None
    assert report["temperature"] == 1.5 and report["by_question_type"]["score"]["mae"] == report["mae"]
    assert json.loads(json.dumps(report))["macro_nll"] == report["macro_nll"]


def test_summarize_without_ordinal_questions():
    report = summarize([{"src": "a", "question_type": "noul", "probs": [0.5, 0.5], "log_probs": [-math.log(2)] * 2, "target": [1.0, 0.0]}])
    assert report["rps"] is None and report["mae"] is None and report["ordinal_count"] == 0


if __name__ == "__main__":
    test_full_target_ce_and_padding()
    test_malformed_batches_fail_loudly()
    test_option_distribution_masks_and_temperature()
    test_summarize_aggregates()
    test_summarize_without_ordinal_questions()
    print("test_metrics: ok")
