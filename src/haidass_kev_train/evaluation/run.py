"""Final evaluation CLI: fit temperature on an independent calibration split, then
report raw and calibrated metrics for a development and/or locked test split.

    python -m haidass_kev_train.evaluation.run --artifact artifacts/sft \
        --suite data/raw/kev-suites/v4/decision-v4 --split test \
        --calibration-split calibration --out artifacts/eval/report.json

The Calibration Artifact is written separately and is bound to the Decision Model
Artifact hash and the calibration split identity; it never mutates the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import types
from copy import deepcopy
from pathlib import Path

import torch
from scipy.optimize import minimize_scalar

from haidass_kev_train.data.packing import encode_record, load_suite
from haidass_kev_train.evaluation.metrics import evaluate, iter_logits, per_question_ce, predict
from haidass_kev_train.model.decision import load_artifact

CALIBRATION_VERSION = 1
REPORT_VERSION = 1
DEFAULT_SUITE = "data/raw/kev-suites/v6/decision-v6"
SPLITS = ("train", "development", "calibration", "test")
TEMPERATURE_BOUNDS = (0.02, 50.0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: str | Path) -> str:
    """Hash only the portable artifact, not optimizer/RNG checkpoint state."""
    target = Path(path)
    manifest = json.loads((target / "decision_model.json").read_text())
    training_mode = manifest.get("training_mode", "lora")
    files = {"lora": ("adapter_config.json", "adapter_model.safetensors", "decision_model.json"),
             "full": ("model.safetensors", "decision_model.json")}
    if training_mode not in files:
        raise ValueError(f"Unsupported artifact training mode: {training_mode!r}")
    digest = hashlib.sha256()
    for name in files[training_mode]:
        digest.update(name.encode())
        digest.update(bytes.fromhex(sha256_file(target / name)))
    return digest.hexdigest()


def split_identity(suite: str | Path, split: str) -> dict:
    """Identity of one frozen suite partition: bytes, manifest, and role."""
    directory = Path(suite)
    data = directory / f"{split}.jsonl"
    if not data.is_file():
        raise SystemExit(f"missing split file: {data}")
    manifest = directory / "manifest.json"
    return {
        "suite": directory.name,
        "path": str(directory.resolve()),
        "split": split,
        "sha256": sha256_file(data),
        "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
    }


def check_no_mixing(calibration: dict, targets: dict[str, dict]) -> None:
    """One split, one role: no calibration on test, no evaluating the training split."""
    if calibration["split"] != "calibration":
        raise SystemExit("temperature fitting requires the independent calibration split")
    for label, identity in targets.items():
        if identity["split"] == "train":
            raise SystemExit(f"refusing to evaluate the training split ({label})")
        if identity["split"] == calibration["split"] or identity["sha256"] == calibration["sha256"]:
            raise SystemExit(f"{label} split {identity['split']} is also the calibration split")


def build_calibration(fit: dict, model_artifact_sha256: str, calibration_split: dict) -> dict:
    return {
        "version": CALIBRATION_VERSION,
        "model_artifact_sha256": model_artifact_sha256,
        "calibration_split": calibration_split,
        "temperature": fit["temperature"],
        "count": fit["count"],
        "nll_raw": fit["nll_raw"],
        "nll_calibrated": fit["nll_calibrated"],
        "success": fit["success"],
    }


def load_calibration(path: str | Path, model_artifact_sha256: str) -> dict:
    calibration = json.loads(Path(path).read_text())
    if calibration.get("version") != CALIBRATION_VERSION:
        raise SystemExit(f"unsupported calibration artifact version {calibration.get('version')}")
    if calibration.get("model_artifact_sha256") != model_artifact_sha256:
        raise SystemExit("calibration artifact is bound to a different Decision Model Artifact")
    if not 0 < float(calibration.get("temperature", 0.0)) < math.inf or not calibration.get("success"):
        raise SystemExit("calibration artifact carries no positive temperature")
    return calibration


def fit_temperature(model, encoded_records, batch_size: int = 4, device: str = "cuda", bounds=TEMPERATURE_BOUNDS) -> dict:
    """Fit one scalar temperature by bounded scalar minimization of calibration NLL.

    Only logits, masks, and targets are cached: a full batch also holds the
    `[B, 1, L, L]` attention bias, which must not be retained across the search.
    """
    cache = [
        (
            logits.detach().float().cpu(),
            types.SimpleNamespace(
                question_mask=batch.question_mask.cpu(),
                option_mask=batch.option_mask.cpu(),
                target_probs=batch.target_probs.detach().float().cpu(),
            ),
        )
        for logits, batch in iter_logits(model, encoded_records, batch_size=batch_size, device=device)
    ]

    def nll(temperature: float) -> tuple[float, int]:
        total = 0.0
        count = 0
        for logits, batch in cache:
            values, valid = per_question_ce(logits, batch, temperature=temperature)
            total += float(values[valid].sum())
            count += int(valid.sum())
        return (total / count if count else math.inf), count

    raw, count = nll(1.0)
    if not count:
        raise SystemExit("calibration split has no scorable question")
    low, high = bounds
    result = minimize_scalar(lambda x: nll(math.exp(x))[0], bounds=(math.log(low), math.log(high)), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fitting failed")
    temperature = float(math.exp(result.x))
    calibrated, _ = nll(temperature)
    return {
        "temperature": temperature,
        "count": count,
        "nll_raw": raw,
        "nll_calibrated": calibrated,
        "success": bool(result.success),
    }


def reorder_record(record: dict) -> dict:
    """Reverse choice options with their full soft targets; never permute ordinal scores."""
    out = deepcopy(record)
    for question in out["questions"].values():
        criteria = question.get("criteria")
        if question.get("type") != "choice" or not isinstance(criteria, dict):
            continue
        question["criteria"] = {key: criteria[key] for key in reversed(list(criteria))}
        if isinstance(question.get("target"), list):
            question["target"] = list(reversed(question["target"]))
    return out


def reorder_stability(model, tokenizer, records, batch_size: int = 4, device: str = "cuda", temperature: float = 1.0) -> dict:
    """Option-order sensitivity of choice questions.

    Re-encodes the suite with reversed options and the target permuted with them,
    then compares each question's distribution mapped back onto the original
    option order. A faithful re-encoding is asserted, not assumed.
    """
    original = predict(
        model,
        [encode_record(record, tokenizer) for record in records],
        batch_size=batch_size,
        device=device,
        temperature=temperature,
    )
    reversed_rows = predict(
        model,
        [encode_record(reorder_record(record), tokenizer) for record in records],
        batch_size=batch_size,
        device=device,
        temperature=temperature,
    )
    index = {(row["record_id"], row["question_name"]): row for row in reversed_rows}
    divergences: list[float] = []
    shifts: list[float] = []
    flips = 0
    for row in original:
        if row["question_type"] != "choice":
            continue
        other = index.get((row["record_id"], row["question_name"]))
        if other is None or len(other["probs"]) != len(row["probs"]):
            continue
        if max(abs(a - b) for a, b in zip(reversed(other["target"]), row["target"])) > 1e-6:
            raise ValueError(
                f"reordered target does not follow option order: {row['record_id']}/{row['question_name']}"
            )
        p = torch.tensor(row["probs"], dtype=torch.float64)
        q = torch.tensor(list(reversed(other["probs"])), dtype=torch.float64)
        divergences.append(float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum()))
        shifts.append(float((p - q).abs().max()))
        flips += int(torch.argmax(p) != torch.argmax(q))
    return {
        "count": len(divergences),
        "mean_kl": (sum(divergences) / len(divergences)) if divergences else None,
        "mean_max_abs_diff": (sum(shifts) / len(shifts)) if shifts else None,
        "flip_rate": (flips / len(divergences)) if divergences else None,
        "temperature": float(temperature),
    }


def write_json(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", required=True, help="Decision Model Artifact directory")
    parser.add_argument("--base", default=None, help="base model path; defaults to the pinned resource")
    parser.add_argument("--suite", default=DEFAULT_SUITE, help="frozen suite directory")
    parser.add_argument("--split", default="test", choices=SPLITS, help="split to evaluate")
    parser.add_argument("--dev-split", default=None, choices=SPLITS, help="optional second split (e.g. development)")
    parser.add_argument("--calibration-split", default="calibration", choices=SPLITS, help="split that fits temperature")
    parser.add_argument("--calibration-in", default=None, help="reuse an existing Calibration Artifact")
    parser.add_argument("--calibration-out", default=None, help="write the fitted Calibration Artifact here")
    parser.add_argument("--out", default=None, help="report path; defaults to stdout")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reorder-stability", action="store_true", help="measure option-order sensitivity")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    suite = Path(args.suite)
    model_artifact_sha256 = artifact_sha256(args.artifact)
    model, tokenizer = load_artifact(args.artifact, args.base, trainable=False)

    targets = {"target": split_identity(suite, args.split)}
    if args.dev_split:
        if args.dev_split == args.split:
            raise SystemExit("--dev-split must differ from --split")
        targets["development"] = split_identity(suite, args.dev_split)

    if args.calibration_in:
        calibration = load_calibration(args.calibration_in, model_artifact_sha256)
        calibration_split = calibration["calibration_split"]
        expected_split = split_identity(suite, args.calibration_split)
        if any(calibration_split[key] != expected_split[key] for key in ("split", "sha256", "manifest_sha256")):
            raise SystemExit("calibration artifact split identity does not match this frozen suite")
    else:
        calibration_split = split_identity(suite, args.calibration_split)
    check_no_mixing(calibration_split, targets)

    records = {label: load_suite(str(suite), identity["split"]) for label, identity in targets.items()}
    encoded = {label: [encode_record(record, tokenizer) for record in rows] for label, rows in records.items()}

    if not args.calibration_in:
        calibration = build_calibration(
            fit_temperature(
                model,
                [encode_record(record, tokenizer) for record in load_suite(str(suite), calibration_split["split"])],
                batch_size=args.batch_size,
                device=args.device,
            ),
            model_artifact_sha256,
            calibration_split,
        )
        if args.calibration_out:
            write_json(args.calibration_out, calibration)

    temperature = float(calibration["temperature"])
    report = {
        "version": REPORT_VERSION,
        "model_artifact": {"path": str(Path(args.artifact).resolve()), "sha256": model_artifact_sha256},
        "calibration": calibration,
        "splits": {
            identity["split"]: {
                "identity": identity,
                "raw": evaluate(
                    model, encoded[label], batch_size=args.batch_size, device=args.device, temperature=1.0
                ),
                "calibrated": evaluate(
                    model, encoded[label], batch_size=args.batch_size, device=args.device, temperature=temperature
                ),
            }
            for label, identity in targets.items()
        },
    }
    if args.reorder_stability:
        report["reorder_stability"] = reorder_stability(
            model,
            tokenizer,
            records["target"],
            batch_size=args.batch_size,
            device=args.device,
            temperature=temperature,
        )

    if args.out:
        write_json(args.out, report)
        for name, entry in report["splits"].items():
            print(
                f"{name}: nll {entry['raw']['nll']:.4f} -> {entry['calibrated']['nll']:.4f} "
                f"(macro {entry['raw']['macro_nll']:.4f} -> {entry['calibrated']['macro_nll']:.4f}) at T={temperature:.3f}"
            )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
