"""Run a small, explicit Decision SFT budget/LR/mode comparison."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import tomllib
import uuid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _toml(config: dict) -> str:
    lines = [f"{key} = {_toml_value(value)}" for key, value in config.items() if not isinstance(value, dict)]
    for table, values in config.items():
        if isinstance(values, dict):
            lines += ["", f"[{table}]", *(f"{key} = {_toml_value(value)}" for key, value in values.items())]
    return "\n".join(lines) + "\n"


def _lr_name(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace("+", "p").replace(".", "p")


def plan(config_path, output, *, budgets, learning_rates, modes, seeds, plan_only=False, wandb_project=None):
    """Create resolved finite comparison arms and optionally train them serially."""
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison {output}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in budgets):
        raise ValueError("Budgets must be positive integers")
    if any(not math.isfinite(value) or value <= 0 for value in learning_rates):
        raise ValueError("Every learning rate must be finite and positive")
    if any(mode not in {"lora", "full"} for mode in modes):
        raise ValueError("Modes must be 'lora' or 'full'")
    base = tomllib.loads(config_path.read_text())
    scheduler = base.get("scheduler")
    if scheduler not in {"cosine", "onecycle"}:
        raise ValueError("Base scheduler must be 'cosine' or 'onecycle'")
    warmup = base.get("warmup_steps")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("Base warmup_steps must be a non-negative integer")
    if scheduler == "onecycle" and warmup != 0:
        raise ValueError("onecycle requires warmup_steps = 0")
    if scheduler == "cosine" and any(warmup >= budget for budget in budgets):
        raise ValueError("Base warmup_steps must be below every comparison budget")
    arms = []
    for budget, learning_rate, mode, seed in itertools.product(budgets, learning_rates, modes, seeds):
        resolved = {**base, "max_steps": budget, "learning_rate": learning_rate,
                    "training_mode": mode, "seed": seed}
        name = f"{mode}-steps{budget}-lr{_lr_name(learning_rate)}-seed{seed}"
        arms.append((name, resolved))
    names = [name for name, _ in arms]
    if len(names) != len(set(names)):
        raise ValueError("Comparison arguments produce duplicate run names")
    if wandb_project and not plan_only:
        from haidass_kev_train.training.tracking import require_entity
        require_entity(wandb_project)

    output.mkdir(parents=True)
    runs = []
    source_sha256 = _sha256(config_path)
    for name, resolved in arms:
        run_dir = output / name
        run_dir.mkdir()
        resolved_path = run_dir / "config.toml"
        resolved_path.write_text(_toml(resolved))
        run = {"name": name, "status": "planned", "config_sha256": _sha256(resolved_path),
               "checkpoint_output": f"{name}/checkpoints"}
        provenance = {"source_config": str(config_path), "source_config_sha256": source_sha256,
                      "resolved_config_sha256": run["config_sha256"], "plan_only": bool(plan_only)}
        (run_dir / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
        runs.append(run)

    comparison_path = output / "comparison.json"
    comparison = {"plan_only": bool(plan_only), "source_config": str(config_path),
                  "source_config_sha256": source_sha256, "runs": runs}
    if wandb_project:
        comparison["wandb_group"] = f"{output.name}-{uuid.uuid4().hex[:8]}"
    comparison_path.write_text(json.dumps(comparison, sort_keys=True, indent=2) + "\n")
    if plan_only:
        return comparison

    from haidass_kev_train.training.sft import train
    for run in runs:
        run["status"] = "running"
        comparison_path.write_text(json.dumps(comparison, sort_keys=True, indent=2) + "\n")
        run_dir = output / run["name"]
        if wandb_project:
            train(run_dir / "config.toml", run_dir / "checkpoints",
                  wandb_project=wandb_project, wandb_name=run["name"],
                  wandb_group=comparison["wandb_group"])
        else:
            train(run_dir / "config.toml", run_dir / "checkpoints")
        best = json.loads((run_dir / "checkpoints" / "best.json").read_text())
        run.update(status="completed", selection=best)
        comparison_path.write_text(json.dumps(comparison, sort_keys=True, indent=2) + "\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--learning-rates", type=float, nargs="+", required=True)
    parser.add_argument("--modes", choices=("lora", "full"), nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--wandb-project")
    args = parser.parse_args()
    plan(args.config, args.output, budgets=args.budgets, learning_rates=args.learning_rates,
         modes=args.modes, seeds=args.seeds, plan_only=args.plan_only, wandb_project=args.wandb_project)


if __name__ == "__main__":
    main()
