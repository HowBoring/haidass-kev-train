"""Single-GPU Stage 2 CE/proper/RLCD training from a trusted Stage 1 checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import tomllib

import torch

from haidass_kev_train.data.packing import collate, encode_record, load_suite
from haidass_kev_train.evaluation.diagnostics import training_diagnostics
from haidass_kev_train.evaluation.metrics import per_question_ce
from haidass_kev_train.evaluation.run import artifact_sha256, sha256_file
from haidass_kev_train.model.decision import load_artifact, save_artifact
from haidass_kev_train.training.objectives import proper_loss, rlcd_loss
from haidass_kev_train.training.sft import (
    _grad_norm,
    _parameter_groups,
    configure_runtime,
    digest,
    restore_rng,
    rng_state,
)
from haidass_kev_train.training.tracking import TrainingTracker, require_entity

MODES = {"B": "ce", "C": "proper", "D": "rlcd"}


def _config(path: str | Path) -> dict:
    config = tomllib.loads(Path(path).read_text())
    required = {
        "base_path", "typed_suite_path", "replay_suite_path", "replay_sources", "seed", "rl_seed",
        "batch_size", "replay_batch_size", "gradient_accumulation", "learning_rate",
        "head_learning_rate", "weight_decay", "max_steps", "warmup_steps", "eval_interval",
        "checkpoint_interval", "max_packed", "eval_batch_size", "max_grad_norm", "training_mode",
        "mode", "ce_weight", "objective_weight", "objective_ramp_steps", "replay_weight",
        "group_size", "sigma", "normalize_advantage", "spherical_weight", "rps_weight", "log_floor",
    }
    if not required <= config.keys():
        raise ValueError(f"Missing Stage 2 config: {required - config.keys()}")
    for key in (
        "seed", "rl_seed", "batch_size", "replay_batch_size", "gradient_accumulation", "max_steps",
        "eval_interval", "checkpoint_interval", "max_packed", "eval_batch_size", "group_size",
    ):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["group_size"] < 2:
        raise ValueError("group_size must be >= 2")
    if not isinstance(config["warmup_steps"], int) or not 0 <= config["warmup_steps"] < config["max_steps"]:
        raise ValueError("warmup_steps must be a non-negative integer below max_steps")
    if not isinstance(config["objective_ramp_steps"], int) or config["objective_ramp_steps"] < 0:
        raise ValueError("objective_ramp_steps must be a non-negative integer")
    if config["training_mode"] not in {"inherit", "lora", "full"}:
        raise ValueError("training_mode must be 'inherit', 'lora', or 'full'")
    if config["mode"] not in MODES or not isinstance(config["normalize_advantage"], bool):
        raise ValueError("mode must be B, C, or D and normalize_advantage must be boolean")
    numeric = (
        "learning_rate", "head_learning_rate", "weight_decay", "max_grad_norm", "ce_weight",
        "objective_weight", "replay_weight", "sigma", "spherical_weight", "rps_weight", "log_floor",
    )
    if any(not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or not math.isfinite(config[key]) for key in numeric):
        raise ValueError("Stage 2 numeric configuration must be finite")
    if config["learning_rate"] <= 0 or config["head_learning_rate"] <= 0 or config["max_grad_norm"] <= 0:
        raise ValueError("learning rates and max_grad_norm must be positive")
    if any(config[key] < 0 for key in ("weight_decay", "ce_weight", "objective_weight", "replay_weight", "spherical_weight", "rps_weight")):
        raise ValueError("loss weights and weight_decay must be non-negative")
    if config["sigma"] <= 0 or config["log_floor"] > 0:
        raise ValueError("sigma must be positive and log_floor non-positive")
    if not isinstance(config["replay_sources"], list) or not config["replay_sources"] or len(set(config["replay_sources"])) != len(config["replay_sources"]):
        raise ValueError("replay_sources must be a non-empty unique list")
    return config


def _encoded(records: list[dict], tokenizer, max_packed: int, label: str) -> list:
    encoded, blockers = [], []
    for record in records:
        try:
            encoded.append(encode_record(record, tokenizer, max_packed=max_packed))
        except ValueError as error:
            blockers.append(str(error))
    if blockers:
        raise ValueError(f"{label}: {len(blockers)} oversized/invalid records; first blocker: {blockers[0]}")
    if not encoded:
        raise ValueError(f"empty {label} data")
    return encoded


def _order(size: int, seed: int, stream: int, epoch: int) -> list[int]:
    order = list(range(size))
    random.Random(seed + 1_000_003 * stream + epoch).shuffle(order)
    return order


def _take(data: list, state: dict, count: int, seed: int, stream: int) -> list:
    result = []
    while len(result) < count:
        order = _order(len(data), seed, stream, state["epoch"])
        remaining = min(count - len(result), len(order) - state["cursor"])
        result.extend(data[index] for index in order[state["cursor"] : state["cursor"] + remaining])
        state["cursor"] += remaining
        if state["cursor"] == len(order):
            state["epoch"] += 1
            state["cursor"] = 0
    return result


def _scheduler(optimizer, config):
    def schedule(current_step):
        warmup = config["warmup_steps"]
        if current_step < warmup:
            return (current_step + 1) / max(1, warmup)
        progress = min(1.0, (current_step - warmup) / (config["max_steps"] - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _parent(parent: Path, config: dict) -> tuple[dict, dict]:
    state_path = parent / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    parent_config = state.get("identity", {}).get("config", {})
    if config["training_mode"] != "inherit" and parent_config.get("training_mode") != config["training_mode"]:
        raise ValueError("Stage 1 artifact mode disagrees with Stage 2 training_mode")
    if parent_config.get("train_sources") != config["replay_sources"]:
        raise ValueError("replay_sources must exactly preserve the Stage 1 selected sources and order")
    if Path(parent_config.get("suite_path", "")).resolve() != Path(config["replay_suite_path"]).resolve():
        raise ValueError("replay_suite_path must be the Stage 1 suite")
    if parent_config.get("weight_decay") != config["weight_decay"]:
        raise ValueError("Stage 2 optimizer continuation requires the Stage 1 weight_decay")
    required = {"optimizer", "rng", "global_step", "identity"}
    if not required <= state.keys():
        raise ValueError(f"Stage 1 checkpoint is incomplete: {required - state.keys()}")
    provenance = {
        "checkpoint": str(parent.resolve()),
        "artifact_sha256": artifact_sha256(parent),
        "training_state_sha256": sha256_file(state_path),
        "global_step": state["global_step"],
        "stage1_identity": state["identity"],
        "transition": {
            "model": "inherited",
            "optimizer_moments": "inherited",
            "optimizer_learning_rates": "stage2_config",
            "scheduler": "reset_for_stage2",
            "global_rng": "inherited",
            "rl_noise_rng": "new_dedicated_stream",
            "data_cursor": "new_stage2_streams",
        },
    }
    return state, provenance


def _identity(config: dict, parent: dict) -> dict:
    typed = Path(config["typed_suite_path"])
    replay = Path(config["replay_suite_path"])
    return {
        "config": config,
        "resources_sha256": digest("configs/resources.toml"),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deterministic": True,
            "cublas_workspace": ":4096:8",
            "tf32": True,
        },
        "data": {
            "typed_manifest_sha256": digest(typed / "manifest.json"),
            "typed_train_sha256": digest(typed / "train.jsonl"),
            "typed_development_sha256": digest(typed / "development.jsonl"),
            "replay_train_sha256": digest(replay / "train.jsonl"),
            "replay_development_sha256": digest(replay / "development.jsonl"),
        },
        "parent": parent,
    }


def _checkpoint(
    model,
    optimizer,
    scheduler,
    rl_generator,
    output: Path,
    step: int,
    current_stream: dict,
    replay_stream: dict,
    best: float,
    identity: dict,
) -> Path:
    destination = output / f"step-{step:06d}"
    temporary = output / f".step-{step:06d}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
    save_artifact(model, temporary)
    checkpoint_artifact_sha256 = artifact_sha256(temporary)
    torch.save(
        {
            "stage": 2,
            "artifact_sha256": checkpoint_artifact_sha256,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": None,
            "global_step": step,
            "current_stream": current_stream,
            "replay_stream": replay_stream,
            "best_typed_development_nll": best,
            "identity": identity,
            "rng": rng_state(),
            "rl_generator_state": rl_generator.get_state(),
        },
        temporary / "training_state.pt",
    )
    os.replace(temporary, destination)
    return destination


def _ramp(step: int, target: float, steps: int) -> float:
    return target if steps == 0 else target * min(1.0, step / steps)


def train(config_path, output, *, parent=None, resume=None, stop_after=None, wandb_project=None, wandb_name=None):
    if bool(parent) == bool(resume):
        raise ValueError("exactly one of parent (Stage 1 handoff) or resume (exact Stage 2 resume) is required")
    config_path, output = Path(config_path), Path(output)
    parent, resume = Path(parent) if parent else None, Path(resume) if resume else None
    config = _config(config_path)
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop-after must be positive")
    require_entity(wandb_project)
    configure_runtime(config["seed"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage 2 requires a BF16-capable CUDA device")
    if parent and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"New run requires an empty output directory: {output}")

    resume_state = None
    if resume:
        resume_state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        if resume_state.get("stage") != 2:
            raise ValueError("--resume requires a Stage 2 checkpoint; use --parent for Stage 1 handoff")
        if resume_state.get("artifact_sha256") != artifact_sha256(resume):
            raise ValueError("Stage 2 training state does not belong to the resume artifact")
        provenance = resume_state["identity"]["parent"]
        identity = _identity(config, provenance)
        if resume_state["identity"] != identity:
            raise ValueError("Stage 2 resume resource/training configuration mismatch")
        model, tokenizer = load_artifact(resume, config["base_path"], trainable=True)
        parent_state = None
    else:
        parent_state, provenance = _parent(parent, config)
        identity = _identity(config, provenance)
        model, tokenizer = load_artifact(parent, config["base_path"], trainable=True)
    artifact_mode = model.manifest.get("training_mode", "lora")
    parent_mode = provenance["stage1_identity"]["config"]["training_mode"]
    expected_mode = parent_mode if config["training_mode"] == "inherit" else config["training_mode"]
    if artifact_mode != expected_mode:
        raise ValueError("loaded artifact training mode disagrees with the parent/configuration")

    output.mkdir(parents=True, exist_ok=True)
    config_copy = output / "config.toml"
    if config_copy.exists() and tomllib.loads(config_copy.read_text()) != tomllib.loads(config_path.read_text()):
        raise ValueError("Output directory belongs to another configuration")
    config_copy.write_text(config_path.read_text())
    log_path = output / "metrics.jsonl"
    tracker = TrainingTracker(log_path, output, project=wandb_project, name=wandb_name)

    current_records = load_suite(config["typed_suite_path"], "train")
    development_records = load_suite(config["typed_suite_path"], "development")
    selected = set(config["replay_sources"])
    replay_records = [
        record for record in load_suite(config["replay_suite_path"], "train")
        if record.get("_meta", {}).get("source") in selected
    ]
    retention_records = [
        record for record in load_suite(config["replay_suite_path"], "development")
        if record.get("_meta", {}).get("source") in selected
    ]
    if not replay_records or not retention_records:
        raise ValueError("Stage 1 replay/retention data is empty after exact source filtering")
    current_data = _encoded(current_records, tokenizer, config["max_packed"], "typed TRAIN")
    replay_data = _encoded(replay_records, tokenizer, config["max_packed"], "Stage 1 replay")
    _encoded(development_records, tokenizer, config["max_packed"], "typed development")
    _encoded(retention_records, tokenizer, config["max_packed"], "Stage 1 retention")

    model.to("cuda")
    parameters, optimizer_groups = _parameter_groups(model, config)
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=config["weight_decay"])
    fresh_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    if parent_state:
        optimizer.load_state_dict(parent_state["optimizer"])
        for group, fresh in zip(optimizer.param_groups, fresh_groups, strict=True):
            group.update(fresh)
            group["initial_lr"] = group["lr"]
    scheduler = _scheduler(optimizer, config)
    rl_generator = torch.Generator(device="cuda")
    rl_generator.manual_seed(config["rl_seed"])

    step, best = 0, math.inf
    current_stream = {"epoch": 0, "cursor": 0}
    replay_stream = {"epoch": 0, "cursor": 0}
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        step = resume_state["global_step"]
        best = resume_state["best_typed_development_nll"]
        current_stream = dict(resume_state["current_stream"])
        replay_stream = dict(resume_state["replay_stream"])
        restore_rng(resume_state["rng"])
        rl_generator.set_state(resume_state["rl_generator_state"])
    else:
        restore_rng(parent_state["rng"])
    for stream, data, label in (
        (current_stream, current_data, "current"),
        (replay_stream, replay_data, "replay"),
    ):
        if stream["epoch"] < 0 or not 0 <= stream["cursor"] < len(data):
            raise ValueError(f"invalid {label} data cursor")

    target_step = min(config["max_steps"], step + stop_after) if stop_after else config["max_steps"]
    tracker.log(
        "ready",
        step=step,
        target_step=target_step,
        mode=config["mode"],
        objective=MODES[config["mode"]],
        typed_train_records=len(current_records),
        typed_development_records=len(development_records),
        replay_records=len(replay_records),
        retention_records=len(retention_records),
        replay_sources=config["replay_sources"],
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        parent=provenance,
        transition=provenance["transition"],
        config=config,
        device=torch.cuda.get_device_name(),
    )
    last_checkpoint = None
    model.train()
    while step < target_step:
        current_batches = [
            _take(current_data, current_stream, config["batch_size"], config["seed"], 1)
            for _ in range(config["gradient_accumulation"])
        ]
        replay_batches = [
            _take(replay_data, replay_stream, config["replay_batch_size"], config["seed"], 2)
            for _ in range(config["gradient_accumulation"])
        ]
        current_questions = sum(len(record.decide_positions) for records in current_batches for record in records)
        replay_questions = sum(len(record.decide_positions) for records in replay_batches for record in records)
        objective_questions = sum(
            len(record.decide_positions) if config["mode"] == "C" else
            sum(len(options) > 1 for options in record.option_end_positions)
            for records in current_batches for record in records
        ) if config["mode"] in {"C", "D"} else 0
        objective_coefficient = _ramp(step, config["objective_weight"], config["objective_ramp_steps"])
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        totals = {"ce": 0.0, "objective": 0.0, "replay_ce": 0.0, "tokens": 0}
        for current_records_batch, replay_records_batch in zip(current_batches, replay_batches, strict=True):
            batch = collate(current_records_batch, pad_token_id=tokenizer.pad_token_id or 0).to("cuda")
            logits = model(batch)
            ce_values, ce_valid = per_question_ce(logits, batch)
            current_loss = config["ce_weight"] * ce_values.masked_select(ce_valid).sum() / current_questions
            totals["ce"] += float(ce_values.masked_select(ce_valid).sum().detach())
            if config["mode"] == "C" and objective_coefficient:
                values, valid = proper_loss(
                    logits, batch,
                    spherical_weight=config["spherical_weight"],
                    rps_weight=config["rps_weight"],
                    log_floor=config["log_floor"],
                )
                current_loss = current_loss + objective_coefficient * values.masked_select(valid).sum() / objective_questions
                totals["objective"] += float(values.masked_select(valid).sum().detach())
            elif config["mode"] == "D" and objective_coefficient and objective_questions:
                values, valid = rlcd_loss(
                    logits,
                    batch,
                    group_size=config["group_size"],
                    sigma=config["sigma"],
                    spherical_weight=config["spherical_weight"],
                    rps_weight=config["rps_weight"],
                    log_floor=config["log_floor"],
                    normalize_advantage=config["normalize_advantage"],
                    generator=rl_generator,
                )
                current_loss = current_loss + objective_coefficient * values.masked_select(valid).sum() / objective_questions
                totals["objective"] += float(values.masked_select(valid).sum().detach())
            if not bool(torch.isfinite(current_loss)):
                raise FloatingPointError(f"non-finite current objective at Stage 2 step {step}")
            current_loss.backward()

            replay_batch = collate(replay_records_batch, pad_token_id=tokenizer.pad_token_id or 0).to("cuda")
            replay_logits = model(replay_batch)
            replay_values, replay_valid = per_question_ce(replay_logits, replay_batch)
            replay_loss = config["replay_weight"] * replay_values.masked_select(replay_valid).sum() / replay_questions
            if not bool(torch.isfinite(replay_loss)):
                raise FloatingPointError(f"non-finite replay CE at Stage 2 step {step}")
            replay_loss.backward()
            totals["replay_ce"] += float(replay_values.masked_select(replay_valid).sum().detach())
            totals["tokens"] += sum(len(record.input_ids) for record in (*current_records_batch, *replay_records_batch))

        gradient_groups = {
            group["name"]: {"grad_norm": _grad_norm(group["params"]), "lr": group["lr"]}
            for group in optimizer.param_groups
        }
        if (step + 1) % config["eval_interval"] == 0 or step + 1 == config["max_steps"]:
            tracker.gradients(step + 1, optimizer.param_groups)
        norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tracker.log(
            "train",
            step=step,
            current_stream=current_stream,
            replay_stream=replay_stream,
            ce=totals["ce"] / current_questions,
            objective=(totals["objective"] / objective_questions if objective_questions and objective_coefficient else None),
            replay_ce=totals["replay_ce"] / replay_questions,
            weights={
                "ce": config["ce_weight"],
                "objective": objective_coefficient if config["mode"] != "B" else 0.0,
                "replay_ce": config["replay_weight"],
            },
            questions={"current": current_questions, "objective": objective_questions, "replay": replay_questions},
            grad_norm=float(norm),
            gradient_groups=gradient_groups,
            seconds=elapsed,
            tokens=totals["tokens"],
            tokens_per_second=totals["tokens"] / elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )

        improved = False
        if step % config["eval_interval"] == 0 or step == config["max_steps"]:
            typed_report = training_diagnostics(
                model, tokenizer, development_records,
                batch_size=config["eval_batch_size"], device="cuda", max_packed=config["max_packed"],
            )
            score = typed_report["all"]["nll"]
            if score is None or not math.isfinite(score):
                raise FloatingPointError("non-finite typed-development NLL")
            improved = score < best
            best = min(best, score)
            tracker.log(
                "typed_development",
                step=step,
                selection={"split": "development", "subset": "all", "metric": "nll", "value": score},
                report=typed_report,
            )
            retention_report = training_diagnostics(
                model, tokenizer, retention_records,
                batch_size=config["eval_batch_size"], device="cuda", max_packed=config["max_packed"],
            )
            tracker.log("stage1_retention", step=step, selection=False, report=retention_report)
        if step % config["checkpoint_interval"] == 0 or step == target_step or improved:
            last_checkpoint = _checkpoint(
                model, optimizer, scheduler, rl_generator, output, step,
                dict(current_stream), dict(replay_stream), best, identity,
            )
            tracker.log("checkpoint", step=step, path=str(last_checkpoint))
            if improved:
                temporary = output / ".best.json.tmp"
                temporary.write_text(json.dumps({
                    "checkpoint": last_checkpoint.name,
                    "selection": {"split": "development", "subset": "all", "metric": "nll", "value": score},
                }, sort_keys=True) + "\n")
                os.replace(temporary, output / "best.json")
    tracker.log(
        "finished" if step == config["max_steps"] else "paused",
        step=step,
        checkpoint=str(last_checkpoint) if last_checkpoint else str(resume),
    )
    tracker.finish()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parent", help="Trusted Stage 1 checkpoint: inherit model/optimizer/RNG and reset scheduler.")
    source.add_argument("--resume", help="Exact Stage 2 checkpoint: restore model/optimizer/scheduler/RNG/cursors.")
    parser.add_argument("--stop-after", type=int, help="Pause after N additional updates without changing the budget.")
    parser.add_argument("--wandb-project", help="Enable W&B tracking in this project")
    parser.add_argument("--wandb-name", help="W&B display name (defaults to output directory)")
    args = parser.parse_args(argv)
    train(args.config, args.output, parent=args.parent, resume=args.resume, stop_after=args.stop_after,
          wandb_project=args.wandb_project, wandb_name=args.wandb_name)


if __name__ == "__main__":
    main()
