"""Single-GPU supervised decision training; checkpoints are trusted local files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import tomllib

import numpy as np
import torch

from haidass_kev_train.data.augmentation import augment_record
from haidass_kev_train.data.packing import collate, encode_record, load_suite
from haidass_kev_train.evaluation.diagnostics import select_probe, training_diagnostics
from haidass_kev_train.evaluation.metrics import per_question_ce
from haidass_kev_train.model.decision import build_model, load_artifact, save_artifact
from haidass_kev_train.training.tracking import TrainingTracker, require_entity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()




def prepare(suite, split, tokenizer, max_packed, sources=None):
    """Load and encode an unaugmented split for standalone profiling callers."""
    records = load_suite(suite, split)
    if sources is not None:
        selected = set(sources)
        records = [record for record in records if record.get("_meta", {}).get("source") in selected]
    encoded = [encode_record(record, tokenizer, max_packed=max_packed) for record in records]
    if not encoded:
        raise ValueError(f"Empty {split} data")
    return encoded


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint(model, optimizer, scheduler, output, step, epoch, cursor, best, identity):
    destination = output / f"step-{step:06d}"
    temporary = output / f".step-{step:06d}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {destination}")
    save_artifact(model, temporary)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": None, "global_step": step, "epoch": epoch, "data_cursor": cursor,
                "best_macro_nll": best, "identity": identity, "rng": rng_state()},
               temporary / "training_state.pt")
    os.replace(temporary, destination)
    return destination


def configure_runtime(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _config(path):
    config = tomllib.loads(Path(path).read_text())
    config.setdefault("scheduler", "cosine")
    required = {"seed", "batch_size", "gradient_accumulation", "learning_rate", "weight_decay",
                "max_steps", "warmup_steps", "scheduler", "eval_interval", "checkpoint_interval",
                "max_packed", "base_path", "suite_path", "eval_batch_size", "max_grad_norm",
                "training_mode", "probe_groups", "development_selection", "augmentation"}
    if not required <= config.keys():
        raise ValueError(f"Missing config: {required - config.keys()}")
    for key in ("seed", "batch_size", "gradient_accumulation", "max_steps", "eval_interval",
                "checkpoint_interval", "max_packed", "eval_batch_size", "probe_groups"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["warmup_steps"], int) or not 0 <= config["warmup_steps"] < config["max_steps"]:
        raise ValueError("warmup_steps must be a non-negative integer below max_steps")
    if config["scheduler"] not in {"cosine", "onecycle"}:
        raise ValueError("scheduler must be 'cosine' or 'onecycle'")
    if config["scheduler"] == "onecycle" and config["warmup_steps"] != 0:
        raise ValueError("onecycle requires warmup_steps = 0")
    config.setdefault("head_learning_rate", config["learning_rate"])
    if config["learning_rate"] <= 0 or config["head_learning_rate"] <= 0 or config["weight_decay"] < 0 or config["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer configuration")
    if config["training_mode"] not in {"lora", "full"}:
        raise ValueError("training_mode must be 'lora' or 'full'")
    if config["development_selection"] not in {"clean", "all"}:
        raise ValueError("development_selection must be 'clean' or 'all'")
    augmentation = config["augmentation"]
    allowed = {"shuffle", "p_none", "p_none_distract", "p_distract", "p_none_pair"}
    if not isinstance(augmentation, dict) or set(augmentation) - allowed:
        raise ValueError(f"Invalid augmentation configuration: {set(augmentation) - allowed if isinstance(augmentation, dict) else augmentation!r}")
    return config


def _epoch_records(records, tokenizer, config, epoch):
    augmented = [item for record in records for item in augment_record(
        record, seed=config["seed"], epoch=epoch, **config["augmentation"])]
    encoded = [encode_record(record, tokenizer, max_packed=config["max_packed"]) for record in augmented]
    if not encoded:
        raise ValueError("Empty augmented train data")
    order = list(range(len(encoded)))
    random.Random(config["seed"] + epoch).shuffle(order)
    return encoded, order


def _parameter_groups(model, config):
    head = [parameter for parameter in model.pointer_head.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head}
    backbone = [parameter for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in head_ids]
    if not head or not backbone:
        raise RuntimeError("Training requires distinct trainable backbone and pointer-head parameters")
    parameters = [*backbone, *head]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("Optimizer parameter groups overlap")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("Trainable parameters must have FP32 master weights")
    groups = [
        {"params": backbone, "lr": config["learning_rate"], "name": "backbone"},
        {"params": head, "lr": config["head_learning_rate"], "name": "head"},
    ]
    return parameters, groups


def _grad_norm(parameters):
    squared = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(squared).sum().sqrt()) if squared else 0.0


def train(config_path, output, resume=None, stop_after=None, *, wandb_project=None, wandb_name=None, wandb_group=None):
    config_path, output = Path(config_path), Path(output)
    config = _config(config_path)
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop-after must be positive")
    require_entity(wandb_project)
    configure_runtime(config["seed"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("SFT requires a BF16-capable CUDA device")
    identity = {"config": config, "resources_sha256": digest("configs/resources.toml"),
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "deterministic": True,
                            "cublas_workspace": ":4096:8", "tf32": True},
                "train_sha256": digest(Path(config["suite_path"]) / "train.jsonl"),
                "development_sha256": digest(Path(config["suite_path"]) / "development.jsonl")}
    state = None
    if resume:
        state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
        state["identity"]["config"].setdefault("scheduler", "cosine")
        if state["identity"] != identity:
            raise ValueError("Resume resource/training configuration mismatch")
        model, tokenizer = load_artifact(resume, config["base_path"], trainable=True)
        artifact_mode = model.manifest.get("training_mode", "lora")
        if artifact_mode != config["training_mode"]:
            raise ValueError("Resume artifact training mode disagrees with configuration")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"New run requires an empty output directory: {output}")
        model, tokenizer = build_model(config["base_path"], training_mode=config["training_mode"])
    output.mkdir(parents=True, exist_ok=True)
    config_copy = output / "config.toml"
    if config_copy.exists() and tomllib.loads(config_copy.read_text()) != tomllib.loads(config_path.read_text()):
        raise ValueError("Output directory belongs to another configuration")
    config_copy.write_text(config_path.read_text())
    log_path = output / "metrics.jsonl"
    tracker = TrainingTracker(log_path, output, project=wandb_project, name=wandb_name, group=wandb_group)

    train_records = load_suite(config["suite_path"], "train")
    if config.get("train_sources") is not None:
        selected = set(config["train_sources"])
        train_records = [record for record in train_records if record.get("_meta", {}).get("source") in selected]
    if not train_records:
        raise ValueError("Empty train data")
    development_records = load_suite(config["suite_path"], "development")
    probe = select_probe(train_records, config["probe_groups"], config["seed"])

    model.to("cuda")
    parameters, optimizer_groups = _parameter_groups(model, config)
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=config["weight_decay"])

    if config["scheduler"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config["learning_rate"], config["head_learning_rate"]],
            total_steps=config["max_steps"],
            pct_start=0.1,
        )
    else:
        def schedule(current_step):
            warmup = config["warmup_steps"]
            if current_step < warmup:
                return (current_step + 1) / max(1, warmup)
            progress = min(1., (current_step - warmup) / (config["max_steps"] - warmup))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    step, epoch, cursor, best = 0, 0, 0, math.inf
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, epoch, cursor = state["global_step"], state["epoch"], state["data_cursor"]
        best = state["best_macro_nll"]
        restore_rng(state["rng"])
    train_data, order = _epoch_records(train_records, tokenizer, config, epoch)
    if cursor > len(order):
        raise ValueError("Resume data cursor exceeds deterministic epoch data")
    target_step = min(config["max_steps"], step + stop_after) if stop_after else config["max_steps"]
    probe_ids = [record["_meta"]["id"] for record in probe]
    tracker.log("ready", step=step, target_step=target_step, train_records=len(train_records),
         augmented_records=len(train_data), development_records=len(development_records),
         train_probe_record_ids=probe_ids,
         trainable_parameters=sum(parameter.numel() for parameter in parameters),
         device=torch.cuda.get_device_name(), config=config)
    last_checkpoint = None
    model.train()
    while step < target_step:
        microbatches = []
        for _ in range(config["gradient_accumulation"]):
            if cursor == len(order):
                epoch += 1
                cursor = 0
                train_data, order = _epoch_records(train_records, tokenizer, config, epoch)
            indices = order[cursor:cursor + config["batch_size"]]
            cursor += len(indices)
            microbatches.append([train_data[index] for index in indices])
        total_questions = sum(len(item.decide_positions) for batch in microbatches for item in batch)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started, loss_sum, tokens = time.perf_counter(), 0., 0
        for records in microbatches:
            batch = collate(records, pad_token_id=tokenizer.pad_token_id or 0).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            values, valid = per_question_ce(logits, batch)
            loss = values[valid].sum() / total_questions
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            loss_sum += float(loss.detach())
            tokens += sum(len(item.input_ids) for item in records)
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
        tracker.log("train", step=step, epoch=epoch, data_cursor=cursor, loss=loss_sum,
             grad_norm=float(norm), gradient_groups=gradient_groups, seconds=elapsed,
             questions=total_questions, tokens=tokens, tokens_per_second=tokens / elapsed,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
             peak_reserved_bytes=torch.cuda.max_memory_reserved())
        improved = False
        if step % config["eval_interval"] == 0 or step == config["max_steps"]:
            probe_report = training_diagnostics(model, tokenizer, probe, batch_size=config["eval_batch_size"],
                                                device="cuda", max_packed=config["max_packed"])
            tracker.log("train_probe", step=step, record_ids=probe_ids, report=probe_report)
            report = training_diagnostics(model, tokenizer, development_records,
                                          batch_size=config["eval_batch_size"], device="cuda",
                                          max_packed=config["max_packed"])
            subset = config["development_selection"]
            score = report[subset]["macro_nll"]
            if score is None or not math.isfinite(score):
                raise FloatingPointError("Nonfinite development macro NLL")
            improved = score < best
            best = min(best, score)
            selection = {"split": "development", "subset": subset, "metric": "macro_nll", "value": score}
            tracker.log("development", step=step, selection=selection, report=report)
        if step % config["checkpoint_interval"] == 0 or step == target_step or improved:
            last_checkpoint = checkpoint(model, optimizer, scheduler, output, step, epoch, cursor, best, identity)
            tracker.log("checkpoint", step=step, path=str(last_checkpoint))
            if improved:
                temp = output / ".best.json.tmp"
                temp.write_text(json.dumps({"checkpoint": last_checkpoint.name,
                                            "selection": selection}, sort_keys=True) + "\n")
                os.replace(temp, output / "best.json")
    tracker.log("finished" if step == config["max_steps"] else "paused", step=step,
                checkpoint=str(last_checkpoint) if last_checkpoint else str(resume))
    tracker.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int, help="Pause after N additional updates without changing the run budget")
    parser.add_argument("--wandb-project", help="Enable W&B tracking in this project")
    parser.add_argument("--wandb-name", help="W&B display name (defaults to output directory)")
    args = parser.parse_args()
    train(args.config, args.output, args.resume, args.stop_after,
          wandb_project=args.wandb_project, wandb_name=args.wandb_name)


if __name__ == "__main__":
    main()
