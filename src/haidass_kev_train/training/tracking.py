"""Optional W&B mirror of the authoritative local training metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import torch


def _scalars(value, prefix):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _scalars(child, f"{prefix}/{key}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value

def require_entity(project):
    if project:
        entity = os.environ.get("WANDB_ENTITY")
        if not entity:
            raise ValueError("WANDB_ENTITY is required when W&B tracking is enabled")
        return entity
    return None


class TrainingTracker:
    def __init__(self, log_path, output, *, project=None, name=None, group=None):
        self.log_path = Path(log_path)
        self.run = None
        self.wandb = None
        if not project:
            return
        entity = require_entity(project)
        output = Path(output)
        identity_path = output / "wandb-run.json"
        if identity_path.exists():
            identity = json.loads(identity_path.read_text())
            if identity["entity"] != entity or identity["project"] != project:
                raise ValueError("W&B entity/project differs from the existing training run")
            if name is not None and name != identity["name"]:
                raise ValueError("W&B name differs from the existing training run")
        else:
            identity = {"entity": entity, "project": project,
                        "name": name or output.name, "id": uuid.uuid4().hex}
            temporary = identity_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(identity, sort_keys=True) + "\n")
            os.replace(temporary, identity_path)
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                entity=entity, project=project, name=identity["name"], id=identity["id"],
                resume="allow", reinit="create_new", group=group, dir=str(output), save_code=False,
                settings=wandb.Settings(console="off", disable_git=True, disable_code=True),
            )
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step")
        except Exception as error:
            self._disable(error)

    def _disable(self, error):
        print(f"W&B tracking disabled: {error}", file=sys.stderr, flush=True)
        run, self.run = self.run, None
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass

    def log(self, event, **values):
        row = {"event": event, **values}
        text = json.dumps(row, allow_nan=False)
        print(text, flush=True)
        with self.log_path.open("a") as handle:
            handle.write(text + "\n")
        if self.run is None or event not in {"train", "train_probe", "development", "typed_development", "stage1_retention"}:
            return
        metrics = {key: value for field, item in values.items() if field != "step"
                   for key, value in _scalars(item, f"{event}/{field}")}
        try:
            self.run.log({"global_step": values["step"], **metrics})
        except Exception as error:
            self._disable(error)

    def gradients(self, step, optimizer_groups):
        """Log pre-clip distributions without collecting all model gradients in memory."""
        if self.run is None:
            return
        try:
            histograms = {}
            for group in optimizer_groups:
                gradients = [p.grad.detach() for p in group["params"] if p.grad is not None]
                if not gradients:
                    continue
                minimum = min(float(gradient.min()) for gradient in gradients)
                maximum = max(float(gradient.max()) for gradient in gradients)
                if minimum == maximum:
                    minimum -= 0.5
                    maximum += 0.5
                counts = torch.zeros(64, dtype=torch.int64, device=gradients[0].device)
                deterministic = torch.are_deterministic_algorithms_enabled()
                warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
                try:
                    # CUDA histc uses atomics; telemetry cannot change optimization determinism.
                    torch.use_deterministic_algorithms(False)
                    for gradient in gradients:
                        counts += torch.histc(gradient.float(), bins=64, min=minimum, max=maximum).to(torch.int64)
                finally:
                    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
                histograms[f"gradients/{group['name']}"] = self.wandb.Histogram(
                    np_histogram=(counts.cpu().numpy(), np.linspace(minimum, maximum, 65)))
            if histograms:
                self.run.log({"global_step": step, **histograms})
        except Exception as error:
            self._disable(error)

    def finish(self):
        if self.run is not None:
            run, self.run = self.run, None
            try:
                run.finish()
            except Exception as error:
                print(f"W&B tracking disabled: {error}", file=sys.stderr, flush=True)
