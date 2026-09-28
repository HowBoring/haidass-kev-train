"""FP32-master Qwen3 decision model with LoRA and full-training modes."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import tomllib

import torch
from torch import nn
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MARKER_TOKENS = dict(zip(
    ["<|kev_state|>", "<|kev_question|>", "<|kev_option|>", "<|kev_option_end|>", "<|kev_decide|>"],
    ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>"],
))
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none", "modules_to_save": ["pointer_head"]}


@lru_cache(maxsize=4)
def resolve_markers(tokenizer):
    result = {}
    for expected, (logical, spelling) in enumerate(MARKER_TOKENS.items(), 6):
        actual = tokenizer.encode(spelling, add_special_tokens=False)
        if actual != [expected] or len(tokenizer) != 64000:
            raise ValueError(f"Pinned structural token mismatch: {spelling} -> {actual}")
        result[logical] = expected
    return result


class PointerHead(nn.Module):
    def __init__(self, hidden_size=576, dimension=256):
        super().__init__()
        self.query = nn.Linear(hidden_size, dimension, bias=False)
        self.key = nn.Linear(hidden_size, dimension, bias=False)
        self.scale = math.sqrt(dimension)

    def forward(self, decisions, options):
        return (self.query(decisions).unsqueeze(-2) * self.key(options)).sum(-1) / self.scale


class DecisionModel(nn.Module):
    def __init__(self, backbone, manifest):
        super().__init__()
        self.backbone, self.manifest = backbone, manifest

    @property
    def pointer_head(self):
        backbone = self.backbone.get_base_model() if isinstance(self.backbone, PeftModel) else self.backbone
        return backbone.pointer_head

    def forward(self, batch):
        device_type = batch.input_ids.device.type
        with torch.autocast(device_type, dtype=torch.bfloat16):
            hidden = self.backbone(input_ids=batch.input_ids, position_ids=batch.position_ids,
                                   attention_mask=batch.attention_bias, use_cache=False,
                                   return_dict=True).last_hidden_state
            b = torch.arange(hidden.shape[0], device=hidden.device)
            decisions = hidden[b[:, None], batch.decide_positions]
            options = hidden[b[:, None, None], batch.option_end_positions]
            logits = self.pointer_head(decisions, options).to(torch.bfloat16)
        logits = logits.masked_fill(~batch.option_mask, float("-inf"))
        return logits.masked_fill(~batch.question_mask.unsqueeze(-1), 0.)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _base(base_path, resources_path):
    resource = tomllib.loads(Path(resources_path).read_text())["model"]["haidass"]
    base_path = Path(base_path or resource["local_dir"])
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    markers = resolve_markers(tokenizer)
    loaded = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32,
                                                 attn_implementation="sdpa", local_files_only=True)
    backbone = loaded.model
    if backbone.config.hidden_size != 576 or backbone.get_input_embeddings().weight.shape != (64000, 576):
        raise ValueError("Pinned backbone shape mismatch")
    backbone.pointer_head = PointerHead()
    manifest = {"interface_version": 1, "base_model": resource,
                "base_weights_sha256": _sha(base_path / "model.safetensors"),
                "base_config_sha256": _sha(base_path / "config.json"),
                "tokenizer": {"length": len(tokenizer), "model_sha256": _sha(base_path / "tokenizer.model"),
                              "config_sha256": _sha(base_path / "tokenizer_config.json")},
                "markers": {key: {"token": MARKER_TOKENS[key], "id": value} for key, value in markers.items()},
                "pointer_head": {"hidden_size": 576, "dimension": 256, "bias": False, "projections": 2},
                "lora": {**LORA, "trainable_token_indices": list(markers.values())}, "weight_dtype": "float32"}
    return backbone, tokenizer, manifest


def build_model(base_path=None, resources_path="configs/resources.toml", training_mode="lora"):
    if training_mode not in {"lora", "full"}:
        raise ValueError("training_mode must be 'lora' or 'full'")
    backbone, tokenizer, manifest = _base(base_path, resources_path)
    if training_mode == "lora":
        backbone = get_peft_model(backbone, LoraConfig(**manifest["lora"]))
    else:
        manifest["training_mode"] = "full"
    return DecisionModel(backbone, manifest), tokenizer


def save_artifact(model, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact {path}")
    training_mode = model.manifest.get("training_mode", "lora")
    if training_mode == "lora":
        weights = get_peft_model_state_dict(model.backbone, save_embedding_layers=False)
        if any(t.dtype != torch.float32 for t in weights.values()):
            raise ValueError("Decision Model Artifact trainable weights must be FP32")
        model.backbone.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    elif training_mode == "full":
        weights = {name: tensor.detach().cpu().contiguous()
                   for name, tensor in model.backbone.state_dict().items()}
        if any(tensor.dtype != torch.float32 for tensor in weights.values()):
            raise ValueError("Decision Model Artifact full-mode weights must be FP32")
        path.mkdir()
        save_file(weights, path / "model.safetensors")
    else:
        raise ValueError(f"Unsupported Decision Model Artifact training mode: {training_mode!r}")
    (path / "decision_model.json").write_text(json.dumps(model.manifest, sort_keys=True, indent=2) + "\n")


def load_artifact(path, base_path=None, resources_path="configs/resources.toml", trainable=False):
    path = Path(path)
    expected = json.loads((path / "decision_model.json").read_text())
    training_mode = expected.get("training_mode", "lora")
    if training_mode not in {"lora", "full"}:
        raise ValueError(f"Unsupported Decision Model Artifact training mode: {training_mode!r}")
    backbone, tokenizer, actual = _base(base_path, resources_path)
    if training_mode == "full":
        actual["training_mode"] = "full"
    if actual != expected:
        raise ValueError("Decision Model Artifact resource, tokenizer, interface, dtype or architecture mismatch")
    if training_mode == "full":
        saved = load_file(path / "model.safetensors")
        restored = backbone.state_dict()
        if saved.keys() != restored.keys() or any(saved[name].shape != restored[name].shape for name in saved):
            raise ValueError("Full-model state disagrees with the manifest architecture")
        if any(tensor.dtype != torch.float32 for tensor in saved.values()):
            raise ValueError("Decision Model Artifact contains non-FP32 weights")
        backbone.load_state_dict(saved, strict=True)
        if any(not torch.equal(saved[name], tensor.cpu()) for name, tensor in backbone.state_dict().items()):
            raise ValueError("Incomplete or inexact full-model restoration")
        backbone.requires_grad_(trainable)
    else:
        adapter_config = LoraConfig.from_pretrained(path).to_dict()
        for key, value in expected["lora"].items():
            observed = adapter_config.get(key)
            if isinstance(value, list):
                matches = isinstance(observed, (list, set)) and sorted(value) == sorted(observed)
            else:
                matches = observed == value
            if not matches:
                raise ValueError(f"Adapter configuration disagrees with manifest: {key}")
        saved = load_file(path / "adapter_model.safetensors")
        if any(t.dtype != torch.float32 for t in saved.values()):
            raise ValueError("Adapter contains non-FP32 weights")
        backbone = PeftModel.from_pretrained(backbone, path, is_trainable=trainable)
        restored = get_peft_model_state_dict(backbone, save_embedding_layers=False)
        if saved.keys() != restored.keys() or any(not torch.equal(saved[key], restored[key].cpu()) for key in saved):
            raise ValueError("Incomplete or inexact adapter restoration (including pointer/token rows)")
        if not trainable:
            backbone.requires_grad_(False)
    return DecisionModel(backbone, actual), tokenizer
