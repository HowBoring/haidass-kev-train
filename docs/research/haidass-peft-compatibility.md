# Haidass1.5-143M × Transformers 5.17 / PEFT 0.21 — Adaptation Compatibility

Resolves [HowBoring/haidass-kev-train#7](https://github.com/HowBoring/haidass-kev-train/issues/7) —
*"Does `DALabCommunity/Haidass1.5-143M` work with the planned Transformers and PEFT adaptation path without hidden incompatibilities?"*

**Verdict: COMPATIBLE.** The full planned chain — load → add tokens → resize → strip `lm_head` → PEFT
(LoRA r=16 + `trainable_token_indices` + `modules_to_save` head) → train → save → reload — runs end-to-end
and round-trips bit-exactly on the pinned stack (`transformers==5.17.0`, `peft==0.21.0`, `torch==2.14.0`,
Python 3.12; `pyproject.toml:1-16`). Every claim below was verified **empirically** against the frozen local
checkpoint (`models/base/haidass1.5-143m`, revision `d8a00d4943971088e6f0f4e08fb317dd5ba33ed1` per
`configs/resources.toml:1-4`) by running the exact construction sequence, plus **primary-source** citations to
the installed library source.

## 1. The model is a stock Qwen3 — module paths resolve

**Inspected local files:**

- `config.json` — `"architectures": ["Qwen3ForCausalLM"]`, `"model_type": "qwen3"`, 30 layers, hidden 576,
  9 heads / 3 KV heads (GQA), head_dim 64, FFN 1536, `vocab_size: 64000`, `tie_word_embeddings: true`,
  `torch_dtype: "bfloat16"`, `max_position_embeddings: 4096`.
- `model.safetensors` header (all 332 tensors inspected): keys are exactly the stock `Qwen3Model` names —
  `model.embed_tokens.weight [64000,576]`, per-layer `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
  `self_attn.{q,k}_norm.weight`, `mlp.{gate,up,down}_proj.weight`, input/post-attention `layernorm.weight`,
  and `model.norm.weight`. **No `lm_head.weight` tensor exists** (tied embeddings, so it is not stored).
- The [model card](https://huggingface.co/DALabCommunity/Haidass1.5-143M) corroborates: Qwen3 architecture,
  tied word embeddings, BF16, custom 64k bilingual vocabulary, trained on Ascend/MindSpeed-LLM.

**Consequence:** `AutoModelForCausalLM.from_pretrained(...)` instantiates `Qwen3ForCausalLM`; `model.model`
is `Qwen3Model` (verified). Every planned LoRA target exists at the standard path. PEFT 0.21 knows the Qwen3
architecture — its LoRA defaults for qwen3 target `q,k,v,o,gate,up,down_proj` (`peft/tuners/lora/config.py:457+`,
"modules will be chosen according to the model architecture"). The default LoRA dtype is fp32 even on a bf16
base (verified in §5).

## 2. Tokenizer: works, but needs `protobuf` (unpinned)

- `tokenizer_config.json` declares `"tokenizer_class": "LlamaTokenizer"` with `"legacy": true` and a
  SentencePiece `tokenizer.model`. Load succeeded only **after** `pip install protobuf` — transformers 5.17
  converts the slow tokenizer and falls back through SentencePiece → TikToken extractors, raising
  `ModuleNotFoundError: No module named 'tiktoken'` / protobuf errors without it. **Deviation #1:** add
  `protobuf` (and optionally `tiktoken`) to `pyproject.toml` dependencies — neither is in `uv.lock` today.
- Post-install the tokenizer loads as a **fast** tokenizer (`is_fast: True`, class `LlamaTokenizer`,
  `vocab_size 64000`), and `add_special_tokens({"additional_special_tokens": [...]})` adds exactly 5 tokens
  (verified: `64000 → 64005`, ids `[64000..64004]`).
- Quirk to respect during data prep: `tokenizer_config.json` sets `model_max_length: 131072` while the model's
  context is `max_position_embeddings: 4096` — truncation must key off the **model** limit.

## 3. Tied embeddings survive resize and are genuinely shared

Verified, with `tie_word_embeddings: true` in the local `config.json`:

- Before resize: `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()` — one storage, no separate
  `lm_head` tensor in the safetensors file.
- After `model.resize_token_embeddings(64005)`: still tied (same `data_ptr`), shape `[64005, 576]` on both
  sides, and the 5 **new rows are bit-identical** between embedding and head (`torch.equal` on both slices).
- New rows are initialized by `mean_resizing=True` (default): "multivariate normal distribution that has old
  embeddings' mean and covariance" — runtime log; API at `transformers/modeling_utils.py:2629-2665`
  (`resize_token_embeddings(..., mean_resizing: bool = True)`). Pass `mean_resizing=False` to opt for
  per-row random init instead; the plan does not require it.

## 4. Removing the language-model head

`backbone = model.model` (verified `Qwen3Model`, no `lm_head` attribute). Because the head is tied, "removing"
it is purely bookkeeping — the embedding table itself must stay (it *is* the output projection's weights).
Two caveats verified:

- **Config flag:** the backbone's config still carries `tie_word_embeddings: true`. This matters for PEFT:
  `peft/utils/other.py:1567` (`model_config.get("tie_word_embeddings", False)`) drives its tied-token logic.
- **Do not use `AutoModel`** as the primary load path. It returns a `Qwen3Model` without an `lm_head` attribute,
  but resize of a bare `AutoModel` does not resize the head-side copy and `tie_word_embeddings` handling in
  PEFT's tied-token wrapper expects the causal-LM wrapper's config contract. Load as `ForCausalLM`, resize,
  then take `.model` — the exact order in §8. *(This paragraph states the verified-empirical sequence; the
  `AutoModel`-direct caution is [INFERENCE] from the resize contract in `transformers/modeling_utils.py:2688-2717`,
  which resizes both `get_input_embeddings()` and `get_output_embeddings()` only on the causal-LM wrapper.)*

## 5. DType / master weights: the single biggest deviation from the plan

**Finding (verified):** `AutoModelForCausalLM.from_pretrained(BASE)` with no dtype argument loads the model in
**bfloat16** — transformers 5.x follows the checkpoint's stored dtype (`torch_dtype: "bfloat16"` in
`config.json`), it does **not** upcast to fp32.

**Consequence for the plan** (`docs/chatgpt/Haidass-Kev-Train-Analysis.md:1921-1936`, "base weights: FP32,
forward/backward: BF16 autocast"): the plan's FP32 master-weight setup requires an **explicit**
`dtype=torch.float32` at load time. Without it, the run silently trains a bf16 base — and a fp32 PointerHead
attached to that bf16 backbone **crashes at the first matmul** with
`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float` (reproduced).

Verified dtype matrix (all empirically):

| Load path | Base weights | LoRA / token-delta params | Master-weight contract met? |
|---|---|---|---|
| `from_pretrained(BASE)` (no dtype) | **bf16** | fp32 (PEFT casts adapters to fp32 regardless) | No — bf16 base |
| `from_pretrained(BASE, dtype=torch.float32)` | fp32 | fp32 | **Yes** |
| `from_pretrained(BASE, dtype=torch.bfloat16)` | bf16 | **fp32** (verified) | No — bf16 base |
| + `torch.autocast("cuda", dtype=torch.bfloat16)` over fp32 base | fp32 | fp32 | **Yes** |

Two supporting primary-source facts: PEFT casts adapter params to fp32 even on bf16 bases (verified; grads
flowed through 61 params in the bf16-base probe), and autocast preserves fp32 master weights (all parameters
remain fp32 after autocast forward/backward — verified; autocast computes in bf16 but stores master weights
fp32 per the [PyTorch autocast
docs](https://pytorch.org/docs/stable/amp.html#autocasting)). Note `torch.autocast("cpu", ...)` casts fewer
op types than CUDA autocast — do not treat CPU-autocast output dtype as a CUDA parity signal.

**Required deviation #2 (make explicit in the training script):** `dtype=torch.float32` must be passed at
load; "FP32 master" is not the transformers 5.x default.

## 6. `trainable_token_indices` — works on Haidass, including tied-head propagation

PEFT 0.21.0's `LoraConfig` supports `trainable_token_indices` (`peft/tuners/lora/config.py:781-794`: "Lets
you specify which token indices to selectively fine-tune without requiring to re-train the whole embedding
matrix"). Verified end-to-end:

- Wraps `embed_tokens` in a `TrainableTokensWrapper`; only the wrapped delta is trainable — the frozen base
  embedding table stays frozen (`embed_tokens.original_module.weight.requires_grad == False`).
- **Tied-head propagation:** because `tie_word_embeddings` is still true on the backbone config, PEFT
  (`peft/utils/related` logic at `peft/utils/other.py:1606-1670`) propagates the token-delta to the tied
  output projection. `LoraConfig.ensure_weight_tying` defaults to **False**
  (`peft/tuners/lora/config.py:989-1001`) but tying is applied when the model itself reports tied weights.
  Since we strip the head, there is no separate `lm_head` module to desynchronize — the delta lands in the
  single shared table. If the head were kept, tied propagation still applies (verified in source:
  `_get_module_names_tied_with_embedding` path).
- Full construction with `trainable_token_indices=[64000..64004]` + LoRA r=16 over all 7 target projections
  + `modules_to_save=["pointer_head"]` yields **5,036,100 trainable params** (verified count).
- Gradients flow to every trainable param after one backward — except the 210 `lora_A` matrices, which have
  zero grad at step 0 because `lora_B` is zero-initialized ("the LoRA B weight being set to 0. This means
  that without further training, the LoRA adapter will be a no-op" — `peft/tuners/lora/config.py:490-493`).
  This is correct standard init, not a bug; do not "fix" it.
- The token delta updated after one step (L1 ≈ 1750 on the probe), confirming the special-token embeddings
  actually train. **The exact failure mode the plan feared ("special embedding 实际没被训练") did not occur.**
- FSDP caveat from the PEFT docs: "training with FSDP requires `use_orig_params=True`" — irrelevant here
  (single GPU, no FSDP planned), noted for completeness.

## 7. LoRA target modules

Verified present at the stock paths and accepted by PEFT:
`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` — 30 layers × 7 modules ×
(A+B) = 420 adapter pairs. `fan_in_fan_out=False` (correct: these are `nn.Linear`, stored `[out, in]`,
matching the inspected safetensors shapes, e.g. `q_proj [576,576]`, `k_proj [192,576]`, `gate_proj [1536,576]`).
`bias="none"` is consistent with the checkpoint having no attention/MLP biases
(`attention_bias: false` in `config.json`). Optionally use `target_modules="all-linear"` (PEFT docs: "all
linear/Conv1D modules are chosen (if the model is a PreTrainedModel, the output layer excluded)") — same
result here since there is no head module to exclude. **No deviation.**

## 8. The exact compatible construction / save–reload contract

Verified end-to-end, bit-exact reload (`maxdiff = 0.00e+00` on hidden states; token delta and head weights
restored exactly):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

BASE = "models/base/haidass1-143m-path"          # local frozen copy
SPECIAL = ["<|kev_state|>", "<|kev_question|>", "<|kev_option|>",
           "<|kev_option_end|>", "<|kev_decide|>"]

# 1. Tokenizer first: know the new ids before resizing
tok = AutoTokenizer.from_pretrained(BASE)
n_added = tok.add_special_tokens({"additional_special_tokens": SPECIAL})   # 5
special_ids = tok.convert_tokens_to_ids(SPECIAL)                           # [64000..64004]

# 2. Load with EXPLICIT fp32 (transformers 5.x defaults to checkpoint dtype = bf16)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)

# 3. Resize BEFORE peft, on the causal-LM wrapper (resizes + re-ties embed & lm_head)
model.resize_token_embeddings(len(tok))          # 64000 -> 64005, stays tied, mean_resizing=True default

# 4. Strip the head by taking the backbone; attach the PointerHead to it
backbone = model.model                            # Qwen3Model, no lm_head attr
backbone.pointer_head = PointerHead(576, 256)     # your own module, fp32

# 5. PEFT wrap the BACKBONE (not the causal-LM wrapper)
peft_model = get_peft_model(backbone, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    trainable_token_indices=special_ids,
    modules_to_save=["pointer_head"],
))

# 6. Train under autocast (master weights stay fp32)
with torch.autocast("cuda", dtype=torch.bfloat16):
    ...  # forward/backward; optimizer sees fp32 params

# 7. Save: adapter-only checkpoint (adapter_config.json + adapter_model.safetensors)
peft_model.save_pretrained(out_dir)

# 8. Reload — replay steps 1-4, then:
pm = PeftModel.from_pretrained(backbone_rebuilt, out_dir, is_trainable=True)
```

### Save/reload contract (verified)

- `save_pretrained` writes `adapter_config.json` (with `trainable_token_indices` and
  `modules_to_save=["pointer_head"]` preserved) plus a small `adapter_model.safetensors` — base weights are
  never duplicated. `task_type` is `None` in the adapter config (backbone-only wrap); harmless.
- **Reload replay contract:** the base side must be rebuilt **identically** — same tokenizer with the same
  5 added tokens, `resize_token_embeddings(len(tok))`, same fp32 load dtype, same `PointerHead` class
  (importable at reload time), then `PeftModel.from_pretrained(backbone, adapter_dir, is_trainable=True)`.
- **`is_trainable=True` is required to resume training.** Verified: default reload yields **0 trainable
  params** (inference-only); `is_trainable=True` restores the full 555,840 trainable params (LoRA + token
  delta + head).
- Round-trip fidelity: hidden states match to `maxdiff = 0.00e+00`; `trainable_tokens_delta` and head weights
  restored bit-exactly. **Optimizer state is NOT saved by `save_pretrained`** — for mid-training resume, use
  Trainer/accelerate checkpointing on top; do not treat the adapter dir as a full training checkpoint.

### Uncertainties / not covered here

- The `pointer_head` reload path requires `modules_to_save` and an importable class; if the head class moves,
  PEFT falls back to constructing a fresh (random) head on `PeftModel.from_pretrained` — the trained head
  weights live only in the adapter checkpoint. [INFERENCE from `modules_to_save` semantics,
  `peft/tuners/lora/config.py:488-489`]
- CUDA-autocast numerics were verified for dtype bookkeeping only; no CUDA GPU was available for this probe
  (CPU-only verification of master-weight fp32 preservation + the bf16-base crash). CUDA parity is a
  one-line check in the first real training run.
- `model_max_length: 131072` in `tokenizer_config.json` vs `max_position_embeddings: 4096` — truncation in
  the data pipeline must use the model's 4096.

## 9. Required deviations from the current plan

1. **Add `protobuf` (and optionally `tiktoken`) to `pyproject.toml`** — the tokenizer cannot load without it;
   neither is currently in `uv.lock`.
2. **Pass `dtype=torch.float32` explicitly at load** — transformers 5.x defaults to the checkpoint's bf16;
   the plan's FP32-master design silently degrades to bf16 otherwise, and a fp32 head attached to the bf16
   backbone crashes outright.
3. **Wrap the backbone (`model.model`), not the causal-LM wrapper, with `get_peft_model`** — matches the plan
   ("strip lm_head"), but note the head-strip must happen *before* PEFT wrap and the backbone's
   `tie_word_embeddings` flag must stay `true` for PEFT's tied-token propagation.
4. **Reload with `is_trainable=True`** — the default reload is inference-only (0 trainable params).
5. No deviation needed for `trainable_token_indices`, LoRA target modules, tied-embedding resize, or the
   mean-resizing default — all work as planned.

## 10. Primary-source index

| Claim | Source |
|---|---|
| Architecture, vocab, tied embeddings, bf16, 4096 ctx | local `config.json`; [model card](https://huggingface.co/DALabCommunity/Haidass1.5-147M) |
| No `lm_head` tensor; stock Qwen3 module names | local `model.safetensors` header (332 tensors inspected) |
| Tokenizer class/legacy, 131072 max length | local `tokenizer_config.json` |
| Tokenizer needs protobuf/tiktoken fallback chain | `transformers/tokenization_utils_tokenizers.py:227-288`; reproduced error |
| Resize API + `mean_resizing=True` default | `transformers/modeling_utils.py:2629-2665` |
| Resize re-ties head; resizes output embeddings on causal-LM wrapper | `transformers/modeling_utils.py:2688-2717` |
| Default load dtype = checkpoint dtype | verified (`from_pretrained` no-dtype → all-bf16 params) |
| `trainable_token_indices` support + FSDP caveat | `peft/tuners/lora/config.py:781-794` |
| `ensure_weight_tying` default False; tied propagation | `peft/tuners/lora/config.py:989-1001`; `peft/utils/other.py:1567-1670` |
| LoRA B zero-init (no-op before training) | `peft/tuners/lora/config.py:490-493` |
| `modules_to_save` semantics | `peft/tuners/lora/config.py:488-489` |
| Reload needs `is_trainable=True` (0 trainable otherwise) | verified (5.04M → 0 vs 555,840 trainable params) |
| Autocast preserves fp32 masters | [PyTorch autocast docs](https://pytorch.org/docs/stable/amp.html#autocasting); verified param dtypes after autocast fwd/bwd |
| FP32-master plan section | repo `docs/chatgpt/Haidass-Kev-Train-Analysis.md:1921-1936` |
| Construction-order plan section | repo `docs/chatgpt/Haidass-Kev-Train-Analysis.md:1833-1860, 2110-2144` |
