# Kev option/question layout: what actually performed best

Research note on `jaredpalmer/kev` (research question), 2026-09-22. Question: which layout —
question-level packing with position resets, sequential options within a question, or
option-level `option_isolation` — produced the best reported numbers, and what should this
project (Haidass Qwen3 attention-only backbone) adopt?

## The three layouts in current `kev/model.py`

1. **Question-level branches in one packed sequence** (default, attention-only models).
   State + every question in one token row; additive block-causal mask lets a question see
   the state and its own branch but never another question; each question's position IDs
   restart just after the state.
   `encode()` + `branch_mask_batch()` — https://github.com/jaredpalmer/kev/blob/main/kev/model.py
   (encode docstring: "Pack one record: [<state> ...] then per-question [...]", and
   `branch_mask`: "attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i])").
   Documented in README "How It Works":
   https://github.com/jaredpalmer/kev/blob/main/README.md#how-it-works

2. **Sequential options within a question** (part of layout 1's default): options are laid
   out consecutively inside the branch with strictly increasing positions
   (`br_pos = list(range(p0, p0 + len(br)))` in `encode`). Options *within* a question can
   affect one another; README: "This does not mean option order is irrelevant."

3. **Option-level `option_isolation`** (flag, default off): every option span becomes its
   own sub-branch that sees only state + instruction + itself; all option spans share the
   same position IDs; `<decide>` sits at one fixed position after the longest span.
   `encode(..., option_isolation=True)` and the `opts=` path of `branch_mask_batch`, same
   file. Gives exact permutation invariance by construction (measured 1.2e-7 on the real
   model, PLAN.md).

A fourth form exists only as serving/training machinery, not a different layout: on hybrid
Qwen3.5 bases (Gated DeltaNet layers ignore attention masks), each question runs as its own
causal row `[state + branch]` continuing from a cached state pass (`rows_of`,
`forward_rows_batch`). On attention-only models it is **bit-identical** to the packed mask
form (max |Δp| = 0.000000, 0 flips, 24 records, fp32 — PLAN.md §10 Phase 1;
`tests/test_model.py::test_rows_match_packed`). Model card confirmation:
https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md ("isolation is exact
by construction (together vs alone within 1e-5) and on attention-only models this form is
bit-identical to the packed one").

## Which layout the best checkpoints use

Every released checkpoint — best or otherwise — uses **question-level isolation with
sequential options**. `option_isolation` defaults to `False` in `DecisionModel.__init__`
and is never enabled in any released recipe. Best reported results:

| Checkpoint | Base | Locked-test acc (dev/test) | Layout |
|---|---|---|---|
| Kev-9B | Qwen3.5-9B (hybrid) | 0.822 / **0.852** (Brier 0.237) | question rows from shared state; sequential options |
| Kev-4B | Qwen3.5-4B (hybrid) | 0.797 / 0.837 (Brier 0.255) | same |
| Kev-0.8B | Qwen3.5-0.8B (hybrid) | 0.652 / 0.684 (Brier 0.460) | same |
| Kev-4B (Qwen3) | Qwen3-4B (attention-only) | 0.790 / 0.806 | packed block-causal branches; sequential options |
| Kev-8B (Qwen3) | Qwen3-8B (attention-only) | 0.796 / 0.780 | packed block-causal branches; sequential options |

Source: README models table — https://github.com/jaredpalmer/kev/blob/main/README.md#models

## Ablation outcomes: option isolation never wins

From PLAN.md, overnight autoresearch round
(https://github.com/jaredpalmer/kev/blob/main/PLAN.md#overnight-autoresearch-branch-researchovernight-1-pr-3):

- **0.6B**: "Option isolation trains normally at 0.6B (dev 0.800) with a measured flip rate
  of exactly 0.0; transfer 0.58–0.60 (parity with the plain encoding; six salvaged
  arch-screen trials all within noise of the incumbent)." → no accuracy cost, no gain.
- **0.6B synthesis**: option isolation listed under "What did not work": "exact permutation
  invariance at no accuracy cost, but no accuracy gain."
- **4B**: "option isolation at low lr 0.729 (−5.8 pp, significant) — **isolation costs
  accuracy at 4B**." (Round `auto-4b-r1` vs the 0.755–0.767 incumbent range.)
- Combining it with low lr / `public_frac` / `synthetic_repeat` "does not stack" (0.748–0.752).

**Why the Qwen3.5 hybrids cannot use it:** `DecisionModel.__init__` raises
`ValueError("option_isolation needs the packed mask; not available on hybrid backbones")` —
the recurrent DeltaNet layers ignore attention masks, so the packed-mask trick behind
option isolation is unimplementable there. The released 0.8B/4B/9B family therefore
questions-as-rows instead, which is question-level (not option-level) isolation by
construction.

## Decisive answer

The best-performing Kev checkpoints (Kev-9B 0.852, Kev-4B 0.837/0.806, Kev-8B 0.780) all use
**question-level isolation with sequential options**, via packed block-causal branches with
per-question position resets (Qwen3) or equivalent per-question rows from a shared state
(Qwen3.5). Option-level `option_isolation` is a measured negative at 4B (−5.8 pp), noise at
0.6B, unsupported on hybrids, and enabled in zero released checkpoints. Question isolation
is the property that matters (verified to 4e-6 together-vs-separate); option order
sensitivity within a question was left alone and even exploited by the
`--perm_kl`-free permutation augmentation in the released recipes.

## Recommendation for the Haidass Qwen3 project

Keep the plan in `docs/agents/training.md` as is: block-causal additive `[B, 1, L, L]` mask
with per-question position resets after the state, sequential options inside each branch,
pointer-head scoring of `</opt>` against `<decide>`. This is exactly the layout of the best
attention-only released checkpoints (Kev-4B/8B Qwen3), and `kev`'s row form is bit-identical
to it — so there is no evidence for and measured evidence against adding `option_isolation`
to this project. Do not implement option-level shared-position sub-branches.
