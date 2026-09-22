# Domain Docs

How engineering skills consume this repository's domain documentation.

## Before exploring, read these

- `CONTEXT.md` at the repository root, when present.
- `docs/adr/`: ADRs relevant to the area being changed.

If these files do not exist, proceed silently. The domain-modeling skill creates them lazily when terminology or decisions are resolved.

## File structure

This is a single-context repository:

```text
/
├── CONTEXT.md
├── docs/adr/
└── src/
```

## Use the glossary's vocabulary

Use domain concepts as defined in `CONTEXT.md`. Avoid synonyms the glossary explicitly rejects. Missing vocabulary should be reconsidered or recorded for domain modeling.

## Flag ADR conflicts

If work contradicts an existing ADR, surface the conflict explicitly rather than silently overriding it.
