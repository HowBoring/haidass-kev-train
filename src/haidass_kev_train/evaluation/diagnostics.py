"""Fixed, group-aware diagnostics for the training loop."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

from haidass_kev_train.data.packing import encode_record
from haidass_kev_train.evaluation.metrics import predict, summarize
from haidass_kev_train.evaluation.run import reorder_record
__all__ = ["select_probe", "training_diagnostics"]


def _record_identity(record: dict) -> tuple[str, str]:
    meta = record.get("_meta") or {}
    record_id = meta.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("diagnostic records require a non-empty _meta.id")
    group_id = meta.get("group_id", record_id)
    if not isinstance(group_id, str) or not group_id:
        raise ValueError(f"diagnostic record {record_id!r} requires a non-empty _meta.group_id")
    return record_id, group_id


def select_probe(records: list[dict], count: int, seed: int) -> list[dict]:
    """Select ``count`` stable groups and return every sibling in those groups."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("probe group count must be a non-negative integer")
    groups: dict[str, list[tuple[str, dict]]] = {}
    for record_id, record in _records_by_id(records).items():
        group_id = record["_meta"].get("group_id", record_id)
        groups.setdefault(group_id, []).append((record_id, record))
    ranked = sorted(groups, key=lambda group: (hashlib.sha256(f"{seed}\0{group}".encode()).digest(), group))[:count]
    return [record for group in ranked for _, record in sorted(groups[group], key=lambda item: item[0])]


def _records_by_id(records: list[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        record_id, _ = _record_identity(record)
        if record_id in indexed:
            raise ValueError(f"duplicate diagnostic record identity {record_id!r}")
        indexed[record_id] = record
    return indexed


def _predicted_key(row: dict) -> str:
    keys = row["option_keys"]
    if len(keys) != len(row["probs"]) or len(keys) != len(set(keys)):
        raise ValueError(f"invalid option keys for {row['record_id']}/{row['question_name']}")
    return keys[max(range(len(keys)), key=row["probs"].__getitem__)]


def _correct(row: dict) -> bool:
    return _predicted_key(row) == row["option_keys"][max(range(len(row["target"])), key=row["target"].__getitem__)]


def _paired_flip(rows: list[dict], records: dict[str, dict]) -> dict:
    candidates: dict[tuple[str, str], dict[str, dict]] = {}
    declared: dict[tuple[str, str], str] = {}
    for row in rows:
        meta = records[row["record_id"]].get("_meta") or {}
        pair_id = meta.get("pair_id")
        if pair_id is None:
            continue
        sibling = meta.get("sibling")
        kind = meta.get("pair_kind")
        if not isinstance(pair_id, str) or not pair_id or sibling not in ("a", "b"):
            raise ValueError(f"invalid official pair metadata on record {row['record_id']!r}")
        if kind is not None and kind not in ("relevant", "irrelevant"):
            raise ValueError(f"invalid official pair metadata on record {row['record_id']!r}")
        key = (pair_id, row["question_name"])
        pair = candidates.setdefault(key, {})
        if sibling in pair:
            raise ValueError(f"duplicate diagnostic pair sibling {pair_id!r}/{row['question_name']}/{sibling}")
        pair[sibling] = row
        if kind is not None:
            if key in declared and declared[key] != kind:
                raise ValueError(f"conflicting diagnostic pair kind {pair_id!r}/{row['question_name']}")
            declared[key] = kind

    pairs = {"relevant": {}, "irrelevant": {}}
    untyped = 0
    for key, pair in candidates.items():
        kind = declared.get(key)
        if kind is None:
            if set(pair) != {"a", "b"}:
                untyped += len(pair)
                continue
            kind = (
                "relevant"
                if pair["a"]["option_keys"][max(range(len(pair["a"]["target"])), key=pair["a"]["target"].__getitem__)]
                != pair["b"]["option_keys"][max(range(len(pair["b"]["target"])), key=pair["b"]["target"].__getitem__)]
                else "irrelevant"
            )
        pairs[kind][key] = pair

    def complete(kind: str) -> tuple[list[dict], int]:
        complete_pairs = [pair for pair in pairs[kind].values() if set(pair) == {"a", "b"}]
        return complete_pairs, len(pairs[kind]) - len(complete_pairs)

    relevant, relevant_incomplete = complete("relevant")
    invariant, invariant_incomplete = complete("irrelevant")
    return {
        "relevant": {
            "n": len(relevant),
            "incomplete": relevant_incomplete,
            "flip_rate": (
                sum(_predicted_key(pair["a"]) != _predicted_key(pair["b"]) for pair in relevant) / len(relevant)
                if relevant else None
            ),
            "both_correct_rate": (
                sum(_correct(pair["a"]) and _correct(pair["b"]) for pair in relevant) / len(relevant)
                if relevant else None
            ),
        },
        "invariant": {
            "n": len(invariant),
            "incomplete": invariant_incomplete,
            "invariance_rate": (
                sum(_predicted_key(pair["a"]) == _predicted_key(pair["b"]) for pair in invariant) / len(invariant)
                if invariant else None
            ),
            "both_correct_rate": (
                sum(_correct(pair["a"]) and _correct(pair["b"]) for pair in invariant) / len(invariant)
                if invariant else None
            ),
        },
        "untyped_records": untyped,
    }


def _choice_reorder(original: list[dict], reordered: list[dict]) -> dict:
    index: dict[tuple[str, str], dict] = {}
    for row in reordered:
        key = (row["record_id"], row["question_name"])
        if key in index:
            raise ValueError(f"duplicate reordered prediction identity {key[0]}/{key[1]}")
        index[key] = row
    deltas: list[float] = []
    flips = 0
    for row in original:
        if row["question_type"] != "choice":
            continue
        identity = (row["record_id"], row["question_name"])
        other = index.get(identity)
        if other is None:
            raise ValueError(f"reordered prediction missing {identity[0]}/{identity[1]}")
        keys = row["option_keys"]
        if len(keys) != len(set(keys)) or set(keys) != set(other["option_keys"]):
            raise ValueError(f"reordered option keys differ for {identity[0]}/{identity[1]}")
        before = dict(zip(keys, row["probs"], strict=True))
        after = dict(zip(other["option_keys"], other["probs"], strict=True))
        before_target = dict(zip(keys, row["target"], strict=True))
        after_target = dict(zip(other["option_keys"], other["target"], strict=True))
        if max(abs(before_target[key] - after_target[key]) for key in keys) > 1e-6:
            raise ValueError(f"reordered target does not follow option keys: {identity[0]}/{identity[1]}")
        deltas.append(max(abs(before[key] - after[key]) for key in keys))
        flips += max(keys, key=before.__getitem__) != max(keys, key=after.__getitem__)
    return {
        "n": len(deltas),
        "flip_rate": flips / len(deltas) if deltas else None,
        "mean_max_delta": sum(deltas) / len(deltas) if deltas else None,
    }


def _canonical_report(rows: list[dict], records: list[dict], model, tokenizer, batch_size: int,
                      device: str, max_packed: int, seed: int) -> dict:
    """Fixed five-K metrics and key-aligned, same-membership permutation flips."""
    cases: dict[str, dict[int, dict]] = {}
    for row in rows:
        cid, k = row["canonical_id"], row["k"]
        if not isinstance(cid, str) or not cid or type(k) is not int or k not in range(2, 7):
            raise ValueError("canonical diagnostics require canonical_id and K=2..6")
        bucket = cases.setdefault(cid, {})
        if k in bucket:
            raise ValueError(f"duplicate fixed view for {cid}/k{k}")
        bucket[k] = row
    if any(set(views) != set(range(2, 7)) for views in cases.values()):
        raise ValueError("canonical diagnostics require all five fixed K views per canonical")
    if any(len({(r["src"], r["group_id"]) for r in views.values()}) != 1
           for views in cases.values()):
        raise ValueError("fixed views of one canonical must share source and group")
    if any(row["question_type"] != "choice" or len(row["option_keys"]) != row["k"] for row in rows):
        raise ValueError("canonical diagnostics require one choice question with K options per view")

    def counts(group: list[dict]) -> dict:
        return {"views": len(group), "canonicals": len({r["canonical_id"] for r in group}),
                "groups": len({r["group_id"] for r in group})}

    def metrics(group: list[dict]) -> dict:
        return {**summarize(group), **counts(group)}

    by_source = {src: {**metrics(group), "by_k": {str(k): metrics([r for r in group if r["k"] == k])
                                                  for k in range(2, 7)}}
                 for src, group in ((src, [r for r in rows if r["src"] == src])
                                    for src in sorted({r["src"] for r in rows}))}
    by_k = {str(k): metrics([r for r in rows if r["k"] == k]) for k in range(2, 7)}

    ranked = sorted(cases, key=lambda cid: (hashlib.sha256(f"{seed}\0permutation\0{cid}".encode()).digest(), cid))
    chosen = set(ranked[:200])
    originals = {record["_meta"]["id"]: record for record in records}
    alternates: list[dict] = []
    for cid in ranked[:200]:
        for k in range(2, 7):
            row = cases[cid][k]
            record = originals[row["record_id"]]
            keys = list(record["questions"]["decision"]["criteria"])
            for offset in range(1, 2 if k == 2 else 3):
                shifted = keys[offset:] + keys[:offset]
                alternate = {**record, "questions": {**record["questions"],
                             "decision": {**record["questions"]["decision"],
                                          "criteria": {key: record["questions"]["decision"]["criteria"][key]
                                                       for key in shifted}}}}
                alternates.append(alternate)
    alternate_rows = predict(model, [encode_record(record, tokenizer, max_packed=max_packed) for record in alternates],
                             batch_size=batch_size, device=device)
    alternate_index: dict[tuple[str, int], list[dict]] = {}
    for row in alternate_rows:
        alternate_index.setdefault((row["canonical_id"], row["k"]), []).append(row)

    def flip_stats(keys: list[tuple[str, int]]) -> dict:
        flipped = sum(len({_predicted_key(cases[cid][k]), *(_predicted_key(row) for row in alternate_index[cid, k])}) > 1
                      for cid, k in keys)
        return {"flips": flipped, "count": len(keys), "rate": flipped / len(keys) if keys else None}

    for cid in chosen:
        for k in range(2, 7):
            original = cases[cid][k]
            reordered = alternate_index[cid, k]
            if len(reordered) != (1 if k == 2 else 2):
                raise ValueError(f"incomplete permutation diagnostic for {cid}/k{k}")
            for other in reordered:
                _choice_reorder([original], [other])

    selected_rows = [row for row in rows if row["canonical_id"] in chosen]
    permutation = {**counts(selected_rows),
                   "source_counts": {src: counts([row for row in selected_rows if row["src"] == src])
                                     for src in sorted({row["src"] for row in selected_rows})},
                   "by_k": {str(k): {**flip_stats([(cid, k) for cid in chosen]), "orders": 2 if k == 2 else 3}
                            for k in range(2, 7)},
                   "by_source": {src: {str(k): flip_stats([(cid, k) for cid in chosen
                                                          if cases[cid][k]["src"] == src])
                                       for k in range(2, 7)}
                                 for src in sorted({cases[cid][2]["src"] for cid in chosen})}}
    return {**metrics(rows), "chance_accuracy": 0.29,
            "evaluation": {"temperature": 1.0, "model_precision": "model_forward",
                           "log_probability_dtype": "float32", "aggregation_dtype": "float64",
                           "nll": "natural_log", "reduction": "equal_case_equal_k"},
            "by_source": by_source, "by_k": by_k, "permutation": permutation}


def training_diagnostics(
    model,
    tokenizer,
    records: list[dict],
    *,
    batch_size: int = 4,
    device: str = "cuda",
    max_packed: int = 2048,
    seed: int = 0,
) -> dict:
    """Return raw training-time diagnostics without consuming training RNG state."""
    indexed = _records_by_id(records)
    was_training = model.training
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        encoded = [encode_record(record, tokenizer, max_packed=max_packed) for record in records]
        rows = predict(model, encoded, batch_size=batch_size, device=device)
        reordered = predict(
            model,
            [encode_record(reorder_record(record), tokenizer, max_packed=max_packed) for record in records],
            batch_size=batch_size,
            device=device,
        )
        clean = [row for row in rows if row["variant"] == "clean"]
        tasks: dict[str, list[dict]] = {}
        for row in clean:
            tasks.setdefault(row["src"], []).append(row)
        report = {
            "all": summarize(rows),
            "clean": summarize(clean),
            "tasks": {task: summarize(task_rows) for task, task_rows in tasks.items()},
            "paired_flip": _paired_flip(clean, indexed),
            "choice_reorder": _choice_reorder(rows, reordered),
        }
        if any((record.get("_meta") or {}).get("canonical_id") is not None for record in records):
            if len(clean) != len(records) or any((record.get("_meta") or {}).get("canonical_id") is None for record in records):
                raise ValueError("canonical diagnostics require only fixed clean views")
            report["canonical"] = _canonical_report(clean, records, model, tokenizer, batch_size, device, max_packed, seed)
        return report
    finally:
        model.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
