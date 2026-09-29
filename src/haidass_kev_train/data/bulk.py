"""Concurrent shard-lane bulk runner: chained offset builder batches -> one unreviewed aggregate suite.

Each of ``shard_count`` lanes builds frozen batches sequentially under
``<output>/lane-00/batch-NNNN`` .. ``lane-<N-1>/batch-NNNN``; the runner assigns ``shard_index``
per lane (any base value is overridden) and the next batch's ``source_start_rows`` continue the
previous batch's start + ``scanned_by_source`` within that lane only. Lanes build concurrently,
but every batch keeps the 100/2000/4h trial caps and an incomplete batch ends its lane, frozen
and untouched. Once the distinct train canonicals over all complete batches reach
``--minimum-train``, the batches are merged exactly once via the unreviewed aggregation path;
no audit evidence is fabricated and no review is implied.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
from threading import Lock
import time
import tomllib

from haidass_kev_train.data.aggregate import aggregate
from haidass_kev_train.data.build import build
from haidass_kev_train.data.canonical import load_canonical_suite

# Batch stop reasons that mean the run itself is broken, not that a lane ran out of source.
_FAILURE_REASONS = {"time_limit", "attempt_limit", "service_error", "configuration_error"}
# Config keys this runner assigns per batch/lane; every other base key must be identical in
# every frozen batch so the whole chain is one policy at different offsets.
_RUNNER_KEYS = {"source_start_rows", "shard_index"}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _batch_dir(root, lane, index):
    return root / f"lane-{lane:02d}" / f"batch-{index:04d}"


def _read_batch(path, base, lane, shard_count):
    """Verify one frozen batch against its data, summary, base config and lane identity."""
    manifest, summary = _load(path / "manifest.json"), _load(path / "summary.json")
    config = manifest.get("build", {}).get("config", {})
    for key, value in base.items():
        if key not in _RUNNER_KEYS and config.get(key) != value:
            raise ValueError(f"{path}: build config {key!r} differs from the base configuration")
    if config.get("shard_index", 0) != lane or config.get("shard_count", 1) != shard_count:
        raise ValueError(f"{path}: batch shard_index/shard_count does not match its lane")
    if summary.get("stop_reason") != manifest.get("stop_reason") or summary.get("splits") != manifest.get("counts"):
        raise ValueError(f"{path}: summary disagrees with the frozen manifest")
    train = load_canonical_suite(path, "train")
    load_canonical_suite(path, "development")
    return manifest, summary, {row["_meta"]["id"] for row in train}


class _Lane:
    """One shard's verified batch chain plus the offset where its next batch must start."""

    def __init__(self, root, lane, base, shard_count):
        self.lane, self.batches, self.train_ids = lane, [], set()
        self.policy = self.split_seed = None
        self.active = True
        offsets = {source: 0 for source in base["sources"]}
        offsets.update(base.get("source_start_rows") or {})
        self.offsets = offsets
        directory = root / f"lane-{lane:02d}"
        seen = set()
        index = 0
        while True:
            path = directory / f"batch-{index:04d}"
            if not path.exists():
                break
            seen.add(path)
            manifest, summary, ids = _read_batch(path, base, lane, shard_count)
            if manifest["build"]["config"].get("source_start_rows") != self.offsets:
                raise ValueError(f"{path}: source_start_rows do not continue the lane chain at {self.offsets}")
            self._identity(manifest, path)
            if manifest.get("complete") is not True:
                reason = manifest.get("stop_reason")
                if reason in _FAILURE_REASONS:
                    raise ValueError(f"{path}: lane stopped on a failed batch ({reason}); resolve before resuming")
                self.active = False  # exhausted source: frozen, excluded, and it must be the last
                if (directory / f"batch-{index + 1:04d}").exists():
                    raise ValueError(f"{path}: incomplete batch does not end its lane chain")
                break
            self.batches.append(path)
            self.train_ids |= ids
            scanned = summary.get("scanned_by_source") or {}
            self.offsets = {source: self.offsets[source] + scanned.get(source, 0) for source in self.offsets}
            index += 1
        self.next_index = index
        if directory.exists() and any(
                child.name.startswith("batch-") and child not in seen for child in directory.iterdir()):
            raise ValueError(f"{directory}: non-contiguous batch numbering")

    def _identity(self, manifest, path):
        policy = manifest.get("build", {}).get("policy_sha256")
        seed = manifest.get("build", {}).get("config", {}).get("split_seed")
        if self.policy is None:
            self.policy, self.split_seed = policy, seed
        elif policy != self.policy or seed != self.split_seed:
            raise ValueError(f"{path}: policy or split_seed changed inside a lane chain")

    def submit(self, pool, base, root, shard_count, progress=None):
        # shard_count/shard_index exist only in the sharded builder; a single lane stays
        # config-identical to the unsharded policy (builder defaults are 1/0).
        config = {key: value for key, value in base.items() if key != "shard_index"}
        if shard_count > 1:
            config.update(shard_count=shard_count, shard_index=self.lane)
        config["source_start_rows"] = self.offsets
        path = _batch_dir(root, self.lane, self.next_index)

        def run_batch():
            build(config, path, progress=(lambda state: progress(self.lane, state)) if progress else None)
            return path

        return pool.submit(run_batch)


def run(config, output, minimum_train=30000):
    """Resume or extend the frozen lane chains under ``output`` and aggregate once at minimum."""
    base = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    if not isinstance(base.get("sources"), dict) or not base["sources"]:
        raise ValueError("base configuration must name sources")
    shard_count = base.get("shard_count", 1)
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    if type(minimum_train) is not int or minimum_train < 1:
        raise ValueError("minimum_train must be a positive integer")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    lanes = [_Lane(root, lane, base, shard_count) for lane in range(shard_count)]
    policies = {lane.policy for lane in lanes if lane.policy is not None}
    seeds = {lane.split_seed for lane in lanes if lane.policy is not None}
    if len(policies) > 1 or len(seeds) > 1:
        raise ValueError("lanes disagree on generation policy or split_seed")
    batches = [path for lane in lanes for path in lane.batches]
    train_ids = set().union(*(lane.train_ids for lane in lanes)) if batches else set()
    suite = root / "suite"
    if suite.exists():
        return {"status": "already_aggregated", "suite": str(suite),
                "distinct_train": len(train_ids), "batches": len(batches)}
    snapshots = {}
    lock = Lock()
    def progress(lane, snapshot):
        with lock:
            snapshots[lane] = snapshot

    def heartbeat(pending):
        with lock:
            active = [snapshots[lane.lane] for lane in pending.values() if lane.lane in snapshots]
        sources = Counter(snapshot["source"] for snapshot in active)
        rejected = Counter()
        for snapshot in active:
            rejected.update(snapshot["rejected"])
        print(json.dumps({"event": "progress", "active_lanes": len(pending),
                          "reporting_lanes": len(active), "batches_complete": len(batches),
                          "distinct_train": len(train_ids),
                          "scanned_inflight": sum(snapshot["scanned"] for snapshot in active),
                          "accepted_inflight": sum(snapshot["accepted"] for snapshot in active),
                          "attempts_inflight": sum(snapshot["attempts"] for snapshot in active),
                          "current_sources": dict(sources),
                          "top_rejections": dict(rejected.most_common(8))}), flush=True)
    with ThreadPoolExecutor(max_workers=shard_count) as pool:
        pending = {lane.submit(pool, base, root, shard_count, progress): lane for lane in lanes
                   if lane.active and len(train_ids) < minimum_train}
        last_log = time.monotonic()
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if time.monotonic() - last_log >= 30:
                heartbeat(pending)
                last_log = time.monotonic()
            for future in done:
                lane = pending.pop(future)
                path = future.result()  # a raised build error halts the run; frozen artifacts stay
                manifest, summary, ids = _read_batch(path, base, lane.lane, shard_count)
                with lock:
                    snapshots.pop(lane.lane, None)
                print(json.dumps({"event": "batch_complete", "lane": lane.lane,
                                  "batch": lane.next_index, "accepted": summary["accepted"],
                                  "attempts": summary["attempts"],
                                  "scanned": summary["scanned"],
                                  "elapsed_seconds": summary["elapsed_seconds"],
                                  "stop_reason": summary["stop_reason"]}), flush=True)
                lane._identity(manifest, path)
                if manifest.get("complete") is not True:
                    lane.active = False
                    reason = manifest.get("stop_reason")
                    if reason in _FAILURE_REASONS:
                        raise ValueError(f"{path}: batch failed ({reason}); frozen artifact left in place")
                    continue  # source exhausted: lane ends without fabricating its target
                lane.batches.append(path)
                batches.append(path)
                lane.train_ids |= ids
                train_ids |= ids
                scanned = summary.get("scanned_by_source") or {}
                lane.offsets = {source: lane.offsets[source] + scanned.get(source, 0)
                                for source in lane.offsets}
                lane.next_index += 1
                if len(train_ids) < minimum_train:
                    pending[lane.submit(pool, base, root, shard_count, progress)] = lane
    if len(train_ids) < minimum_train:
        raise ValueError(f"all lanes exhausted at {len(train_ids)} distinct train canonicals, "
                         f"below minimum {minimum_train}; nothing aggregated")
    aggregation = aggregate(batches, None, None, None, None, minimum_train, suite, unreviewed=True)
    return {"status": "aggregated", "suite": str(suite), "distinct_train": len(train_ids),
            "batches": len(batches), "quality_status": aggregation.get("quality_status"),
            "counts": aggregation.get("counts")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="base TOML build configuration (offsets/shard_index assigned per lane)")
    parser.add_argument("--output", required=True, help="lane/batch root; suite is written to <output>/suite")
    parser.add_argument("--minimum-train", type=int, default=30000,
                        help="distinct train canonical IDs required before unreviewed aggregation")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, args.output, args.minimum_train), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
