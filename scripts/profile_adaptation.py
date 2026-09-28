"""Measure real packed forward/backward/optimizer updates on the available GPU."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from haidass_kev_train.data.packing import collate
from haidass_kev_train.evaluation.metrics import per_question_ce
from haidass_kev_train.model.decision import build_model
from haidass_kev_train.training.sft import prepare, configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/reports/adaptation-profile.json")
    args = parser.parse_args()
    configure_runtime(42)
    model, tokenizer = build_model("models/base/haidass1.5-143m")
    model.cuda().train()
    records = prepare("data/raw/kev-suites/v6/decision-v6", "train", tokenizer, 2048)
    ordered = sorted(records, key=lambda record: len(record.input_ids))
    representatives = {"median": ordered[len(ordered) // 2], "maximum": ordered[-1]}
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    rows = []
    for label, record in representatives.items():
        for size in (1, 2, 4):
            batch = collate([record] * size, pad_token_id=tokenizer.pad_token_id or 0).to("cuda")
            elapsed = []
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                for iteration in range(4):
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(batch)
                    values, valid = per_question_ce(logits, batch)
                    loss = values[valid].mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite profiling loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    if iteration:
                        elapsed.append(time.perf_counter() - start)
                seconds = statistics.median(elapsed)
                row = {"length_class": label, "batch_size": size, "packed_length": len(record.input_ids),
                       "seconds": seconds, "tokens_per_second": size * len(record.input_ids) / seconds,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "status": "ok"}
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                row = {"length_class": label, "batch_size": size,
                       "packed_length": len(record.input_ids), "status": "out_of_memory"}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del batch
            torch.cuda.empty_cache()
    result = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "records": len(records), "measurements": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
