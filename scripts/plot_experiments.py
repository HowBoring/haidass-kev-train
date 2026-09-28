"""Plot existing experiment logs without training or accessing any dataset split.

Run without changing project dependencies:
uv run --no-project --with matplotlib python scripts/plot_experiments.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports/experiment-figures"))
    args = parser.parse_args()
    root = args.root
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "artifacts/comparisons/sft-exploration/summary.json").read_text())
    logs, paths = {}, {}
    for arm in summary["first_wave_ranked"] + summary["predeclared_repeats"] + summary["matched_short_budget_runs"]:
        paths[arm["name"]] = Path(arm["best_checkpoint"]).parent / "metrics.jsonl"
    paths["onecycle-2190-seed42"] = Path("artifacts/checkpoints/decision-v7-aug-onecycle/metrics.jsonl")
    for name in ("b-ce", "c-proper", "d-rlcd"):
        paths[name] = Path("artifacts/comparisons/stage2") / name / "metrics.jsonl"
    for name, path in paths.items():
        logs[name] = [json.loads(line) for line in (root / path).read_text().splitlines()]
        assert logs[name][-1]["event"] == "finished", f"Incomplete run: {name}"
    extracted = {}

    def series(name, event, *keys):
        xs, ys = [], []
        for row in logs[name]:
            if row["event"] != event:
                continue
            value = row
            for key in keys:
                value = value[key]
            if value is not None:
                xs.append(row["step"])
                ys.append(float(value))
        assert xs and all(b > a for a, b in zip(xs, xs[1:])), (name, event, keys)
        extracted[(name, event, ".".join(keys))] = list(zip(xs, ys))
        return xs, ys

    def moving_average(values, window):
        total, result = 0.0, []
        for index, value in enumerate(values):
            total += value
            if index >= window:
                total -= values[index - window]
            result.append(total / min(index + 1, window))
        return result

    def curve(ax, name, event, keys, color, label=None, smooth=0, style="-", selected=False):
        xs, ys = series(name, event, *keys)
        if smooth:
            ax.plot(xs, ys, color=color, alpha=0.055, lw=0.5, rasterized=True)
            ax.plot(xs, moving_average(ys, smooth), color=color, lw=1.9, ls=style, label=label)
        else:
            ax.plot(xs, ys, color=color, lw=1.8, ls=style, marker="o", ms=3, label=label)
        if selected:
            best = min((row for row in logs[name] if row["event"] == event), key=lambda row: row["selection"]["value"])
            index = xs.index(best["step"])
            ax.scatter([xs[index]], [ys[index]], marker="*", s=145, color=color, edgecolor="white", linewidth=0.8, zorder=5)

    def setup(ax, title, ylabel, percent=False):
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.17)
        ax.spines[["top", "right"]].set_visible(False)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    outputs = []
    def save(fig, name, title, note, handles, labels, columns=3):
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=columns, frameon=False, fontsize=10)
        fig.text(0.5, 0.013, note, ha="center", va="bottom", fontsize=9, color="#444444")
        fig.tight_layout(rect=(0.01, 0.14, 0.99, 0.94), h_pad=2.3, w_pad=2.6)
        for extension in ("png", "svg", "pdf"):
            path = out / f"{name}.{extension}"
            fig.savefig(path, dpi=180, facecolor="white")
            outputs.append(str(path.relative_to(root)))
        plt.close(fig)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10})
    recipe = [
        ("decision-v7-aug-cosine-seed42-2190", "LoRA cosine | 2190 | LR 2e-4"),
        ("onecycle-2190-seed42", "LoRA OneCycle | 2190 | LR 2e-4"),
        ("lora-4380-lr2e4-seed42", "LoRA cosine | 4380 | LR 2e-4"),
        ("lora-4380-lr1e4-seed42", "LoRA cosine | 4380 | LR 1e-4"),
        ("full-4380-lr5e5-seed42", "Full cosine | 4380 | LR 5e-5"),
        ("full-4380-lr2e5-seed42", "Full cosine | 4380 | LR 2e-5"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for (name, label), color in zip(recipe, COLORS):
        curve(axes[0, 0], name, "train", ("loss",), color, label, smooth=75)
        curve(axes[0, 1], name, "development", ("report", "clean", "accuracy"), color, selected=True)
        curve(axes[1, 0], name, "train_probe", ("report", "clean", "accuracy"), color)
        curve(axes[1, 1], name, "development", ("selection", "value"), color, selected=True)
    setup(axes[0, 0], "A  Online augmented training loss", "Cross-entropy (75-update trailing mean)")
    setup(axes[0, 1], "B  Development accuracy", "Clean accuracy (BF16)", True)
    setup(axes[1, 0], "C  Fixed training-probe accuracy", "Clean probe accuracy (BF16)", True)
    setup(axes[1, 1], "D  Development selection loss", "Clean macro-NLL (BF16)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    save(fig, "sft_recipes", "SFT recipe comparison | seed 42", "Faint traces: raw training loss. Stars: minimum development macro-NLL checkpoints. Probe is a small fixed training subset, not full-train accuracy.", handles, labels)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for column, seed in enumerate((42, 43, 44)):
        short = "decision-v7-aug-cosine-seed42-2190" if seed == 42 else f"baseline-2190-lr2e4-seed{seed}"
        long = f"lora-4380-lr2e4-seed{seed}"
        for name, label, color, style in [(short, "2190-update budget", COLORS[0], "--"), (long, "4380-update budget", COLORS[1], "-")]:
            curve(axes[0, column], name, "train", ("loss",), color, label, smooth=75, style=style)
            curve(axes[1, column], name, "development", ("report", "clean", "accuracy"), color, style=style, selected=True)
        setup(axes[0, column], f"Seed {seed} | Training CE", "75-update trailing mean")
        setup(axes[1, column], f"Seed {seed} | Development accuracy", "Clean accuracy (BF16)", True)
    for row in axes:
        low = min(ax.get_ylim()[0] for ax in row)
        high = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(low, high)
            ax.set_xlim(0, 4450)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    save(fig, "sft_matched_seeds", "SFT matched-seed comparison | LoRA cosine, LR 2e-4", "Same seed and data recipe; budgets change the cosine trajectory. Stars select minimum dev macro-NLL, not maximum accuracy. No curves extrapolated.", handles, labels, columns=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for name, label, color in [("b-ce", "B: CE + replay", COLORS[0]), ("c-proper", "C: CE + proper + replay", COLORS[1]), ("d-rlcd", "D: CE + RLCD + replay", COLORS[2])]:
        curve(axes[0, 0], name, "train", ("ce",), color, label, smooth=15)
        curve(axes[0, 1], name, "typed_development", ("report", "all", "accuracy"), color, selected=True)
        curve(axes[1, 0], name, "train", ("replay_ce",), color, smooth=15)
        curve(axes[1, 1], name, "stage1_retention", ("report", "clean", "accuracy"), color)
    setup(axes[0, 0], "A  Typed-decisions training CE", "Soft-target CE (15-update trailing mean)")
    setup(axes[0, 1], "B  Typed-decisions development accuracy", "Teacher-argmax agreement (BF16)", True)
    setup(axes[1, 0], "C  Stage 1 replay training CE", "Replay CE, unweighted (15-update trailing mean)")
    setup(axes[1, 1], "D  Replay-source development retention", "Clean accuracy (BF16; filtered sources)", True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    save(fig, "stage2_objectives", "Stage 2 comparison | shared SFT parent, 300 updates", "Comparable CE components shown, not the mixed CE/proper/PG scalar. Stars: minimum typed-dev NLL. No train accuracy is logged in Stage 2.", handles, labels)

    with (out / "curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "event", "metric", "step", "raw_value"])
        for (name, event, metric), values in extracted.items():
            writer.writerows((name, event, metric, step, value) for step, value in values)
    manifest = {"figures": outputs, "series": len(extracted), "source_logs": {name: {"path": str(path), "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()} for name, path in paths.items()}, "smoothing": {"sft": "trailing mean, 75 updates, partial windows at start", "stage2": "trailing mean, 15 updates, partial windows at start"}, "accuracy": "evaluation-time BF16, uncalibrated; fixed train probe only for SFT; no fabricated training accuracy", "selection_stars": "development NLL minimum, not accuracy maximum", "stage2_loss": "separate unweighted current CE and replay CE, not incomparable mixed objective scalars"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": outputs, "series": len(extracted), "csv": str(out / "curves.csv")}, indent=2))


if __name__ == "__main__":
    main()
