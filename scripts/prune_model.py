#!/usr/bin/env python3
"""Structured channel pruning study at 10/20/30/50% with recovery fine-tune."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from modules.efficiency_utils import (
    benchmark_inference,
    build_eval_model,
    count_parameters,
    ensure_subdirs,
    evaluate_model_metrics,
    make_split_loaders,
    resolve_model_context,
)


def prune_and_finetune(
    ctx,
    prune_ratio: float,
    device: torch.device,
    *,
    finetune_epochs: int = 3,
    batch_size: int = 32,
) -> dict:
    model = build_eval_model(ctx, device)
    base_params = count_parameters(model)["total_parameters"]

    example_inputs = torch.randn(1, 3, ctx.image_size, ctx.image_size, device=device)
    imp = tp.importance.MagnitudeImportance(p=1)
    ignored = []
    if hasattr(model, "head"):
        ignored.append(model.head)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored,
    )
    pruner.step()

    train_loader, val_loader, _ = make_split_loaders(ctx, batch_size=batch_size, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    model.train()
    for _ in range(finetune_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    pruned_params = count_parameters(model)["total_parameters"]
    metrics = evaluate_model_metrics(model, val_loader, device, ctx.num_classes)
    bench = benchmark_inference(model, device, 1, ctx.image_size, warmup=20, runs=100)

    return {
        "prune_ratio_pct": prune_ratio * 100,
        "parameters": pruned_params,
        "parameter_reduction_pct": (1 - pruned_params / base_params) * 100,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "latency_ms": bench["latency_ms_mean"],
        "throughput": bench["throughput_samples_per_sec"],
    }


def plot_tradeoff(df: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["prune_ratio_pct"], df["accuracy"], "o-", color="#4C72B0", label="Accuracy")
    ax1.set_xlabel("Pruning ratio (%)")
    ax1.set_ylabel("Accuracy", color="#4C72B0")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["prune_ratio_pct"], df["latency_ms"], "s--", color="#C44E52", label="Latency (ms)")
    ax2.set_ylabel("Latency (ms)", color="#C44E52")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    ax1.set_title("Pruning Trade-off: Accuracy vs Latency")
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Structured pruning study")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.50])
    p.add_argument("--finetune_epochs", type=int, default=3)
    args = p.parse_args()

    dirs = ensure_subdirs()
    pdir = dirs["pruning"]
    ctx = resolve_model_context(checkpoint=args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline (no pruning)
    baseline_model = build_eval_model(ctx, device)
    _, val_loader, _ = make_split_loaders(ctx, batch_size=32, device=device)
    base_metrics = evaluate_model_metrics(baseline_model, val_loader, device, ctx.num_classes)
    base_bench = benchmark_inference(baseline_model, device, 1, ctx.image_size, warmup=20, runs=100)
    base_params = count_parameters(baseline_model)["total_parameters"]

    rows = [{
        "prune_ratio_pct": 0.0,
        "parameters": base_params,
        "parameter_reduction_pct": 0.0,
        "accuracy": base_metrics["accuracy"],
        "f1": base_metrics["f1"],
        "latency_ms": base_bench["latency_ms_mean"],
        "throughput": base_bench["throughput_samples_per_sec"],
    }]

    for ratio in args.ratios:
        print(f"Pruning {ratio*100:.0f}% ...")
        row = prune_and_finetune(ctx, ratio, device, finetune_epochs=args.finetune_epochs)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(pdir / "pruning_results.csv", index=False)
    plot_tradeoff(df, pdir / "pruning_tradeoff_curve.png")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
