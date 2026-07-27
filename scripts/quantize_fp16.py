#!/usr/bin/env python3
"""FP16 quantization evaluation: size, latency, accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from modules.efficiency_utils import (
    benchmark_inference,
    build_eval_model,
    count_parameters,
    ensure_subdirs,
    evaluate_model_metrics,
    file_size_mb,
    make_split_loaders,
    resolve_model_context,
)


def plot_comparison(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["model_size_mb", "latency_ms", "accuracy"]
    titles = ["Model Size (MB)", "GPU Latency (ms, batch=1)", "Accuracy"]
    colors = ["#4C72B0", "#55A868"]
    for ax, col, title in zip(axes, metrics, titles):
        ax.bar(df["variant"], df[col], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="FP16 quantization evaluation")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=500)
    args = p.parse_args()

    dirs = ensure_subdirs()
    qdir = dirs["quantization"]
    ctx = resolve_model_context(checkpoint=args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = make_split_loaders(ctx, batch_size=32, device=device)

    rows = []

    # Original FP32
    model_fp32 = build_eval_model(ctx, device)
    metrics_fp32 = evaluate_model_metrics(model_fp32, val_loader, device, ctx.num_classes, use_amp=False)
    bench_fp32 = benchmark_inference(model_fp32, device, 1, ctx.image_size, warmup=args.warmup, runs=args.runs)
    rows.append({
        "variant": "Original (FP32)",
        "dtype": "float32",
        "parameters": count_parameters(model_fp32)["total_parameters"],
        "model_size_mb": file_size_mb(ctx.checkpoint),
        "latency_ms": bench_fp32["latency_ms_mean"],
        "throughput": bench_fp32["throughput_samples_per_sec"],
        "accuracy": metrics_fp32["accuracy"],
        "f1": metrics_fp32["f1"],
        "auc_roc": metrics_fp32.get("auc_roc"),
    })

    # FP16
    model_fp16 = build_eval_model(ctx, device, dtype=torch.float16)
    metrics_fp16 = evaluate_model_metrics(model_fp16, val_loader, device, ctx.num_classes, use_amp=False)
    bench_fp16 = benchmark_inference(
        model_fp16, device, 1, ctx.image_size,
        warmup=args.warmup, runs=args.runs, dtype=torch.float16,
    )
    fp16_ckpt = qdir / "model_fp16.pt"
    torch.save({"model_state_dict": model_fp16.state_dict()}, fp16_ckpt)
    rows.append({
        "variant": "FP16",
        "dtype": "float16",
        "parameters": count_parameters(model_fp16)["total_parameters"],
        "model_size_mb": file_size_mb(str(fp16_ckpt)),
        "latency_ms": bench_fp16["latency_ms_mean"],
        "throughput": bench_fp16["throughput_samples_per_sec"],
        "accuracy": metrics_fp16["accuracy"],
        "f1": metrics_fp16["f1"],
        "auc_roc": metrics_fp16.get("auc_roc"),
    })

    df = pd.DataFrame(rows)
    df.to_csv(qdir / "fp16_results.csv", index=False)
    plot_comparison(df, qdir / "fp16_comparison.png")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
