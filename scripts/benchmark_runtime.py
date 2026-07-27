#!/usr/bin/env python3
"""GPU/CPU runtime benchmarking and memory footprint analysis."""

from __future__ import annotations

import argparse
import json
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
    file_size_mb,
    measure_gpu_memory,
    resolve_model_context,
)


def plot_gpu_latency(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["batch_size"], df["latency_ms_mean"], "o-", label="Mean latency", color="#4C72B0")
    ax.fill_between(
        df["batch_size"],
        df["latency_ms_mean"] - df["latency_ms_std"],
        df["latency_ms_mean"] + df["latency_ms_std"],
        alpha=0.2,
        color="#4C72B0",
    )
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("GPU Inference Latency vs Batch Size")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cpu_latency(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["batch_size"].astype(str), df["latency_ms_mean"], color="#55A868")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("CPU Inference Latency")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_memory(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"checkpoint": "#4C72B0", "inference_gpu": "#55A868", "training_gpu": "#C44E52"}
    for _, row in df.iterrows():
        ax.bar(row["category"], row["peak_memory_mb"], color=colors.get(row["category"], "#8172B3"))
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Footprint")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark GPU/CPU runtime and memory")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--gpu_batches", type=int, nargs="+", default=[1, 8, 16, 32])
    p.add_argument("--cpu_batches", type=int, nargs="+", default=[1, 8])
    args = p.parse_args()

    dirs = ensure_subdirs()
    bench_dir = dirs["benchmarks"]
    ctx = resolve_model_context(checkpoint=args.checkpoint)

    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    # GPU benchmark
    gpu_rows = []
    if torch.cuda.is_available():
        gpu_model = build_eval_model(ctx, gpu_device)
        for bs in args.gpu_batches:
            print(f"GPU benchmark batch={bs} ...")
            stats = benchmark_inference(
                gpu_model, gpu_device, bs, ctx.image_size,
                warmup=args.warmup, runs=args.runs,
            )
            stats["device"] = "cuda"
            gpu_rows.append(stats)
        gpu_df = pd.DataFrame(gpu_rows)
        gpu_df.to_csv(bench_dir / "gpu_runtime_benchmark.csv", index=False)
        plot_gpu_latency(gpu_df, bench_dir / "gpu_latency_vs_batchsize.png")
    else:
        print("CUDA not available; skipping GPU benchmark.")

    # CPU benchmark
    cpu_model = build_eval_model(ctx, cpu_device)
    cpu_rows = []
    for bs in args.cpu_batches:
        print(f"CPU benchmark batch={bs} ...")
        stats = benchmark_inference(
            cpu_model, cpu_device, bs, ctx.image_size,
            warmup=min(args.warmup, 20), runs=min(args.runs, 200),
        )
        stats["device"] = "cpu"
        cpu_rows.append(stats)
    cpu_df = pd.DataFrame(cpu_rows)
    cpu_df.to_csv(bench_dir / "cpu_runtime_benchmark.csv", index=False)
    plot_cpu_latency(cpu_df, bench_dir / "cpu_latency_chart.png")

    params = count_parameters(cpu_model)
    table_row = {
        "Model": f"ResNet50 ({params['total_parameters']:,} params)",
        "Parameters": params["total_parameters"],
        "CPU Latency (ms)": cpu_df.loc[cpu_df["batch_size"] == 1, "latency_ms_mean"].iloc[0],
        "CPU Throughput": cpu_df.loc[cpu_df["batch_size"] == 1, "throughput_samples_per_sec"].iloc[0],
        "Peak Memory (MB)": cpu_df.loc[cpu_df["batch_size"] == 1, "peak_memory_mb"].iloc[0],
    }
    table_df = pd.DataFrame([table_row])
    table_df.to_csv(bench_dir / "table_cpu_benchmark.csv", index=False)
    table_df.to_csv(dirs["tables"] / "table_cpu_benchmark.csv", index=False)

    # Memory footprint
    mem_rows = [
        {"category": "checkpoint", "peak_memory_mb": file_size_mb(ctx.checkpoint), "notes": "PyTorch .pt file"},
    ]
    if torch.cuda.is_available():
        gpu_model = build_eval_model(ctx, gpu_device)
        inf_mem = measure_gpu_memory(gpu_model, gpu_device, 1, ctx.image_size, mode="inference")
        train_mem = measure_gpu_memory(gpu_model, gpu_device, 8, ctx.image_size, mode="training")
        mem_rows.append({"category": "inference_gpu", **inf_mem, "notes": "batch=1 FP32"})
        mem_rows.append({"category": "training_gpu", **train_mem, "notes": "batch=8 FP32 with backward"})
    onnx_path = dirs["onnx"] / "best_model.onnx"
    if onnx_path.is_file():
        mem_rows.append({"category": "onnx", "peak_memory_mb": file_size_mb(str(onnx_path)), "notes": "ONNX export"})

    mem_df = pd.DataFrame(mem_rows)
    mem_df.to_csv(bench_dir / "memory_footprint.csv", index=False)
    plot_memory(mem_df, bench_dir / "memory_usage_chart.png")

    meta = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warmup": args.warmup,
        "runs": args.runs,
    }
    (bench_dir / "benchmark_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Benchmarks written to {bench_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
