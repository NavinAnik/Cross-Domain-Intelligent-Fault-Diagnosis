#!/usr/bin/env python3
"""INT8 post-training quantization via ONNX Runtime static PTQ."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from modules.efficiency_utils import (
    ensure_subdirs,
    evaluate_model_metrics,
    file_size_mb,
    make_split_loaders,
    resolve_model_context,
    build_eval_model,
)
from sklearn.metrics import f1_score


class SpectrogramCalibrationReader(CalibrationDataReader):
    def __init__(self, loader, max_samples: int = 200):
        self._batches = []
        count = 0
        for x, _ in loader:
            self._batches.append({"input": x.numpy()})
            count += x.size(0)
            if count >= max_samples:
                break
        self._iter = iter(self._batches)

    def get_next(self):
        return next(self._iter, None)


def run_onnx_inference(onnx_path: Path, loader, num_classes: int) -> dict:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    all_y, all_pred, all_prob = [], [], []
    for x, y in loader:
        out = sess.run(None, {"input": x.numpy()})[0]
        prob = _softmax(out)
        pred = np.argmax(prob, axis=1)
        all_y.extend(y.numpy().tolist())
        all_pred.extend(pred.tolist())
        all_prob.append(prob)
    probs = np.vstack(all_prob)
    y_t = np.array(all_y)
    y_p = np.array(all_pred)
    return {
        "accuracy": float(np.mean(y_t == y_p)),
        "f1": float(f1_score(y_t, y_p, zero_division=0)),
        "y_true": y_t,
        "y_pred": y_p,
        "probs": probs,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def benchmark_onnx(onnx_path: Path, batch_size: int = 1, image_size: int = 224, runs: int = 200) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)
    for _ in range(20):
        sess.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def plot_comparison(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    cols = ["model_size_mb", "latency_ms", "accuracy"]
    titles = ["Model Size (MB)", "CPU Latency (ms)", "Accuracy"]
    for ax, col, title in zip(axes, cols, titles):
        ax.bar(df["variant"], df[col], color=["#4C72B0", "#55A868", "#C44E52"][: len(df)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="INT8 PTQ evaluation")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--calibration_samples", type=int, default=200)
    args = p.parse_args()

    dirs = ensure_subdirs()
    qdir = dirs["quantization"]
    onnx_dir = dirs["onnx"]
    ctx = resolve_model_context(checkpoint=args.checkpoint)

    fp32_onnx = onnx_dir / "best_model.onnx"
    if not fp32_onnx.is_file():
        raise FileNotFoundError(f"Run export_onnx.py first: {fp32_onnx}")

    _, val_loader, _ = make_split_loaders(ctx, batch_size=8, device=None)
    cal_loader = make_split_loaders(ctx, batch_size=8, device=None)[0]

    int8_path = qdir / "best_model_int8.onnx"
    quantize_static(
        model_input=str(fp32_onnx),
        model_output=str(int8_path),
        calibration_data_reader=SpectrogramCalibrationReader(cal_loader, args.calibration_samples),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
    )

    rows = []
    for variant, path in [("Original (ONNX FP32)", fp32_onnx), ("INT8 (PTQ)", int8_path)]:
        metrics = run_onnx_inference(path, val_loader, ctx.num_classes)
        lat = benchmark_onnx(path, 1, ctx.image_size)
        rows.append({
            "variant": variant,
            "model_size_mb": file_size_mb(str(path)),
            "latency_ms": lat,
            "throughput": 1.0 / (lat / 1000.0),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(qdir / "int8_results.csv", index=False)
    plot_comparison(df, qdir / "int8_comparison.png")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
