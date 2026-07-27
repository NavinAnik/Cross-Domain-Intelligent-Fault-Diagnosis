"""
Shared utilities for model-efficiency evaluation (runtime, memory, ONNX, quantization, pruning).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure src/ is on path when imported from scripts/
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.eval_spectrogram_checkpoint import SpectrogramImageFolderDataset
from modules.evaluation import compute_classification_metrics
from modules.spectrogram_cnn_models import build_model
from modules.source_model_locator import find_best_cwru_checkpoint


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def results_efficiency_dir() -> Path:
    d = repo_root() / "results" / "model_efficiency"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_subdirs() -> Dict[str, Path]:
    base = results_efficiency_dir()
    out = {
        "root": base,
        "benchmarks": base / "benchmarks",
        "figures": base / "figures",
        "tables": base / "tables",
        "onnx": base / "onnx",
        "quantization": base / "quantization",
        "pruning": base / "pruning",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class ModelContext:
    checkpoint: str
    fold: int
    run_dir: str
    fold_dir: str
    config: Dict[str, Any]
    split: Dict[str, Any]
    metrics: Dict[str, Any]
    dataset_path: str
    model_name: str
    backbone: str
    num_classes: int
    image_size: int


def default_dataset_path() -> str:
    candidates = [
        repo_root() / "data" / "cwru_spectrograms_full_png",
        repo_root() / "data" / "cwru_spectrograms_png",
    ]
    for p in candidates:
        if p.is_dir():
            return str(p.resolve())
    raise FileNotFoundError(f"CWRU spectrogram dataset not found under {candidates}")


def resolve_model_context(
    *,
    checkpoint: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> ModelContext:
    info = find_best_cwru_checkpoint() if checkpoint is None else None
    ckpt = checkpoint or info["checkpoint"]  # type: ignore[index]
    ckpt = os.path.normpath(os.path.abspath(ckpt))
    fold_dir = os.path.dirname(ckpt)
    run_dir = os.path.dirname(fold_dir)

    cfg_path = os.path.join(fold_dir, "config.json")
    split_path = os.path.join(fold_dir, "split_indices.json")
    folds_csv = os.path.join(run_dir, "folds_grouped.csv")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    fold = int(config.get("fold", info["fold"] if info else 0))  # type: ignore[index]
    metrics: Dict[str, Any] = {}
    if os.path.isfile(folds_csv):
        import pandas as pd

        df = pd.read_csv(folds_csv)
        row = df[df["fold"] == fold]
        if not row.empty:
            metrics = row.iloc[0].to_dict()

    data_root = dataset_path or default_dataset_path()
    # Verify split indices are compatible with dataset ordering
    ds_probe = SpectrogramImageFolderDataset(data_root, image_size=int(config.get("image_size", 224)))
    max_idx = max(max(split["train_idx"]), max(split["val_idx"]))
    if max_idx >= len(ds_probe):
        for alt in [
            repo_root() / "data" / "cwru_spectrograms_full_png",
            repo_root() / "data" / "cwru_spectrograms_png",
        ]:
            alt_s = str(alt.resolve())
            if alt_s == data_root or not alt.is_dir():
                continue
            probe = SpectrogramImageFolderDataset(alt_s, image_size=int(config.get("image_size", 224)))
            if max_idx < len(probe):
                data_root = alt_s
                break
        else:
            raise ValueError(
                f"Split indices require dataset with >{max_idx + 1} samples; "
                f"got {len(ds_probe)} at {data_root}"
            )

    return ModelContext(
        checkpoint=ckpt,
        fold=fold,
        run_dir=run_dir,
        fold_dir=fold_dir,
        config=config,
        split=split,
        metrics=metrics,
        dataset_path=data_root,
        model_name=str(config.get("model", "resnet50")),
        backbone=str(config.get("backbone", "resnet50")),
        num_classes=int(config.get("num_classes", 2)),
        image_size=int(config.get("image_size", 224)),
    )


def build_eval_model(
    ctx: ModelContext,
    device: torch.device,
    *,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    model = build_model(
        ctx.model_name,
        num_classes=ctx.num_classes,
        backbone=ctx.backbone,
        pretrained=False,
    )
    try:
        blob = torch.load(ctx.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ctx.checkpoint, map_location="cpu")
    state = blob.get("model_state_dict", blob)
    model.load_state_dict(state, strict=True)
    model.eval()
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device)


def load_full_dataset(ctx: ModelContext) -> SpectrogramImageFolderDataset:
    return SpectrogramImageFolderDataset(ctx.dataset_path, image_size=ctx.image_size)


def make_split_loaders(
    ctx: ModelContext,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, SpectrogramImageFolderDataset]:
    ds = load_full_dataset(ctx)
    train_idx = ctx.split["train_idx"]
    val_idx = ctx.split["val_idx"]
    pin = device is not None and device.type == "cuda"
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, ds


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
    }


def checkpoint_size_bytes(path: str) -> int:
    return os.path.getsize(path)


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


@torch.no_grad()
def evaluate_model_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    use_amp: bool = False,
) -> Dict[str, Any]:
    model.eval()
    all_y: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        if x.dtype != next(model.parameters()).dtype:
            x = x.to(dtype=next(model.parameters()).dtype)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and use_amp):
            logits = model(x)
        prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        all_y.extend(y.numpy().tolist())
        all_pred.extend(pred.tolist())
        all_prob.append(prob)
    probs = np.vstack(all_prob) if all_prob else np.zeros((0, num_classes))
    y_t = np.array(all_y)
    y_p = np.array(all_pred)
    metrics = compute_classification_metrics(y_t, y_p, probs, n_classes=num_classes)
    return metrics


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int = 224,
    *,
    warmup: int = 100,
    runs: int = 1000,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if dtype is not None:
        x = x.to(dtype=dtype)
        model = model.to(dtype=dtype)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        sync_device(device)

        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            times_ms: List[float] = []
            for _ in range(runs):
                starter.record()
                _ = model(x)
                ender.record()
                sync_device(device)
                times_ms.append(starter.elapsed_time(ender))
        else:
            times_ms = []
            proc = psutil.Process(os.getpid())
            mem_before = proc.memory_info().rss
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = model(x)
                sync_device(device)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            mem_after = proc.memory_info().rss
            peak_mem_mb = max(mem_before, mem_after) / (1024 * 1024)

    arr = np.array(times_ms, dtype=np.float64)
    latency_ms = float(np.mean(arr))
    throughput = batch_size / (latency_ms / 1000.0)
    out = {
        "batch_size": batch_size,
        "latency_ms_mean": latency_ms,
        "latency_ms_std": float(np.std(arr)),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "throughput_samples_per_sec": throughput,
        "warmup_runs": warmup,
        "benchmark_runs": runs,
    }
    if device.type == "cpu":
        out["peak_memory_mb"] = float(peak_mem_mb)  # type: ignore[possibly-undefined]
    return out


def measure_gpu_memory(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int = 224,
    *,
    mode: str = "inference",
) -> Dict[str, float]:
    if device.type != "cuda":
        return {"peak_memory_mb": 0.0, "mode": mode}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if mode == "inference":
        model.eval()
        with torch.no_grad():
            _ = model(x)
    else:
        model.train()
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()

    sync_device(device)
    peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return {"peak_memory_mb": float(peak), "mode": mode, "batch_size": batch_size}


def write_source_model_summary(ctx: ModelContext, out_dir: Path) -> None:
    params = count_parameters(build_eval_model(ctx, torch.device("cpu")))
    ckpt_mb = file_size_mb(ctx.checkpoint)
    arch = f"{ctx.model_name} ({ctx.backbone} backbone, CNNBaseline + linear head)"

    def _fmt(x: Any) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "N/A"
        try:
            v = float(x)
            return f"{v * 100:.2f}%" if v <= 1 else f"{v:.4f}"
        except (TypeError, ValueError):
            return str(x)

    lines = [
        "# Source Model Summary",
        "",
        "Auto-located best CWRU ResNet50 spectrogram classifier checkpoint for efficiency evaluation.",
        "",
        f"- **Architecture:** {arch}",
        f"- **Total parameters:** {params['total_parameters']:,}",
        f"- **Trainable parameters:** {params['trainable_parameters']:,}",
        f"- **Checkpoint path:** `{ctx.checkpoint}`",
        f"- **Checkpoint size:** {ckpt_mb:.2f} MB",
        f"- **Dataset:** CWRU 12 kHz drive-end log-Mel spectrograms (grouped k-fold, fold {ctx.fold})",
        f"- **Dataset path:** `{ctx.dataset_path}`",
        f"- **Accuracy:** {_fmt(ctx.metrics.get('accuracy'))}",
        f"- **F1-score:** {_fmt(ctx.metrics.get('f1'))}",
        f"- **ROC-AUC:** {_fmt(ctx.metrics.get('auc_roc'))}",
        f"- **Metrics source:** `{os.path.join(ctx.run_dir, 'folds_grouped.csv')}`",
        "",
        "## Notes",
        "",
        "Spectrogram images are 224×224 RGB log-Mel representations. The classifier uses a "
        "ResNet50 backbone (timm) with global average pooling and a 2-class linear head.",
        "",
    ]
    summary_path = out_dir / "source_model_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "architecture": arch,
        "checkpoint": ctx.checkpoint,
        "checkpoint_size_mb": ckpt_mb,
        "dataset": "CWRU 12 kHz drive-end log-Mel spectrograms",
        "dataset_path": ctx.dataset_path,
        "fold": ctx.fold,
        "accuracy": ctx.metrics.get("accuracy"),
        "f1": ctx.metrics.get("f1"),
        "auc_roc": ctx.metrics.get("auc_roc"),
        **params,
    }
    (out_dir / "source_model_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
