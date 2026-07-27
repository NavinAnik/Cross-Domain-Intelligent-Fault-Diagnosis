"""
Physics-aware Grad-CAM evaluation for CWRU spectrogram fault diagnosis.

Computes localization fidelity, CAM energy at BPFI/BPFO/BSF, and fault-frequency overlay figures.

Run from ``src/``:
  python -m experiments.evaluate_gradcam_physics \\
    --dataset_path ../data/cwru_spectrograms_full_png \\
    --checkpoint results/cwru_reviewer_response/<run>/kfold/grouped/fold_0/best.pt \\
    --mat_dir ../data/cwru_mat
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.export_cwru_spectrogram_images import pick_vibration_series
from modules.cwru_bearing_physics import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_N_MELS,
    DEFAULT_SR,
    class_to_fault_lines,
    energy_in_freq_bands,
    fault_frequencies,
    freq_to_col,
    localization_fidelity,
    save_fault_frequencies_csv,
)
from modules.cwru_experiment_utils import load_manifest, prepare_cwru_image_dataset, set_global_seeds
from modules.cwru_metadata import LOAD_TO_RPM, stem_from_png_basename
from modules.spectrogram_cnn_models import build_model
from modules.spectrogram_fault_data import get_fault_transforms
from modules.visualization import compute_grad_cam_cnn
from utils.manuscript_io import PLOT_MODEL_LABEL, save_figure

logger = logging.getLogger(__name__)

WIN_RE = re.compile(r"_w(\d+)\.", re.I)
FAULT_COLORS = {"BPFI": "#e41a1c", "BPFO": "#377eb8", "BSF": "#4daf4a", "BSF2": "#4daf4a", "FTF": "#984ea3"}


def find_default_checkpoint(search_root: str = "results/cwru_reviewer_response") -> Optional[str]:
    pattern = os.path.join(search_root, "*", "kfold", "grouped", "fold_0", "best.pt")
    cands = sorted(glob.glob(pattern), reverse=True)
    for ck in cands:
        if os.path.isfile(ck):
            return ck
    return None


def _window_index_from_basename(basename: str) -> int:
    m = WIN_RE.search(basename)
    return int(m.group(1)) if m else 0


def _load_mat_segment(
    mat_dir: str,
    stem: str,
    win_idx: int,
    *,
    channel: str = "de",
    sample_rate: int = DEFAULT_SR,
    window: int = 2048,
    hop: int = 2048,
) -> np.ndarray:
    mat_path = Path(mat_dir) / f"{stem}.mat"
    if not mat_path.is_file():
        mat_path = Path(mat_dir) / f"X{stem}.mat"
    if not mat_path.is_file():
        raise FileNotFoundError(f"MAT file not found for stem {stem}: {mat_dir}")
    mat = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    sig = pick_vibration_series(mat, channel=channel)
    start = win_idx * hop
    if start + window > sig.size:
        start = max(0, sig.size - window)
    return sig[start : start + window]


def _denorm_image(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().cpu().float()
    if img.dim() == 4:
        img = img[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = np.clip(img.numpy(), 0, 1)
    return np.transpose(img, (1, 2, 0))


def _load_png_gray(path: str, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _resize_cam(cam: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    from PIL import Image as PILImage

    cam = np.asarray(cam, dtype=np.float32)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    pil = PILImage.fromarray((cam * 255).astype(np.uint8))
    pil = pil.resize((shape[1], shape[0]), PILImage.BILINEAR)
    out = np.asarray(pil, dtype=np.float32) / 255.0
    return out


def _draw_fault_vlines(ax, rpm: float, width_px: int, *, sr: int = DEFAULT_SR, n_mels: int = DEFAULT_N_MELS) -> None:
    freqs = fault_frequencies(rpm)
    for name, color in FAULT_COLORS.items():
        if name not in freqs:
            continue
        x = freq_to_col(freqs[name], width_px, sr=sr, n_mels=n_mels)
        ax.axvline(x, color=color, linestyle="--", linewidth=1.2, alpha=0.85, label=name)


def plot_frequency_markers(
    segment: np.ndarray,
    rpm: float,
    output_path: str,
    *,
    sr: int = DEFAULT_SR,
    title: str = "",
) -> None:
    n = len(segment)
    fft_mag = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, fft_mag, color="0.2", linewidth=0.8)
    ff = fault_frequencies(rpm)
    for name, color in FAULT_COLORS.items():
        if name not in ff:
            continue
        ax.axvline(ff[name], color=color, linestyle="--", linewidth=1.5, label=f"{name} ({ff[name]:.1f} Hz)")
    ax.set_xlim(0, min(500, sr / 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|FFT|")
    ax.set_title(title or f"Raw spectrum with fault markers ({PLOT_MODEL_LABEL})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_spectrogram_fault_overlay(
    spec_gray: np.ndarray,
    rpm: float,
    output_path: str,
    *,
    title: str = "",
) -> None:
    h, w = spec_gray.shape
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(spec_gray, aspect="auto", cmap="magma", origin="upper")
    _draw_fault_vlines(ax, rpm, w)
    ax.set_xlabel("Mel frequency (→ Hz)")
    ax.set_ylabel("Time")
    ax.set_title(title or f"Spectrogram + fault frequency overlay ({PLOT_MODEL_LABEL})")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_gradcam_fault_overlay(
    spec_gray: np.ndarray,
    cam: np.ndarray,
    rpm: float,
    output_path: str,
    *,
    title: str = "",
) -> None:
    h, w = spec_gray.shape
    cam_r = _resize_cam(cam, (h, w))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(spec_gray, aspect="auto", cmap="gray", origin="upper", alpha=0.55)
    ax.imshow(cam_r, aspect="auto", cmap="jet", origin="upper", alpha=0.45)
    _draw_fault_vlines(ax, rpm, w)
    ax.set_xlabel("Mel frequency (→ Hz)")
    ax.set_ylabel("Time")
    ax.set_title(title or f"Grad-CAM + fault frequency overlay ({PLOT_MODEL_LABEL})")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_path)


def _sample_indices_by_fault(
    manifest: Any,
    samples: List[str],
    per_class: int,
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    stem_rows = manifest.copy()
    stem_rows["stem"] = stem_rows["stem"].astype(str)
    stem_rows["basename"] = stem_rows["filepath"].apply(os.path.basename)
    basename_to_idx = {os.path.basename(p): i for i, p in enumerate(samples)}

    buckets: Dict[str, List[int]] = defaultdict(list)
    for _, row in stem_rows.iterrows():
        b = row["basename"]
        if b not in basename_to_idx:
            continue
        ft = str(row.get("fault_type", "unknown"))
        if ft == "normal":
            key = "normal"
        elif ft in ("IR", "OR", "B"):
            key = ft
        else:
            key = "abnormal"
        buckets[key].append(basename_to_idx[b])

    out: Dict[str, List[int]] = {}
    for key, idxs in buckets.items():
        idxs = list(dict.fromkeys(idxs))
        rng.shuffle(idxs)
        out[key] = idxs[:per_class]
    return out


def _plot_localization_barplot(per_class: Dict[str, List[float]], output_path: str) -> None:
    classes = [c for c in ("IR", "OR", "B", "normal") if c in per_class and per_class[c]]
    means = [float(np.mean(per_class[c])) for c in classes]
    stds = [float(np.std(per_class[c])) for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(classes))
    ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Localization Fidelity")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Grad-CAM Localization Fidelity ({PLOT_MODEL_LABEL})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path)


def _plot_physics_energy_barplot(
    energy_by_class: Dict[str, Dict[str, List[float]]],
    output_path: str,
) -> None:
    bands = ["BPFI", "BPFO", "BSF"]
    classes = [c for c in ("IR", "OR", "B") if c in energy_by_class]
    if not classes:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.25
    x = np.arange(len(classes))
    for bi, band in enumerate(bands):
        means = [float(np.mean(energy_by_class[c].get(band, [0]))) for c in classes]
        ax.bar(x + (bi - 1) * w, means, width=w, label=f"Energy around {band}")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("CAM energy fraction")
    ax.set_title(f"Grad-CAM energy at fault frequencies ({PLOT_MODEL_LABEL})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_path)


def run_gradcam_physics(
    dataset_path: str,
    output_dir: str,
    *,
    checkpoint: Optional[str] = None,
    mat_dir: str = "../data/cwru_mat",
    per_class_samples: int = 8,
    seed: int = 42,
    model_name: str = "resnet50",
) -> Dict[str, Any]:
    set_global_seeds(seed)
    os.makedirs(output_dir, exist_ok=True)

    ck = checkpoint or find_default_checkpoint()
    if not ck or not os.path.isfile(ck):
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint or run K-fold CV first "
            "(kfold/grouped/fold_0/best.pt)."
        )
    logger.info("Using checkpoint: %s", ck)

    save_fault_frequencies_csv(os.path.join(output_dir, "fault_frequencies.csv"))

    manifest = load_manifest(dataset_path)
    train_ds, _, data_root = prepare_cwru_image_dataset(dataset_path, augment="off")
    tfm = get_fault_transforms(image_size=DEFAULT_IMAGE_SIZE, augment="off")["val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = max(train_ds.labels) + 1
    model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    try:
        blob = torch.load(ck, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ck, map_location=device)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.eval()

    manifest_idx = manifest.copy()
    manifest_idx["stem"] = manifest_idx["stem"].astype(str)
    manifest_idx["basename"] = manifest_idx["filepath"].apply(os.path.basename)
    meta_by_base = manifest_idx.set_index("basename").to_dict("index")

    sampled = _sample_indices_by_fault(manifest, train_ds.samples, per_class_samples, seed)

    loc_rows: List[Dict[str, Any]] = []
    physics_rows: List[Dict[str, Any]] = []
    loc_by_class: Dict[str, List[float]] = defaultdict(list)
    energy_by_class: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    rep_saved = False

    for fault_class, idx_list in sampled.items():
        for ds_idx in idx_list:
            path = train_ds.samples[ds_idx]
            basename = os.path.basename(path)
            meta = meta_by_base.get(basename, {})
            stem = str(meta.get("stem", stem_from_png_basename(basename) or ""))
            fault_type = str(meta.get("fault_type", "unknown"))
            load_hp = int(meta.get("load_hp", 0))
            rpm = float(LOAD_TO_RPM.get(load_hp, 1797))
            win_idx = _window_index_from_basename(basename)

            img = Image.open(path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            label = train_ds.labels[ds_idx]
            with torch.enable_grad():
                cam = compute_grad_cam_cnn(model, x, target_class=label, backbone_attr="backbone")
            cam2d = _resize_cam(cam, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
            spec_gray = _load_png_gray(path)

            fid = localization_fidelity(cam2d, rpm, fault_type)
            loc_rows.append(
                {
                    "filepath": basename,
                    "fault_type": fault_type,
                    "load_hp": load_hp,
                    "localization_fidelity": round(fid, 4),
                }
            )
            loc_by_class[fault_class].append(fid)

            energies = energy_in_freq_bands(cam2d, rpm, bands=("BPFI", "BPFO", "BSF"))
            physics_rows.append(
                {
                    "filepath": basename,
                    "fault_type": fault_type,
                    "load_hp": load_hp,
                    **{f"energy_{k}": round(v, 4) for k, v in energies.items()},
                }
            )
            if fault_class in ("IR", "OR", "B"):
                for k, v in energies.items():
                    energy_by_class[fault_class][k].append(v)

            if fault_class == "IR" and not rep_saved:
                try:
                    seg = _load_mat_segment(
                        mat_dir,
                        stem,
                        win_idx,
                        window=int(meta.get("window", 2048)),
                        hop=int(meta.get("hop", 2048)),
                        sample_rate=int(meta.get("sample_rate", DEFAULT_SR)),
                    )
                    plot_frequency_markers(
                        seg,
                        rpm,
                        os.path.join(output_dir, "frequency_markers"),
                        title=f"Spectrum ({fault_type}, {load_hp} HP) — {PLOT_MODEL_LABEL}",
                    )
                except Exception as e:
                    logger.warning("frequency_markers skipped for %s: %s", basename, e)

                plot_spectrogram_fault_overlay(
                    spec_gray,
                    rpm,
                    os.path.join(output_dir, "spectrogram_fault_overlay"),
                    title=f"Spectrogram overlay ({fault_type}, {load_hp} HP) — {PLOT_MODEL_LABEL}",
                )
                plot_gradcam_fault_overlay(
                    spec_gray,
                    cam2d,
                    rpm,
                    os.path.join(output_dir, "gradcam_fault_overlay"),
                    title=f"Grad-CAM overlay ({fault_type}, {load_hp} HP) — {PLOT_MODEL_LABEL}",
                )
                rep_saved = True

    # Save localization metrics
    loc_csv = os.path.join(output_dir, "localization_metrics.csv")
    with open(loc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["filepath", "fault_type", "load_hp", "localization_fidelity"],
        )
        w.writeheader()
        w.writerows(loc_rows)

    all_fids = [r["localization_fidelity"] for r in loc_rows]
    loc_summary = {
        "model_plot_label": PLOT_MODEL_LABEL,
        "checkpoint": ck,
        "n_samples": len(loc_rows),
        "mean": float(np.mean(all_fids)) if all_fids else 0.0,
        "std": float(np.std(all_fids)) if all_fids else 0.0,
        "per_class": {
            c: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for c, v in loc_by_class.items()
            if v
        },
    }
    with open(os.path.join(output_dir, "localization_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(loc_summary, f, indent=2)

    _plot_localization_barplot(dict(loc_by_class), os.path.join(output_dir, "localization_fidelity_barplot"))

    phys_csv = os.path.join(output_dir, "gradcam_physics_metrics.csv")
    pfields = ["filepath", "fault_type", "load_hp", "energy_BPFI", "energy_BPFO", "energy_BSF"]
    with open(phys_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        w.writerows(physics_rows)

    _plot_physics_energy_barplot(dict(energy_by_class), os.path.join(output_dir, "gradcam_physics_barplot"))

    return {
        "localization_summary": loc_summary,
        "n_physics_rows": len(physics_rows),
        "output_dir": output_dir,
        "checkpoint": ck,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grad-CAM physics + localization evaluation (CWRU)")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="results/gradcam_physics")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--mat_dir", type=str, default="../data/cwru_mat")
    p.add_argument("--per_class_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default="resnet50")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    run_gradcam_physics(
        args.dataset_path,
        args.output_dir,
        checkpoint=args.checkpoint,
        mat_dir=args.mat_dir,
        per_class_samples=args.per_class_samples,
        seed=args.seed,
        model_name=args.model,
    )
    logger.info("Grad-CAM physics evaluation complete: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
