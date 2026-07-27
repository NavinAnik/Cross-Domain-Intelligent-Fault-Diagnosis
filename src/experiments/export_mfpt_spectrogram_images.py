"""
Export MFPT MATLAB vibration files to spectrogram PNGs for ImageDataset (normal/abnormal).

Run from ``src/``:
  python -m experiments.export_mfpt_spectrogram_images --mat_dir ../data/mfpt --out_dir ../data/mfpt_spectrograms

Baseline files ``baseline_*.mat`` are labeled **normal** (0); InnerRaceFault / OuterRaceFault
files are **abnormal** (1) for binary evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.export_cwru_spectrogram_images import log_mel_spectrogram, mel_to_rgb_png
from modules.mfpt_metadata import meta_for_stem

logger = logging.getLogger(__name__)


def pick_mfpt_vibration_series(mat: Dict[str, Any]) -> Tuple[np.ndarray, int, Optional[float], Optional[float]]:
    """
    Extract 1D vibration trace and sample rate from an MFPT ``bearing`` struct or flat keys.
    """
    bearing = mat.get("bearing")
    if bearing is not None:
        try:
            gs = np.asarray(bearing.gs, dtype=np.float64).squeeze()
            sr = int(np.asarray(bearing.sr).squeeze())
            load = float(np.asarray(getattr(bearing, "load", np.nan)).squeeze()) if hasattr(bearing, "load") else None
            rate = float(np.asarray(getattr(bearing, "rate", np.nan)).squeeze()) if hasattr(bearing, "rate") else None
            if gs.ndim != 1:
                gs = gs.reshape(-1)
            if gs.size >= 256:
                return gs, sr, load, rate
        except Exception as e:
            logger.debug("bearing struct parse failed: %s", e)

    best_key: Optional[str] = None
    best_arr: Optional[np.ndarray] = None
    best_len = -1
    for k, v in mat.items():
        if str(k).startswith("__"):
            continue
        if not isinstance(v, np.ndarray):
            continue
        arr = np.asarray(v, dtype=np.float64).squeeze()
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.size < 256:
            continue
        if arr.size > best_len:
            best_len = arr.size
            best_key = k
            best_arr = arr

    if best_arr is None:
        keys = [k for k in mat.keys() if not str(k).startswith("__")]
        raise KeyError(f"No 1D vibration array found in MAT file. Keys (sample): {keys[:20]}")

    sr = 48828
    for k in ("sr", "SR", "sample_rate", "fs"):
        if k in mat:
            try:
                sr = int(np.asarray(mat[k]).squeeze())
                break
            except (TypeError, ValueError):
                pass
    logger.info("Using MAT key %r (n=%d, sr=%d)", best_key, best_arr.size, sr)
    return best_arr, sr, None, None


def export_mat_file(
    mat_path: Path,
    out_root: Path,
    *,
    window: int,
    hop: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    image_size: int,
    sample_rate_override: Optional[int] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    sig, sr_file, load_mat, rate_mat = pick_mfpt_vibration_series(mat)
    n = sig.size
    if n < window:
        logger.warning("Skip %s: signal length %d < window %d", mat_path.name, n, window)
        return 0, []

    stem = re.sub(r"[^0-9a-zA-Z._-]+", "_", mat_path.stem)
    meta = meta_for_stem(stem)
    sr = int(sample_rate_override or (meta["sample_rate"] if meta else sr_file))
    if meta is not None:
        label = int(meta["label"])
        sub = "normal" if label == 0 else "abnormal"
        fault_type = meta["fault_type"]
        load_lbs = meta.get("load_lbs")
        shaft_hz = meta.get("shaft_hz", 25.0)
    else:
        sl = stem.lower()
        label = 0 if "baseline" in sl else 1
        sub = "normal" if label == 0 else "abnormal"
        fault_type = "normal" if label == 0 else "unknown"
        load_lbs = load_mat
        shaft_hz = rate_mat if rate_mat is not None else 25.0

    out_dir = out_root / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    written = 0
    start = 0
    win_idx = 0
    while start + window <= n:
        seg = sig[start : start + window]
        lm = log_mel_spectrogram(seg, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        rgb = mel_to_rgb_png(lm, image_size=image_size)
        fname = f"{stem}_w{win_idx:06d}.png"
        out_path = out_dir / fname
        rgb.save(out_path)
        rel = os.path.join(sub, fname).replace("\\", "/")
        rows.append(
            {
                "filepath": rel,
                "stem": stem,
                "label": label,
                "load_lbs": load_lbs if load_lbs is not None else "",
                "shaft_hz": shaft_hz if shaft_hz is not None else "",
                "fault_type": fault_type,
                "sample_rate": sr,
                "window": window,
                "hop": hop,
            }
        )
        written += 1
        win_idx += 1
        start += hop
    return written, rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export MFPT .mat to spectrogram PNGs (normal/abnormal folders).",
        epilog=(
            "Typical use (from src/):\n"
            "  python -m experiments.export_mfpt_spectrogram_images "
            "--mat_dir ../data/mfpt --out_dir ../data/mfpt_spectrograms\n"
            "Then run domain adaptation or k-fold CV on ../data/mfpt_spectrograms."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mat_dir", type=str, required=True, help="Directory containing downloaded .mat files")
    p.add_argument("--out_dir", type=str, required=True, help="Output root with normal/ and abnormal/ subfolders")
    p.add_argument(
        "--sample_rate",
        type=int,
        default=None,
        help="Override sample rate for all files (default: per-file from metadata or MAT struct)",
    )
    p.add_argument("--window", type=int, default=2048, help="Samples per window")
    p.add_argument("--hop", type=int, default=2048, help="Window hop in samples (default: non-overlapping)")
    p.add_argument("--n_fft", type=int, default=512, help="FFT size for mel spectrogram")
    p.add_argument("--hop_length", type=int, default=128, help="STFT hop for mel spectrogram")
    p.add_argument("--n_mels", type=int, default=128, help="Number of mel bands")
    p.add_argument("--image_size", type=int, default=224, help="Saved PNG square side (matches training default)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)

    mat_dir = Path(os.path.expandvars(os.path.expanduser(args.mat_dir))).resolve()
    out_root = Path(os.path.expandvars(os.path.expanduser(args.out_dir))).resolve()
    if not mat_dir.is_dir():
        raise FileNotFoundError(f"mat_dir not found: {mat_dir}")

    mats = sorted(mat_dir.glob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files under {mat_dir}")

    meta: Dict[str, Any] = {
        "mat_dir": str(mat_dir),
        "out_dir": str(out_root),
        "sample_rate_override": args.sample_rate,
        "window": args.window,
        "hop": args.hop,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "image_size": args.image_size,
        "files": [p.name for p in mats],
    }

    total = 0
    manifest_rows: List[Dict[str, Any]] = []
    for mp in mats:
        n, rows = export_mat_file(
            mp,
            out_root,
            window=int(args.window),
            hop=int(args.hop),
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            n_mels=int(args.n_mels),
            image_size=int(args.image_size),
            sample_rate_override=args.sample_rate,
        )
        manifest_rows.extend(rows)
        logger.info("%s -> %d windows", mp.name, n)
        total += n

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"
    if manifest_rows:
        fieldnames = list(manifest_rows[0].keys())
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(manifest_rows)
        meta["manifest_csv"] = str(manifest_path)
        meta["n_manifest_rows"] = len(manifest_rows)
    with open(out_root / "export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(
        "Wrote %d PNG windows under %s (manifest.csv, export_meta.json)",
        total,
        out_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
