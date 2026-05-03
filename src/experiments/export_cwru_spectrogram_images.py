"""
Export CWRU MATLAB vibration files to spectrogram PNGs for ImageDataset (normal/abnormal).

Run from ``src/``:
  python -m experiments.export_cwru_spectrogram_images --mat_dir ../data/cwru_mat --out_dir ../data/cwru_spectrograms_png

Baseline files ``97.mat``–``100.mat`` are labeled **normal** (0); all other ``.mat`` in
``--mat_dir`` are **abnormal** (1) for binary evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.io import loadmat

logger = logging.getLogger(__name__)

BASELINE_STEMS = {"97", "98", "99", "100"}


def _is_baseline_mat(path: Path) -> bool:
    stem = path.stem.lstrip("Xx")
    return stem in BASELINE_STEMS


def pick_vibration_series(mat: Dict[str, Any], channel: str) -> np.ndarray:
    """Pick the primary 1D vibration trace (DE / FE / BA) from a CWRU loadmat dict."""
    ch = channel.lower().strip()
    if ch not in {"de", "fe", "ba"}:
        raise ValueError("channel must be one of: de, fe, ba")
    suffix = f"_{ch.upper()}_time".lower()

    best_key: Optional[str] = None
    best_arr: Optional[np.ndarray] = None
    best_len = -1

    for k, v in mat.items():
        if str(k).startswith("__"):
            continue
        if not isinstance(v, np.ndarray):
            continue
        kl = k.lower()
        if not kl.endswith(suffix):
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
        raise KeyError(
            f"No *{suffix.upper()} key found in MAT file. Available keys (sample): {keys[:20]}"
        )
    logger.info("Using MAT key %r (n=%d)", best_key, best_arr.size)
    return best_arr


def log_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    return log_mel


def mel_to_rgb_png(log_mel: np.ndarray, image_size: int) -> Image.Image:
    """Min–max normalize log-mel, resize to square RGB uint8 (3 identical channels)."""
    x = log_mel.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo + 1e-8:
        g = np.zeros_like(x, dtype=np.uint8)
    else:
        g = ((x - lo) / (hi - lo + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(g, mode="L")
    pil = pil.resize((image_size, image_size), Image.BILINEAR)
    return pil.convert("RGB")


def export_mat_file(
    mat_path: Path,
    out_root: Path,
    *,
    channel: str,
    sample_rate: int,
    window: int,
    hop: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    image_size: int,
    label_is_normal: bool,
) -> int:
    mat = loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    sig = pick_vibration_series(mat, channel=channel)
    n = sig.size
    if n < window:
        logger.warning("Skip %s: signal length %d < window %d", mat_path.name, n, window)
        return 0

    sub = "normal" if label_is_normal else "abnormal"
    out_dir = out_root / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = re.sub(r"[^0-9a-zA-Z._-]+", "_", mat_path.stem)
    written = 0
    start = 0
    win_idx = 0
    while start + window <= n:
        seg = sig[start : start + window]
        lm = log_mel_spectrogram(seg, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        rgb = mel_to_rgb_png(lm, image_size=image_size)
        out_path = out_dir / f"{stem}_w{win_idx:06d}.png"
        rgb.save(out_path)
        written += 1
        win_idx += 1
        start += hop
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export CWRU .mat to spectrogram PNGs (normal/abnormal folders).",
        epilog=(
            "Typical use (from src/):\n"
            "  python -m experiments.export_cwru_spectrogram_images "
            "--mat_dir ../data/cwru_mat --out_dir ../data/cwru_spectrograms_png\n"
            "Then run eval_spectrogram_checkpoint on ../data/cwru_spectrograms_png."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mat_dir", type=str, required=True, help="Directory containing downloaded .mat files")
    p.add_argument("--out_dir", type=str, required=True, help="Output root with normal/ and abnormal/ subfolders")
    p.add_argument("--channel", type=str, default="de", choices=["de", "fe", "ba"], help="Accelerometer channel")
    p.add_argument("--sample_rate", type=int, default=12000, help="Sampling rate in Hz (12000 for 12k CWRU)")
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
        "channel": args.channel,
        "sample_rate": args.sample_rate,
        "window": args.window,
        "hop": args.hop,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "image_size": args.image_size,
        "baseline_stems": sorted(BASELINE_STEMS),
        "files": [p.name for p in mats],
    }

    total = 0
    for mp in mats:
        is_normal = _is_baseline_mat(mp)
        n = export_mat_file(
            mp,
            out_root,
            channel=args.channel,
            sample_rate=int(args.sample_rate),
            window=int(args.window),
            hop=int(args.hop),
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            n_mels=int(args.n_mels),
            image_size=int(args.image_size),
            label_is_normal=is_normal,
        )
        logger.info("%s -> %d windows (%s)", mp.name, n, "normal" if is_normal else "abnormal")
        total += n

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %d PNG windows under %s (metadata: export_meta.json)", total, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
