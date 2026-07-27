#!/usr/bin/env python3
"""
Thin CLI wrapper for MFPT spectrogram export (mirrors generate_cwru_spectrograms naming).

Run from repo root:
  python scripts/generate_mfpt_spectrograms.py
  python scripts/generate_mfpt_spectrograms.py --mat_dir data/mfpt --out_dir data/mfpt_spectrograms
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parent.parent
    src = repo / "src"
    parser = argparse.ArgumentParser(
        description="Generate MFPT log-mel spectrogram PNGs (wrapper around export_mfpt_spectrogram_images).",
        epilog=(
            "Example:\n"
            "  python scripts/download_mfpt.py\n"
            "  python scripts/generate_mfpt_spectrograms.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mat_dir", type=str, default="data/mfpt")
    parser.add_argument("--out_dir", type=str, default="data/mfpt_spectrograms")
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=2048)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--sample_rate", type=int, default=None)
    args, extra = parser.parse_known_args(argv)

    cmd = [
        sys.executable,
        "-m",
        "experiments.export_mfpt_spectrogram_images",
        "--mat_dir",
        str((repo / args.mat_dir).resolve()),
        "--out_dir",
        str((repo / args.out_dir).resolve()),
        "--window",
        str(args.window),
        "--hop",
        str(args.hop),
        "--n_fft",
        str(args.n_fft),
        "--hop_length",
        str(args.hop_length),
        "--n_mels",
        str(args.n_mels),
        "--image_size",
        str(args.image_size),
    ]
    if args.sample_rate is not None:
        cmd.extend(["--sample_rate", str(args.sample_rate)])
    cmd.extend(extra)

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(src))


if __name__ == "__main__":
    raise SystemExit(main())
