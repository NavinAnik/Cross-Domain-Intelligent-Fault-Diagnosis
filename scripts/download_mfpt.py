#!/usr/bin/env python3
"""
Download MFPT bearing fault dataset (.mat files) for spectrogram export.

Primary source: MathWorks RollingElementBearingFaultDiagnosis-Data mirror
(known-condition train/test splits used in MATLAB fault-diagnosis examples).

Data attribution: Machinery Failure Prevention Technology (MFPT) Society
https://www.mfpt.org/fault-data-sets/

Run from repo root:
  python scripts/download_mfpt.py
  python scripts/download_mfpt.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_URL = (
    "https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/"
    "archive/refs/heads/master.zip"
)
ATTRIBUTION = (
    "Data source: MFPT Society Bearing Fault Data Sets\n"
    "https://www.mfpt.org/fault-data-sets/\n"
    "MathWorks mirror: RollingElementBearingFaultDiagnosis-Data\n"
    "Use of this data should follow the data center's terms and cite appropriately."
)

# Curated presets matching MathWorks known-condition splits.
PRESETS: Dict[str, Sequence[str]] = {
    "mathworks_known": [
        "train_data/baseline_1.mat",
        "train_data/baseline_2.mat",
        "train_data/InnerRaceFault_vload_1.mat",
        "train_data/InnerRaceFault_vload_2.mat",
        "train_data/InnerRaceFault_vload_3.mat",
        "train_data/InnerRaceFault_vload_4.mat",
        "train_data/InnerRaceFault_vload_5.mat",
        "train_data/OuterRaceFault_vload_1.mat",
        "train_data/OuterRaceFault_vload_2.mat",
        "train_data/OuterRaceFault_vload_3.mat",
        "train_data/OuterRaceFault_vload_4.mat",
        "train_data/OuterRaceFault_vload_5.mat",
        "train_data/OuterRaceFault_vload_6.mat",
        "train_data/OuterRaceFault_vload_7.mat",
        "test_data/baseline_3.mat",
        "test_data/InnerRaceFault_vload_6.mat",
        "test_data/InnerRaceFault_vload_7.mat",
        "test_data/OuterRaceFault_1.mat",
        "test_data/OuterRaceFault_2.mat",
        "test_data/OuterRaceFault_3.mat",
    ],
    "mathworks_train_only": [
        "train_data/baseline_1.mat",
        "train_data/baseline_2.mat",
        "train_data/InnerRaceFault_vload_1.mat",
        "train_data/OuterRaceFault_vload_1.mat",
    ],
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(
    url: str,
    dest: Path,
    *,
    force: bool,
    timeout_sec: float,
    retries: int,
    backoff_sec: float,
    resume: bool = True,
) -> str:
    """Return ``skipped`` if left unchanged, ``downloaded`` if bytes were written."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"skip (exists): {dest.name}")
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    if force and tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
        existing = 0

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            headers = {"User-Agent": "contrastive_ssl_mfpt_downloader/1.0"}
            if resume and existing > 0:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                mode = "ab" if resume and existing > 0 and resp.status == 206 else "wb"
                if mode == "wb" and existing > 0:
                    existing = 0
                    tmp.unlink(missing_ok=True)
                with open(tmp, mode) as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
            tmp.replace(dest)
            print(f"ok: {dest.name} ({dest.stat().st_size} bytes)")
            return "downloaded"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            wait = backoff_sec * (2 ** (attempt - 1))
            print(f"retry {attempt}/{retries} for {dest.name}: {e} (sleep {wait:.1f}s)", file=sys.stderr)
            time.sleep(wait)
            existing = tmp.stat().st_size if tmp.exists() else 0
    raise RuntimeError(f"failed to download {url}: {last_err}")


def _extract_zip(zip_path: Path, extract_root: Path, *, force: bool) -> Path:
    """Extract zip and return the top-level extracted directory."""
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".extracted"
    if marker.exists() and not force:
        # Find existing extracted tree
        for child in extract_root.iterdir():
            if child.is_dir() and child.name.startswith("RollingElement"):
                return child
        return extract_root

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    marker.write_text(str(zip_path), encoding="utf-8")
    for child in extract_root.iterdir():
        if child.is_dir() and child.name.startswith("RollingElement"):
            return child
    return extract_root


def _flatten_mats(
    src_root: Path,
    out_dir: Path,
    rel_paths: Sequence[str],
    *,
    force: bool,
) -> List[str]:
    """Copy selected .mat files into flat out_dir using stem as filename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for rel in rel_paths:
        src = src_root / rel
        if not src.is_file():
            # try glob by basename under extracted tree
            hits = list(src_root.rglob(Path(rel).name))
            if not hits:
                print(f"WARN: missing {rel}", file=sys.stderr)
                continue
            src = hits[0]
        stem = src.stem
        dest = out_dir / f"{stem}.mat"
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"skip (exists): {dest.name}")
        else:
            shutil.copy2(src, dest)
            print(f"copied: {dest.name}")
        written.append(dest.name)
    return written


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download MFPT .mat files (MathWorks mirror) and organize under data/mfpt/.",
        epilog=(
            "Example pipeline (from repo root):\n"
            "  python scripts/download_mfpt.py --out_dir data/mfpt\n"
            "  cd src\n"
            "  python -m experiments.export_mfpt_spectrogram_images "
            "--mat_dir ../data/mfpt --out_dir ../data/mfpt_spectrograms"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/mfpt",
        help="Directory to write flat .mat files (default: data/mfpt)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/mfpt_cache",
        help="Directory for downloaded zip and extraction (default: data/mfpt_cache)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="ZIP download URL (default: MathWorks GitHub mirror)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="mathworks_known",
        choices=sorted(PRESETS.keys()),
        help="Which curated file bundle to extract (default: mathworks_known)",
    )
    parser.add_argument(
        "--checksum",
        type=str,
        default=None,
        help="Optional SHA-256 hex digest of the downloaded zip (verified after download)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract even if present")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds")
    parser.add_argument("--retries", type=int, default=4, help="Retries on network errors")
    parser.add_argument("--backoff", type=float, default=1.5, help="Initial backoff seconds (exponential)")
    args = parser.parse_args(argv)

    print(ATTRIBUTION)
    print()

    out_dir = Path(args.out_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    zip_path = cache_dir / "mfpt_mirror.zip"

    try:
        download_one(
            args.url,
            zip_path,
            force=args.force,
            timeout_sec=args.timeout,
            retries=max(1, int(args.retries)),
            backoff_sec=max(0.1, float(args.backoff)),
        )
    except Exception as e:
        print(f"ERROR: download failed: {e}", file=sys.stderr)
        return 1

    if args.checksum:
        digest = _sha256_file(zip_path)
        expected = args.checksum.strip().lower()
        if digest != expected:
            print(f"ERROR: checksum mismatch (got {digest}, expected {expected})", file=sys.stderr)
            return 1
        print(f"checksum ok: {digest[:16]}...")

    try:
        extracted = _extract_zip(zip_path, cache_dir / "extracted", force=args.force)
    except zipfile.BadZipFile as e:
        print(f"ERROR: bad zip: {e}", file=sys.stderr)
        return 1

    files = list(PRESETS[args.preset])
    written = _flatten_mats(extracted, out_dir, files, force=args.force)

    if not written:
        print("ERROR: no .mat files were written", file=sys.stderr)
        return 1

    print(f"\nDone. {len(written)} file(s) under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
