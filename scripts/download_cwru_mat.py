#!/usr/bin/env python3
"""
Download official CWRU Bearing Data Center MATLAB (.mat) files.

Files are served from Case School of Engineering static URLs:
  https://engineering.case.edu/sites/default/files/<name>.mat

Data attribution: Case Western Reserve University Bearing Data Center
https://engineering.case.edu/bearingdatacenter

No HTML scraping — filenames are curated from the official site tables.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

BASE_URL = "https://engineering.case.edu/sites/default/files/"
ATTRIBUTION = (
    "Data source: Case Western Reserve University Bearing Data Center\n"
    "https://engineering.case.edu/bearingdatacenter\n"
    "Use of this data should follow the data center's terms and cite appropriately."
)

# Normal baseline (12k / healthy) — see Normal Baseline Data page.
BASELINE_MAT = ["97.mat", "98.mat", "99.mat", "100.mat"]

# All 12k Drive End fault .mat filenames from the official table (unique).
FULL_12K_DE_FAULT_MAT: List[str] = sorted(
    {
        # 0.007"
        "105.mat",
        "106.mat",
        "107.mat",
        "108.mat",
        "118.mat",
        "119.mat",
        "120.mat",
        "121.mat",
        "130.mat",
        "131.mat",
        "132.mat",
        "133.mat",
        "144.mat",
        "145.mat",
        "146.mat",
        "147.mat",
        "156.mat",
        "158.mat",
        "159.mat",
        "160.mat",
        # 0.014"
        "169.mat",
        "170.mat",
        "171.mat",
        "172.mat",
        "185.mat",
        "186.mat",
        "187.mat",
        "188.mat",
        "197.mat",
        "198.mat",
        "199.mat",
        "200.mat",
        # 0.021"
        "209.mat",
        "210.mat",
        "211.mat",
        "212.mat",
        "222.mat",
        "223.mat",
        "224.mat",
        "225.mat",
        "234.mat",
        "235.mat",
        "236.mat",
        "237.mat",
        "246.mat",
        "247.mat",
        "248.mat",
        "249.mat",
        "258.mat",
        "259.mat",
        "260.mat",
        "261.mat",
        # 0.028"
        "3001.mat",
        "3002.mat",
        "3003.mat",
        "3004.mat",
        "3005.mat",
        "3006.mat",
        "3007.mat",
        "3008.mat",
    },
    key=lambda s: int(s.replace(".mat", "")),
)

# One inner / ball / outer @6 fault at 0 HP for a compact binary test alongside baselines.
MINIMAL_FAULT_12K_DE = ["105.mat", "118.mat", "130.mat"]

PRESETS: Dict[str, Sequence[str]] = {
    "baseline_only": list(BASELINE_MAT),
    "minimal_binary_12k_de": list(BASELINE_MAT) + list(MINIMAL_FAULT_12K_DE),
    "full_12k_de_faults": list(FULL_12K_DE_FAULT_MAT),
    # Baselines + every 12k DE fault file in the official table.
    "full_12k_de": list(BASELINE_MAT) + list(FULL_12K_DE_FAULT_MAT),
}


def _normalize_files_arg(files: str | None) -> List[str]:
    if not files or not str(files).strip():
        return []
    out: List[str] = []
    for part in str(files).split(","):
        p = part.strip()
        if not p:
            continue
        if not p.lower().endswith(".mat"):
            p = f"{p}.mat"
        out.append(p)
    return out


def download_one(
    url: str,
    dest: Path,
    *,
    force: bool,
    timeout_sec: float,
    retries: int,
    backoff_sec: float,
) -> str:
    """Return ``skipped`` if left unchanged, ``downloaded`` if bytes were written."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"skip (exists): {dest.name}")
        return "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "contrastive_ssl_cwru_downloader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = resp.read()
            tmp.write_bytes(data)
            tmp.replace(dest)
            print(f"ok: {dest.name} ({len(data)} bytes)")
            return "downloaded"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            wait = backoff_sec * (2 ** (attempt - 1))
            print(f"retry {attempt}/{retries} for {dest.name}: {e} (sleep {wait:.1f}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download CWRU .mat files from Case Engineering static hosting.",
        epilog=(
            "Example pipeline (from repo root):\n"
            "  python scripts/download_cwru_mat.py --preset minimal_binary_12k_de --out_dir data/cwru_mat\n"
            "  cd src\n"
            "  python -m experiments.export_cwru_spectrogram_images "
            "--mat_dir ../data/cwru_mat --out_dir ../data/cwru_spectrograms_png\n"
            "  python -m experiments.eval_spectrogram_checkpoint "
            "--run_id YOUR_RUN_TIMESTAMP --dataset_path ../data/cwru_spectrograms_png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/cwru_mat",
        help="Directory to write .mat files (default: data/cwru_mat)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="minimal_binary_12k_de",
        choices=sorted(PRESETS.keys()),
        help="Which curated file bundle to download (default: minimal_binary_12k_de)",
    )
    parser.add_argument(
        "--files",
        type=str,
        default=None,
        help="Comma-separated .mat filenames (or stems) to download; overrides --preset when non-empty",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists and is non-empty")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout seconds")
    parser.add_argument("--retries", type=int, default=4, help="Retries per file on network errors")
    parser.add_argument("--backoff", type=float, default=1.5, help="Initial backoff seconds (exponential)")
    args = parser.parse_args(argv)

    print(ATTRIBUTION)
    print()

    files = _normalize_files_arg(args.files)
    if not files:
        files = list(PRESETS[args.preset])

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    failed: List[str] = []
    n_downloaded = 0
    n_skipped = 0
    for name in files:
        url = BASE_URL + name
        dest = out_dir / name
        try:
            status = download_one(
                url,
                dest,
                force=args.force,
                timeout_sec=args.timeout,
                retries=max(1, int(args.retries)),
                backoff_sec=max(0.1, float(args.backoff)),
            )
            if status == "downloaded":
                n_downloaded += 1
            else:
                n_skipped += 1
        except Exception as e:
            print(f"ERROR: {name}: {e}", file=sys.stderr)
            failed.append(name)

    if failed:
        print(f"\nFailed ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
        return 1
    print(
        f"\nDone. {len(files)} file(s) under {out_dir} "
        f"(newly downloaded: {n_downloaded}, skipped (already present): {n_skipped})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
