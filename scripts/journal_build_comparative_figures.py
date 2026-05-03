"""
Build comparative bar charts from journal_experiment_index.json.

Writes PDF and PNG under docs/journal_submission_figures/ by default.

Usage:
  python scripts/journal_build_comparative_figures.py
  python scripts/journal_build_comparative_figures.py --index docs/journal_aggregation/journal_experiment_index.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _short_label(row: Dict[str, Any], max_len: int = 42) -> str:
    parts = [
        str(row.get("line_of_work", ""))[:20],
        str(row.get("run_id", ""))[:22],
    ]
    s = " / ".join(p for p in parts if p)
    return (s[: max_len - 2] + "…") if len(s) > max_len else s


def build_figures(
    index_path: Path,
    out_dir: Path,
    *,
    top_n: int = 24,
) -> List[Path]:
    if not index_path.is_file():
        return []
    rows: List[Dict[str, Any]] = json.loads(index_path.read_text(encoding="utf-8"))
    scored = [r for r in rows if r.get("primary_metric_value") is not None]
    scored.sort(key=lambda r: -float(r["primary_metric_value"]))
    top = scored[:top_n]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []

    if not top:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No rows with primary_metric_value", ha="center", va="center")
        ax.axis("off")
        for ext in (".pdf", ".png"):
            p = out_dir / f"comparison_top_runs{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            out_paths.append(p)
        plt.close(fig)
        return out_paths

    labels = [_short_label(r) for r in top]
    vals = [float(r["primary_metric_value"]) for r in top]

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(top))))
    colors = plt.cm.viridis([0.2 + 0.6 * i / max(len(top) - 1, 1) for i in range(len(top))])
    ax.barh(range(len(top)), vals, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Primary metric value (mixed metrics across rows — see table)")
    ax.set_title("Top indexed runs by reported primary metric")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    for ext in (".pdf", ".png"):
        p = out_dir / f"comparison_top_runs{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        out_paths.append(p)
    plt.close(fig)

    # Second figure: count by line_of_work
    from collections import Counter

    c = Counter(str(r.get("line_of_work", "?")) for r in rows)
    if c:
        fig2, ax2 = plt.subplots(figsize=(7, 3.2))
        labs = list(c.keys())
        cnts = [c[k] for k in labs]
        ax2.bar(labs, cnts, color="#3D5A80", alpha=0.85)
        ax2.set_ylabel("Indexed rows")
        ax2.set_title("Harvested rows by line_of_work")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right")
        fig2.tight_layout()
        for ext in (".pdf", ".png"):
            p = out_dir / f"comparison_rows_by_line{ext}"
            fig2.savefig(p, dpi=200, bbox_inches="tight")
            out_paths.append(p)
        plt.close(fig2)

    return out_paths


def maybe_run_publication_bundle(results_dir: Path, fig_out: Path) -> None:
    """Optionally call experiments.generate_publication_figures if results_dir looks populated."""
    if not results_dir.is_dir():
        return
    marker = results_dir / "results.json"
    if not marker.is_file():
        return
    import subprocess
    import sys

    fig_out.mkdir(parents=True, exist_ok=True)
    src = _repo_root() / "src"
    cmd = [
        sys.executable,
        "-m",
        "experiments.generate_publication_figures",
        "--results_dir",
        str(results_dir),
        "--output_dir",
        str(fig_out),
    ]
    subprocess.run(cmd, cwd=str(src), check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument(
        "--publication-results-dir",
        type=Path,
        default=None,
        help="If set and contains results.json, also run generate_publication_figures into out-dir/publication/",
    )
    args = parser.parse_args()
    root = _repo_root()
    index = args.index or (root / "docs" / "journal_aggregation" / "journal_experiment_index.json")
    out_dir = args.out_dir or (root / "docs" / "journal_submission_figures")
    paths = build_figures(index, out_dir, top_n=args.top_n)
    for p in paths:
        print("Wrote:", p)
    if args.publication_results_dir:
        maybe_run_publication_bundle(
            args.publication_results_dir,
            out_dir / "publication_bundle",
        )


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
