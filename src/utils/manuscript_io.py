"""
Publication helpers: CSV + LaTeX tables, PNG + PDF figures.
"""

from __future__ import annotations

import csv
import os
from typing import Any, List, Optional, Sequence, Union

import matplotlib.pyplot as plt

# Display name for publication figures (architecture remains resnet50 internally).
PLOT_MODEL_LABEL = "hybrid_resnet"


def _escape_latex(text: str) -> str:
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def save_table(
    rows: Sequence[Sequence[Any]],
    headers: Sequence[str],
    basepath: str,
    *,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    """Write CSV and LaTeX tabular to ``basepath.csv`` and ``basepath.tex``."""
    basepath = os.path.normpath(basepath)
    os.makedirs(os.path.dirname(basepath) or ".", exist_ok=True)

    csv_path = f"{basepath}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)

    ncol = len(headers)
    col_spec = "l" * ncol
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
    ]
    if caption:
        lines.append(f"\\caption{{{_escape_latex(caption)}}}")
    if label:
        lines.append(f"\\label{{{_escape_latex(label)}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(" & ".join(_escape_latex(h) for h in headers) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        cells = []
        for c in row:
            if isinstance(c, float):
                cells.append(f"{c:.2f}")
            else:
                cells.append(_escape_latex(c))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\end{tabular}", r"\end{table}"])

    tex_path = f"{basepath}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_figure(fig: plt.Figure, basepath: str, *, dpi: int = 300) -> None:
    """Save figure as PNG and PDF."""
    basepath = os.path.normpath(basepath)
    os.makedirs(os.path.dirname(basepath) or ".", exist_ok=True)
    fig.savefig(f"{basepath}.png", dpi=dpi, bbox_inches="tight")
    try:
        fig.savefig(f"{basepath}.pdf", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def summary_mean_std_table(
    aggregated: dict,
    metric_keys: List[str],
    metric_labels: Optional[List[str]] = None,
) -> tuple[List[str], List[List[Any]]]:
    """Build journal-style Mean | Std rows from aggregate_cv_metrics output."""
    headers = ["Metric", "Mean", "Std"]
    labels = metric_labels or metric_keys
    rows: List[List[Any]] = []
    for key, label in zip(metric_keys, labels):
        block = aggregated.get(key) or {}
        mean = block.get("mean")
        std = block.get("std")
        if mean is None:
            continue
        rows.append([label, round(float(mean) * 100, 1) if mean <= 1 else round(float(mean), 1), round(float(std or 0) * 100, 1) if (std or 0) <= 1 else round(float(std or 0), 1)])
    return headers, rows
