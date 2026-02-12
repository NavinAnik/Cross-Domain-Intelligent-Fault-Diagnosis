"""
Generate comparison tables with all metrics for Q1 journal publication.

Produces:
  Table 1: Model Comparison (Params, FLOPs, Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, Inference Time)
  Table 2: Ablation Study - Fusion Strategies (all metrics)
  Table 3: Ablation Study - Contrastive Losses (all metrics)
  Table 4: Cross-validation Results (mean ± std) per model and metric
  Table 5: Statistical Significance Matrix (p-values)

Usage:
  python -m experiments.generate_tables --results_dir results --output_dir tables
  python -m experiments.generate_tables --results_dir results --output_dir tables --demo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.statistical_tests import aggregate_cv_metrics, significance_matrix


# Standard classification metrics (from compute_classification_metrics)
STANDARD_METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc_roc",
    "auc_pr",
    "mcc",
    "cohen_kappa",
]

# Table 1 display columns (plan: Params | FLOPs | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Inference Time)
TABLE1_COLUMNS = [
    "Model",
    "Params",
    "FLOPs",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC-ROC",
    "AUC-PR",
    "Inference Time",
]

# Map internal keys to Table 1 column names
TABLE1_KEY_MAP = {
    "model_name": "Model",
    "params_M": "Params",
    "param_count": "Params",
    "flops_G": "FLOPs",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "auc_roc": "AUC-ROC",
    "auc_pr": "AUC-PR",
    "inference_time_ms": "Inference Time",
}


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_number(val: Any, decimals: int = 4, na_str: str = "—") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return na_str
    if isinstance(val, (int, float)):
        if abs(val) >= 1e4 or (abs(val) < 0.0001 and val != 0):
            return f"{val:.2e}"
        return f"{val:.{decimals}f}"
    return str(val)


def _get_metrics_from_result(r: Dict[str, Any]) -> Dict[str, Any]:
    """Extract flat metrics from a result dict (evaluation_metrics or top-level)."""
    out = {}
    em = r.get("evaluation_metrics") or {}
    for k in STANDARD_METRIC_KEYS:
        if k in em and em[k] is not None:
            out[k] = em[k]
        elif k in r and r[k] is not None:
            out[k] = r[k]
    # Fallback: best_val_acc -> accuracy
    if "accuracy" not in out and r.get("best_val_acc") is not None:
        out["accuracy"] = r["best_val_acc"]
    return out


def _get_params_display(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        if val >= 1e6:
            return f"{val / 1e6:.0f}M"
        if val >= 1e3:
            return f"{val / 1e3:.1f}K"
        return f"{val:.0f}"
    return str(val)


def _get_flops_display(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.0f}G" if val >= 1 else f"{val:.2f}G"
    return str(val)


def _get_inference_time_display(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.1f} ms"
    return str(val)


def table1_model_comparison(
    results: List[Dict[str, Any]],
    include_params_flops: bool = True,
) -> Tuple[str, List[List[str]]]:
    """
    Build Table 1: Model Comparison with Params, FLOPs, Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, Inference Time.

    Returns:
        (markdown_string, rows_for_csv)
    """
    rows = []
    for r in results:
        if r.get("error"):
            continue
        name = r.get("model_name") or r.get("model", "?")
        metrics = _get_metrics_from_result(r)
        # Params: params_M or param_count
        params_raw = r.get("params_M")
        if params_raw is None and r.get("param_count") is not None:
            params_raw = r["param_count"] / 1e6
        params_str = _get_params_display(params_raw) if include_params_flops else "—"
        flops_raw = r.get("flops_G") or r.get("flops")
        flops_str = _get_flops_display(flops_raw) if include_params_flops else "—"
        inf_time = r.get("inference_time_ms") or r.get("inference_time")
        inf_str = _get_inference_time_display(inf_time) if include_params_flops else "—"

        row = [
            name,
            params_str,
            flops_str,
            _format_number(metrics.get("accuracy")),
            _format_number(metrics.get("precision")),
            _format_number(metrics.get("recall")),
            _format_number(metrics.get("f1")),
            _format_number(metrics.get("auc_roc")),
            _format_number(metrics.get("auc_pr")),
            inf_str,
        ]
        rows.append(row)

    header = TABLE1_COLUMNS
    if not rows:
        md = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |\n*(no data)*"
        return md, [header]

    # Markdown
    col_widths = [max(len(str(h)), 4) for h in header]
    for row in rows:
        for i, c in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(c)))
    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines = [
        "|" + "|".join(str(h).ljust(col_widths[i]) for i, h in enumerate(header)) + "|",
        sep,
    ]
    for row in rows:
        lines.append("|" + "|".join(str(c).ljust(col_widths[i]) for i, c in enumerate(row)) + "|")
    return "\n".join(lines), [header] + rows


def table2_ablation_fusion(
    results: Dict[str, Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> Tuple[str, List[List[str]]]:
    """
    Build Table 2: Ablation Study - Fusion Strategies (all metrics).

    results: fusion_name -> { accuracy, precision, recall, f1, auc_roc, auc_pr, ... }
    """
    metric_keys = metric_keys or STANDARD_METRIC_KEYS
    header = ["Fusion"] + [m.replace("_", " ").title() for m in metric_keys]
    rows = []
    for name, r in sorted(results.items()):
        if r.get("error"):
            rows.append([name, "error", r.get("error", "")] + [""] * (len(metric_keys) - 2))
            continue
        metrics = _get_metrics_from_result(r)
        row = [name] + [_format_number(metrics.get(k)) for k in metric_keys]
        rows.append(row)
    if not rows:
        md = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |\n*(no data)*"
        return md, [header]
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines), [header] + rows


def table3_ablation_contrastive(
    results: Dict[str, Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> Tuple[str, List[List[str]]]:
    """
    Build Table 3: Ablation Study - Contrastive Losses (all metrics).
    """
    metric_keys = metric_keys or STANDARD_METRIC_KEYS
    header = ["Loss"] + [m.replace("_", " ").title() for m in metric_keys]
    rows = []
    for name, r in sorted(results.items()):
        if r.get("error"):
            rows.append([name, "error"] + [""] * (len(metric_keys) - 1))
            continue
        metrics = _get_metrics_from_result(r)
        row = [name] + [_format_number(metrics.get(k)) for k in metric_keys]
        rows.append(row)
    if not rows:
        md = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |\n*(no data)*"
        return md, [header]
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines), [header] + rows


def table4_cv_results(
    metrics_by_model: Dict[str, List[Dict[str, Any]]],
    metric_keys: Optional[List[str]] = None,
    confidence: float = 0.95,
) -> Tuple[str, List[List[str]]]:
    """
    Build Table 4: Cross-validation Results (mean ± std) per model and metric.
    """
    metric_keys = metric_keys or STANDARD_METRIC_KEYS
    header = ["Model"] + [m.replace("_", " ").title() for m in metric_keys]
    rows = []
    for name in sorted(metrics_by_model.keys()):
        per_fold = metrics_by_model[name]
        agg = aggregate_cv_metrics(per_fold, confidence=confidence)
        row = [name]
        for k in metric_keys:
            if k not in agg or agg[k]["mean"] is None:
                row.append("—")
            else:
                mean, std = agg[k]["mean"], agg[k]["std"]
                row.append(f"{mean:.4f} ± {std:.4f}")
        rows.append(row)
    if not rows:
        md = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * len(header)) + " |\n*(no data)*"
        return md, [header]
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines), [header] + rows


def table5_significance_matrix(
    metrics_by_model: Dict[str, List[Dict[str, Any]]],
    metric_name: str = "accuracy",
    tests: Tuple[str, ...] = ("ttest", "wilcoxon"),
) -> Tuple[str, str, List[List[str]], List[List[str]]]:
    """
    Build Table 5: Statistical Significance Matrix (p-values).
    Returns (markdown_ttest, markdown_wilcoxon, rows_ttest, rows_wilcoxon).
    """
    # Extract score list per model (same order of folds)
    scores = {}
    for name, per_fold in metrics_by_model.items():
        vals = [d.get(metric_name) for d in per_fold if d.get(metric_name) is not None]
        if vals:
            scores[name] = vals
    if len(scores) < 2:
        empty = "*(need at least 2 models with same number of folds)*"
        return empty, empty, [], []

    n_folds = len(next(iter(scores.values())))
    if not all(len(v) == n_folds for v in scores.values()):
        return "*(all models must have same number of folds)*", "", [], []

    sm = significance_matrix(scores, metric_name=metric_name, tests=tests)
    names = sm["model_names"]

    def _matrix_to_md(arr: np.ndarray, title: str) -> str:
        lines = [f"**{title}** (p-values)", ""]
        header = [""] + names
        sep = "| " + " | ".join(["---"] * (len(names) + 1)) + " |"
        lines.append("| " + " | ".join(header) + " |")
        lines.append(sep)
        for i, name in enumerate(names):
            row = [name] + [f"{arr[i, j]:.4f}" for j in range(len(names))]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _matrix_to_rows(arr: np.ndarray) -> List[List[str]]:
        header = [""] + names
        rows = [header]
        for i, name in enumerate(names):
            rows.append([name] + [f"{arr[i, j]:.4f}" for j in range(len(names))])
        return rows

    md_ttest = ""
    md_wilcoxon = ""
    rows_ttest = []
    rows_wilcoxon = []
    if "pvalue_ttest" in sm:
        md_ttest = _matrix_to_md(sm["pvalue_ttest"], "Paired t-test")
        rows_ttest = _matrix_to_rows(sm["pvalue_ttest"])
    if "pvalue_wilcoxon" in sm:
        md_wilcoxon = _matrix_to_md(sm["pvalue_wilcoxon"], "Wilcoxon signed-rank")
        rows_wilcoxon = _matrix_to_rows(sm["pvalue_wilcoxon"])
    return md_ttest, md_wilcoxon, rows_ttest, rows_wilcoxon


def load_results_dir(results_dir: str) -> Dict[str, Any]:
    """Load all result files from results_dir."""
    data = {
        "model_results": [],
        "ablation_fusion": {},
        "ablation_contrastive": {},
        "ablation_modality": {},
        "cv_results": {},
    }
    # Single model results
    results_json = os.path.join(results_dir, "results.json")
    if os.path.isfile(results_json):
        raw = _load_json(results_json)
        data["model_results"] = raw if isinstance(raw, list) else [raw]
    if os.path.isdir(results_dir):
        for name in os.listdir(results_dir):
            if name == "results.json":
                continue
            if name.endswith(".json"):
                path = os.path.join(results_dir, name)
                try:
                    raw = _load_json(path)
                    if name == "ablation_fusion.json" and isinstance(raw, dict):
                        data["ablation_fusion"] = raw
                    elif name == "ablation_contrastive.json" and isinstance(raw, dict):
                        data["ablation_contrastive"] = raw
                    elif name == "ablation_modality.json" and isinstance(raw, dict):
                        data["ablation_modality"] = raw
                    elif name in ("cv_results.json", "cv_metrics.json") and isinstance(raw, (dict, list)):
                        if isinstance(raw, dict):
                            data["cv_results"] = raw
                        else:
                            for item in raw:
                                n = item.get("model_name") or item.get("model", "unknown")
                                folds = item.get("cv_metrics") or item.get("per_fold_metrics") or []
                                data["cv_results"][n] = list(folds)
                except (json.JSONDecodeError, TypeError):
                    pass
    return data


def get_demo_data() -> Dict[str, Any]:
    """Synthetic data for all five tables when no real results exist."""
    np.random.seed(42)
    model_names = ["DINOv2-Base", "MAE ViT-Base", "Swin-Base", "BEiT-Base", "AST"]
    model_results = []
    for i, name in enumerate(model_names):
        model_results.append({
            "model_name": name,
            "params_M": 86 + (i % 3) * 2,
            "flops_G": 18 - i,
            "inference_time_ms": 12 - i * 0.5,
            "accuracy": 0.82 + 0.02 * i + 0.01 * np.random.rand(),
            "precision": 0.80 + 0.02 * i + 0.01 * np.random.rand(),
            "recall": 0.79 + 0.02 * i + 0.01 * np.random.rand(),
            "f1": 0.81 + 0.02 * i + 0.01 * np.random.rand(),
            "auc_roc": 0.85 + 0.02 * i + 0.01 * np.random.rand(),
            "auc_pr": 0.83 + 0.02 * i + 0.01 * np.random.rand(),
            "mcc": 0.6 + 0.05 * i + 0.02 * np.random.rand(),
            "cohen_kappa": 0.58 + 0.05 * i + 0.02 * np.random.rand(),
        })
    fusion_names = ["Early", "Late", "Cross-Attention", "ConFu", "ProMoBal"]
    ablation_fusion = {}
    for i, name in enumerate(fusion_names):
        ablation_fusion[name] = {
            "accuracy": 0.80 + 0.02 * i + 0.01 * np.random.rand(),
            "precision": 0.78 + 0.02 * i + 0.01 * np.random.rand(),
            "recall": 0.77 + 0.02 * i + 0.01 * np.random.rand(),
            "f1": 0.79 + 0.02 * i + 0.01 * np.random.rand(),
            "auc_roc": 0.82 + 0.02 * i + 0.01 * np.random.rand(),
            "auc_pr": 0.80 + 0.02 * i + 0.01 * np.random.rand(),
            "mcc": 0.55 + 0.05 * i,
            "cohen_kappa": 0.52 + 0.05 * i,
        }
    loss_names = ["SimCLR", "MoCo v3", "BYOL", "VICReg"]
    ablation_contrastive = {}
    for i, name in enumerate(loss_names):
        ablation_contrastive[name] = {
            "accuracy": 0.79 + 0.02 * i + 0.01 * np.random.rand(),
            "precision": 0.77 + 0.02 * i + 0.01 * np.random.rand(),
            "recall": 0.76 + 0.02 * i + 0.01 * np.random.rand(),
            "f1": 0.78 + 0.02 * i + 0.01 * np.random.rand(),
            "auc_roc": 0.81 + 0.02 * i + 0.01 * np.random.rand(),
            "auc_pr": 0.79 + 0.02 * i + 0.01 * np.random.rand(),
            "mcc": 0.52 + 0.05 * i,
            "cohen_kappa": 0.50 + 0.05 * i,
        }
    n_folds = 5
    cv_results = {}
    for name in model_names:
        cv_results[name] = []
        for _ in range(n_folds):
            cv_results[name].append({
                "accuracy": 0.80 + 0.05 * np.random.rand(),
                "precision": 0.78 + 0.05 * np.random.rand(),
                "recall": 0.77 + 0.05 * np.random.rand(),
                "f1": 0.79 + 0.05 * np.random.rand(),
                "auc_roc": 0.82 + 0.05 * np.random.rand(),
                "auc_pr": 0.80 + 0.05 * np.random.rand(),
                "mcc": 0.55 + 0.08 * np.random.rand(),
                "cohen_kappa": 0.52 + 0.08 * np.random.rand(),
            })
    return {
        "model_results": model_results,
        "ablation_fusion": ablation_fusion,
        "ablation_contrastive": ablation_contrastive,
        "cv_results": cv_results,
    }


def write_csv(rows: List[List[str]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)


def generate_all_tables(
    results_dir: str,
    output_dir: str,
    use_demo: bool = False,
    write_csv_files: bool = True,
) -> str:
    """
    Generate all five comparison tables (Markdown + optional CSV) and return combined report.
    """
    if use_demo:
        data = get_demo_data()
    else:
        data = load_results_dir(results_dir)

    os.makedirs(output_dir, exist_ok=True)
    sections = []

    # Table 1: Model Comparison
    md1, rows1 = table1_model_comparison(data["model_results"])
    sections.append("## Table 1: Model Comparison\n\n" + md1)
    if write_csv_files and rows1:
        write_csv(rows1, os.path.join(output_dir, "table1_model_comparison.csv"))
    with open(os.path.join(output_dir, "table1_model_comparison.md"), "w", encoding="utf-8") as f:
        f.write("# Table 1: Model Comparison\n\n" + md1)

    # Table 2: Ablation - Fusion
    md2, rows2 = table2_ablation_fusion(data["ablation_fusion"])
    sections.append("\n## Table 2: Ablation Study - Fusion Strategies\n\n" + md2)
    if write_csv_files and rows2:
        write_csv(rows2, os.path.join(output_dir, "table2_ablation_fusion.csv"))
    with open(os.path.join(output_dir, "table2_ablation_fusion.md"), "w", encoding="utf-8") as f:
        f.write("# Table 2: Ablation Study - Fusion Strategies\n\n" + md2)

    # Table 3: Ablation - Contrastive
    md3, rows3 = table3_ablation_contrastive(data["ablation_contrastive"])
    sections.append("\n## Table 3: Ablation Study - Contrastive Losses\n\n" + md3)
    if write_csv_files and rows3:
        write_csv(rows3, os.path.join(output_dir, "table3_ablation_contrastive.csv"))
    with open(os.path.join(output_dir, "table3_ablation_contrastive.md"), "w", encoding="utf-8") as f:
        f.write("# Table 3: Ablation Study - Contrastive Losses\n\n" + md3)

    # Table 4: Cross-validation (mean ± std)
    md4, rows4 = table4_cv_results(data["cv_results"])
    sections.append("\n## Table 4: Cross-validation Results (mean ± std)\n\n" + md4)
    if write_csv_files and rows4:
        write_csv(rows4, os.path.join(output_dir, "table4_cv_results.csv"))
    with open(os.path.join(output_dir, "table4_cv_results.md"), "w", encoding="utf-8") as f:
        f.write("# Table 4: Cross-validation Results (mean ± std)\n\n" + md4)

    # Table 5: Statistical Significance Matrix
    md5_ttest, md5_wilcoxon, rows5_ttest, rows5_wilcoxon = table5_significance_matrix(
        data["cv_results"], metric_name="accuracy"
    )
    sections.append("\n## Table 5: Statistical Significance Matrix (p-values)\n\n" + md5_ttest)
    if md5_wilcoxon and "*(need" not in md5_wilcoxon and "*(all" not in md5_wilcoxon:
        sections.append("\n\n" + md5_wilcoxon)
    if write_csv_files and rows5_ttest:
        write_csv(rows5_ttest, os.path.join(output_dir, "table5_significance_ttest.csv"))
    if write_csv_files and rows5_wilcoxon:
        write_csv(rows5_wilcoxon, os.path.join(output_dir, "table5_significance_wilcoxon.csv"))
    with open(os.path.join(output_dir, "table5_significance_matrix.md"), "w", encoding="utf-8") as f:
        f.write("# Table 5: Statistical Significance Matrix\n\n" + md5_ttest)
        if md5_wilcoxon and "*(need" not in md5_wilcoxon and "*(all" not in md5_wilcoxon:
            f.write("\n\n" + md5_wilcoxon)

    report = "\n".join(sections)
    report_path = os.path.join(output_dir, "comparison_tables_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Comparison Tables – All Metrics\n\n")
        f.write(report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison tables with all metrics for publication"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing results.json, ablation_*.json, cv_results.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tables",
        help="Output directory for Markdown and CSV tables",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo/synthetic data to generate all tables",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Do not write CSV files, only Markdown",
    )
    args = parser.parse_args()
    report = generate_all_tables(
        args.results_dir,
        args.output_dir,
        use_demo=args.demo,
        write_csv_files=not args.no_csv,
    )
    print(report)
    print(f"\nTables written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
