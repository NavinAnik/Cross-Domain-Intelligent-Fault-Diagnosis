"""
Comparative analysis of experiment results for publication.

Loads saved results (from run_all_models, run_ablations, hyperparameter_search)
and produces tables and figures: model comparison, ablation summaries,
confidence intervals, and statistical significance.
"""

import os
import sys
import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load JSON results from file or directory."""
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f) if path.endswith(".json") else [json.load(f)]
    results = []
    for name in os.listdir(path):
        if name.endswith(".json"):
            with open(os.path.join(path, name)) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
    return results


def table_model_comparison(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> str:
    """
    Build a markdown table comparing models (e.g. Table 1 in the plan).
    Expects each result to have model_name (or 'model') and best_val_acc, etc.
    """
    metrics = metrics or ["best_val_acc", "final_val_acc"]
    rows = []
    for r in results:
        if r.get("error"):
            continue
        name = r.get("model_name") or r.get("model", "?")
        row = [name]
        for m in metrics:
            val = r.get(m)
            row.append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
        rows.append(row)
    header = ["Model"] + metrics
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
    return "\n".join(lines)


def table_ablation_fusion(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Table 2: Ablation Study - Fusion Strategies."""
    rows = []
    for name, r in results.items():
        if r.get("error"):
            rows.append([name, "error", r["error"]])
            continue
        rows.append([
            name,
            f"{r.get('best_val_acc', 0):.4f}",
            f"{r.get('final_val_acc', r.get('val_accuracies', [0])[-1] if r.get('val_accuracies') else 0):.4f}",
        ])
    header = ["Fusion", "Best Val Acc", "Final Val Acc"]
    return "|" + "|".join(header) + "|\n" + "|" + "|".join(["---"] * 3) + "|\n" + "\n".join(
        "|" + "|".join(str(c) for c in row) for row in rows
    )


def table_ablation_contrastive(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Table 3: Ablation Study - Contrastive Losses (final train loss)."""
    rows = []
    for name, r in results.items():
        if r.get("error"):
            rows.append([name, "error"])
            continue
        losses = r.get("train_losses", [])
        final_loss = losses[-1] if losses else 0
        rows.append([name, f"{final_loss:.4f}"])
    return "|Loss|Final Train Loss|\n|---|---|\n" + "\n".join(
        "|" + "|".join(str(c) for c in row) for row in rows
    )


def table_ablation_modality(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Table: Ablation Study - Modality (Audio-only vs Vision-only vs Multimodal)."""
    rows = []
    for name, r in results.items():
        if r.get("error"):
            rows.append([name, "error", r["error"]])
            continue
        best = r.get("best_val_acc")
        final = r.get("final_val_acc")
        if final is None and r.get("val_accuracies"):
            final = r["val_accuracies"][-1] if r["val_accuracies"] else None
        rows.append([
            name,
            f"{best:.4f}" if isinstance(best, (int, float)) else str(best),
            f"{final:.4f}" if isinstance(final, (int, float)) else str(final),
        ])
    return "|Modality|Best Val Acc|Final Val Acc|\n|---------|------------|-------------|\n" + "\n".join(
        "|" + "|".join(str(c) for c in row) for row in rows
    )


def summary_statistics(
    results: List[Dict[str, Any]],
    metric: str = "best_val_acc",
) -> Dict[str, Any]:
    """Compute mean, std, and 95% CI for a metric across runs."""
    values = [r[metric] for r in results if isinstance(r.get(metric), (int, float))]
    if not values:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if n >= 2:
        try:
            from scipy import stats
            se = std / np.sqrt(n)
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
            ci95 = (float(ci[0]), float(ci[1]))
        except ImportError:
            ci95 = (mean - 1.96 * std / np.sqrt(n), mean + 1.96 * std / np.sqrt(n))
    else:
        ci95 = (mean, mean)
    return {"mean": mean, "std": std, "ci95": ci95, "n": n}


def generate_report(
    results_dir: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a markdown report from results in results_dir.
    Looks for results.json, ablation_fusion.json, ablation_contrastive.json.
    """
    output_path = output_path or os.path.join(results_dir, "comparative_report.md")
    sections = []

    # Model comparison
    model_results_path = os.path.join(results_dir, "results.json")
    if os.path.exists(model_results_path):
        results = load_results(model_results_path)
        sections.append("## Model Comparison\n\n" + table_model_comparison(results))
        by_model = {}
        for r in results:
            name = r.get("model_name") or r.get("model", "?")
            if name not in by_model:
                by_model[name] = []
            by_model[name].append(r)
        sections.append("\n## Per-model summary (best_val_acc)\n")
        for name, runs in by_model.items():
            stats = summary_statistics(runs, "best_val_acc")
            mean_s = f"{stats['mean']:.4f}" if stats["mean"] is not None else "N/A"
            std_s = f"{stats['std']:.4f}" if stats["std"] is not None else "N/A"
            sections.append(f"- **{name}**: mean={mean_s}, std={std_s}, n={stats['n']}")

    # Fusion ablation
    fusion_path = os.path.join(results_dir, "ablation_fusion.json")
    if os.path.exists(fusion_path):
        with open(fusion_path) as f:
            fusion_results = json.load(f)
        sections.append("\n## Ablation: Fusion Strategies\n\n" + table_ablation_fusion(fusion_results))

    # Contrastive ablation
    contrastive_path = os.path.join(results_dir, "ablation_contrastive.json")
    if os.path.exists(contrastive_path):
        with open(contrastive_path) as f:
            contrastive_results = json.load(f)
        sections.append("\n## Ablation: Contrastive Losses\n\n" + table_ablation_contrastive(contrastive_results))

    # Modality ablation
    modality_path = os.path.join(results_dir, "ablation_modality.json")
    if os.path.exists(modality_path):
        with open(modality_path) as f:
            modality_results = json.load(f)
        sections.append("\n## Ablation: Modality (Audio vs Vision vs Multimodal)\n\n" + table_ablation_modality(modality_results))

    report = "\n".join(sections)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Comparative analysis of experiment results")
    parser.add_argument("--results_dir", type=str, default="experiments/hp_search",
                        help="Directory or JSON file with results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output report path (default: results_dir/comparative_report.md)")
    args = parser.parse_args()
    report = generate_report(args.results_dir, args.output)
    print(report)


if __name__ == "__main__":
    main()
