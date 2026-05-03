"""
Generate docs/Journal_Full_Comparative_Study.docx — IMRaD-style comparative manuscript.

Optionally re-runs aggregation and comparative figures before assembling the Word file.

Usage (from repository root):
  python scripts/journal_aggregate_experiment_results.py --with-plots
  python scripts/journal_generate_full_paper_docx.py

Or in one step (harvest + figures + docx):
  python scripts/journal_generate_full_paper_docx.py --run-harvest --with-plots

Requires: pip install python-docx
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _run(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=False)


def _add_para(doc, text: str, *, bold: bool = False, italic: bool = False) -> None:
    from docx.shared import Pt

    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(11)


def _add_heading(doc, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def _add_picture_if_exists(doc, path: Path, width_in: float = 6.0) -> bool:
    from docx.shared import Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    if not path.is_file():
        doc.add_paragraph(f"[Figure not found: {path}]").runs[0].italic = True
        return False
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path.resolve()), width=Inches(width_in))
    return True


def _caption(doc, text: str) -> None:
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _results_table(doc, rows: List[Dict[str, Any]], max_rows: int = 35) -> None:
    from docx.enum.table import WD_TABLE_ALIGNMENT

    cols = [
        "line_of_work",
        "run_id",
        "primary_metric_name",
        "primary_metric_value",
        "split_protocol",
    ]
    subset = rows[:max_rows]
    t = doc.add_table(rows=len(subset) + 1, cols=len(cols))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(cols):
        t.rows[0].cells[j].text = c
        t.rows[0].cells[j].paragraphs[0].runs[0].bold = True
    for i, r in enumerate(subset, start=1):
        for j, c in enumerate(cols):
            v = r.get(c, "")
            if c == "primary_metric_value" and v is not None:
                v = f"{float(v):.4f}"
            t.rows[i].cells[j].text = str(v)


def build_docx(
    output_path: Path,
    *,
    index: Optional[List[Dict[str, Any]]],
    best: Optional[Dict[str, Any]],
    inventory_path: Path,
) -> None:
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt, RGBColor
    except ImportError as e:
        raise SystemExit("Install python-docx: pip install python-docx") from e

    root = _repo_root()
    inv = _load_json(inventory_path) or {}
    entries = inv.get("experiment_entrypoints") or []

    doc = Document()
    t = doc.add_heading("Journal full comparative study", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st = doc.add_paragraph()
    r = st.add_run(
        "Industrial anomaly / fault detection: unified inventory of experiments in this repository "
        "and best-performing configuration per comparability bucket (automated harvest)."
    )
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # Abstract
    _add_heading(doc, "Abstract", 1)
    n_rows = len(index or [])
    n_buckets = len((best or {}).get("buckets") or {})
    abs_p = doc.add_paragraph()
    abs_p.add_run(
        f"We systematically inventoried {len(entries)} experiment families implemented in a single "
        f"codebase for industrial acoustic and image-based fault screening. "
        f"An automated harvest indexed {n_rows} distinct result rows from available JSON/CSV artifacts "
        f"under the configured scan roots; {n_buckets} non-empty comparability buckets yielded a ranked "
        f"winner under a fixed tie rule (maximize reported primary metric; see Methods). "
        "Experiments differ by dataset, modality, split protocol, and metric; therefore no single global "
        "winner is asserted across incompatible settings. "
        "Where local `results/` trees are absent (common in minimal git clones because `results/` is "
        "gitignored), the index may be empty and the manuscript must be populated after local execution. "
    )

    _add_heading(doc, "Introduction", 1)
    _add_para(
        doc,
        "Early detection of machine faults from operational sound reduces downtime. "
        "This repository implements multiple lines of work—supervised vision encoders, contrastive "
        "self-supervised learning on mel features, multimodal fusion, rigorous cross-validation utilities, "
        "and supervised classification on spectrogram images—with shared dataset loaders for MIMII-style "
        "folder layouts. "
        "The present document consolidates what was attempted in code, which components are novel relative "
        "to a plain supervised baseline, and which harvested runs achieve the strongest reported score "
        "within each comparability bucket.",
    )

    _add_heading(doc, "Methods", 1)
    _add_para(
        doc,
        "Experiment entry points, default output locations, and differentiated modules are listed in "
        f"`{inventory_path.as_posix()}` (machine-readable JSON and companion Markdown). "
        "Novel or engineering contributions implemented in software include asymmetric focal contrastive "
        "components (AFCL), curriculum training hooks, spectrogram-image models with optional spatial "
        "attention and CNN–Transformer hybrids, group-aware stratified validation for grouped corpora, "
        "and fusion architectures with optional contrastive pairing between modalities.",
        italic=False,
    )
    _add_heading(doc, "Harvest and bucketing", 2)
    _add_para(
        doc,
        "The harvester (`scripts/journal_aggregate_experiment_results.py`) walks user-provided roots "
        "and parses: (i) directories containing both `config.json` and `metrics.json` (spectrogram fault runs); "
        "(ii) `single_model_experiments_summary.json`; (iii) `results.json` list bundles; (iv) "
        "`rigorous_cv_results.json`; (v) `fusion_summary.json`; (vi) `exp*_results.json`; (vii) generic "
        "`ablation_results*.json`; (viii) spectrogram ablation CSVs. "
        "Each row carries `line_of_work`, `split_protocol`, `dataset_hint`, and a `primary_metric_value` "
        "chosen by priority: validation balanced accuracy, balanced accuracy, ROC AUC, best validation "
        "accuracy, accuracy, macro-F1, then MCC. "
        "Buckets are formed as `line_of_work|split_protocol|dataset_hint` (truncated). "
        "Within each bucket the harvester selects the row with the largest primary metric value; "
        "ties break by lexicographic `run_id`.",
    )

    _add_heading(doc, "Figures", 2)
    fig_dir = root / "docs" / "journal_submission_figures"
    paper_fig = root / "src" / "paper_assets" / "figures"
    anchor = paper_fig / "from_anchor_run_20260407_193339"

    if _add_picture_if_exists(doc, fig_dir / "comparison_top_runs.png", 6.2):
        _caption(doc, "Fig. 1. Top indexed runs by harvested primary metric (metrics may differ across rows—interpret with Table 1).")
    if _add_picture_if_exists(doc, fig_dir / "comparison_rows_by_line.png", 5.5):
        _caption(doc, "Fig. 2. Count of harvested index rows by line of work.")

    if _add_picture_if_exists(doc, paper_fig / "methodology_pipeline.png", 6.2):
        _caption(doc, "Fig. 3. Spectrogram-image supervised pipeline schematic (repository figure generator).")
    if _add_picture_if_exists(doc, paper_fig / "architecture_cnn_baseline.png", 6.0):
        _caption(doc, "Fig. 4. CNN baseline architecture (timm backbone + GAP + linear head).")

    fig_num = 5
    for name in ["grad_cam_0.png", "roc_curve.png", "confusion_matrix.png"]:
        pth = anchor / name
        if pth.is_file() and _add_picture_if_exists(doc, pth, 5.4):
            _caption(
                doc,
                f"Fig. {fig_num}. Example qualitative or validation figure from mirrored anchor run (`{name}`).",
            )
            fig_num += 1

    pub_dir = fig_dir / "publication_bundle"
    if pub_dir.is_dir():
        extras = sorted(pub_dir.glob("*.png"))[:4]
        for pth in extras:
            if _add_picture_if_exists(doc, pth, 5.8):
                _caption(
                    doc,
                    f"Fig. {fig_num}. Publication bundle figure (`{pth.name}`) from generate_publication_figures.",
                )
                fig_num += 1

    _add_heading(doc, "Experimental setup", 1)
    _add_para(
        doc,
        "Software: Python 3 with PyTorch; see `src/requirements.txt`. Hardware and wall-clock times are not "
        "inferred by the harvester and must be filled by authors from training logs. "
        "Dataset paths recorded in each JSON row reflect the machine that produced the artifact.",
    )

    _add_heading(doc, "Results", 1)
    _add_para(
        doc,
        "Table 1 lists up to 35 indexed rows (truncated). Per-bucket winners are summarized in prose below.",
    )
    _add_heading(doc, "Table 1. Harvested experiment index (truncated)", 2)
    if index:
        _results_table(
            doc,
            sorted(index, key=lambda r: -float(r.get("primary_metric_value") if r.get("primary_metric_value") is not None else -1.0)),
        )
    else:
        doc.add_paragraph("No rows harvested.")

    _add_heading(doc, "Best configuration per bucket", 2)
    if best and best.get("buckets"):
        for bkey, payload in sorted(best["buckets"].items(), key=lambda x: x[0])[:40]:
            w = payload.get("winner") or {}
            val = w.get("primary_metric_value")
            mname = w.get("primary_metric_name", "")
            rid = w.get("run_id", "")
            doc.add_paragraph(
                f"Bucket `{bkey[:200]}`: best {mname} = {val} (run_id={rid}; n_candidates={payload.get('n_candidates', 0)}).",
                style="List Bullet",
            )
    else:
        _add_para(doc, "No buckets with finite primary metrics. Run experiments and re-harvest.", italic=True)

    _add_heading(doc, "Discussion", 1)
    _add_para(
        doc,
        "The automated index does not replace domain review: different scripts report different primary "
        "scores, and some rows may be incomparable even within a bucket if `dataset_hint` is coarse. "
        "Novel modules in the codebase do not guarantee superior performance; ablation tables in "
        "`generate_tables.py` should be cited when available.",
    )

    _add_heading(doc, "Limitations", 1)
    _add_para(
        doc,
        "Gitignore excludes `results/`, `*.png`, `*.csv`, and checkpoints; clones may lack evidence files. "
        "Group labels may be imperfect. External validation on held-out sites is not performed by the harvester.",
    )

    _add_heading(doc, "Data and code availability", 1)
    _add_para(
        doc,
        "Source code: this repository. Regenerate the index: "
        "`python scripts/journal_aggregate_experiment_results.py --roots <paths> --with-plots`. "
        "Regenerate comparative figures: `python scripts/journal_build_comparative_figures.py`. "
        "Regenerate this document: `python scripts/journal_generate_full_paper_docx.py`.",
    )

    _add_heading(doc, "Checklist before journal submission", 1)
    checklist = [
        "Replace placeholder affiliation and author list.",
        "Verify every numeric claim against raw logs; do not submit with empty index if the study is empirical.",
        "Unify notation for metrics (e.g. always report balanced accuracy for imbalanced tasks).",
        "Add IRB / data-use statements if human data ever appear.",
        "Run venue-specific reference formatting and disclosure statements.",
        "Spell-check; consider professional language editing.",
    ]
    for item in checklist:
        doc.add_paragraph(item, style="List Bullet")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print("Wrote:", output_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--run-harvest",
        action="store_true",
        help="Run journal_aggregate_experiment_results.py before building (default roots).",
    )
    parser.add_argument(
        "--with-plots",
        action="store_true",
        help="If combined with --run-harvest, pass --with-plots to the harvester.",
    )
    parser.add_argument(
        "--publication-results-dir",
        type=Path,
        default=None,
        help="If set, run generate_tables + generate_publication_figures from this directory.",
    )
    args = parser.parse_args()
    root = _repo_root()
    agg_dir = root / "docs" / "journal_aggregation"
    index_path = agg_dir / "journal_experiment_index.json"
    best_path = agg_dir / "journal_best_by_bucket.json"
    inv_path = root / "docs" / "journal_experiment_inventory.json"

    if args.run_harvest:
        cmd = [sys.executable, str(root / "scripts" / "journal_aggregate_experiment_results.py")]
        if args.with_plots:
            cmd.append("--with-plots")
        _run(cmd, root)

    if not index_path.is_file() and not args.run_harvest:
        _run([sys.executable, str(root / "scripts" / "journal_aggregate_experiment_results.py")], root)

    idx = _load_json(index_path)
    if not isinstance(idx, list):
        idx = []

    fig_script = root / "scripts" / "journal_build_comparative_figures.py"
    _run([sys.executable, str(fig_script), "--index", str(index_path)], root)

    if args.publication_results_dir and args.publication_results_dir.is_dir():
        src = root / "src"
        tables_out = root / "docs" / "journal_submission_tables"
        fig_out = root / "docs" / "journal_submission_figures" / "publication_bundle"
        _run(
            [
                sys.executable,
                "-m",
                "experiments.generate_tables",
                "--results_dir",
                str(args.publication_results_dir),
                "--output_dir",
                str(tables_out),
            ],
            src,
        )
        _run(
            [
                sys.executable,
                "-m",
                "experiments.generate_publication_figures",
                "--results_dir",
                str(args.publication_results_dir),
                "--output_dir",
                str(fig_out),
            ],
            src,
        )

    best = _load_json(best_path)
    if not isinstance(best, dict):
        best = {}

    out = args.output or (root / "docs" / "Journal_Full_Comparative_Study.docx")
    build_docx(out, index=idx, best=best, inventory_path=inv_path)


if __name__ == "__main__":
    main()
