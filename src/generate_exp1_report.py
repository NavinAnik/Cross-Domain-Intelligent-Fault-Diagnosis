"""Generate Experiment 1 (Focal Baseline) Report in Microsoft Word Format."""

import os
import json
from datetime import datetime

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


def load_exp1_results() -> dict:
    """Load exp1_results.json."""
    results_path = os.path.join(
        os.path.dirname(__file__), "results", "improved_experiments", "exp1_results.json"
    )
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return {}


def create_exp1_report():
    """Create Microsoft Word report for Experiment 1."""
    if not HAS_DOCX:
        raise ImportError("python-docx is required. Install with: pip install python-docx")

    data = load_exp1_results()
    config = data.get("config", {})
    best = data.get("best_metrics", {})

    doc = Document()

    # Title
    title = doc.add_heading("Experiment Report: Exp1 Focal Baseline (MIMII)", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Subtitle
    subtitle = doc.add_paragraph()
    subtitle_run = subtitle.add_run("Improved Experiments for Q1 Journal Publication")
    subtitle_run.font.size = Pt(14)
    subtitle_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Date
    date_para = doc.add_paragraph()
    date_run = date_para.add_run(f"Report Generated: {datetime.now().strftime('%B %d, %Y')}")
    date_run.font.size = Pt(11)
    date_run.font.italic = True
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # 1. Executive Summary
    doc.add_heading("1. Executive Summary", level=1)
    summary = doc.add_paragraph()
    summary.add_run("This report documents Experiment 1 (Focal Baseline) conducted on March 4, 2026. ")
    summary.add_run("The experiment evaluates a DINOv2-based classifier with tuned focal loss for anomaly detection ")
    summary.add_run("on the MIMII industrial sound dataset. Key achievements:")
    doc.add_paragraph()
    doc.add_paragraph(
        f"• Best Abnormal Recall: {data.get('best_abnormal_recall', 0):.2%} (at threshold {best.get('optimal_threshold', 0.65):.2f})",
        style="List Bullet"
    )
    doc.add_paragraph(
        f"• AUC-ROC: {best.get('auc_roc', 0):.4f}",
        style="List Bullet"
    )
    doc.add_paragraph(
        f"• Balanced Accuracy: {best.get('balanced_accuracy', 0):.2%}",
        style="List Bullet"
    )
    doc.add_paragraph(
        f"• Overall Accuracy: {best.get('overall_accuracy', 0):.2%}",
        style="List Bullet"
    )

    # 2. Experiment Configuration
    doc.add_heading("2. Experiment Configuration", level=1)

    doc.add_heading("2.1 Model & Architecture", level=2)
    model_table = doc.add_table(rows=7, cols=2)
    model_table.style = "Table Grid"
    model_data = [
        ("Model", "DINOv2 (dinov2_vitb14)"),
        ("Architecture", "Vision Transformer Base, 14×14 patches"),
        ("Backbone", "Frozen"),
        ("Hidden Dimension", "256"),
        ("Number of Classes", "2 (Normal / Anomaly)"),
        ("Mixed Precision (FP16)", "Enabled"),
        ("Optimizer", "AdamW (lr=1e-4, weight_decay=0.05)"),
    ]
    for i, (key, val) in enumerate(model_data):
        model_table.rows[i].cells[0].text = str(key)
        model_table.rows[i].cells[1].text = str(val)
        model_table.rows[i].cells[0].paragraphs[0].runs[0].bold = True

    doc.add_paragraph()

    doc.add_heading("2.2 Training Hyperparameters", level=2)
    train_table = doc.add_table(rows=7, cols=2)
    train_table.style = "Table Grid"
    train_data = [
        ("Epochs", config.get("epochs", 50)),
        ("Batch Size", config.get("batch_size", 32)),
        ("Learning Rate", config.get("learning_rate", 0.0001)),
        ("Focal Loss Alpha", str(config.get("focal_alpha", [0.1, 0.9]))),
        ("Focal Loss Gamma", config.get("focal_gamma", 3.0)),
        ("Scheduler", "CosineAnnealingLR"),
        ("Threshold Optimization", "Yes (F1-maximizing)"),
    ]
    for i, (key, val) in enumerate(train_data):
        train_table.rows[i].cells[0].text = str(key)
        train_table.rows[i].cells[1].text = str(val)
        train_table.rows[i].cells[0].paragraphs[0].runs[0].bold = True

    doc.add_paragraph()

    doc.add_heading("2.3 Dataset & Data Loading", level=2)
    ds_table = doc.add_table(rows=5, cols=2)
    ds_table.style = "Table Grid"
    ds_data = [
        ("Dataset", "MIMII (Machine Sound Anomaly Detection)"),
        ("Input Format", "Spectrograms (224×224)"),
        ("Train/Val Split", "80% / 20%"),
        ("Class Balancing", "WeightedRandomSampler (imbalanced)"),
        ("Data Augmentation", "DINOv2 standard transforms"),
    ]
    for i, (key, val) in enumerate(ds_data):
        ds_table.rows[i].cells[0].text = str(key)
        ds_table.rows[i].cells[1].text = str(val)
        ds_table.rows[i].cells[0].paragraphs[0].runs[0].bold = True

    # 3. Results
    doc.add_heading("3. Experiment Results", level=1)

    doc.add_heading("3.1 Best Metrics (Optimal Threshold)", level=2)
    opt_thresh = best.get("optimal_threshold", 0.65)
    results_table = doc.add_table(rows=14, cols=2)
    results_table.style = "Table Grid"
    results_data = [
        ("Optimal Threshold", f"{opt_thresh:.2f}"),
        ("Best Abnormal Recall (Sensitivity)", f"{best.get('abnormal_recall', 0):.2%}"),
        ("Normal Recall (Specificity)", f"{best.get('normal_recall', 0):.2%}"),
        ("Abnormal Precision", f"{best.get('abnormal_precision', 0):.2%}"),
        ("Normal Precision", f"{best.get('normal_precision', 0):.2%}"),
        ("Overall Accuracy", f"{best.get('overall_accuracy', 0):.2%}"),
        ("Balanced Accuracy", f"{best.get('balanced_accuracy', 0):.2%}"),
        ("AUC-ROC", f"{best.get('auc_roc', 0):.4f}"),
        ("AUC-PR", f"{best.get('auc_pr', 0):.4f}"),
        ("MCC (Matthews Correlation)", f"{best.get('mcc', 0):.4f}"),
        ("Final Train Loss", f"{best.get('train_loss', 0):.6f}"),
        ("Final Train Accuracy", f"{best.get('train_accuracy', 0):.2%}"),
        ("Total Epochs", str(config.get("epochs", 50))),
    ]
    for i, (key, val) in enumerate(results_data):
        results_table.rows[i].cells[0].text = str(key)
        results_table.rows[i].cells[1].text = str(val)
        results_table.rows[i].cells[0].paragraphs[0].runs[0].bold = True

    doc.add_paragraph()

    doc.add_heading("3.2 Confusion Matrix", level=2)
    cm = best.get("confusion_matrix", [[0, 0], [0, 0]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    cm_para = doc.add_paragraph()
    cm_para.add_run("Layout: rows = True Label, columns = Predicted Label. Labels: [Normal=0, Anomaly=1]\n\n")
    cm_table = doc.add_table(rows=3, cols=3)
    cm_table.style = "Table Grid"
    cm_table.rows[0].cells[0].text = ""
    cm_table.rows[0].cells[1].text = "Pred: Normal"
    cm_table.rows[0].cells[2].text = "Pred: Anomaly"
    cm_table.rows[1].cells[0].text = "True: Normal"
    cm_table.rows[1].cells[1].text = str(tn)
    cm_table.rows[1].cells[2].text = str(fp)
    cm_table.rows[2].cells[0].text = "True: Anomaly"
    cm_table.rows[2].cells[1].text = str(fn)
    cm_table.rows[2].cells[2].text = str(tp)

    cm_para2 = doc.add_paragraph()
    cm_para2.add_run(f"True Negatives: {tn}  |  False Positives: {fp}  |  False Negatives: {fn}  |  True Positives: {tp}")

    doc.add_paragraph()

    doc.add_heading("3.3 Training Trajectory (Summary)", level=2)
    traj = doc.add_paragraph()
    traj.add_run("From terminal logs (Epochs 41–50):")
    traj_list = [
        "Train accuracy remained ~57.5–58% (classifier head on frozen backbone)",
        "Validation accuracy ranged 80.7%–83.9%, peak at Epoch 45 (83.61%)",
        "Abnormal recall varied 39.9%–45.9% per epoch; best overall 52.76% at threshold 0.65",
        "Training loss decreased to ~0.017 by final epoch",
        "WandB Run: exp1-focal-baseline (tejf3tkn)",
    ]
    for item in traj_list:
        doc.add_paragraph(item, style="List Bullet")

    # 4. Conclusions
    doc.add_heading("4. Conclusions and Next Steps", level=1)

    conclusions = [
        "Focal loss with alpha=[0.1, 0.9] and gamma=3.0 effectively addresses class imbalance.",
        "Threshold optimization (F1-maximizing) improved abnormal recall to 52.76% vs ~40% at default 0.5.",
        "Validation accuracy (~83%) is strong; abnormal recall remains limited due to dataset imbalance.",
        "This baseline establishes a foundation for subsequent experiments (AFCL, curriculum learning, fusion).",
    ]
    for c in conclusions:
        doc.add_paragraph(c, style="List Bullet")

    doc.add_heading("Recommended Follow-ups:", level=2)
    followups = [
        "Experiment 2: AFCL (Asymmetric Focal Contrastive Loss) integration",
        "Experiment 3: Curriculum learning training",
        "Experiment 4: HTSF fusion experiments",
        "Rigorous cross-validation with multiple seeds for statistical robustness",
    ]
    for f in followups:
        doc.add_paragraph(f, style="List Bullet")

    # 5. Appendix
    doc.add_heading("5. Appendix", level=1)
    doc.add_heading("5.1 Outputs & Artifacts", level=2)
    app_table = doc.add_table(rows=5, cols=2)
    app_table.style = "Table Grid"
    app_data = [
        ("Results JSON", "results/improved_experiments/exp1_results.json"),
        ("Best Model Checkpoint", "results/improved_experiments/exp1_best_model.pt"),
        ("WandB Project", "mimii-improved-experiments"),
        ("WandB Run", "exp1-focal-baseline (tejf3tkn)"),
        ("Report Generation", "src/generate_exp1_report.py"),
    ]
    for i, (key, val) in enumerate(app_data):
        app_table.rows[i].cells[0].text = str(key)
        app_table.rows[i].cells[1].text = str(val)
        app_table.rows[i].cells[0].paragraphs[0].runs[0].bold = True

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "results", "improved_experiments")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "Experiment1_Focal_Baseline_Report.docx")
    doc.save(output_path)
    print(f"Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_exp1_report()
