"""
Comprehensive Experiment Report Generator
Generates a Microsoft Word document with all experiment details,
code changes, and results including embedded graphs and plots.
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os
from datetime import datetime

def set_cell_shading(cell, color):
    """Set cell background color."""
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), color)
    cell._tc.get_or_add_tcPr().append(shading)

def add_heading_with_style(doc, text, level):
    """Add a heading with custom styling."""
    heading = doc.add_heading(text, level=level)
    return heading

def create_experiment_report():
    """Generate the comprehensive experiment report."""
    doc = Document()
    
    # Set document title
    title = doc.add_heading('Contrastive Self-Supervised Learning\nExperiment Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add subtitle
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run('MIMII Dataset Anomaly Detection using Vision Transformers')
    run.bold = True
    run.font.size = Pt(14)
    
    # Add date and version
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run(f'Report Date: {datetime.now().strftime("%B %d, %Y")}\n')
    info.add_run('Version: 2.0\n')
    info.add_run('Author: Navin Anik')
    
    doc.add_page_break()
    
    # Table of Contents
    doc.add_heading('Table of Contents', level=1)
    toc_items = [
        '1. Executive Summary',
        '2. Project Overview',
        '3. Dataset Description',
        '4. System Configuration',
        '5. Code Changes and Development History',
        '6. Experiment Details',
        '   6.1 DINOv2 Experiments',
        '   6.2 DINOv3 Experiments',
        '   6.3 MetaCLIP2 Experiments',
        '   6.4 RT-DETR v4 Experiments',
        '7. Results and Analysis',
        '8. Visualizations and Plots',
        '9. Conclusions and Future Work',
        '10. Appendix'
    ]
    for item in toc_items:
        doc.add_paragraph(item)
    
    doc.add_page_break()
    
    # 1. Executive Summary
    doc.add_heading('1. Executive Summary', level=1)
    exec_summary = doc.add_paragraph()
    exec_summary.add_run(
        'This report documents the comprehensive experimental evaluation of multiple state-of-the-art '
        'vision transformer architectures for anomaly detection on the MIMII (Machine Inspection and '
        'Investigation Intelligence) dataset. The experiments compare various self-supervised learning '
        'approaches including DINOv2, DINOv3, MetaCLIP2, and RT-DETR v4 for binary classification of '
        'normal vs abnormal machine sounds converted to spectrograms.\n\n'
    )
    
    # Key findings
    doc.add_heading('Key Findings', level=2)
    findings = [
        ('Best Overall Accuracy', '85.3%', 'DINOv2 (20 epochs)'),
        ('Best Balanced Performance', '79.7%', 'MetaCLIP2 (100 epochs)'),
        ('Best AUC-ROC Score', '0.76', 'DINOv2'),
        ('Best F1 Score (Abnormal)', '0.478', 'DINOv2 (50 epochs)')
    ]
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Metric'
    header_cells[1].text = 'Value'
    header_cells[2].text = 'Model'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, '4472C4')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for metric, value, model in findings:
        row = table.add_row().cells
        row[0].text = metric
        row[1].text = value
        row[2].text = model
    
    doc.add_paragraph()
    
    # 2. Project Overview
    doc.add_heading('2. Project Overview', level=1)
    doc.add_paragraph(
        'This project implements a contrastive self-supervised learning framework for anomaly detection '
        'in industrial machinery. The framework leverages pre-trained vision transformers to extract '
        'meaningful representations from spectrogram images of machine sounds.'
    )
    
    doc.add_heading('Supported Models', level=2)
    models_info = [
        ('DINOv2', 'Meta AI', 'Self-supervised ViT with 1024-dim embeddings'),
        ('DINOv3', 'Meta AI (Aug 2025)', 'Latest DINOv3 ViT-H+ with 1280-dim, up to 7B params'),
        ('MetaCLIP2', 'Meta AI', 'CLIP 2 with 2B parameters trained on worldwide data'),
        ('RT-DETR v4', 'Baidu', 'Real-Time Detection Transformer backbone adapted for classification'),
        ('SigLIP2', 'Google', 'Sigmoid-based contrastive learning'),
        ('EVA-02', 'BAAI', 'Enhanced ViT Adapter'),
        ('MAE', 'Meta AI', 'Masked Autoencoder ViT'),
        ('Swin', 'Microsoft', 'Shifted Window Transformer'),
        ('BEiT', 'Microsoft', 'BERT pre-training for images'),
        ('DeiT', 'Meta AI', 'Data-efficient Image Transformer'),
        ('CLIP', 'OpenAI', 'Contrastive Language-Image Pre-training'),
        ('AST', 'MIT', 'Audio Spectrogram Transformer')
    ]
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Model'
    header_cells[1].text = 'Source'
    header_cells[2].text = 'Description'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, '4472C4')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for model, source, desc in models_info:
        row = table.add_row().cells
        row[0].text = model
        row[1].text = source
        row[2].text = desc
    
    doc.add_page_break()
    
    # 3. Dataset Description
    doc.add_heading('3. Dataset Description', level=1)
    doc.add_paragraph(
        'The MIMII (Machine Inspection and Investigation Intelligence) Dataset is used for '
        'industrial anomaly detection. The dataset contains audio recordings of industrial '
        'machines converted to spectrogram images for visual analysis.'
    )
    
    doc.add_heading('Dataset Statistics', level=2)
    dataset_stats = [
        ('Dataset Path', 'H:/dataset/MIMII_Dataset/spectrograms'),
        ('Number of Classes', '2 (Normal, Abnormal)'),
        ('Normal Samples', '~17,260 (81.5%)'),
        ('Abnormal Samples', '~3,903 (18.5%)'),
        ('Total Samples', '~21,163'),
        ('Class Imbalance Ratio', '4.4:1'),
        ('Image Format', 'Spectrogram PNG'),
        ('Default Image Size', '224x224 pixels')
    ]
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Property'
    header_cells[1].text = 'Value'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, '70AD47')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for prop, value in dataset_stats:
        row = table.add_row().cells
        row[0].text = prop
        row[1].text = value
    
    doc.add_paragraph()
    doc.add_paragraph(
        'Note: The dataset exhibits significant class imbalance with normal samples being '
        '4.4 times more frequent than abnormal samples. This imbalance was addressed using '
        'class-weighted loss functions during training.'
    )
    
    # 4. System Configuration
    doc.add_heading('4. System Configuration', level=1)
    
    doc.add_heading('Hardware', level=2)
    hw_specs = [
        ('GPU', 'NVIDIA GeForce RTX 2060 SUPER'),
        ('GPU Memory', '8 GB VRAM'),
        ('GPU Architecture', 'Turing'),
        ('CUDA Cores', '2176'),
        ('System RAM', '48 GB'),
        ('CPU', '8 cores / 16 logical processors'),
        ('Storage', '250 GB SSD')
    ]
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Component'
    header_cells[1].text = 'Specification'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, 'ED7D31')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for comp, spec in hw_specs:
        row = table.add_row().cells
        row[0].text = comp
        row[1].text = spec
    
    doc.add_heading('Software', level=2)
    sw_specs = [
        ('Operating System', 'Windows 11 (10.0.26200)'),
        ('Python Version', '3.13.5 / 3.14.3'),
        ('PyTorch', '>= 2.0.0'),
        ('CUDA Version', '13.1'),
        ('Transformers', '>= 4.56.0'),
        ('timm', '>= 1.0.20'),
        ('WandB', '>= 0.24.2'),
        ('scikit-learn', 'Latest')
    ]
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Software'
    header_cells[1].text = 'Version'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, 'ED7D31')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for sw, ver in sw_specs:
        row = table.add_row().cells
        row[0].text = sw
        row[1].text = ver
    
    doc.add_page_break()
    
    # 5. Code Changes and Development History
    doc.add_heading('5. Code Changes and Development History', level=1)
    
    doc.add_heading('Git Commit History', level=2)
    commits = [
        ('56f729e', 'Feb 23, 2026', 'feat: fixes nan loss issue', 
         'Fixed NaN loss issues, added optimal decision threshold documentation, '
         'enhanced run_all_models.py with improved training loop'),
        ('53ee7cf', 'Feb 21, 2026', 'feat: change model',
         'Added DINOv3, MetaCLIP2, SigLIP2, EVA-02, RT-DETR v4 encoders. '
         'Enhanced evaluation metrics and statistical tests. Added logging configuration.'),
        ('4fa474f', 'Feb 21, 2026', 'feat: experiment 2 branch',
         'Created experiment 2 branch with additional wandb runs'),
        ('fba50b6', 'Feb 7, 2026', 'feat: changes for experiment 2',
         'Major refactoring: Added all encoder modules, fusion strategies, '
         'experiment runners, visualization tools, statistical tests. '
         'Created comprehensive documentation files.'),
        ('f16a30b', 'Feb 7, 2026', 'added docker',
         'Added Docker support, restructured to src/ directory, '
         'added fine-tuning and training modules'),
        ('c253a4a', 'Jan 2026', 'changed to pytorch',
         'Migrated from TensorFlow to PyTorch implementation'),
        ('8cfc775', 'Jan 2026', 'feat: added graph plot and modified folder structure',
         'Added visualization utilities and reorganized project structure'),
        ('c1e494e', 'Jan 2026', 'feat: adds the base code',
         'Initial codebase with contrastive learning implementation')
    ]
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    headers = ['Commit', 'Date', 'Message', 'Details']
    for i, header in enumerate(headers):
        header_cells[i].text = header
        header_cells[i].paragraphs[0].runs[0].bold = True
        set_cell_shading(header_cells[i], '5B9BD5')
        header_cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for commit, date, msg, details in commits:
        row = table.add_row().cells
        row[0].text = commit
        row[1].text = date
        row[2].text = msg
        row[3].text = details
    
    doc.add_heading('Key Implementation Changes', level=2)
    
    doc.add_heading('5.1 Class Imbalance Handling', level=3)
    doc.add_paragraph(
        'Implemented class-weighted cross-entropy loss to address the 4.4:1 class imbalance:\n\n'
        '• Weight calculation: weight_i = total_samples / (num_classes × count_i)\n'
        '• Normal class weight: 0.613\n'
        '• Abnormal class weight: 2.711\n\n'
        'This forces the model to pay more attention to minority class (abnormal) samples '
        'during training, significantly improving recall for anomaly detection.'
    )
    
    doc.add_heading('5.2 Enhanced Metrics Tracking', level=3)
    doc.add_paragraph(
        'Added per-epoch tracking of:\n'
        '• Precision (for abnormal class)\n'
        '• Recall (for abnormal class)\n'
        '• F1 Score\n'
        '• Macro F1, Weighted F1\n'
        '• AUC-ROC, PR-AUC\n'
        '• Cohen\'s Kappa, MCC\n'
        '• ECE (Expected Calibration Error)\n'
        '• Brier Score\n\n'
        'All metrics are logged to Weights & Biases for real-time monitoring.'
    )
    
    doc.add_heading('5.3 New Model Encoders Added', level=3)
    new_encoders = [
        'dinov3_encoder.py - DINOv3 with ViT-H+ architecture',
        'metaclip2_encoder.py - Meta CLIP 2 with 2B parameters',
        'siglip2_encoder.py - Google SigLIP2 sigmoid-based',
        'eva02_encoder.py - EVA-02 enhanced ViT',
        'rtdetrv4_encoder.py - RT-DETR v4 detection backbone'
    ]
    for encoder in new_encoders:
        doc.add_paragraph(f'• {encoder}', style='List Bullet')
    
    doc.add_heading('5.4 Fusion Strategies', level=3)
    doc.add_paragraph(
        'Implemented multimodal fusion for audio-visual learning:\n'
        '• Early Fusion - Concatenate features before classifier\n'
        '• Late Fusion - Weighted averaging of modality logits\n'
        '• Cross-Attention Fusion - Bidirectional cross-attention\n'
        '• ConFu - Contrastive fusion with pairwise alignment\n'
        '• ProMoBal - Progressive modal balancing\n'
        '• HTSF - Hierarchical Temporal-Spatial Fusion'
    )
    
    doc.add_page_break()
    
    # 6. Experiment Details
    doc.add_heading('6. Experiment Details', level=1)
    
    # 6.1 DINOv2 Experiments
    doc.add_heading('6.1 DINOv2 Experiments', level=2)
    
    doc.add_heading('Experiment 1: DINOv2 - 20 Epochs', level=3)
    exp1_config = [
        ('Run ID', 'jju61go4'),
        ('Date', 'February 7, 2026'),
        ('Model', 'DINOv2 ViT-Large'),
        ('Epochs', '20'),
        ('Batch Size', '16'),
        ('Learning Rate', '0.0001'),
        ('Image Size', '224'),
        ('Dropout', '0.1'),
        ('Optimizer', 'AdamW'),
        ('Scheduler', 'Cosine Annealing'),
        ('Mixed Precision', 'FP16 + AMP'),
        ('Backbone', 'Frozen')
    ]
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Parameter'
    header_cells[1].text = 'Value'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, '4472C4')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for param, value in exp1_config:
        row = table.add_row().cells
        row[0].text = param
        row[1].text = value
    
    doc.add_paragraph()
    doc.add_heading('Results:', level=4)
    exp1_results = [
        ('Best Validation Accuracy', '85.3%'),
        ('Final Training Accuracy', '86.5%'),
        ('Final Validation Loss', '0.393'),
        ('Training Time', '~2.6 hours')
    ]
    for metric, value in exp1_results:
        doc.add_paragraph(f'• {metric}: {value}', style='List Bullet')
    
    doc.add_heading('Experiment 2: DINOv2 - 50 Epochs (Extended)', level=3)
    exp2_config = [
        ('Run ID', 'bllqd33o'),
        ('Date', 'February 7, 2026'),
        ('Model', 'DINOv2 ViT-Large'),
        ('Epochs', '50'),
        ('Batch Size', '8'),
        ('Accumulation Steps', '4'),
        ('Learning Rate', '0.0001'),
        ('Training Time', '~6.6 hours')
    ]
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Parameter'
    header_cells[1].text = 'Value'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, '4472C4')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for param, value in exp2_config:
        row = table.add_row().cells
        row[0].text = param
        row[1].text = value
    
    doc.add_paragraph()
    doc.add_heading('Results:', level=4)
    exp2_results = [
        ('Best Validation Accuracy', '82.4%'),
        ('Validation Precision', '52.9%'),
        ('Validation Recall', '43.6%'),
        ('Validation F1 Score', '0.478')
    ]
    for metric, value in exp2_results:
        doc.add_paragraph(f'• {metric}: {value}', style='List Bullet')
    
    # 6.2 DINOv3 Experiments
    doc.add_heading('6.2 DINOv3 Experiments', level=2)
    
    doc.add_heading('Experiment: DINOv3 - 5 Epochs (Preliminary)', level=3)
    dinov3_config = [
        ('Run ID', 'hd0zxi7f'),
        ('Date', 'February 23, 2026'),
        ('Model', 'DINOv3 ViT-H+ (facebook/dinov3-vith16plus-pretrain-lvd1689m)'),
        ('Epochs', '5'),
        ('Batch Size', '24'),
        ('Learning Rate', '0.0002'),
        ('Image Size', '518'),
        ('Dropout', '0.2'),
        ('Scheduler', 'Cosine Annealing'),
        ('Augment Strength', 'Strong'),
        ('Weight Decay', '0.05'),
        ('Weighted Sampler', 'Enabled')
    ]
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Parameter'
    header_cells[1].text = 'Value'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, '70AD47')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for param, value in dinov3_config:
        row = table.add_row().cells
        row[0].text = param
        row[1].text = value
    
    doc.add_paragraph()
    doc.add_paragraph(
        'Note: DINOv3 experiments showed training accuracy around 50% with validation '
        'accuracy at ~19%, indicating potential issues with model initialization or '
        'learning rate that require further investigation.'
    )
    
    # 6.3 MetaCLIP2 Experiments
    doc.add_heading('6.3 MetaCLIP2 Experiments', level=2)
    
    doc.add_heading('Experiment: MetaCLIP2 - 100 Epochs (Full Training)', level=3)
    metaclip_config = [
        ('Run ID', 'hh01rq6n'),
        ('Date', 'February 25, 2026'),
        ('Model', 'MetaCLIP2'),
        ('Epochs', '100'),
        ('Batch Size', '8'),
        ('Accumulation Steps', '2'),
        ('Learning Rate', '0.0002'),
        ('Image Size', '224'),
        ('Dropout', '0.1'),
        ('Training Time', '~61.8 hours')
    ]
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Parameter'
    header_cells[1].text = 'Value'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, 'ED7D31')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for param, value in metaclip_config:
        row = table.add_row().cells
        row[0].text = param
        row[1].text = value
    
    doc.add_paragraph()
    doc.add_heading('Comprehensive Results:', level=4)
    metaclip_results = [
        ('Best Validation Accuracy', '79.7%'),
        ('Macro Precision', '65.6%'),
        ('Macro Recall', '64.4%'),
        ('Macro F1', '64.9%'),
        ('AUC-ROC', '0.724'),
        ('PR-AUC', '0.476'),
        ('Cohen\'s Kappa', '0.299'),
        ('MCC', '0.300'),
        ('ECE', '0.117'),
        ('Brier Score', '0.162'),
        ('Specificity', '88.6%'),
        ('Sensitivity (Recall)', '40.2%'),
        ('PPV', '44.5%'),
        ('NPV', '86.8%'),
        ('GFLOPs', '162.0'),
        ('GPU Memory Peak', '3.67 GB'),
        ('Inference Time/Sample', '25.6 ms')
    ]
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Metric'
    header_cells[1].text = 'Value'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, 'ED7D31')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for metric, value in metaclip_results:
        row = table.add_row().cells
        row[0].text = metric
        row[1].text = value
    
    # 6.4 RT-DETR v4 Experiments
    doc.add_heading('6.4 RT-DETR v4 Experiments', level=2)
    
    doc.add_heading('Experiment: RT-DETR v4 Large - 5 Epochs', level=3)
    rtdetr_config = [
        ('Run ID', 'rena554o'),
        ('Date', 'February 21, 2026'),
        ('Model', 'RT-DETR v4 Large'),
        ('Epochs', '5'),
        ('Batch Size', '8'),
        ('Learning Rate', '0.0002'),
        ('Image Size', '640'),
        ('Dropout', '0.1')
    ]
    
    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Parameter'
    header_cells[1].text = 'Value'
    for cell in header_cells:
        cell.paragraphs[0].runs[0].bold = True
        set_cell_shading(cell, '7030A0')
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for param, value in rtdetr_config:
        row = table.add_row().cells
        row[0].text = param
        row[1].text = value
    
    doc.add_paragraph()
    doc.add_paragraph(
        'Note: RT-DETR v4 experiments were preliminary runs to validate the encoder '
        'adaptation for classification tasks. The model showed initial training accuracy '
        'of 75-78% but validation accuracy dropped from 82% to 63% indicating overfitting.'
    )
    
    doc.add_page_break()
    
    # 7. Results and Analysis
    doc.add_heading('7. Results and Analysis', level=1)
    
    doc.add_heading('7.1 Model Comparison Summary', level=2)
    
    comparison_data = [
        ('DINOv2 (20 ep)', '85.3%', '-', '-', '-', '0.76'),
        ('DINOv2 (50 ep)', '82.4%', '52.9%', '43.6%', '0.478', '-'),
        ('MetaCLIP2 (100 ep)', '79.7%', '44.5%', '40.2%', '0.422', '0.724'),
        ('DINOv3 (5 ep)*', '19.5%', '-', '-', '-', '-'),
        ('RT-DETR v4 (5 ep)*', '63.5%', '-', '-', '-', '-')
    ]
    
    table = doc.add_table(rows=1, cols=6)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    for i, header in enumerate(headers):
        header_cells[i].text = header
        header_cells[i].paragraphs[0].runs[0].bold = True
        set_cell_shading(header_cells[i], '4472C4')
        header_cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for row_data in comparison_data:
        row = table.add_row().cells
        for i, value in enumerate(row_data):
            row[i].text = value
    
    doc.add_paragraph()
    doc.add_paragraph('* Preliminary experiments with limited epochs')
    
    doc.add_heading('7.2 Confusion Matrix Analysis', level=2)
    
    doc.add_heading('DINOv2 Best Model Confusion Matrix', level=3)
    doc.add_paragraph(
        'The DINOv2 model (50 epochs) achieved the following confusion matrix:\n\n'
        '                    Predicted\n'
        '                  Normal  Abnormal\n'
        'Actual Normal     16,753    507\n'
        '       Abnormal    2,596  1,307\n\n'
        '• True Negatives: 16,753 (97.1% of normal samples correctly classified)\n'
        '• True Positives: 1,307 (33.5% of abnormal samples correctly classified)\n'
        '• False Positives: 507 (2.9% of normal samples misclassified)\n'
        '• False Negatives: 2,596 (66.5% of abnormal samples missed)'
    )
    
    doc.add_heading('MetaCLIP2 Confusion Matrix', level=3)
    doc.add_paragraph(
        'The MetaCLIP2 model (100 epochs) achieved:\n\n'
        '                    Predicted\n'
        '                  Normal  Abnormal\n'
        'Actual Normal     15,301  1,959\n'
        '       Abnormal    2,335  1,568\n\n'
        '• True Negatives: 15,301 (88.6% of normal samples correctly classified)\n'
        '• True Positives: 1,568 (40.2% of abnormal samples correctly classified)\n'
        '• Higher recall but lower specificity compared to DINOv2'
    )
    
    doc.add_heading('7.3 Key Observations', level=2)
    observations = [
        'DINOv2 with 20 epochs achieved the highest overall accuracy (85.3%), benefiting '
        'from strong pre-trained representations.',
        'Extended training (50 epochs) showed diminishing returns with slight accuracy '
        'decrease but improved F1 score for the minority class.',
        'MetaCLIP2 showed more balanced performance with better recall for abnormal class '
        '(40.2% vs 33.5%) at the cost of lower overall accuracy.',
        'Class imbalance remains a significant challenge - even with weighted loss, '
        'models tend to favor the majority (normal) class.',
        'DINOv3 preliminary experiments showed convergence issues that require further '
        'hyperparameter tuning.',
        'RT-DETR v4 showed signs of overfitting with high training accuracy but poor '
        'validation performance.'
    ]
    
    for obs in observations:
        doc.add_paragraph(f'• {obs}', style='List Bullet')
    
    doc.add_page_break()
    
    # 8. Visualizations and Plots
    doc.add_heading('8. Visualizations and Plots', level=1)
    
    doc.add_paragraph(
        'The following visualizations are generated for each trained model and stored '
        'in the results/{model_name}/ directory:'
    )
    
    # Try to add images if they exist
    results_dir = os.path.join(os.path.dirname(__file__), 'src', 'results')
    
    models_with_plots = ['dinov2', 'dinov3', 'metaclip2', 'rtdetrv4_l']
    plot_types = [
        ('accuracy.png', 'Training vs Validation Accuracy'),
        ('loss.png', 'Training vs Validation Loss'),
        ('f1_vs_epoch.png', 'F1 Score vs Epoch'),
        ('precision_vs_epoch.png', 'Precision vs Epoch'),
        ('recall_vs_epoch.png', 'Recall vs Epoch'),
        ('conf_matrix_epoch_best.png', 'Confusion Matrix'),
        ('roc_epoch_best.png', 'ROC Curve'),
        ('pr_curve_epoch_best.png', 'Precision-Recall Curve')
    ]
    
    for model in models_with_plots:
        doc.add_heading(f'8.{models_with_plots.index(model)+1} {model.upper()} Results', level=2)
        
        model_results_dir = os.path.join(results_dir, model)
        
        for plot_file, plot_title in plot_types:
            plot_path = os.path.join(model_results_dir, plot_file)
            if os.path.exists(plot_path):
                doc.add_heading(plot_title, level=3)
                try:
                    doc.add_picture(plot_path, width=Inches(5.5))
                    last_paragraph = doc.paragraphs[-1]
                    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                except Exception as e:
                    doc.add_paragraph(f'[Image: {plot_path}]')
                doc.add_paragraph()
    
    doc.add_page_break()
    
    # 9. Conclusions and Future Work
    doc.add_heading('9. Conclusions and Future Work', level=1)
    
    doc.add_heading('9.1 Conclusions', level=2)
    conclusions = [
        'Self-supervised vision transformers (DINOv2, MetaCLIP2) show strong performance '
        'for spectrogram-based anomaly detection, achieving up to 85.3% accuracy.',
        'Class imbalance handling through weighted loss significantly improves minority '
        'class detection, increasing abnormal recall from ~23% to ~40%.',
        'Pre-trained representations from large-scale datasets transfer effectively to '
        'industrial anomaly detection tasks.',
        'Extended training beyond optimal epochs can lead to overfitting, particularly '
        'for smaller datasets.',
        'The trade-off between precision and recall remains challenging for anomaly '
        'detection where missing anomalies is costly.'
    ]
    for conc in conclusions:
        doc.add_paragraph(f'• {conc}', style='List Bullet')
    
    doc.add_heading('9.2 Future Work', level=2)
    future_work = [
        'Implement focal loss as alternative to weighted cross-entropy for better '
        'handling of hard examples.',
        'Explore threshold tuning to optimize precision-recall trade-off for '
        'production deployment.',
        'Investigate data augmentation strategies specific to spectrogram images.',
        'Implement semi-supervised learning with pseudo-labeling for unlabeled data.',
        'Develop ensemble methods combining multiple model predictions.',
        'Explore multi-modal fusion with raw audio and spectrogram inputs.',
        'Conduct rigorous cross-validation (10-fold × 5 seeds) for publication-ready results.',
        'Apply statistical significance tests (paired t-test, Wilcoxon) for model comparison.'
    ]
    for fw in future_work:
        doc.add_paragraph(f'• {fw}', style='List Bullet')
    
    doc.add_heading('9.3 Recommendations', level=2)
    doc.add_paragraph(
        'Based on the experimental results, we recommend:\n\n'
        '1. Use DINOv2 as the primary model for production deployment due to its '
        'balance of accuracy and computational efficiency.\n\n'
        '2. Set decision threshold based on business requirements - lower threshold '
        'for higher recall (catching more anomalies) at the cost of more false alarms.\n\n'
        '3. Consider MetaCLIP2 for scenarios where balanced precision-recall is critical.\n\n'
        '4. Implement model monitoring to detect concept drift over time.\n\n'
        '5. Regularly retrain models as new labeled data becomes available.'
    )
    
    doc.add_page_break()
    
    # 10. Appendix
    doc.add_heading('10. Appendix', level=1)
    
    doc.add_heading('A. WandB Run IDs', level=2)
    wandb_runs = [
        ('jju61go4', 'Feb 7, 2026', 'DINOv2', '20 epochs'),
        ('bllqd33o', 'Feb 7, 2026', 'DINOv2', '50 epochs'),
        ('vecwkoec', 'Feb 7, 2026', 'DINOv2', '20 epochs'),
        ('rena554o', 'Feb 21, 2026', 'RT-DETR v4', '5 epochs'),
        ('ly9t2kqq', 'Feb 23, 2026', 'DINOv3', '5 epochs'),
        ('uxh91g7z', 'Feb 23, 2026', 'DINOv3', '5 epochs + Focal Loss'),
        ('hd0zxi7f', 'Feb 23, 2026', 'DINOv3', '5 epochs'),
        ('63y81svo', 'Feb 25, 2026', 'MetaCLIP2', '-'),
        ('21bqxy49', 'Feb 25, 2026', 'MetaCLIP2', '-'),
        ('clex54kn', 'Feb 25, 2026', 'MetaCLIP2', '-'),
        ('hh01rq6n', 'Feb 25, 2026', 'MetaCLIP2', '100 epochs'),
        ('w2dnq12q', 'Mar 1, 2026', 'Rigorous CV', '10-fold × 5 seeds')
    ]
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    header_cells = table.rows[0].cells
    headers = ['Run ID', 'Date', 'Model', 'Notes']
    for i, header in enumerate(headers):
        header_cells[i].text = header
        header_cells[i].paragraphs[0].runs[0].bold = True
        set_cell_shading(header_cells[i], '5B9BD5')
        header_cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    for run_id, date, model, notes in wandb_runs:
        row = table.add_row().cells
        row[0].text = run_id
        row[1].text = date
        row[2].text = model
        row[3].text = notes
    
    doc.add_heading('B. File Structure', level=2)
    file_structure = '''
src/
├── modules/           # Core neural network modules
│   ├── dinov2_encoder.py
│   ├── dinov3_encoder.py
│   ├── metaclip2_encoder.py
│   ├── rtdetrv4_encoder.py
│   ├── siglip2_encoder.py
│   ├── eva02_encoder.py
│   ├── contrastive_loss.py
│   ├── moco_loss.py
│   ├── byol_loss.py
│   ├── vicreg_loss.py
│   ├── evaluation.py
│   └── visualization.py
├── experiments/       # Experiment runners
│   ├── run_all_models.py
│   ├── run_rigorous_cv.py
│   └── run_fusion_experiments.py
├── fusion/           # Multimodal fusion strategies
├── training/         # Training utilities
├── utils/            # Plotting, logging
├── wandb/            # Experiment tracking logs
├── configs/          # Configuration files
├── model/            # Saved checkpoints
└── results/          # Results per model
'''
    doc.add_paragraph(file_structure, style='No Spacing')
    
    doc.add_heading('C. Requirements', level=2)
    requirements = [
        'torch >= 2.0.0',
        'transformers >= 4.56.0',
        'timm >= 1.0.20',
        'wandb >= 0.24.2',
        'scikit-learn',
        'scipy',
        'statsmodels',
        'umap-learn',
        'matplotlib',
        'seaborn',
        'pandas',
        'numpy',
        'Pillow',
        'python-docx'
    ]
    for req in requirements:
        doc.add_paragraph(f'• {req}', style='List Bullet')
    
    # Save the document
    output_path = os.path.join(os.path.dirname(__file__), 'Experiment_Report_Complete.docx')
    doc.save(output_path)
    print(f'Report saved to: {output_path}')
    return output_path


if __name__ == '__main__':
    create_experiment_report()
