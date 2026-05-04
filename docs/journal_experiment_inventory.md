# Journal experiment inventory

This file mirrors [`journal_experiment_inventory.json`](journal_experiment_inventory.json) in prose form. It lists **every major training/evaluation entry point** in the repository, where outputs typically land, and which **novel or differentiated components** the Methods section of a comparative paper should cite.

## Novel / differentiated components (code)

| Theme | Module / path |
|--------|----------------|
| Asymmetric focal contrastive loss | `src/modules/afcl_loss.py` |
| Curriculum learning | `src/modules/curriculum_learning.py` |
| Spectrogram CNN, spatial attention, CNN–Transformer hybrid | `src/modules/spectrogram_cnn_models.py` |
| Spectrogram transforms, group-aware splits | `src/modules/spectrogram_fault_data.py` |
| Audio–vision fusion (incl. optional contrastive term) | `src/experiments/run_fusion_experiments.py` |
| Publication figure/table generators | `src/experiments/generate_publication_figures.py`, `generate_tables.py` |

## Experiment entry points

| ID | Line of work | Script | Typical artifacts |
|----|----------------|--------|---------------------|
| vision_run_all_models | Vision supervised zoo | `experiments/run_all_models.py` | Per-model checkpoints, eval under `results/` |
| vision_single_model_batch | Vision supervised zoo | `experiments/run_single_model_experiments.py` | `single_model_experiments_summary.json` |
| improved_exp_1_6 | Improved MIMII / DINOv2 | `experiments/run_improved_experiments.py` | `src/results/improved_experiments/exp*_results.json` |
| q1_pipeline | Q1 rigorous pipeline | `experiments/run_q1_experiments.py` | `results/q1_*/**` |
| fusion_multimodal | Fusion | `experiments/run_fusion_experiments.py` | `fusion_summary.json` |
| ablations_contrastive | Ablations | `experiments/run_ablations.py` | `results/ablation_*.json` |
| ablation_comparison_runner | Ablations | `experiments/run_ablation_comparison.py` | JSON + markdown tables |
| rigorous_cv | Cross-validation | `experiments/run_rigorous_cv.py` | `rigorous_cv_results.json` |
| cv_statistical_analysis | CV + stats | `experiments/run_cv_and_statistical_analysis.py` | `cv_results.json` |
| spectrogram_fault_train | Spectrogram images | `experiments/train_spectrogram_fault.py` | `results/spectrogram_fault/<run_id>/metrics.json` |
| spectrogram_ablation_grid | Spectrogram images | `experiments/run_spectrogram_ablation.py` | `ablation_results_*.csv` / `.json` |
| contrastive_mimii_train | Contrastive SSL (audio) | `train.py` | logs / wandb |
| main_pretrain | Contrastive SSL (image) | `main_pretrain.py` | logs |
| dinov2_main | DINOv2 line | `main_dinov2.py` | `results/**` |
| image_cv | Image CV | `experiments/run_image_cv.py` | fold outputs |
| hyperparameter_search | HP search | `experiments/hyperparameter_search.py` | `experiments/hp_search/**` |

## Aggregated outputs (generated)

Running `python scripts/journal_aggregate_experiment_results.py` writes:

- `docs/journal_aggregation/journal_experiment_index.json` and `.csv`
- `docs/journal_aggregation/journal_best_by_bucket.json`
- `docs/journal_aggregation/MISSING_DATA.md` when no runs are found under the scanned roots

Running `python scripts/journal_generate_full_paper_docx.py` writes:

- `docs/Journal_Full_Comparative_Study.docx`

Pass `--roots` with comma-separated paths to your local `results/` trees if they are outside the repo clone.

## Git tracking of generated assets

The repository `.gitignore` ignores most `*.png`, `*.csv`, and `results/` trees. Narrow **exceptions** were added so you can commit:

- `docs/journal_submission_figures/` (comparative plots, optional `publication_bundle/` from `generate_publication_figures`)
- `docs/journal_submission_tables/` (Markdown/CSV from `generate_tables`)
- `docs/journal_aggregation/` (`journal_experiment_index.*`, `journal_best_by_bucket.json`, `MISSING_DATA.md`)

Regenerate after new runs: `python scripts/journal_aggregate_experiment_results.py --roots <your_results> --with-plots` then `python scripts/journal_generate_full_paper_docx.py`.

## Comparability caveat

Experiments differ by **dataset**, **split** (random vs group-aware vs CV), **modality** (mel vs image vs fusion), and **reported metric**. The automated **best-by-bucket** file only ranks runs **within** a bucket string derived from line-of-work + dataset hint + split hint; it does **not** assert a single global winner across the whole codebase.
