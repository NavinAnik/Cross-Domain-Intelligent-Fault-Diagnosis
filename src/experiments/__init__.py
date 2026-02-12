"""
Unified experiment runners for all models and ablations.

- run_all_models: Train/evaluate any single model (DINOv2, MAE, Swin, BEiT, DeiT, CLIP, AST)
- run_fusion_experiments: Train and evaluate all fusion strategies (Early, Late, Cross-Attention, ConFu, ProMoBal)
- run_ablations: Fusion strategies and contrastive loss ablations
- hyperparameter_search: Grid/random search over key hyperparameters
- comparative_analysis: Load results and produce tables/figures for publication
"""

from .run_all_models import run_single_model, list_models, get_default_config

def run_fusion_experiment(fusion_name, config=None, run_name=None):
    """Train and evaluate a single fusion strategy. Lazy import to avoid loading heavy deps."""
    from .run_fusion_experiments import run_fusion_experiment as _run
    return _run(fusion_name, config=config, run_name=run_name)


def list_fusion_strategies():
    """Return list of fusion strategy names (early, late, cross_attention, confu, promobal)."""
    from .run_fusion_experiments import list_fusion_strategies as _list
    return _list()


__all__ = [
    "run_single_model",
    "list_models",
    "get_default_config",
    "run_fusion_experiment",
    "list_fusion_strategies",
]


def run_all_models_sequential(models=None, config_overrides=None, run_name=None):
    """Run multiple models sequentially. models=None means all."""
    names = models or list_models()
    return {m: run_single_model(m, config=config_overrides, run_name=run_name or m) for m in names}


def run_fusion_ablations(audio_path, image_path, config=None):
    """Run all fusion strategy ablations."""
    from .run_ablations import run_all_fusion_ablations
    return run_all_fusion_ablations(audio_path, image_path, config)


def run_contrastive_ablations(dataset_path, config=None):
    """Run all contrastive loss ablations."""
    from .run_ablations import run_all_contrastive_ablations
    return run_all_contrastive_ablations(dataset_path, config)
