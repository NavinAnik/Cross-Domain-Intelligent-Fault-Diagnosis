"""
Weights & Biases (wandb) integration for experiment tracking.

Use environment variables to control behavior:
- WANDB_DISABLED=true or WANDB_MODE=disabled: disable wandb (no-op)
- WANDB_PROJECT: override default project name
- WANDB_ENTITY: optional team/username
"""

from typing import Any, Dict, Optional

_WANDB_RUN = None


def is_wandb_available() -> bool:
    """Return True if wandb is installed and not disabled by env."""
    import os
    if os.getenv("WANDB_DISABLED", "").lower() in ("true", "1", "yes"):
        return False
    if os.getenv("WANDB_MODE", "").lower() == "disabled":
        return False
    try:
        import wandb
        return True
    except ImportError:
        return False


def init_wandb(
    project: str = "mimii-anomaly-detection",
    config: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    entity: Optional[str] = None,
    tags: Optional[list] = None,
    reinit: bool = True,
) -> bool:
    """
    Initialize a Weights & Biases run. Safe to call even when wandb is disabled.

    Args:
        project: W&B project name.
        config: Dict of hyperparameters and run config (logged for reproducibility).
        name: Optional run name.
        entity: Optional W&B entity (team/username).
        tags: Optional list of tags.
        reinit: If True, allow re-initializing in the same process.

    Returns:
        True if wandb was initialized, False otherwise (e.g. disabled or not installed).
    """
    global _WANDB_RUN
    if not is_wandb_available():
        return False
    import wandb
    kwargs = {"project": project, "reinit": reinit}
    if config is not None:
        kwargs["config"] = config
    if name is not None:
        kwargs["name"] = name
    if entity is not None:
        kwargs["entity"] = entity
    if tags is not None:
        kwargs["tags"] = tags
    _WANDB_RUN = wandb.init(**kwargs)
    return _WANDB_RUN is not None


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
    """
    Log metrics to the current wandb run. No-op if wandb is disabled or not initialized.

    Args:
        metrics: Dict of metric name -> value (e.g. {"train_loss": 0.5, "val_acc": 0.9}).
        step: Optional step (e.g. epoch number).
        commit: If True, commit this log step.
    """
    if not is_wandb_available():
        return
    import wandb
    if wandb.run is None:
        return
    kwargs = {}
    if step is not None:
        kwargs["step"] = step
    wandb.log(metrics, commit=commit, **kwargs)


def log_artifact(path: str, name: str, artifact_type: str = "model") -> None:
    """Log a file or directory as a wandb artifact. No-op if wandb disabled."""
    import os as _os
    if not is_wandb_available():
        return
    import wandb
    if wandb.run is None:
        return
    artifact = wandb.Artifact(name=name, type=artifact_type)
    if _os.path.isfile(path):
        artifact.add_file(path)
    elif _os.path.isdir(path):
        artifact.add_dir(path)
    wandb.log_artifact(artifact)


def finish_wandb() -> None:
    """Finish the current wandb run. No-op if wandb disabled or not initialized."""
    if not is_wandb_available():
        return
    import wandb
    if wandb.run is not None:
        wandb.finish()
