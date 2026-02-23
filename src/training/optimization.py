"""
Training optimization utilities for memory-efficient training on limited GPU (e.g. 8GB).

Provides:
- Gradient checkpointing: trade compute for memory (40-60% memory reduction, ~20-30% slower).
- Mixed precision (FP16/BF16): autocast + GradScaler for faster training and lower memory.
- Gradient accumulation: effective batch size = batch_size * accumulation_steps.
"""

import torch
import contextlib
from typing import Optional, Callable, Any


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """
    Enable gradient checkpointing on a model to reduce memory at the cost of extra compute.

    - For HuggingFace-style models: calls model.gradient_checkpointing_enable() if present.
    - For timm/other backbones: calls the same if the attribute exists.
    - Does not wrap custom nn.Modules in checkpoint (encoder-specific logic stays in encoders).

    Call this before training when the backbone is trainable (not frozen).
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return
    # Recursively enable for submodules (e.g. encoder.backbone)
    for child in model.children():
        if hasattr(child, "gradient_checkpointing_enable"):
            child.gradient_checkpointing_enable()
        else:
            enable_gradient_checkpointing(child)


def create_amp_scaler(
    use_amp: bool,
    device: torch.device,
    enabled: Optional[bool] = None,
) -> Optional[torch.amp.GradScaler]:
    """
    Create a GradScaler for mixed precision training when using FP16.

    Use with: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update().

    Args:
        use_amp: Whether to use mixed precision.
        device: torch device (scaler only needed for CUDA).
        enabled: Override for scaler.enabled (defaults to use_amp and device.type == 'cuda').

    Returns:
        GradScaler if use_amp and CUDA, else None.
    """
    if not use_amp or device.type != "cuda":
        return None
    if enabled is None:
        enabled = True
    return torch.amp.GradScaler("cuda", enabled=enabled)


@contextlib.contextmanager
def autocast_context(
    use_amp: bool, device: torch.device, dtype: Optional[torch.dtype] = None
):
    """
    Context manager for mixed precision forward pass.

    Use around model forward and loss computation, e.g.:

        with autocast_context(use_amp, device):
            logits = model(x)
            loss = criterion(logits, y)

    Args:
        use_amp: Whether to use autocast.
        device: torch device (autocast only on CUDA).
        dtype: Optional torch dtype (default: torch.float16 for CUDA).
    """
    if not use_amp or device.type != "cuda":
        yield
        return
    if dtype is None:
        dtype = torch.float16
    with torch.amp.autocast("cuda", dtype=dtype):
        yield


class TrainingStepHelper:
    """
    Helper to run training steps with optional mixed precision and gradient accumulation.

    Usage:

        helper = TrainingStepHelper(
            optimizer=optimizer,
            use_amp=True,
            device=device,
            accumulation_steps=4,
        )
        for batch_idx, (x, y) in enumerate(train_loader):
            loss = helper.accumulation_step(
                model, x, y,
                loss_fn=lambda logits, y: criterion(logits, y),
                forward_fn=lambda m, x: m(x),
            )
            if loss is not None:  # step was performed this batch
                print(f"Loss: {loss:.4f}")
        helper.finalize_step()  # call at end of epoch if last batch didn't step
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        use_amp: bool = True,
        device: torch.device = None,
        accumulation_steps: int = 1,
    ):
        self.optimizer = optimizer
        self.use_amp = use_amp and (device is not None and device.type == "cuda")
        self.device = device or torch.device("cpu")
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.scaler = create_amp_scaler(self.use_amp, self.device)
        self._step_count = 0

    def zero_grad(self) -> None:
        """Zero optimizer gradients. Call at start of epoch or after step."""
        self.optimizer.zero_grad(set_to_none=True)

    def _backward(self, loss: torch.Tensor) -> None:
        """Scale loss and backward (for AMP)."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self, clip_grad_norm: Optional[float] = 1.0) -> None:
        """Unscale gradients and step optimizer (for AMP)."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if clip_grad_norm is not None and clip_grad_norm > 0:
            all_params = [p for g in self.optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(all_params, clip_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def accumulation_step(
        self,
        model: torch.nn.Module,
        *batch_inputs,
        loss_fn: Callable[..., torch.Tensor],
        forward_fn: Optional[Callable[..., torch.Tensor]] = None,
        **batch_kwargs,
    ) -> Optional[float]:
        """
        Run one accumulation step: forward, loss, backward. Step optimizer every accumulation_steps.

        Args:
            model: The model to train.
            *batch_inputs: Inputs to forward_fn (e.g. x, y).
            loss_fn: Callable that takes model outputs and batch targets and returns loss.
                     Signature: loss = loss_fn(outputs, *targets) or loss_fn(**kwargs).
            forward_fn: Callable(model, *inputs) -> outputs. If None, uses model(*batch_inputs).
            **batch_kwargs: Keyword arguments for loss_fn (e.g. y=labels).

        Returns:
            Unscaled loss value (float) for logging on the step that actually updates weights,
            else None for accumulation-only steps.
        """
        model.train()
        effective_batch_size = 1
        if (
            batch_inputs
            and hasattr(batch_inputs[0], "size")
            and batch_inputs[0].dim() > 0
        ):
            effective_batch_size = batch_inputs[0].size(0)

        with autocast_context(self.use_amp, self.device):
            if forward_fn is not None:
                outputs = forward_fn(model, *batch_inputs)
            else:
                outputs = model(*batch_inputs)
            loss = loss_fn(outputs, **batch_kwargs)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss = loss / self.accumulation_steps

        self._backward(loss)
        self._step_count += 1
        loss_to_log = (loss.detach() * self.accumulation_steps).item()

        if self._step_count % self.accumulation_steps == 0:
            self._optimizer_step(clip_grad_norm=1.0)
            self.zero_grad()
            return loss_to_log
        return None

    def step(
        self,
        loss: torch.Tensor,
        clip_grad_norm: Optional[float] = 1.0,
    ) -> None:
        """
        Backward and optimizer step for a single precomputed loss.
        Use when you compute loss yourself and want AMP + gradient accumulation.

        Loss is divided by accumulation_steps before backward so that the effective
        gradient is the average over accumulation_steps batches.

        Args:
            loss: Scalar loss tensor for the current batch.
            clip_grad_norm: If set, clip gradients before step (AMP unscale is done internally).
        """
        scaled_loss = loss / self.accumulation_steps
        self._backward(scaled_loss)
        self._step_count += 1
        if self._step_count % self.accumulation_steps == 0:
            self._optimizer_step(clip_grad_norm=clip_grad_norm)
            self.zero_grad()

    def finalize_step(self) -> None:
        """
        Call at end of epoch if the last batch index was not a multiple of accumulation_steps,
        so that remaining gradients are stepped.
        """
        if self._step_count % self.accumulation_steps != 0:
            self._optimizer_step(clip_grad_norm=1.0)
            self.zero_grad()
        self._step_count = 0

    def reset_step_count(self) -> None:
        """Reset accumulation step counter (e.g. at start of new epoch)."""
        self._step_count = 0
