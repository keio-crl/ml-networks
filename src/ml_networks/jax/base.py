"""ベースモジュール."""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytorch_lightning as pl
from flax import nnx


def numpy_collate(batch: Any) -> Any:
    """Collate function that returns NumPy arrays instead of torch.Tensors.

    Drop-in replacement for PyTorch's default_collate. Used with DataLoader
    so that batches are NumPy arrays, ready for conversion to JAX arrays.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], tuple | list):
        transposed = zip(*batch, strict=False)
        return [numpy_collate(samples) for samples in transposed]
    if isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    return np.array(batch)


class BaseModule(nnx.Module):
    """Base module for JAX/Flax NNX."""

    def freeze_weights(self) -> None:
        """Freeze all weight parameters (kernel in Flax)."""
        _graph_def, state = nnx.split(self)
        flat_state = state.flat_state()
        for key, value in flat_state.items():
            if "kernel" in key and isinstance(value, nnx.VariableState):
                value.type = nnx.VariableState  # type: ignore[assignment]
        state = nnx.State.from_flat_path(flat_state)
        nnx.update(self, state)

    def freeze_biases(self) -> None:
        """Freeze all bias parameters."""
        _graph_def, state = nnx.split(self)
        flat_state = state.flat_state()
        for key, value in flat_state.items():
            if "bias" in key and isinstance(value, nnx.VariableState):
                value.type = nnx.VariableState  # type: ignore[assignment]
        state = nnx.State.from_flat_path(flat_state)
        nnx.update(self, state)

    def unfreeze_weights(self) -> None:
        """Unfreeze all weight parameters (kernel in Flax)."""
        _graph_def, state = nnx.split(self)
        flat_state = state.flat_state()
        for key, value in flat_state.items():
            if "kernel" in key and isinstance(value, nnx.VariableState):
                value.type = nnx.Param  # type: ignore[assignment]
        state = nnx.State.from_flat_path(flat_state)
        nnx.update(self, state)

    def unfreeze_biases(self) -> None:
        """Unfreeze all bias parameters."""
        _graph_def, state = nnx.split(self)
        flat_state = state.flat_state()
        for key, value in flat_state.items():
            if "bias" in key and isinstance(value, nnx.VariableState):
                value.type = nnx.Param  # type: ignore[assignment]
        state = nnx.State.from_flat_path(flat_state)
        nnx.update(self, state)


class JaxLightningModule(pl.LightningModule):
    """Base class for JAX models trained with PyTorch Lightning.

    Subclasses should:
    1. Define JAX models (nnx.Module) and optimizers (nnx.Optimizer) in __init__
    2. Implement static @nnx.jit methods for train/val steps
    3. Call those methods from training_step/validation_step
    4. Implement _get_jax_state/_set_jax_state for checkpointing
    """

    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization = False
        self._jax_step_count = 0

    def configure_optimizers(self) -> list:
        """Return empty optimizer list. JAX manages its own optimizers."""
        return []

    def step_jax(self) -> None:
        """Increment JAX step counter.

        Call this at the end of each training_step so that
        JaxModelCheckpoint can correctly track progress and save checkpoints.
        """
        self._jax_step_count += 1

    def step_global_step(self) -> None:
        """Deprecated: use step_jax() instead."""
        warnings.warn("step_global_step() is deprecated, use step_jax() instead", DeprecationWarning, stacklevel=2)
        self.step_jax()

    @property
    def jax_step_count(self) -> int:
        return self._jax_step_count

    def log_jax_dict(self, metrics: dict[str, Any], prefix: str, **kwargs: Any) -> None:
        """Log a dict of JAX scalars to Lightning, converting to Python floats."""
        self.log_dict(
            {f"{prefix}/{k}": float(v) for k, v in metrics.items()},
            prog_bar=True,
            sync_dist=True,
            **kwargs,
        )

    # --- Checkpoint hooks ---

    @staticmethod
    def _jax_to_numpy(x: Any) -> np.ndarray:
        """Convert JAX array to numpy, handling PRNGKey arrays."""
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.integer) and x.dtype.name.startswith("key"):
            return np.asarray(jax.random.key_data(x))
        try:
            return np.asarray(x)
        except TypeError:
            return np.asarray(jax.random.key_data(x))

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        jax_state = self._get_jax_state()
        checkpoint["jax_state"] = jax.tree.map(self._jax_to_numpy, jax_state)
        opt_state = self._get_opt_state()
        if opt_state is not None:
            checkpoint["jax_opt_state"] = jax.tree.map(self._jax_to_numpy, opt_state)
        checkpoint["jax_step_count"] = self._jax_step_count
        # Remove empty PyTorch state_dict to avoid confusion
        checkpoint["state_dict"] = {}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if "jax_state" in checkpoint:
            jax_state = jax.tree.map(jnp.asarray, checkpoint["jax_state"])
            self._set_jax_state(jax_state)
            # Also update persistent params if used
            if hasattr(self, "_params"):
                self._params = jax_state
        if "jax_opt_state" in checkpoint:
            opt_state = jax.tree.map(jnp.asarray, checkpoint["jax_opt_state"])
            self._set_opt_state(opt_state)
        if "jax_step_count" in checkpoint:
            self._jax_step_count = checkpoint["jax_step_count"]

    # --- Abstract methods for subclasses ---

    def _get_jax_state(self) -> Any:
        """Return the JAX model state for checkpointing."""
        raise NotImplementedError

    def _set_jax_state(self, state: Any) -> None:
        """Restore JAX model state from a checkpoint."""
        raise NotImplementedError

    def _get_opt_state(self) -> Any:
        """Return the optimizer state for checkpointing. None if not applicable."""
        return None

    def _set_opt_state(self, state: Any) -> None:
        """Restore optimizer state from a checkpoint."""
