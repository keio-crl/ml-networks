"""ベースモジュール."""

from __future__ import annotations

from flax import nnx


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
