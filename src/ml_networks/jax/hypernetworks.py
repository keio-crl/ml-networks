"""ハイパーネットワークを扱うモジュール."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ml_networks.config import MLPConfig
from ml_networks.jax.layers import MLPLayer

Shape = tuple[int, ...]

InputMode = Literal["cos|sin", "z|1-z"] | None

encoding_multiplier: dict[InputMode, int] = {
    None: 1,
    "cos|sin": 2,
    "z|1-z": 2,
}


def _same_keys(
    required: dict[str, Any],
    provided: dict[str, Any],
    name: str = "",
) -> None:
    """Check if two dictionaries have the same keys."""
    missing = required.keys() - provided.keys()
    if len(missing) > 0:
        msg = f"Missing {name}: {list(missing)}"
        raise ValueError(msg)

    unexpected = provided.keys() - required.keys()
    if len(unexpected) > 0:
        msg = f"Unexpected {name}: {list(unexpected)}"
        raise ValueError(msg)


class HyperNetMixin:
    """Mixin class for hypernetwork functionality."""

    input_shapes: dict[str, Shape]
    output_shapes: dict[str, Shape]
    encoding: InputMode

    def encode_inputs(self, inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Encode the inputs using the specified encoding mode."""
        if self.encoding is None:
            return inputs
        return {k: encode_input(x, mode=self.encoding) for k, x in inputs.items()}

    def flat_input_size(self) -> int:
        """Return the flat input size after encoding."""
        flat_input_size = sum(int(np.prod(v)) for v in self.input_shapes.values())
        flat_input_size *= encoding_multiplier[self.encoding]
        return flat_input_size

    def flat_output_size(self) -> int:
        """Return the flat output size."""
        return sum(int(np.prod(v)) for v in self.output_shapes.values())

    def output_offsets(self) -> dict[str, tuple[int, int]]:
        """Return the output offsets."""
        offset = 0
        offsets = {}
        for name, shape in self.output_shapes.items():
            size = int(np.prod(shape))
            offsets[name] = (offset, size)
            offset += size
        return offsets

    def _validate_inputs(self, inputs: dict[str, jax.Array]) -> int:
        """Validate the inputs and return batch dimension."""
        _same_keys(self.input_shapes, inputs, name="inputs")

        for name, required_shape in self.input_shapes.items():
            shape = inputs[name].shape

            if shape[1:] == required_shape:
                continue

            if shape == required_shape:
                inputs[name] = inputs[name][None]
                continue

            error_msg = f"Wrong shape for {name}, expected {required_shape} or {('B', *required_shape)}, got {shape}"
            raise ValueError(error_msg)

        batch_dims = [x.shape[0] for x in inputs.values()]
        if len(set(batch_dims)) > 1:
            msg = f"Multiple batch dimensions found: {batch_dims}"
            raise ValueError(msg)

        return batch_dims[0]

    def flatten_inputs(self, inputs: dict[str, jax.Array]) -> jax.Array:
        """Flatten the input tensors into a single tensor along the last dimension."""
        batch_dim = self._validate_inputs(inputs)
        return jnp.concatenate(
            [inputs[hp].reshape(batch_dim, -1) for hp in sorted(self.input_shapes)],
            axis=-1,
        )

    def unflatten_output(self, flat_output: jax.Array) -> dict[str, jax.Array]:
        """Unflatten the output tensor into a dictionary of named tensors."""
        outputs = {}
        batch_dim = flat_output.shape[0]
        for name, shape in self.output_shapes.items():
            shape_tuple = (batch_dim, *shape)
            offsets = self.output_offsets()[name]
            outputs[name] = jax.lax.dynamic_slice_in_dim(
                flat_output,
                offsets[0],
                offsets[1],
                axis=1,
            ).reshape(shape_tuple)
        return outputs


class HyperNet(nnx.Module, HyperNetMixin):
    """A hypernetwork that generates weights for a target network.

    Parameters
    ----------
    input_dim : int
        Dimension of the input.
    output_shapes : dict[str, Shape]
        Shapes of the primary network weights being predicted.
    fc_cfg : MLPConfig | None
        Configuration for the MLP backbone.
    encoding : InputMode
        The input encoding mode. Default is None.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        input_dim: int,
        output_shapes: dict[str, Shape],
        fc_cfg: MLPConfig | None = None,
        encoding: InputMode = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.input_dim = input_dim
        self.output_shapes = output_shapes
        self.encoding = encoding

        self._output_offsets = self.output_offsets()

        self.backbone: nnx.Module
        if fc_cfg is not None:
            self.backbone = MLPLayer(
                self.input_dim,
                self.flat_output_size(),
                fc_cfg,
                rngs=rngs,
            )
        else:
            self.backbone = nnx.Linear(
                self.input_dim,
                self.flat_output_size(),
                rngs=rngs,
            )

    def __call__(self, inputs: jax.Array) -> dict[str, jax.Array]:
        """Forward pass.

        Parameters
        ----------
        inputs : jax.Array
            Input tensor.

        Returns
        -------
        dict[str, jax.Array]
            Dictionary of output tensors.
        """
        if self.encoding is not None:
            inputs = encode_input(inputs, self.encoding)

        flat_output = self.backbone(inputs)
        return self.unflatten_output(flat_output)


def encode_input(input: jax.Array, mode: InputMode = "cos|sin") -> jax.Array:
    """Encode the input tensor based on the specified mode.

    Parameters
    ----------
    input : jax.Array
        The input tensor.
    mode : InputMode
        The encoding mode.

    Returns
    -------
    jax.Array
        The encoded tensor.

    Raises
    ------
    ValueError
        If an unsupported encoding mode is given.
    """
    z = input

    if mode is None:
        return input

    if mode == "cos|sin":
        _check_tensor_range(input, low=0, high=1)
        scaled_z = jnp.pi * z / 2
        return jnp.concatenate([jnp.cos(scaled_z), jnp.sin(scaled_z)], axis=-1)

    if mode == "z|1-z":
        return jnp.concatenate([z, 1 - z], axis=-1)

    msg = f"Unsupported encoding mode: {mode}"
    raise ValueError(msg)


def _check_tensor_range(tensor: jax.Array, low: float, high: float) -> None:
    """Check if the input tensor is within specified range."""
    if (tensor < low).any() or (tensor > high).any():
        warnings.warn(f"Input tensor has dimensions outside of [{low},{high}].", stacklevel=2)


Nonlinearity = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]


def initialize_weight(
    key: jax.Array,
    shape: tuple[int, ...],
    distribution: str | None,
    nonlinearity: str | None = "LeakyReLU",
) -> jax.Array:
    """Initialize weight tensor using the specified distribution.

    Parameters
    ----------
    key : jax.Array
        JAX PRNG key.
    shape : tuple[int, ...]
        Shape of the weight tensor.
    distribution : str | None
        Distribution to use for initialization.
    nonlinearity : str | None
        Nonlinearity function. Default is "LeakyReLU".

    Returns
    -------
    jax.Array
        Initialized weight tensor.

    Raises
    ------
    ValueError
        When the specified distribution is not supported.
    """
    if distribution is None:
        return jax.random.normal(key, shape) * 0.01

    if distribution == "zeros":
        return jnp.zeros(shape)
    if distribution == "kaiming_normal":
        return jax.nn.initializers.kaiming_normal()(key, shape)
    if distribution == "kaiming_uniform":
        return jax.nn.initializers.he_uniform()(key, shape)
    if distribution == "kaiming_normal_fanout":
        return jax.nn.initializers.kaiming_normal()(key, shape)
    if distribution == "kaiming_uniform_fanout":
        return jax.nn.initializers.he_uniform()(key, shape)
    if distribution == "glorot_normal":
        return jax.nn.initializers.glorot_normal()(key, shape)
    if distribution == "glorot_uniform":
        return jax.nn.initializers.glorot_uniform()(key, shape)
    if distribution == "orthogonal":
        return jax.nn.initializers.orthogonal()(key, shape)
    msg = f"Unsupported weight distribution '{distribution}'"
    raise ValueError(msg)


def initialize_bias(
    shape: tuple[int, ...],
    distribution: float | None = 0.0,
) -> jax.Array:
    """Initialize the bias tensor.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the bias tensor.
    distribution : float | None
        The distribution to use (constant value). Default is 0.

    Returns
    -------
    jax.Array
        Initialized bias tensor.

    Raises
    ------
    ValueError
        When the specified distribution is not supported.
    """
    if distribution is None:
        return jnp.zeros(shape)

    if isinstance(distribution, int | float):
        return jnp.full(shape, distribution)

    msg = f"Unsupported bias distribution '{distribution}'"
    raise ValueError(msg)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
