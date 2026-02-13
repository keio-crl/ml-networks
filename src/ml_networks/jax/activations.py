"""活性化関数を扱うモジュール."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


class Activation(nnx.Module):
    """Generic activation function."""

    def __init__(self, activation: str, **kwargs: Any) -> None:
        if "glu" not in activation.lower():
            kwargs.pop("dim", None)

        builtin_activations: dict[str, Any] = {
            "ReLU": jax.nn.relu,
            "GELU": jax.nn.gelu,
            "GeLU": jax.nn.gelu,
            "SiLU": jax.nn.silu,
            "Tanh": jnp.tanh,
            "Sigmoid": jax.nn.sigmoid,
            "ELU": jax.nn.elu,
            "LeakyReLU": jax.nn.leaky_relu,
            "Mish": lambda x: x * jnp.tanh(jax.nn.softplus(x)),
            "Softplus": jax.nn.softplus,
            "Identity": lambda x: x,
        }

        if activation in builtin_activations:
            self._fn = builtin_activations[activation]
            self._module: nnx.Module | None = None
        elif activation == "TanhExp":
            self._fn = None
            self._module = TanhExp()
        elif activation == "REReLU":
            self._fn = None
            self._module = REReLU(**kwargs)
        elif activation in {"SiGLU", "SwiGLU"}:
            self._fn = None
            self._module = SiGLU(**kwargs)
        elif activation == "CRReLU":
            self._fn = None
            self._module = CRReLU(**kwargs)
        elif activation == "L2Norm":
            self._fn = None
            self._module = L2Norm()
        else:
            msg = f"Activation: '{activation}' is not implemented yet."
            raise NotImplementedError(msg)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self._module is not None:
            return self._module(x)
        return self._fn(x)


class L2Norm(nnx.Module):
    """
    L2 Normalization layer.

    Examples
    --------
    >>> l2norm = L2Norm()
    >>> x = jnp.array([[3.0, 4.0]])
    >>> output = l2norm(x)
    >>> output.shape
    (1, 2)
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class REReLU(nnx.Module):
    """
    Reparametarized ReLU activation function. This backward pass is differentiable.

    References
    ----------
    https://openreview.net/forum?id=lNCnZwcH5Z

    Parameters
    ----------
    reparametarize_fn : str
        Reparametarization function. Default is GELU.

    Examples
    --------
    >>> rerelu = REReLU()
    >>> x = jnp.array([[1.0, -1.0, 0.5]])
    >>> output = rerelu(x)
    >>> output.shape
    (1, 3)
    """

    def __init__(self, reparametarize_fn: str = "gelu") -> None:
        reparam_fns: dict[str, Any] = {
            "gelu": jax.nn.gelu,
            "relu": jax.nn.relu,
            "silu": jax.nn.silu,
            "elu": jax.nn.elu,
        }
        reparametarize_fn = reparametarize_fn.lower()
        if reparametarize_fn not in reparam_fns:
            msg = f"Reparametarization function '{reparametarize_fn}' is not supported."
            raise ValueError(msg)
        self.reparametarize_fn = reparam_fns[reparametarize_fn]

    def __call__(self, x: jax.Array) -> jax.Array:
        return (
            jax.lax.stop_gradient(jax.nn.relu(x))
            + self.reparametarize_fn(x)
            - jax.lax.stop_gradient(self.reparametarize_fn(x))
        )


class CRReLU(nnx.Module):
    """
    Correction Regularized ReLU activation function. This is a variant of ReLU activation function.

    References
    ----------
    https://openreview.net/forum?id=7TZYM6Hm9p

    Parameters
    ----------
    lr : float
        Learning rate. Default is 0.01.

    Examples
    --------
    >>> crrelu = CRReLU()
    >>> x = jnp.array([[1.0, -1.0, 0.5]])
    >>> output = crrelu(x)
    >>> output.shape
    (1, 3)
    """

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = nnx.Param(jnp.array(lr, dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.relu(x) + self.lr.value * x * jnp.exp(-(x**2) / 2)


class SiGLU(nnx.Module):
    """
    SiGLU activation function.

    This is equivalent to SwiGLU (Swish variant of Gated Linear Unit) activation function.

    References
    ----------
    https://arxiv.org/abs/2102.11972

    Parameters
    ----------
    dim : int
        Dimension to split the tensor. Default is -1.

    Examples
    --------
    >>> siglu = SiGLU()
    >>> x = jnp.ones((1, 4))
    >>> output = siglu(x)
    >>> output.shape
    (1, 2)

    >>> siglu = SiGLU(dim=0)
    >>> x = jnp.ones((4, 1))
    >>> output = siglu(x)
    >>> output.shape
    (2, 1)
    """

    def __init__(self, dim: int = -1) -> None:
        self.dim = dim

    def __call__(self, x: jax.Array) -> jax.Array:
        x1, x2 = jnp.split(x, 2, axis=self.dim)
        return x1 * jax.nn.silu(x2)


@jax.custom_vjp
def _tanhexp(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jnp.exp(x))


def _tanhexp_fwd(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _tanhexp(x), x


def _tanhexp_bwd(x: jax.Array, g: jax.Array) -> tuple[jax.Array]:
    tanh_exp_x = jnp.tanh(jnp.exp(x))
    grad = tanh_exp_x - x * jnp.exp(x) * (tanh_exp_x**2 - 1)
    return (g * grad,)


_tanhexp.defvjp(_tanhexp_fwd, _tanhexp_bwd)


class TanhExp(nnx.Module):
    """
    TanhExp activation function.

    Examples
    --------
    >>> tanhexp = TanhExp()
    >>> x = jnp.array([[1.0, -1.0, 0.5]])
    >>> output = tanhexp(x)
    >>> output.shape
    (1, 3)
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        return _tanhexp(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
