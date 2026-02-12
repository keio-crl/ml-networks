"""JAX 関連のユーティリティ関数を扱うモジュール."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import rearrange

from ml_networks.config import SoftmaxTransConfig

if TYPE_CHECKING:
    pass


class MinMaxNormalize:
    """MinMax 正規化変換.

    JAX/NumPy版。入力の値域 [old_min, old_max] を [min_val, max_val] に変換する。
    """

    def __init__(self, min_val: float, max_val: float, old_min: float = 0.0, old_max: float = 1.0) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.scale = (max_val - min_val) / (old_max - old_min)
        self.shift = min_val - old_min * self.scale

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.scale + self.shift


def get_optimizer(
    name: str,
    **kwargs: Any,
) -> optax.GradientTransformation:
    """
    Get optimizer from optax.

    Parameters
    ----------
    name : str
        Optimizer name (e.g. "adam", "sgd", "adamw", "lamb", "rmsprop").
    kwargs : dict
        Optimizer arguments (e.g. learning_rate=0.01).

    Returns
    -------
    optax.GradientTransformation

    Examples
    --------
    >>> opt = get_optimizer("adam", learning_rate=0.01)
    """
    # Map common PyTorch-style names to optax equivalents
    name_map: dict[str, str] = {
        "Adam": "adam",
        "AdamW": "adamw",
        "SGD": "sgd",
        "RMSprop": "rmsprop",
        "Lamb": "lamb",
        "Lars": "lars",
        "Adagrad": "adagrad",
    }

    optax_name = name_map.get(name, name)

    # Map common PyTorch kwargs to optax kwargs
    mapped_kwargs = dict(kwargs)
    if "lr" in mapped_kwargs:
        mapped_kwargs["learning_rate"] = mapped_kwargs.pop("lr")

    if hasattr(optax, optax_name):
        optimizer_fn = getattr(optax, optax_name)
    else:
        msg = f"Optimizer {name} is not implemented in optax. "
        msg += "Please check the name."
        raise NotImplementedError(msg)
    return optimizer_fn(**mapped_kwargs)


def softmax(
    inputs: jax.Array,
    axis: int,
    temperature: float = 1.0,
) -> jax.Array:
    """
    Softmax function with temperature. This prevents overflow and underflow.

    Parameters
    ----------
    inputs : jax.Array
        Input tensor.
    axis : int
        Axis to apply softmax.
    temperature : float
        Temperature. Default is 1.0.

    Returns
    -------
    jax.Array
        Softmaxed tensor.

    Raises
    ------
    ValueError
        If the softmax is inf or nan.
    """
    x = inputs / temperature
    x = jnp.exp(jax.nn.log_softmax(x, axis=axis))
    if jnp.isinf(x).any() or jnp.isnan(x).any():
        msg = "softmax is inf or nan"
        raise ValueError(msg)
    return x


def gumbel_softmax(
    inputs: jax.Array,
    key: jax.Array,
    axis: int,
    temperature: float = 1.0,
) -> jax.Array:
    """
    Gumbel softmax function with temperature.

    Parameters
    ----------
    inputs : jax.Array
        Input tensor.
    key : jax.Array
        JAX PRNG key.
    axis : int
        Axis to apply softmax.
    temperature : float
        Temperature. Default is 1.0.

    Returns
    -------
    jax.Array
        Gumbel softmaxed tensor (hard one-hot with straight-through gradient).

    Raises
    ------
    ValueError
        If the gumbel_softmax is inf or nan.
    """
    x = inputs - jnp.max(jax.lax.stop_gradient(inputs), axis=-1, keepdims=True)
    # Gumbel noise
    u = jax.random.uniform(key, shape=x.shape, minval=1e-10, maxval=1.0)
    gumbel_noise = -jnp.log(-jnp.log(u))
    y = (x + gumbel_noise) / temperature
    y_soft = jax.nn.softmax(y, axis=axis)

    # Hard one-hot with straight-through
    index = jnp.argmax(y_soft, axis=axis)
    y_hard = jax.nn.one_hot(index, y_soft.shape[axis])
    # Straight-through estimator
    result = y_hard + y_soft - jax.lax.stop_gradient(y_soft)

    if jnp.isinf(result).any() or jnp.isnan(result).any():
        msg = "gumbel_softmax is inf or nan"
        raise ValueError(msg)
    return result


def jax_fix_seed(seed: int = 42) -> jax.Array:
    """
    乱数を固定する関数.

    Parameters
    ----------
    seed : int
        Random seed.

    Returns
    -------
    jax.Array
        JAX PRNG key.
    """
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


class SoftmaxTransformation:
    """Softmax 変換クラス."""

    def __init__(
        self,
        cfg: SoftmaxTransConfig,
    ) -> None:
        super().__init__()
        self.vector = cfg.vector
        self.sigma = cfg.sigma
        self.n_ignore = cfg.n_ignore
        self.max = cfg.max
        self.min = cfg.min
        self.k = jnp.linspace(self.min, self.max, self.vector)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.transform(x)

    def get_transformed_dim(self, dim: int) -> int:
        return (dim - self.n_ignore) * self.vector + self.n_ignore

    def transform(self, x: jax.Array) -> jax.Array:
        """
        SoftmaxTransformation の実行.

        Parameters
        ----------
        x : jax.Array
            入力テンソル.

        Returns
        -------
        jax.Array
            出力テンソル.

        Examples
        --------
        >>> trans = SoftmaxTransformation(SoftmaxTransConfig(vector=16, sigma=0.01, n_ignore=1, min=-1.0, max=1.0))
        >>> x = jnp.ones((2, 3, 4))
        >>> transformed = trans(x)
        >>> transformed.shape
        (2, 3, 49)

        >>> trans = SoftmaxTransformation(SoftmaxTransConfig(vector=11, sigma=0.05, n_ignore=0, min=-1.0, max=1.0))
        >>> x = jnp.ones((2, 3, 4))
        >>> transformed = trans(x)
        >>> transformed.shape
        (2, 3, 44)

        """
        *batch, dim = x.shape
        x = x.reshape(-1, dim)
        if self.n_ignore:
            data, ignored = x[:, : -self.n_ignore], x[:, -self.n_ignore :]
        else:
            data = x

        negative = jnp.stack([jnp.exp((-((data - self.k[v]) ** 2)) / self.sigma) for v in range(self.vector)])
        negative_sum = negative.sum(axis=0)

        transformed = negative / (negative_sum + 1e-8)
        transformed = rearrange(transformed, "v b d -> b (d v)")

        transformed = jnp.concatenate([transformed, ignored], axis=-1) if self.n_ignore else transformed
        return transformed.reshape(*batch, self.get_transformed_dim(dim))

    def inverse(self, x: jax.Array) -> jax.Array:
        """
        SoftmaxTransformation の逆変換.

        Parameters
        ----------
        x : jax.Array
            入力テンソル.

        Returns
        -------
        jax.Array
            出力テンソル.
        """
        *batch, dim = x.shape
        x = x.reshape(-1, dim)
        if self.n_ignore:
            data, ignored = x[:, : -self.n_ignore], x[:, -self.n_ignore :]
        else:
            data = x

        data = data.reshape(len(data), -1, self.vector)

        data = rearrange(data, "b d v -> v b d")

        data = jnp.stack([data[v] * self.k[v] for v in range(self.vector)]).sum(axis=0)

        data = jnp.concatenate([data, ignored], axis=-1) if self.n_ignore else data
        return data.reshape(*batch, -1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
