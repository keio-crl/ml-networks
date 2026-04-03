"""JAX 関連のユーティリティ関数を扱うモジュール."""

from __future__ import annotations

import logging
import random
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytorch_lightning as pl
from einops import rearrange
from flax import nnx
from pytorch_lightning.callbacks.model_summary import ModelSummary as _PLModelSummary

from ml_networks.config import SoftmaxTransConfig

_log = logging.getLogger(__name__)


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


def numpy_collate(batch: tuple | list) -> np.ndarray | list | dict:
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

    """
    x = inputs / temperature
    return jnp.exp(jax.nn.log_softmax(x, axis=axis))


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


def count_nnx_params(model: nnx.Module) -> tuple[int, int]:
    """Flax NNX モデルのパラメータ数をカウントする.

    Parameters
    ----------
    model : nnx.Module
        Flax NNX モデル.

    Returns
    -------
    tuple[int, int]
        (total_params, trainable_params)
    """
    _, state = nnx.split(model)
    flat = state.flat_state()
    total = 0
    trainable = 0
    for leaf in flat.leaves:
        if isinstance(leaf, nnx.VariableState):
            size = leaf.value.size
            total += size
            if issubclass(leaf.type, nnx.Param):
                trainable += size
    return total, trainable


def _find_nnx_modules(
    lightning_module: pl.LightningModule,
) -> list[tuple[str, nnx.Module]]:
    """JaxLightningModule の直接属性から nnx.Module を探索する."""
    found: list[tuple[str, nnx.Module]] = []
    seen_ids: set[int] = set()
    for name, attr in vars(lightning_module).items():
        if isinstance(attr, nnx.Module) and id(attr) not in seen_ids:
            found.append((name, attr))
            seen_ids.add(id(attr))
    return found


def _compute_model_size_mb(model: nnx.Module) -> float:
    """NNX モデルの推定サイズ (MB) を計算する."""
    _, state = nnx.split(model)
    flat = state.flat_state()
    total_bytes = 0
    for leaf in flat.leaves:
        if isinstance(leaf, nnx.VariableState):
            arr = leaf.value
            total_bytes += arr.size * arr.dtype.itemsize
    return total_bytes / 1e6


def _human_readable_count(number: int) -> str:
    """整数を K, M, B, T 付きの読みやすい文字列に変換する."""
    abbrevs = [(1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]
    for threshold, suffix in abbrevs:
        if abs(number) >= threshold:
            return f"{number / threshold:.1f} {suffix}"
    return f"{number}  "


def _format_jax_summary(
    summary_data: list[tuple[str, list[str]]],
    total_params: int,
    trainable_params: int,
    model_size_mb: float,
) -> str:
    """JAX モデルサマリーテーブルを生成する."""
    n_rows = len(summary_data[0][1])

    col_widths = []
    for header, values in summary_data:
        w = max((len(str(v)) for v in values), default=0)
        col_widths.append(max(w, len(header)))

    fmt = "{:<{}}"
    total_width = sum(col_widths) + 3 * (1 + len(summary_data))
    header_line = " | ".join(fmt.format(h, w) for (h, _), w in zip(summary_data, col_widths, strict=True))

    lines = [header_line, "-" * total_width]
    for i in range(n_rows):
        row = " | ".join(fmt.format(str(vals[i]), w) for (_, vals), w in zip(summary_data, col_widths, strict=True))
        lines.append(row)
    lines.append("-" * total_width)

    s = "{:<10}"
    lines.extend((
        s.format(_human_readable_count(trainable_params)) + "Trainable params",
        s.format(_human_readable_count(total_params - trainable_params)) + "Non-trainable params",
        s.format(_human_readable_count(total_params)) + "Total params",
        s.format(f"{model_size_mb:,.3f}") + "Total estimated model params size (MB)",
    ))

    return "\n".join(lines)


class JaxModelSummary(_PLModelSummary):
    """Flax NNX パラメータを正しくカウントする ModelSummary コールバック.

    JaxLightningModule の場合、NNX モデルのパラメータをカウントして表示する。
    通常の LightningModule の場合はデフォルトの PL サマリーにフォールバックする。
    """

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        from ml_networks.jax.base import JaxLightningModule

        if not self._max_depth:
            return

        if not isinstance(pl_module, JaxLightningModule):
            super().on_fit_start(trainer, pl_module)
            return

        nnx_modules = _find_nnx_modules(pl_module)
        if not nnx_modules:
            super().on_fit_start(trainer, pl_module)
            return

        # 各 NNX モジュールのパラメータ数を集計
        names: list[str] = []
        types: list[str] = []
        param_counts: list[int] = []
        total_params = 0
        trainable_params = 0
        total_size_mb = 0.0

        for name, module in nnx_modules:
            t, tr = count_nnx_params(module)
            names.append(name)
            types.append(type(module).__name__)
            param_counts.append(t)
            total_params += t
            trainable_params += tr
            total_size_mb += _compute_model_size_mb(module)

        summary_data: list[tuple[str, list[str]]] = [
            (" ", [str(i) for i in range(len(names))]),
            ("Name", names),
            ("Type", types),
            ("Params", [_human_readable_count(c) for c in param_counts]),
        ]

        if trainer.is_global_zero:
            summary_table = _format_jax_summary(
                summary_data,
                total_params,
                trainable_params,
                total_size_mb,
            )
            _log.info("\n" + summary_table)


class JaxTrainer(pl.Trainer):
    """Flax NNX パラメータ数を正しく表示する PyTorch Lightning Trainer ラッパー.

    JaxLightningModule を使用するモデルで ``fit()`` を呼び出すと、
    Flax NNX モデルのパラメータ数が正しく表示される。

    Examples
    --------
    >>> trainer = JaxTrainer(max_epochs=10)
    >>> trainer.fit(my_jax_lightning_module, dataloader)
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_model_summary", False)

        callbacks = kwargs.get("callbacks")
        if callbacks is None:
            callbacks = []
        elif not isinstance(callbacks, list):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)

        has_summary = any(isinstance(cb, _PLModelSummary) for cb in callbacks)
        if not has_summary:
            callbacks.append(JaxModelSummary())

        kwargs["callbacks"] = callbacks
        super().__init__(**kwargs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
