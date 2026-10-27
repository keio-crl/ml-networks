"""PyTorch 関連のユーティリティ関数を扱うモジュール."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytorch_lightning as pl  # type: ignore[import-untyped]
import pytorch_optimizer  # type: ignore[import-untyped]
import schedulefree  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchvision import transforms  # type: ignore[import-untyped]

from ml_networks.config import SoftmaxTransConfig

if TYPE_CHECKING:
    from collections.abc import Iterator


class MinMaxNormalize(transforms.Normalize):
    """MinMax 正規化変換."""

    def __init__(self, min_val: float, max_val: float, old_min: float = 0.0, old_max: float = 1.0) -> None:
        """
        MinMaxNormalize の初期化.

        Args:
        -----
        min_val : float
            最小値.
        max_val : float
            最大値.
        old_min : float
            元の最小値.
        old_max : float
            元の最大値.


        """
        scale = (max_val - min_val) / (old_max - old_min)
        shift = min_val - old_min * scale  # new = scale·x + shift

        #   Normalize does (x - mean)/std  =>  x · (1/std)  - mean/std
        mean = -shift / scale  # invert the affine form
        std = 1.0 / scale

        super().__init__(mean=[mean] * 3, std=[std] * 3)
        self.min = min_val
        self.max = max_val


def get_optimizer(
    param: Iterator[nn.Parameter],
    name: str,
    **kwargs: float | str | bool,
) -> torch.optim.Optimizer:
    """
    Get optimizer from torch.optim or pytorch_optimizer.

    Args:
    -----
    param : Iterator[nn.Parameter]
        Parameters of models to optimize.
    name : str
        Optimizer name.
    kwargs : dict
        Optimizer arguments(settings).

    Returns
    -------
    torch.optim.Optimizer

    Examples
    --------
    >>> get_optimizer([nn.Parameter(torch.randn(1, 3))], "Adam", lr=0.01)
    Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        capturable: False
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 0.01
        maximize: False
        weight_decay: 0
    )
    """
    if hasattr(schedulefree, name):
        optimizer = getattr(schedulefree, name)
    elif hasattr(torch.optim, name):
        optimizer = getattr(torch.optim, name)
    elif hasattr(pytorch_optimizer, name):
        optimizer = getattr(pytorch_optimizer, name)
    else:
        msg = f"Optimizer {name} is not implemented in torch.optim or pytorch_optimizer, schedulefree. "
        msg += "Please check the name and capitalization."
        raise NotImplementedError(msg)
    return optimizer(param, **kwargs)


def log_softmax(
    inputs: torch.Tensor,
    dim: int = -1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Log-softmax function with temperature, robust to exponent underflow.

    The argmax entry is excluded from the normalizer so that the log-probability
    is computed via ``softplus`` of the logit margin. This keeps the argmax entry
    exactly 0 before the correction and avoids ``log(1 + tiny)`` losing precision.

    Parameters
    ----------
    inputs : torch.Tensor
        Input logits.
    dim : int
        Dimension to apply log-softmax. Default is -1.
    temperature : float
        Temperature. Default is 1.0.

    Returns
    -------
    torch.Tensor
        Log-softmaxed tensor.

    """
    z = (inputs / temperature).movedim(dim, -1)
    i_max = torch.argmax(z, dim=-1, keepdim=True)
    z_max = torch.gather(z, -1, i_max)  # Max logit.
    is_max = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, i_max, True)  # Max mask.
    shifted = torch.where(is_max, torch.zeros_like(z), z - z_max)  # Exact 0 at argmax.
    others = shifted.masked_fill(is_max, -torch.inf)  # Exclude argmax from sum.
    m = -torch.logsumexp(others, dim=-1, keepdim=True)  # Logit margin.
    return (shifted - F.softplus(-m)).movedim(-1, dim)


def softmax(
    inputs: torch.Tensor,
    dim: int = -1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Softmax function with temperature. This prevents overflow and underflow.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor.
    dim : int
        Dimension to apply softmax. Default is -1.
    temperature : float
        Temperature. Default is 1.0.

    Returns
    -------
    torch.Tensor
        Softmaxed tensor.

    Raises
    ------
    ValueError
        If the softmax is inf or nan.
    """
    x = torch.exp(log_softmax(inputs, dim=dim, temperature=temperature))
    if torch.isinf(x).any() or torch.isnan(x).any():
        msg = "softmax is inf or nan"
        raise ValueError(msg)
    return x


def softmax_log_ratio(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Log-ratio between two softmax distributions, robust to exponent underflow.

    With ``q = softmax(logits)`` and ``p = softmax(target_logits)`` along the last
    dimension, ``log(q / p)`` and ``q - p`` are computed from the logit difference
    so that they stay accurate when ``q`` and ``p`` are close or tiny.

    Parameters
    ----------
    logits : torch.Tensor
        Logits of ``q``.
    target_logits : torch.Tensor
        Logits of ``p``. Same shape as ``logits``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(log_q, p, s, g)``. ``log_q`` and ``p`` keep the computation graph,
        ``s = log(q / p)`` and ``g = q - p`` (with ``sum(g) = 0``) are detached.
    """
    log_q, log_p = log_softmax(logits), log_softmax(target_logits)
    q, p = torch.exp(log_q), torch.exp(log_p)
    with torch.no_grad():
        delta = logits - target_logits  # Logit difference.
        d = delta - torch.sum(p * delta, dim=-1, keepdim=True)  # Center under p.
        # c = log E_p[exp(d)] (= KL(p || q) + ...): log1p form for precision, logsumexp form when large.
        c_large = torch.logsumexp(log_p + d, dim=-1, keepdim=True)
        pe = torch.where(d < 1, p * torch.expm1(d.clamp(max=1)), torch.exp(log_p + d) - p)  # p * expm1(d).
        c = torch.where(c_large > 1, c_large, torch.log1p(torch.sum(pe, dim=-1, keepdim=True)))
        s = d - c  # log(q / p).
        u = torch.where(s > 0, -q * torch.expm1(-s.clamp(min=0)), p * torch.expm1(s.clamp(max=0)))  # q - p.
        g = u - q * torch.sum(u, dim=-1, keepdim=True)  # Ensure sum(g) = 0.
    return log_q, p, s, g


def gumbel_softmax(
    inputs: torch.Tensor,
    dim: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Gumbel softmax function with temperature. This prevents overflow and underflow.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor.
    dim : int
        Dimension to apply softmax.
    temperature : float
        Temperature. Default is 1.0.

    Returns
    -------
    torch.Tensor
        Gumbel softmaxed tensor (hard one-hot with straight-through gradient).

    Raises
    ------
    ValueError
        If the gumbel_softmax is inf or nan.
    """
    gumbels = -torch.empty_like(inputs).exponential_().log()  # Gumbel noise.
    y_soft = torch.exp(log_softmax(inputs + gumbels, dim=dim, temperature=temperature))
    index = torch.argmax(y_soft, dim=dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    # Straight-through estimator
    x = y_hard - y_soft.detach() + y_soft
    if torch.isinf(x).any() or torch.isnan(x).any():
        msg = "gumbel_softmax is inf or nan"
        raise ValueError(msg)
    return x


def torch_fix_seed(seed: int = 42) -> None:
    """
    乱数を固定する関数.

    References
    ----------
    - https://qiita.com/north_redwing/items/1e153139125d37829d2d
    """
    random.seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SoftmaxTransformation:
    """Softmax 変換クラス."""

    def __init__(
        self,
        cfg: SoftmaxTransConfig,
    ) -> None:
        """
        SoftmaxTransformation の初期化.

        Args:
        -----
        cfg : SoftmaxTransConfig
            SoftmaxTransformation の設定.
        """
        super().__init__()
        self.vector = cfg.vector
        self.sigma = cfg.sigma
        self.n_ignore = cfg.n_ignore
        self.max = cfg.max
        self.min = cfg.min
        self.k = torch.linspace(self.min, self.max, self.vector)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    def get_transformed_dim(self, dim: int) -> int:
        return (dim - self.n_ignore) * self.vector + self.n_ignore

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        SoftmaxTransformation の実行.

        Args:
        -----
        x : torch.Tensor
            入力テンソル.

        Returns
        -------
        torch.Tensor
            出力テンソル.

        Examples
        --------
        >>> trans = SoftmaxTransformation(SoftmaxTransConfig(vector=16, sigma=0.01, n_ignore=1, min=-1.0, max=1.0))
        >>> x = torch.randn(2, 3, 4)
        >>> transformed = trans(x)
        >>> transformed.shape
        torch.Size([2, 3, 49])

        >>> trans = SoftmaxTransformation(SoftmaxTransConfig(vector=11, sigma=0.05, n_ignore=0, min=-1.0, max=1.0))
        >>> x = torch.randn(2, 3, 4)
        >>> transformed = trans(x)
        >>> transformed.shape
        torch.Size([2, 3, 44])

        """
        *batch, dim = x.shape
        x = x.reshape(-1, dim)
        if self.n_ignore:
            data, ignored = x[:, : -self.n_ignore], x[:, -self.n_ignore :]
        else:
            data = x

        negative = torch.stack([torch.exp((-((data - self.k[v]) ** 2)) / self.sigma) for v in range(self.vector)])
        negative_sum = negative.sum(dim=0)

        transformed = negative / (negative_sum + 1e-8)
        transformed = rearrange(transformed, "v b d -> b (d v)")

        transformed = torch.cat([transformed, ignored], dim=-1) if self.n_ignore else transformed
        return transformed.reshape(*batch, self.get_transformed_dim(dim))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        SoftmaxTransformation の逆変換.

        Args:
        -----
        x : torch.Tensor
            入力テンソル.

        Returns
        -------
        torch.Tensor
            出力テンソル.
        """
        *batch, dim = x.shape
        x = x.reshape(-1, dim)
        if self.n_ignore:
            data, ignored = x[:, : -self.n_ignore], x[:, -self.n_ignore :]
        else:
            data = x

        data = data.reshape([len(data), -1, self.vector])

        data = rearrange(data, "b d v -> v b d")

        data = torch.stack([data[v] * self.k[v] for v in range(self.vector)]).sum(dim=0)

        data = torch.cat([data, ignored], dim=-1) if self.n_ignore else data
        return data.reshape(*batch, -1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
