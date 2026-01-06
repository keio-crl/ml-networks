"""確率分布と確率的状態を扱うモジュール."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ml_networks.utils import save_blosc2, softmax


def get_dist(state: StochState) -> D.Independent:
    """確率的状態から分布を取得する.

    Parameters
    ----------
    state : StochState
        正規分布・カテゴリカル分布・ベルヌーイ分布のいずれかの確率的状態。

    Returns
    -------
    D.Independent
        与えられた状態に対応する`torch.distributions.Independent`オブジェクト。
    """
    if isinstance(state, NormalStoch):
        dist = D.Normal(state.mean, state.std)
        return D.Independent(dist, 1)
    if isinstance(state, CategoricalStoch):
        dist = D.OneHotCategoricalStraightThrough(probs=state.probs)
        return D.Independent(dist, 1)
    if isinstance(state, BernoulliStoch):
        dist = BernoulliStraightThrough(probs=state.probs)
        return D.Independent(dist, 2)
    raise NotImplementedError


@dataclass
class NormalShape:
    """
    Shapes of the parameters of a normal distribution and its stochastic sample.

    Attributes
    ----------
    mean : torch.Size
        Shape of the mean of the normal distribution.
    std : torch.Size
        Shape of the standard deviation of the normal distribution.
    stoch : torch.Size
        Shape of the sample from the normal distribution with reparametrization trick.

    """

    mean: torch.Size
    std: torch.Size
    stoch: torch.Size


@dataclass
class NormalStoch:
    """
    Parameters of a normal distribution and its stochastic sample.

    Attributes
    ----------
    mean : torch.Tensor
        Mean of the normal distribution.
    std : torch.Tensor
        Standard deviation of the normal distribution.
    stoch : torch.Tensor
        sample from the normal distribution with reparametrization trick.

    """

    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor

    def __post_init__(self) -> None:
        """初期化後の処理.

        Raises
        ------
        ValueError
            `mean` と `std` のshapeが異なる場合、または`std`に負の値が含まれる場合。
        """
        if self.mean.shape != self.std.shape:
            msg = f"mean.shape {self.mean.shape} and std.shape {self.std.shape} must be the same."
            raise ValueError(msg)
        if (self.std < 0).any():
            msg = "std must be non-negative."
            raise ValueError(msg)

    def __getitem__(self, idx: int | slice | tuple) -> NormalStoch:
        """インデックスアクセス.

        Parameters
        ----------
        idx : int or slice or tuple
            インデックス指定。

        Returns
        -------
        NormalStoch
            指定されたインデックスに対応する`NormalStoch`。
        """
        return NormalStoch(self.mean[idx], self.std[idx], self.stoch[idx])

    def __len__(self) -> int:
        """長さを返す.

        Returns
        -------
        int
            バッチ次元の長さ。
        """
        return self.stoch.shape[0]

    @property
    def shape(self) -> NormalShape:
        """mean, std, stoch の shape をタプルで返す."""
        return NormalShape(self.mean.shape, self.std.shape, self.stoch.shape)

    def __getattr__(self, name: str) -> Any:
        """torch.Tensor に含まれるメソッドを呼び出したら、各メンバに適用する.

        例: normal.flatten() → NormalStoch(mean.flatten(), std.flatten(), stoch.flatten()).

        Parameters
        ----------
        name : str
            メソッド名。

        Returns
        -------
        callable
            torch.Tensorのメソッドを各メンバに適用する関数。

        Raises
        ------
        AttributeError
            指定された名前がtorch.Tensorのメソッドでない場合。
        """
        if hasattr(torch.Tensor, name):  # torch.Tensor のメソッドか確認

            def method(*args: Any, **kwargs: Any) -> NormalStoch:
                return NormalStoch(
                    getattr(self.mean, name)(*args, **kwargs),
                    getattr(self.std, name)(*args, **kwargs),
                    getattr(self.stoch, name)(*args, **kwargs),
                )

            return method
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def unsqueeze(self, dim: int) -> NormalStoch:
        """
        Unsqueeze the parameters of the normal distribution.

        Parameters
        ----------
        dim : int
            Dimension to unsqueeze.

        Returns
        -------
        NormalStoch
            Unsqueezed normal distribution.

        """
        return NormalStoch(
            self.mean.unsqueeze(dim),
            self.std.unsqueeze(dim),
            self.stoch.unsqueeze(dim),
        )

    def squeeze(self, dim: int) -> NormalStoch:
        """
        Squeeze the parameters of the normal distribution.

        Parameters
        ----------
        dim : int
            Dimension to squeeze.

        Returns
        -------
        NormalStoch
            Squeezed normal distribution.

        """
        return NormalStoch(
            self.mean.squeeze(dim),
            self.std.squeeze(dim),
            self.stoch.squeeze(dim),
        )

    def save(self, path: str) -> None:
        """
        Save the parameters of the normal distribution to the specified path.

        Parameters
        ----------
        path : str
            Path to save the parameters.

        """
        os.makedirs(path, exist_ok=True)

        save_blosc2(f"{path}/mean.blosc2", self.mean.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/std.blosc2", self.std.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/stoch.blosc2", self.stoch.detach().clone().cpu().numpy())

    def get_distribution(self, independent: int = 1) -> D.Independent:
        return D.Independent(D.Normal(self.mean, self.std), independent)


@dataclass
class CategoricalShape:
    """
    Shapes of the parameters of a categorical distribution and its stochastic sample.

    Attributes
    ----------
    logits : torch.Size
        Shape of the logits of the categorical distribution.
    probs : torch.Size
        Shape of the probabilities of the categorical distribution.
    stoch : torch.Size
        Shape of the sample from the categorical distribution with Straight-Through Estimator.

    """

    logits: torch.Size
    probs: torch.Size
    stoch: torch.Size


@dataclass
class CategoricalStoch:
    """
    Parameters of a categorical distribution and its stochastic sample.

    Attributes
    ----------
    logits : torch.Tensor
        Logits of the categorical distribution.
    probs : torch.Tensor
        Probabilities of the categorical distribution.
    stoch : torch.Tensor
        sample from the categorical distribution with Straight-Through Estimator.

    """

    logits: torch.Tensor
    probs: torch.Tensor
    stoch: torch.Tensor

    def __post_init__(self) -> None:
        """初期化後の処理.

        Raises
        ------
        ValueError
            `logits` と `probs` のshapeが異なる場合、
            あるいは`probs`が[0, 1]の範囲外、または和が1から大きくずれている場合。
        """
        if self.logits.shape != self.probs.shape:
            msg = f"logits.shape {self.logits.shape} and probs.shape {self.probs.shape} must be the same."
            raise ValueError(msg)
        if (self.probs < 0).any() or (self.probs > 1).any():
            msg = "probs must be in the range [0, 1]."
            raise ValueError(msg)
        if (self.probs.sum(dim=-1) - 1).abs().max() > 1e-6:
            msg = "probs must sum to 1."
            raise ValueError(msg)

    def __getitem__(self, idx: int | slice | tuple) -> CategoricalStoch:
        """インデックスアクセス.

        Parameters
        ----------
        idx : int or slice or tuple
            インデックス指定。

        Returns
        -------
        CategoricalStoch
            指定されたインデックスに対応する`CategoricalStoch`。
        """
        return CategoricalStoch(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self) -> int:
        """長さを返す.

        Returns
        -------
        int
            バッチ次元の長さ。
        """
        return self.stoch.shape[0]

    def squeeze(self, dim: int) -> CategoricalStoch:
        """
        Squeeze the parameters of the categorical distribution.

        Parameters
        ----------
        dim : int
            Dimension to squeeze.

        Returns
        -------
        CategoricalStoch
            Squeezed categorical distribution.

        """
        return CategoricalStoch(
            self.logits.squeeze(dim),
            self.probs.squeeze(dim),
            self.stoch.squeeze(dim),
        )

    def unsqueeze(self, dim: int) -> CategoricalStoch:
        """
        Unsqueeze the parameters of the categorical distribution.

        Parameters
        ----------
        dim : int
            Dimension to unsqueeze.

        Returns
        -------
        CategoricalStoch
            Unsqueezed categorical distribution.

        """
        return CategoricalStoch(
            self.logits.unsqueeze(dim),
            self.probs.unsqueeze(dim),
            self.stoch.unsqueeze(dim),
        )

    @property
    def shape(self) -> CategoricalShape:
        """mean, std, stoch の shape をタプルで返す."""
        return CategoricalShape(self.logits.shape, self.probs.shape, self.stoch.shape)

    def __getattr__(self, name: str) -> Any:
        """torch.Tensor に含まれるメソッドを呼び出したら、各メンバに適用する.

        例: normal.flatten() → NormalStoch(mean.flatten(), std.flatten(), stoch.flatten()).

        Parameters
        ----------
        name : str
            メソッド名。

        Returns
        -------
        callable
            torch.Tensorのメソッドを各メンバに適用する関数。

        Raises
        ------
        AttributeError
            指定された名前がtorch.Tensorのメソッドでない場合。
        """
        if hasattr(torch.Tensor, name):  # torch.Tensor のメソッドか確認

            def method(*args: Any, **kwargs: Any) -> CategoricalStoch:
                return CategoricalStoch(
                    getattr(self.logits, name)(*args, **kwargs),
                    getattr(self.probs, name)(*args, **kwargs),
                    getattr(self.stoch, name)(*args, **kwargs),
                )

            return method
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def save(self, path: str) -> None:
        """
        Save the parameters of the categorical distribution to the specified path.

        Parameters
        ----------
        path : str
            Path to save the parameters.
        """
        os.makedirs(path, exist_ok=True)

        save_blosc2(f"{path}/logits.blosc2", self.logits.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/probs.blosc2", self.probs.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/stoch.blosc2", self.stoch.detach().clone().cpu().numpy())

    def get_distribution(self, independent: int = 1) -> D.Independent:
        return D.Independent(D.OneHotCategoricalStraightThrough(self.probs), independent)


@dataclass
class BernoulliShape:
    """
    Shapes of the parameters of a Bernoulli distribution and its stochastic sample.

    Attributes
    ----------
    logits : torch.Size
        Shape of the logits of the Bernoulli distribution.
    probs : torch.Size
        Shape of the probabilities of the Bernoulli distribution.
    stoch : torch.Size
        Shape of the sample from the Bernoulli distribution with Straight-Through Estimator.

    """

    logits: torch.Size
    probs: torch.Size
    stoch: torch.Size


@dataclass
class BernoulliStoch:
    """
    Parameters of a Bernoulli distribution and its stochastic sample.

    Attributes
    ----------
    logits : torch.Tensor
        Logits of the Bernoulli distribution.
    probs : torch.Tensor
        Probabilities of the Bernoulli distribution.
    stoch : torch.Tensor
        sample from the Bernoulli distribution with Straight-Through Estimator.

    """

    logits: torch.Tensor
    probs: torch.Tensor
    stoch: torch.Tensor

    def __post_init__(self) -> None:
        """初期化後の処理.

        Raises
        ------
        ValueError
            `logits` と `probs` のshapeが異なる場合、あるいは`probs`が[0, 1]の範囲外の場合。
        """
        if self.logits.shape != self.probs.shape:
            msg = f"logits.shape {self.logits.shape} and probs.shape {self.probs.shape} must be the same."
            raise ValueError(msg)
        if (self.probs < 0).any() or (self.probs > 1).any():
            msg = "probs must be in the range [0, 1]."
            raise ValueError(msg)

    def __getitem__(self, idx: int | slice | tuple) -> CategoricalStoch:
        """インデックスアクセス.

        Parameters
        ----------
        idx : int or slice or tuple
            インデックス指定。

        Returns
        -------
        CategoricalStoch
            指定されたインデックスに対応する`CategoricalStoch`。
        """
        return CategoricalStoch(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self) -> int:
        """長さを返す.

        Returns
        -------
        int
            バッチ次元の長さ。
        """
        return self.stoch.shape[0]

    @property
    def shape(self) -> BernoulliShape:
        """mean, std, stoch の shape をタプルで返す."""
        return BernoulliShape(self.logits.shape, self.probs.shape, self.stoch.shape)

    def __getattr__(self, name: str) -> Any:
        """torch.Tensor に含まれるメソッドを呼び出したら、各メンバに適用する.

        例: normal.flatten() → NormalStoch(mean.flatten(), std.flatten(), stoch.flatten()).

        Parameters
        ----------
        name : str
            メソッド名。

        Returns
        -------
        callable
            torch.Tensorのメソッドを各メンバに適用する関数。

        Raises
        ------
        AttributeError
            指定された名前がtorch.Tensorのメソッドでない場合。
        """
        if hasattr(torch.Tensor, name):  # torch.Tensor のメソッドか確認

            def method(*args: Any, **kwargs: Any) -> CategoricalStoch:
                return CategoricalStoch(
                    getattr(self.logits, name)(*args, **kwargs),
                    getattr(self.probs, name)(*args, **kwargs),
                    getattr(self.stoch, name)(*args, **kwargs),
                )

            return method
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def squeeze(self, dim: int) -> BernoulliStoch:
        """
        Squeeze the parameters of the Bernoulli distribution.

        Parameters
        ----------
        dim : int
            Dimension to squeeze.

        Returns
        -------
        BernoulliStoch
            Squeezed Bernoulli distribution.

        """
        return BernoulliStoch(
            self.logits.squeeze(dim),
            self.probs.squeeze(dim),
            self.stoch.squeeze(dim),
        )

    def unsqueeze(self, dim: int) -> BernoulliStoch:
        """
        Unsqueeze the parameters of the Bernoulli distribution.

        Parameters
        ----------
        dim : int
            Dimension to unsqueeze.

        Returns
        -------
        BernoulliStoch
            Unsqueezed Bernoulli distribution.

        """
        return BernoulliStoch(
            self.logits.unsqueeze(dim),
            self.probs.unsqueeze(dim),
            self.stoch.unsqueeze(dim),
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        save_blosc2(f"{path}/logits.blosc2", self.logits.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/probs.blosc2", self.probs.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/stoch.blosc2", self.stoch.detach().clone().cpu().numpy())

    def get_distribution(self, independent: int = 1) -> D.Independent:
        return D.Independent(BernoulliStraightThrough(self.probs), independent)


StochState = NormalStoch | CategoricalStoch | BernoulliStoch


def cat_dist(stochs: tuple[StochState, ...], dim: int = -1) -> StochState | None:
    """
    Concatenate the parameters of the distributions.

    Parameters
    ----------
    stochs : Tuple[StochState, ...]
        Tuple of the distributions.
    dim : int, optional
        Dimension to concatenate the parameters of the distributions.
        Default is -1.

    Returns
    -------
    StochState
        Concatenated distribution.

    Examples
    --------
    >>> dist1 = NormalStoch(torch.randn(2, 3), torch.rand(2, 3), torch.randn(2, 3))
    >>> dist2 = NormalStoch(torch.randn(2, 3), torch.rand(2, 3), torch.randn(2, 3))
    >>> cat_dist = cat_dist((dist1, dist2))
    >>> cat_dist.shape
    NormalShape(mean=torch.Size([2, 6]), std=torch.Size([2, 6]), stoch=torch.Size([2, 6]))

    """
    if isinstance(stochs[0], NormalStoch):
        return NormalStoch(
            torch.cat([stoch.mean for stoch in stochs], dim=dim),
            torch.cat([stoch.std for stoch in stochs], dim=dim),
            torch.cat([stoch.stoch for stoch in stochs], dim=dim),
        )
    if isinstance(stochs[0], CategoricalStoch):
        return CategoricalStoch(
            torch.cat([stoch.logits for stoch in stochs], dim=dim),
            torch.cat([stoch.probs for stoch in stochs], dim=dim),
            torch.cat([stoch.stoch for stoch in stochs], dim=dim),
        )
    if isinstance(stochs[0], BernoulliStoch):
        return BernoulliStoch(
            torch.cat([stoch.logits for stoch in stochs], dim=dim),
            torch.cat([stoch.probs for stoch in stochs], dim=dim),
            torch.cat([stoch.stoch for stoch in stochs], dim=dim),
        )
    return None


def stack_dist(stochs: tuple[StochState, ...], dim: int = 0) -> StochState | None:
    """
    Stack the parameters of the distributions.

    Parameters
    ----------
    stochs : Tuple[StochState, ...]
        Tuple of the distributions.
    dim : int, optional
        Dimension to stack the parameters of the distributions. Default is 0.

    Returns
    -------
    StochState
        Stacked distribution.

    Examples
    --------
    >>> dist1 = NormalStoch(torch.randn(2, 3), torch.rand(2, 3), torch.randn(2, 3))
    >>> dist2 = NormalStoch(torch.randn(2, 3), torch.rand(2, 3), torch.randn(2, 3))
    >>> stack_dist = stack_dist((dist1, dist2))
    >>> stack_dist.shape
    NormalShape(mean=torch.Size([2, 2, 3]), std=torch.Size([2, 2, 3]), stoch=torch.Size([2, 2, 3]))

    """
    if isinstance(stochs[0], NormalStoch):
        return NormalStoch(
            torch.stack([stoch.mean for stoch in stochs], dim=dim),
            torch.stack([stoch.std for stoch in stochs], dim=dim),
            torch.stack([stoch.stoch for stoch in stochs], dim=dim),
        )
    if isinstance(stochs[0], CategoricalStoch):
        return CategoricalStoch(
            torch.stack([stoch.logits for stoch in stochs], dim=dim),
            torch.stack([stoch.probs for stoch in stochs], dim=dim),
            torch.stack([stoch.stoch for stoch in stochs], dim=dim),
        )
    if isinstance(stochs[0], BernoulliStoch):
        return BernoulliStoch(
            torch.stack([stoch.logits for stoch in stochs], dim=dim),
            torch.stack([stoch.probs for stoch in stochs], dim=dim),
            torch.stack([stoch.stoch for stoch in stochs], dim=dim),
        )
    return None


class BernoulliStraightThrough(D.Bernoulli):
    has_rsample = True

    def rsample(
        self,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """
        Generate a sample from the Bernoulli distribution with Straight-Through Estimator.

        Parameters
        ----------
        sample_shape : torch.Size, optional
            Shape of the sample. Default is torch.Size().

        Returns
        -------
        torch.Tensor
            Sample from the Bernoulli distribution with Straight-Through Estimator.

        Examples
        --------
        >>> logits = torch.randn(2, 3)
        >>> probs = torch.sigmoid(logits)
        >>> dist = BernoulliStraightThrough(probs=probs)
        >>> sample = dist.rsample()
        >>> sample.shape
        torch.Size([2, 3])

        """
        samples = self.sample(sample_shape)
        probs = self.probs
        return samples + (probs - probs.detach())


class Distribution(nn.Module):
    """
    A distribution function.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    dist : Literal["normal", "categorical", "bernoulli"]
        Distribution type.
    n_groups : int, optional
        Number of groups. Default is 1. This is used for the categorical and Bernoulli distributions.
    spherical : bool, optional
        Whether to project samples to the unit sphere. Default is False.
        This is used for the categorical and Bernoulli distributions.
        If True and dist=="categorical", the samples are projected from {0, 1} to {-1, 1}.
        If True and dist=="bernoulli", the samples are projected from {0, 1} to the unit sphere.

        refer to https://arxiv.org/abs/2406.07548

    Examples
    --------
    >>> dist = Distribution(10, "normal")
    >>> data = torch.randn(2, 20)
    >>> posterior = dist(data)
    >>> posterior.__class__.__name__
    'NormalStoch'
    >>> posterior.shape
    NormalShape(mean=torch.Size([2, 10]), std=torch.Size([2, 10]), stoch=torch.Size([2, 10]))

    >>> dist = Distribution(10, "categorical", n_groups=2)
    >>> data = torch.randn(2, 10)
    >>> posterior = dist(data)
    >>> posterior.__class__.__name__
    'CategoricalStoch'
    >>> posterior.shape
    CategoricalShape(logits=torch.Size([2, 2, 5]), probs=torch.Size([2, 2, 5]), stoch=torch.Size([2, 10]))

    >>> dist = Distribution(10, "bernoulli", n_groups=2)
    >>> data = torch.randn(2, 10)
    >>> posterior = dist(data)
    >>> posterior.__class__.__name__
    'BernoulliStoch'
    >>> posterior.shape
    BernoulliShape(logits=torch.Size([2, 2, 5]), probs=torch.Size([2, 2, 5]), stoch=torch.Size([2, 10]))

    """

    def __init__(
        self,
        in_dim: int,
        dist: Literal["normal", "categorical", "bernoulli"],
        n_groups: int = 1,
        spherical: bool = False,
    ) -> None:
        super().__init__()

        self.dist = dist
        self.spherical = spherical
        self.n_class = in_dim // n_groups
        self.in_dim = in_dim
        self.n_groups = n_groups

        if dist == "normal":
            self.posterior = self.normal  # type: ignore[assignment]
        elif dist == "categorical":
            self.posterior = self.categorical  # type: ignore[assignment]
        elif dist == "bernoulli":
            self.posterior = self.bernoulli  # type: ignore[assignment]
        else:
            raise NotImplementedError

        if spherical:
            self.codebook = BSQCodebook(self.n_class)

    def normal(self, mu_std: torch.Tensor, deterministic: bool = False, inv_tmp: float = 1.0) -> NormalStoch:
        assert mu_std.shape[-1] == self.in_dim * 2, (
            f"mu_std.shape[-1] {mu_std.shape[-1]} and in_dim {self.in_dim} must be the same."
        )

        mu, std = torch.chunk(mu_std, 2, dim=-1)
        std = F.softplus(std) + 1e-6

        posterior_dist = D.Normal(mu, std)
        posterior_dist = D.Independent(posterior_dist, 1)

        sample = posterior_dist.rsample() if not deterministic else mu

        return NormalStoch(mu, std, sample if not deterministic else mu)

    def categorical(self, logits: torch.Tensor, deterministic: bool = False, inv_tmp: float = 1.0) -> CategoricalStoch:
        batch_shape = logits.shape[:-1]
        logits_chunk = torch.chunk(logits, self.n_groups, dim=-1)
        logits = torch.stack(logits_chunk, dim=-2)
        logits = logits
        probs = softmax(logits, dim=-1, temperature=1 / inv_tmp)
        posterior_dist = D.OneHotCategoricalStraightThrough(probs=probs)
        posterior_dist = D.Independent(posterior_dist, 1)

        sample = posterior_dist.rsample()

        if self.spherical:
            sample = sample * 2 - 1

        return CategoricalStoch(
            logits,
            probs,
            sample.reshape([*batch_shape, -1])
            if not deterministic
            else self.deterministic_onehot(probs).reshape([*batch_shape, -1]),
        )

    def bernoulli(self, logits: torch.Tensor, deterministic: bool = False, inv_tmp: float = 1.0) -> BernoulliStoch:
        batch_shape = logits.shape[:-1]
        chunked_logits = torch.chunk(logits, self.n_groups, dim=-1)
        logits = torch.stack(chunked_logits, dim=-2)
        logits = logits * inv_tmp
        probs = torch.sigmoid(logits)

        posterior_dist = BernoulliStraightThrough(probs=probs)
        posterior_dist = D.Independent(posterior_dist, 1)

        sample = posterior_dist.rsample()

        if self.spherical:
            sample = self.codebook.bits_to_codes(sample)

        if deterministic:
            sample = (
                torch.where(sample > 0.5, torch.ones_like(sample), torch.zeros_like(sample)) + probs - probs.detach()
            )

        return BernoulliStoch(
            logits,
            probs,
            sample.reshape([*batch_shape, -1]),
        )

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        inv_tmp: float = 1.0,
    ) -> StochState:
        """
        Compute the posterior distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        deterministic : bool, optional
            Whether to use the deterministic mode. Default is False.
            if True and dist=="normal", the mean is returned.
            if True and dist=="categorical", the one-hot vector computed by argmax is returned.
            if True and dist=="bernoulli", 1 is returned if x > 0.5 or 0 is returned if x <= 0.5.

        inv_tmp : float, optional
            Inverse temperature. Default is 1.0.
            This is used for the categorical and Bernoulli distributions.

        Returns
        -------
        StochState
            Posterior distribution.


        """
        return self.posterior(x, deterministic=deterministic, inv_tmp=inv_tmp)

    def deterministic_onehot(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the one-hot vector by argmax.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            One-hot vector.

        Examples
        --------
        >>> input = torch.arange(6).reshape(2, 3) / 5.0
        >>> dist = Distribution(3, "categorical")
        >>> onehot = dist.deterministic_onehot(input)
        >>> onehot
        tensor([[0., 0., 1.],
                [0., 0., 1.]])
        """
        return F.one_hot(input.argmax(dim=-1), num_classes=self.n_class) + input - input.detach()


class BSQCodebook(nn.Module):
    """
    Binary Spherical Quantization codebook.

    Reference
    ---------
    https://arxiv.org/abs/2406.07548

    Parameters
    ----------
    codebook_dim : int
        Dimension of the codebook.

    Attributes
    ----------
    codebook_dim : int
        Dimension of the codebook.
    codebook_size : int
        Size of the codebook. This is equal to 2 ** codebook_dim.
    codebook : torch.Tensor
        Codebook.

    """

    def __init__(
        self,
        codebook_dim: int,
    ) -> None:
        super().__init__()
        self.codebook_dim = codebook_dim
        self.codebook_size = 2**codebook_dim
        self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)
        self.register_buffer("codebook", codebook.float(), persistent=False)

    @staticmethod
    def bits_to_codes(bits: torch.Tensor) -> torch.Tensor:
        """Convert bits to codes, which are bits of either 0 or 1.

        Parameters
        ----------
        bits : torch.Tensor
            Bits of either 0 or 1.

        Returns
        -------
        torch.Tensor
            Codes, which are bits depending on codebook_dim(dimension of the sphery)

        Examples
        --------
        >>> bits = torch.tensor([[0., 1.], [1., 0.]])
        >>> BSQCodebook.bits_to_codes(bits)
        tensor([[-0.7071,  0.7071],
                [ 0.7071, -0.7071]])

        >>> bits = torch.tensor([[0., 0., 1.], [1., 0., 0.]])
        >>> BSQCodebook.bits_to_codes(bits)
        tensor([[-0.5774, -0.5774,  0.5774],
                [ 0.5774, -0.5774, -0.5774]])


        """
        bits = bits * 2 - 1
        return F.normalize(bits, dim=-1)

    def indices_to_codes(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert indices to codes, which are bits of either -1 or 1.

        Parameters
        ----------
        indices : torch.Tensor
            Indices.

        Returns
        -------
        torch.Tensor
            Codes, which are bits depending on codebook_dim(dimension of the sphery)

        Examples
        --------
        >>> indices = torch.tensor([[31], [19]])
        >>> codebook = BSQCodebook(5)
        >>> codebook.indices_to_codes(indices)
        tensor([[ 0.4472,  0.4472,  0.4472,  0.4472,  0.4472],
                [ 0.4472, -0.4472, -0.4472,  0.4472,  0.4472]])

        """
        indices = indices.squeeze(-1)

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).float()

        return self.bits_to_codes(bits)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
