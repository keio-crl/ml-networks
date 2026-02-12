"""確率分布と確率的状態を扱うモジュール."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from ml_networks.jax.jax_utils import softmax
from ml_networks.utils import save_blosc2


def get_dist(state: StochState) -> distrax.Distribution:
    """確率的状態から分布を取得する.

    Parameters
    ----------
    state : StochState
        正規分布・カテゴリカル分布・ベルヌーイ分布のいずれかの確率的状態。

    Returns
    -------
    distrax.Distribution
        与えられた状態に対応するdistraxオブジェクト。
    """
    if isinstance(state, NormalStoch):
        return distrax.Independent(distrax.Normal(state.mean, state.std), 1)
    if isinstance(state, CategoricalStoch):
        return distrax.Independent(
            distrax.OneHotCategorical(probs=state.probs),
            1,
        )
    if isinstance(state, BernoulliStoch):
        return distrax.Independent(distrax.Bernoulli(probs=state.probs), 2)
    raise NotImplementedError


@dataclass
class NormalShape:
    """Shapes of the parameters of a normal distribution and its stochastic sample."""

    mean: tuple[int, ...]
    std: tuple[int, ...]
    stoch: tuple[int, ...]


@dataclass
class NormalStoch:
    """Parameters of a normal distribution and its stochastic sample."""

    mean: jax.Array
    std: jax.Array
    stoch: jax.Array

    def __post_init__(self) -> None:
        if self.mean.shape != self.std.shape:
            msg = f"mean.shape {self.mean.shape} and std.shape {self.std.shape} must be the same."
            raise ValueError(msg)
        if (self.std < 0).any():
            msg = "std must be non-negative."
            raise ValueError(msg)

    def __getitem__(self, idx: int | slice | tuple) -> NormalStoch:
        return NormalStoch(self.mean[idx], self.std[idx], self.stoch[idx])

    def __len__(self) -> int:
        return self.stoch.shape[0]

    @property
    def shape(self) -> NormalShape:
        return NormalShape(self.mean.shape, self.std.shape, self.stoch.shape)

    def detach(self) -> NormalStoch:
        """Stop gradient equivalent."""
        return NormalStoch(
            jax.lax.stop_gradient(self.mean),
            jax.lax.stop_gradient(self.std),
            jax.lax.stop_gradient(self.stoch),
        )

    def reshape(self, *shape: int) -> NormalStoch:
        return NormalStoch(
            self.mean.reshape(*shape),
            self.std.reshape(*shape),
            self.stoch.reshape(*shape),
        )

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> NormalStoch:
        """Flatten along specified dimensions."""
        ndim = self.mean.ndim
        if end_dim < 0:
            end_dim = ndim + end_dim
        new_shape = self.mean.shape[:start_dim] + (-1,) + self.mean.shape[end_dim + 1 :]
        return self.reshape(*new_shape)

    def save(self, path: str) -> None:
        import numpy as np

        os.makedirs(path, exist_ok=True)
        save_blosc2(f"{path}/mean.blosc2", np.asarray(self.mean))
        save_blosc2(f"{path}/std.blosc2", np.asarray(self.std))
        save_blosc2(f"{path}/stoch.blosc2", np.asarray(self.stoch))

    def get_distribution(self, independent: int = 1) -> distrax.Distribution:
        return distrax.Independent(distrax.Normal(self.mean, self.std), independent)


@dataclass
class CategoricalShape:
    """Shapes of the parameters of a categorical distribution and its stochastic sample."""

    logits: tuple[int, ...]
    probs: tuple[int, ...]
    stoch: tuple[int, ...]


@dataclass
class CategoricalStoch:
    """Parameters of a categorical distribution and its stochastic sample."""

    logits: jax.Array
    probs: jax.Array
    stoch: jax.Array

    def __post_init__(self) -> None:
        if self.logits.shape != self.probs.shape:
            msg = f"logits.shape {self.logits.shape} and probs.shape {self.probs.shape} must be the same."
            raise ValueError(msg)

    def __getitem__(self, idx: int | slice | tuple) -> CategoricalStoch:
        return CategoricalStoch(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self) -> int:
        return self.stoch.shape[0]

    def squeeze(self, axis: int) -> CategoricalStoch:
        return CategoricalStoch(
            jnp.squeeze(self.logits, axis=axis),
            jnp.squeeze(self.probs, axis=axis),
            jnp.squeeze(self.stoch, axis=axis),
        )

    def detach(self) -> CategoricalStoch:
        """Stop gradient equivalent."""
        return CategoricalStoch(
            jax.lax.stop_gradient(self.logits),
            jax.lax.stop_gradient(self.probs),
            jax.lax.stop_gradient(self.stoch),
        )

    def reshape(self, *shape: int) -> CategoricalStoch:
        return CategoricalStoch(
            self.logits.reshape(*shape),
            self.probs.reshape(*shape),
            self.stoch.reshape(*shape),
        )

    @property
    def shape(self) -> CategoricalShape:
        return CategoricalShape(self.logits.shape, self.probs.shape, self.stoch.shape)

    def save(self, path: str) -> None:
        import numpy as np

        os.makedirs(path, exist_ok=True)
        save_blosc2(f"{path}/logits.blosc2", np.asarray(self.logits))
        save_blosc2(f"{path}/probs.blosc2", np.asarray(self.probs))
        save_blosc2(f"{path}/stoch.blosc2", np.asarray(self.stoch))

    def get_distribution(self, independent: int = 1) -> distrax.Distribution:
        return distrax.Independent(
            distrax.OneHotCategorical(probs=self.probs),
            independent,
        )


@dataclass
class BernoulliShape:
    """Shapes of the parameters of a Bernoulli distribution and its stochastic sample."""

    logits: tuple[int, ...]
    probs: tuple[int, ...]
    stoch: tuple[int, ...]


@dataclass
class BernoulliStoch:
    """Parameters of a Bernoulli distribution and its stochastic sample."""

    logits: jax.Array
    probs: jax.Array
    stoch: jax.Array

    def __post_init__(self) -> None:
        if self.logits.shape != self.probs.shape:
            msg = f"logits.shape {self.logits.shape} and probs.shape {self.probs.shape} must be the same."
            raise ValueError(msg)

    def __getitem__(self, idx: int | slice | tuple) -> BernoulliStoch:
        return BernoulliStoch(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self) -> int:
        return self.stoch.shape[0]

    @property
    def shape(self) -> BernoulliShape:
        return BernoulliShape(self.logits.shape, self.probs.shape, self.stoch.shape)

    def squeeze(self, axis: int) -> BernoulliStoch:
        return BernoulliStoch(
            jnp.squeeze(self.logits, axis=axis),
            jnp.squeeze(self.probs, axis=axis),
            jnp.squeeze(self.stoch, axis=axis),
        )

    def detach(self) -> BernoulliStoch:
        """Stop gradient equivalent."""
        return BernoulliStoch(
            jax.lax.stop_gradient(self.logits),
            jax.lax.stop_gradient(self.probs),
            jax.lax.stop_gradient(self.stoch),
        )

    def reshape(self, *shape: int) -> BernoulliStoch:
        return BernoulliStoch(
            self.logits.reshape(*shape),
            self.probs.reshape(*shape),
            self.stoch.reshape(*shape),
        )

    def save(self, path: str) -> None:
        import numpy as np

        os.makedirs(path, exist_ok=True)
        save_blosc2(f"{path}/logits.blosc2", np.asarray(self.logits))
        save_blosc2(f"{path}/probs.blosc2", np.asarray(self.probs))
        save_blosc2(f"{path}/stoch.blosc2", np.asarray(self.stoch))

    def get_distribution(self, independent: int = 1) -> distrax.Distribution:
        return distrax.Independent(distrax.Bernoulli(probs=self.probs), independent)


StochState = NormalStoch | CategoricalStoch | BernoulliStoch


def cat_dist(stochs: tuple[StochState, ...], axis: int = -1) -> StochState | None:
    """
    Concatenate the parameters of the distributions.

    Parameters
    ----------
    stochs : Tuple[StochState, ...]
        Tuple of the distributions.
    axis : int, optional
        Axis to concatenate. Default is -1.

    Returns
    -------
    StochState
        Concatenated distribution.
    """
    if isinstance(stochs[0], NormalStoch):
        return NormalStoch(
            jnp.concatenate([s.mean for s in stochs], axis=axis),
            jnp.concatenate([s.std for s in stochs], axis=axis),
            jnp.concatenate([s.stoch for s in stochs], axis=axis),
        )
    if isinstance(stochs[0], CategoricalStoch):
        return CategoricalStoch(
            jnp.concatenate([s.logits for s in stochs], axis=axis),
            jnp.concatenate([s.probs for s in stochs], axis=axis),
            jnp.concatenate([s.stoch for s in stochs], axis=axis),
        )
    if isinstance(stochs[0], BernoulliStoch):
        return BernoulliStoch(
            jnp.concatenate([s.logits for s in stochs], axis=axis),
            jnp.concatenate([s.probs for s in stochs], axis=axis),
            jnp.concatenate([s.stoch for s in stochs], axis=axis),
        )
    return None


def stack_dist(stochs: tuple[StochState, ...], axis: int = 0) -> StochState | None:
    """
    Stack the parameters of the distributions.

    Parameters
    ----------
    stochs : Tuple[StochState, ...]
        Tuple of the distributions.
    axis : int, optional
        Axis to stack. Default is 0.

    Returns
    -------
    StochState
        Stacked distribution.
    """
    if isinstance(stochs[0], NormalStoch):
        return NormalStoch(
            jnp.stack([s.mean for s in stochs], axis=axis),
            jnp.stack([s.std for s in stochs], axis=axis),
            jnp.stack([s.stoch for s in stochs], axis=axis),
        )
    if isinstance(stochs[0], CategoricalStoch):
        return CategoricalStoch(
            jnp.stack([s.logits for s in stochs], axis=axis),
            jnp.stack([s.probs for s in stochs], axis=axis),
            jnp.stack([s.stoch for s in stochs], axis=axis),
        )
    if isinstance(stochs[0], BernoulliStoch):
        return BernoulliStoch(
            jnp.stack([s.logits for s in stochs], axis=axis),
            jnp.stack([s.probs for s in stochs], axis=axis),
            jnp.stack([s.stoch for s in stochs], axis=axis),
        )
    return None


class BSQCodebook(nnx.Module):
    """
    Binary Spherical Quantization codebook.

    Reference
    ---------
    https://arxiv.org/abs/2406.07548

    Parameters
    ----------
    codebook_dim : int
        Dimension of the codebook.
    """

    def __init__(self, codebook_dim: int) -> None:
        self.codebook_dim = codebook_dim
        self.codebook_size = 2**codebook_dim
        self.mask = 2 ** jnp.arange(codebook_dim - 1, -1, -1)
        all_codes = jnp.arange(self.codebook_size)
        bits = ((all_codes[..., None].astype(jnp.int32) & self.mask) != 0).astype(jnp.float32)
        self.codebook = self.bits_to_codes(bits)

    @staticmethod
    def bits_to_codes(bits: jax.Array) -> jax.Array:
        """Convert bits to codes on the unit sphere.

        Parameters
        ----------
        bits : jax.Array
            Bits of either 0 or 1.

        Returns
        -------
        jax.Array
            Codes on the unit sphere.
        """
        bits = bits * 2 - 1
        return bits / (jnp.linalg.norm(bits, axis=-1, keepdims=True) + 1e-12)

    def indices_to_codes(self, indices: jax.Array) -> jax.Array:
        """Convert indices to codes.

        Parameters
        ----------
        indices : jax.Array
            Indices.

        Returns
        -------
        jax.Array
            Codes on the unit sphere.
        """
        indices = jnp.squeeze(indices, axis=-1)
        bits = ((indices[..., None].astype(jnp.int32) & self.mask) != 0).astype(jnp.float32)
        return self.bits_to_codes(bits)


class Distribution(nnx.Module):
    """
    A distribution function.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    dist : Literal["normal", "categorical", "bernoulli"]
        Distribution type.
    n_groups : int, optional
        Number of groups. Default is 1.
    spherical : bool, optional
        Whether to project samples to the unit sphere. Default is False.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_dim: int,
        dist: Literal["normal", "categorical", "bernoulli"],
        n_groups: int = 1,
        spherical: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dist_type = dist
        self.spherical = spherical
        self.n_class = in_dim // n_groups
        self.in_dim = in_dim
        self.n_groups = n_groups
        self.rngs = rngs

        if spherical:
            self.codebook = BSQCodebook(self.n_class)

    def normal(self, mu_std: jax.Array, deterministic: bool = False, inv_tmp: float = 1.0) -> NormalStoch:
        assert mu_std.shape[-1] == self.in_dim * 2, (
            f"mu_std.shape[-1] {mu_std.shape[-1]} and in_dim {self.in_dim} must be the same."
        )

        mu, std = jnp.split(mu_std, 2, axis=-1)
        std = jax.nn.softplus(std) + 1e-6

        if deterministic:
            sample = mu
        else:
            key = self.rngs()
            sample = mu + std * jax.random.normal(key, mu.shape)

        return NormalStoch(mu, std, sample)

    def categorical(self, logits: jax.Array, deterministic: bool = False, inv_tmp: float = 1.0) -> CategoricalStoch:
        batch_shape = logits.shape[:-1]
        logits_chunks = jnp.split(logits, self.n_groups, axis=-1)
        logits_stacked = jnp.stack(logits_chunks, axis=-2)
        probs = softmax(logits_stacked, axis=-1, temperature=1 / inv_tmp)

        key = self.rngs()
        # Sample using Gumbel-max trick for one-hot with straight-through
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, probs.shape, minval=1e-10, maxval=1.0)))
        y = jnp.log(probs + 1e-10) + gumbel_noise
        hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), self.n_class)
        sample = hard + probs - jax.lax.stop_gradient(probs)

        if self.spherical:
            sample = sample * 2 - 1

        if deterministic:
            stoch = self.deterministic_onehot(probs).reshape(*batch_shape, -1)
        else:
            stoch = sample.reshape(*batch_shape, -1)

        return CategoricalStoch(logits_stacked, probs, stoch)

    def bernoulli(self, logits: jax.Array, deterministic: bool = False, inv_tmp: float = 1.0) -> BernoulliStoch:
        batch_shape = logits.shape[:-1]
        chunked_logits = jnp.split(logits, self.n_groups, axis=-1)
        logits_stacked = jnp.stack(chunked_logits, axis=-2)
        logits_stacked = logits_stacked * inv_tmp
        probs = jax.nn.sigmoid(logits_stacked)

        key = self.rngs()
        # Bernoulli sampling with straight-through
        u = jax.random.uniform(key, probs.shape)
        sample = (u < probs).astype(jnp.float32)
        # Straight-through estimator
        sample = sample + probs - jax.lax.stop_gradient(probs)

        if self.spherical:
            sample = self.codebook.bits_to_codes(sample)

        if deterministic:
            sample = (
                jnp.where(sample > 0.5, jnp.ones_like(sample), jnp.zeros_like(sample))
                + probs
                - jax.lax.stop_gradient(probs)
            )

        return BernoulliStoch(
            logits_stacked,
            probs,
            sample.reshape(*batch_shape, -1),
        )

    def __call__(
        self,
        x: jax.Array,
        deterministic: bool = False,
        inv_tmp: float = 1.0,
    ) -> StochState:
        """
        Compute the posterior distribution.

        Parameters
        ----------
        x : jax.Array
            Input tensor.
        deterministic : bool, optional
            Whether to use the deterministic mode. Default is False.
        inv_tmp : float, optional
            Inverse temperature. Default is 1.0.

        Returns
        -------
        StochState
            Posterior distribution.
        """
        if self.dist_type == "normal":
            return self.normal(x, deterministic=deterministic, inv_tmp=inv_tmp)
        elif self.dist_type == "categorical":
            return self.categorical(x, deterministic=deterministic, inv_tmp=inv_tmp)
        elif self.dist_type == "bernoulli":
            return self.bernoulli(x, deterministic=deterministic, inv_tmp=inv_tmp)
        raise NotImplementedError

    def deterministic_onehot(self, input: jax.Array) -> jax.Array:
        """Compute the one-hot vector by argmax with straight-through."""
        hard = jax.nn.one_hot(jnp.argmax(input, axis=-1), self.n_class)
        return hard + input - jax.lax.stop_gradient(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
