"""コントラスティブ学習を扱うモジュール."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ml_networks.config import ContrastiveLearningConfig
from ml_networks.jax.layers import MLPLayer


class ContrastiveLearningLoss(nnx.Module):
    """
    Contrastive learning module.

    Parameters
    ----------
    dim_input1 : int
        Dimension of first input.
    dim_input2 : int
        Dimension of second input.
    cfg : ContrastiveLearningConfig
        Configuration for contrastive learning.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        dim_input1: int,
        dim_input2: int,
        cfg: ContrastiveLearningConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = cfg
        self.dim_feature = cfg.dim_feature
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.is_ce_like = cfg.cross_entropy_like

        self.eval_func = MLPLayer(dim_input1, cfg.dim_feature, cfg.eval_func, rngs=rngs)
        if self.dim_input1 != self.dim_input2:
            self.eval_func2 = MLPLayer(dim_input2, cfg.dim_feature, cfg.eval_func, rngs=rngs)
        else:
            self.eval_func2 = self.eval_func

    def calc_timeseries_nce(
        self,
        feature1: jax.Array,
        feature2: jax.Array,
        positive_range_self: int = 0,
        positive_range_tgt: int = 0,
        return_emb: bool = False,
    ) -> dict[str, jax.Array] | tuple[dict[str, jax.Array], tuple[jax.Array, jax.Array]]:
        """
        Calculate the NCE loss for time series data.

        Parameters
        ----------
        feature1 : jax.Array
            First input tensor of shape (*batch, length, dim_input1)
        feature2 : jax.Array
            Second input tensor of shape (*batch, length, dim_input2)
        positive_range_self : int
            Range for self-positive samples. Default is 0.
        positive_range_tgt : int
            Range for target-positive samples. Default is 0.
        return_emb : bool
            Whether to return embeddings. Default is False.

        Returns
        -------
        dict or tuple
            Loss dictionary, optionally with embeddings.
        """
        if not positive_range_self and not positive_range_tgt:
            return self.calc_nce(feature1, feature2, return_emb)

        feature1 = feature1.reshape(-1, feature1.shape[-2], self.dim_input1)
        feature2 = feature2.reshape(-1, feature2.shape[-2], self.dim_input2)
        batch, length, _ = feature1.shape

        emb_1 = self.eval_func(feature1)
        emb_2 = self.eval_func2(feature2)

        loss_dict: dict[str, jax.Array] = {}

        positive = jnp.sum(
            emb_1.reshape(-1, emb_1.shape[-1]) * emb_2.reshape(-1, emb_2.shape[-1]),
            axis=-1,
        )
        loss_dict["positive"] = jax.lax.stop_gradient(positive).mean()

        if positive_range_self > 0:
            self_positive_1, self_positive_2 = self._calculate_self_positive_pairs(
                emb_1, emb_2, batch, length, positive_range_self,
            )
            positive = positive + self_positive_1.reshape(-1) + self_positive_2.reshape(-1)
            loss_dict["self_positive_1"] = jax.lax.stop_gradient(self_positive_1).mean()
            loss_dict["self_positive_2"] = jax.lax.stop_gradient(self_positive_2).mean()

        if positive_range_tgt > 0:
            tgt_positive = self._calculate_target_positive_pairs(
                emb_1, emb_2, batch, length, positive_range_tgt,
            )
            positive = positive + tgt_positive.reshape(-1)
            loss_dict["tgt_positive"] = jax.lax.stop_gradient(tgt_positive).mean()

        flat_emb1 = emb_1.reshape(-1, emb_1.shape[-1])
        flat_emb2 = emb_2.reshape(-1, emb_2.shape[-1])
        sim_matrix = flat_emb1 @ flat_emb2.T
        negative = jax.nn.logsumexp(sim_matrix, axis=-1) - np.log(len(sim_matrix))
        loss_dict["negative"] = jax.lax.stop_gradient(negative).mean()

        nce_loss = -positive + negative
        nce_loss = nce_loss.mean()
        loss_dict["nce"] = nce_loss

        if return_emb:
            return loss_dict, (emb_1, emb_2)
        return loss_dict

    def _calculate_self_positive_pairs(
        self,
        emb_1: jax.Array,
        emb_2: jax.Array,
        batch: int,
        length: int,
        positive_range: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Calculate self-positive pairs for time series data."""
        self_positive_1_list = []
        self_positive_2_list = []
        for i in range(batch):
            sim1 = emb_1[i] @ emb_1[i].T
            sim2 = emb_2[i] @ emb_2[i].T
            self_pos_1 = jnp.stack([
                sim1[j, max(0, j - positive_range) : j + positive_range + 1].mean()
                for j in range(length)
            ])
            self_positive_1_list.append(self_pos_1)
            self_pos_2 = jnp.stack([
                sim2[j, max(0, j - positive_range) : j + positive_range + 1].mean()
                for j in range(length)
            ])
            self_positive_2_list.append(self_pos_2)
        return jnp.stack(self_positive_1_list), jnp.stack(self_positive_2_list)

    def _calculate_target_positive_pairs(
        self,
        emb_1: jax.Array,
        emb_2: jax.Array,
        batch: int,
        length: int,
        positive_range: int,
    ) -> jax.Array:
        """Calculate target-positive pairs for time series data."""
        tgt_positive_list = []
        for i in range(batch):
            sim = emb_1[i] @ emb_2[i].T
            tgt_pos = jnp.stack([
                sim[j, max(0, j - positive_range) : j + positive_range + 1].mean()
                for j in range(length)
            ])
            tgt_positive_list.append(tgt_pos)
        return jnp.stack(tgt_positive_list)

    def calc_nce(
        self,
        feature1: jax.Array,
        feature2: jax.Array,
        return_emb: bool = False,
    ) -> dict[str, jax.Array] | tuple[dict[str, jax.Array], tuple[jax.Array, jax.Array]]:
        """
        Calculate the Noise Contrastive Estimation (NCE) loss.

        Parameters
        ----------
        feature1 : jax.Array
            First input tensor of shape (*, dim_input1)
        feature2 : jax.Array
            Second input tensor of shape (*, dim_input2)
        return_emb : bool
            Whether to return embeddings. Default is False.

        Returns
        -------
        dict or tuple
            Loss dictionary, optionally with embeddings.
        """
        loss_dict: dict[str, jax.Array] = {}
        batch_shape = feature1.shape[:-1]
        emb_1 = self.eval_func(feature1.reshape(-1, self.dim_input1))
        emb_2 = self.eval_func2(feature2.reshape(-1, self.dim_input2))

        if self.is_ce_like:
            labels = jnp.arange(len(emb_1))
            sim_matrix = emb_1 @ emb_2.T
            # Cross entropy loss
            log_softmax = jax.nn.log_softmax(sim_matrix, axis=-1)
            nce_loss = -log_softmax[jnp.arange(len(labels)), labels] - np.log(len(sim_matrix))
            loss_dict["nce"] = nce_loss
        else:
            positive = jnp.sum(emb_1 * emb_2, axis=-1)
            loss_dict["positive"] = jax.lax.stop_gradient(positive).mean()

            sim_matrix = emb_1 @ emb_2.T
            negative = jax.nn.logsumexp(sim_matrix, axis=-1) - np.log(len(sim_matrix))
            loss_dict["negative"] = jax.lax.stop_gradient(negative).mean()

            nce_loss = -positive + negative
            loss_dict["nce"] = nce_loss.reshape(batch_shape)

        if return_emb:
            return loss_dict, (emb_1, emb_2)
        return loss_dict

    def calc_sigmoid(
        self,
        feature1: jax.Array,
        feature2: jax.Array,
        return_emb: bool = False,
        temperature: float | jax.Array = 0.1,
        bias: float | jax.Array = 0.0,
    ) -> dict[str, jax.Array] | tuple[dict[str, jax.Array], tuple[jax.Array, jax.Array]]:
        """
        Calculate the Sigmoid loss for contrastive learning.

        Parameters
        ----------
        feature1 : jax.Array
            First input tensor of shape (*, dim_input1)
        feature2 : jax.Array
            Second input tensor of shape (*, dim_input2)
        return_emb : bool
            Whether to return embeddings. Default is False.
        temperature : float
            Temperature. Default is 0.1.
        bias : float
            Bias. Default is 0.0.

        Returns
        -------
        dict or tuple
            Loss dictionary, optionally with embeddings.
        """
        loss_dict: dict[str, jax.Array] = {}
        batch_shape = feature1.shape[:-1]
        emb_1 = self.eval_func(feature1.reshape(-1, self.dim_input1))
        emb_2 = self.eval_func2(feature2.reshape(-1, self.dim_input2))

        logits = emb_1 @ emb_2.T * temperature + bias
        labels = jnp.eye(len(logits)) * 2 - 1
        loss = -jax.nn.log_sigmoid(logits * labels).sum(axis=-1)
        loss_dict["sigmoid"] = loss.reshape(batch_shape)
        if return_emb:
            return loss_dict, (emb_1, emb_2)
        return loss_dict


if __name__ == "__main__":
    import doctest

    doctest.testmod()
