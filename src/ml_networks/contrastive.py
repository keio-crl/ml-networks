from __future__ import annotations
from typing import Any, Dict, List, Literal, Tuple, Union, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ml_networks.config import ContrastiveLearningConfig, MLPConfig, LinearConfig
from ml_networks.layers import MLPLayer


class ContrastiveLearningLoss(pl.LightningModule):
    """
    Contrastive learning module.

    Parameters
    ----------
    cfg : ContrastiveLearningConfig
        Configuration for contrastive learning.

    Examples
    --------
    >>> from ml_networks.config import ContrastiveLearningConfig, MLPConfig, LinearConfig
    >>> cfg = ContrastiveLearningConfig(
    ...     dim_feature=128,
    ...     dim_input1=256,
    ...     dim_input2=256,
    ...     eval_func=MLPConfig(
    ...         hidden_dim=256,
    ...         n_layers=2,
    ...         output_activation="ReLU",
    ...         linear_cfg=LinearConfig(
    ...             activation="ReLU",
    ...             norm="layer",
    ...             norm_cfg={"eps": 1e-05, "elementwise_affine": True, "bias": True},
    ...             dropout=0.1,
    ...             norm_first=False,
    ...             bias=True
    ...         )
    ...     ),
    ...     cross_entropy_like=False
    ... )
    >>> model = ContrastiveLearningLoss(cfg)
    >>> x1 = torch.randn(2, 256)
    >>> x2 = torch.randn(2, 256)
    >>> output = model.calc_nce(x1, x2)
    >>> output["nce"].shape
    torch.Size([])
    >>> output, embeddings = model.calc_nce(x1, x2, return_emb=True)
    >>> embeddings[0].shape, embeddings[1].shape
    (torch.Size([2, 128]), torch.Size([2, 128]))
    """

    def __init__(
        self,
        dim_input1: int,
        dim_input2: int,
        cfg: ContrastiveLearningConfig
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim_feature = cfg.dim_feature
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.is_ce_like = cfg.cross_entropy_like

        self.eval_func = MLPLayer(dim_input1, cfg.dim_feature, cfg.eval_func)
        if self.dim_input1 != self.dim_input2:
            self.eval_func2 = MLPLayer(dim_input2, cfg.dim_feature, cfg.eval_func)
        else:
            self.eval_func2 = self.eval_func

    def calc_timeseries_nce(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor,
        positive_range_self: int = 0,
        positive_range_tgt: int = 0,
        return_emb: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Calculate the Noise Contrastive Estimation (NCE) loss for time series data.

        Parameters
        ----------
        feature1 : torch.Tensor
            First input tensor of shape (*batch, length, dim_input1)
        feature2 : torch.Tensor
            Second input tensor of shape (*batch, length, dim_input2)
        positive_range_self : int, optional
            Range for self-positive samples, by default 0
        positive_range_tgt : int, optional
            Range for target-positive samples, by default 0
        return_emb : bool, optional
            Whether to return embeddings, by default False

        Returns
        -------
        Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
            If return_emb is False, returns loss dictionary.
            If return_emb is True, returns (loss dictionary, (embeddings1, embeddings2))
        """
        if not positive_range_self and not positive_range_tgt:
            return self.calc_nce(feature1, feature2, return_emb)

        # Reshape inputs
        feature1 = feature1.reshape(-1, feature1.shape[-2], self.dim_input1)
        feature2 = feature2.reshape(-1, feature2.shape[-2], self.dim_input2)
        batch, length, _ = feature1.shape

        # Calculate embeddings
        emb_1 = self.eval_func(feature1)
        emb_2 = self.eval_func2(feature2)

        # Initialize loss dictionary
        loss_dict: Dict[str, torch.Tensor] = {}

        # Calculate positive pairs
        positive = torch.sum(emb_1.flatten(0, 1) * emb_2.flatten(0, 1), dim=-1)  # (batch*length)
        loss_dict["positive"] = positive.detach().clone().mean()

        # Calculate self-positive pairs if needed
        if positive_range_self > 0:
            self_positive_1, self_positive_2 = self._calculate_self_positive_pairs(
                emb_1, emb_2, batch, length, positive_range_self
            )
            positive += self_positive_1.flatten(0, 1) + self_positive_2.flatten(0, 1)
            loss_dict["self_positive_1"] = self_positive_1.detach().clone().mean()
            loss_dict["self_positive_2"] = self_positive_2.detach().clone().mean()

        # Calculate target-positive pairs if needed
        if positive_range_tgt > 0:
            tgt_positive = self._calculate_target_positive_pairs(
                emb_1, emb_2, batch, length, positive_range_tgt
            )
            positive += tgt_positive.flatten(0, 1)
            loss_dict["tgt_positive"] = tgt_positive.detach().clone().mean()

        # Calculate negative pairs
        sim_matrix = torch.mm(emb_1.flatten(0, 1), emb_2.flatten(0, 1).T)
        negative = torch.logsumexp(sim_matrix, dim=-1) - np.log(len(sim_matrix))
        loss_dict["negative"] = negative.detach().clone().mean()

        # Calculate final loss
        nce_loss = -positive + negative
        nce_loss = nce_loss.mean()
        loss_dict["nce"] = nce_loss

        if return_emb:
            return loss_dict, (emb_1, emb_2)
        return loss_dict

    def _calculate_self_positive_pairs(
        self,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        batch: int,
        length: int,
        positive_range: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate self-positive pairs for time series data.

        Parameters
        ----------
        emb_1 : torch.Tensor
            First embeddings
        emb_2 : torch.Tensor
            Second embeddings
        batch : int
            Batch size
        length : int
            Sequence length
        positive_range : int
            Range for positive pairs

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Self-positive pairs for both embeddings
        """
        self_positive_1 = []
        self_positive_2 = []
        for i in range(batch):
            self_pos_1 = torch.stack([
                torch.mm(emb_1[i], emb_1[i].T)[j, j-positive_range:j+positive_range+1].mean()
                for j in range(length)
            ])
            self_positive_1.append(self_pos_1)
            self_pos_2 = torch.stack([
                torch.mm(emb_2[i], emb_2[i].T)[j, j-positive_range:j+positive_range+1].mean()
                for j in range(length)
            ])
            self_positive_2.append(self_pos_2)
        return torch.stack(self_positive_1), torch.stack(self_positive_2)

    def _calculate_target_positive_pairs(
        self,
        emb_1: torch.Tensor,
        emb_2: torch.Tensor,
        batch: int,
        length: int,
        positive_range: int,
    ) -> torch.Tensor:
        """
        Calculate target-positive pairs for time series data.

        Parameters
        ----------
        emb_1 : torch.Tensor
            First embeddings
        emb_2 : torch.Tensor
            Second embeddings
        batch : int
            Batch size
        length : int
            Sequence length
        positive_range : int
            Range for positive pairs

        Returns
        -------
        torch.Tensor
            Target-positive pairs
        """
        tgt_positive = []
        for i in range(batch):
            tgt_pos = torch.stack([
                torch.mm(emb_1[i], emb_2[i].T)[j, j-positive_range:j+positive_range+1].mean()
                for j in range(length)
            ])
            tgt_positive.append(tgt_pos)
        return torch.stack(tgt_positive)

    def calc_nce(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor,
        return_emb: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Calculate the Noise Contrastive Estimation (NCE) loss.

        Parameters
        ----------
        feature1 : torch.Tensor
            First input tensor of shape (*, dim_input1)
        feature2 : torch.Tensor
            Second input tensor of shape (*, dim_input2)
        return_emb : bool, optional
            Whether to return embeddings, by default False

        Returns
        -------
        Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]
            If return_emb is False, returns loss dictionary.
            If return_emb is True, returns (loss dictionary, (embeddings1, embeddings2))
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        batch_shape = feature1.shape[:-1]
        emb_1 = self.eval_func(feature1.reshape(-1, self.dim_input1))
        emb_2 = self.eval_func2(feature2.reshape(-1, self.dim_input2))

        if self.is_ce_like:
            labels = torch.arange(len(emb_1), device=emb_1.device)
            sim_matrix = torch.mm(emb_1, emb_2.T)
            nce_loss = F.cross_entropy(sim_matrix, labels, reduction="none") - np.log(len(sim_matrix))
            loss_dict["nce"] = nce_loss
        else:
            positive = torch.sum(emb_1 * emb_2, dim=-1)
            loss_dict["positive"] = positive.detach().clone().mean()

            sim_matrix = torch.mm(emb_1, emb_2.T)
            negative = torch.logsumexp(sim_matrix, dim=-1) - np.log(len(sim_matrix))
            loss_dict["negative"] = negative.detach().clone().mean()

            nce_loss = -positive + negative
            loss_dict["nce"] = nce_loss.reshape(batch_shape)

        if return_emb:
            return loss_dict, (emb_1, emb_2)
        return loss_dict


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    




