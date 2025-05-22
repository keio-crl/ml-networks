import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ml_networks.config import ContrastiveLearningConfig
from ml_networks.layers import MLPLayer

class ContrastiveLearningLoss(nn.Module):
    """
    Contrastive Learning Loss
    Args:
        cfg (ContrastiveLearningConfig): Configuration object for contrastive learning.


    """

    def __init__(self, cfg: ContrastiveLearningConfig):
        """

        """
        super(ContrastiveLearningLoss, self).__init__()
        self.dim_feature = cfg.dim_feature
        self.dim_input1 = cfg.dim_input1
        self.dim_input2 = cfg.dim_input2
        self.is_ce_like = cfg.cross_entropy_like

        self.eval_func = MLPLayer(
            self.dim_input1,
            self.dim_feature,
            cfg.eval_func
        )
        if cfg._shuld_diferrent_func:
            self.eval_func2 = MLPLayer(
                self.dim_input2,
                self.dim_feature,
                cfg.eval_func
            )
        else:
            self.eval_func2 = self.eval_func

    def calc_timeseries_nce(
            self, 
            feature1: torch.Tensor, 
            feature2: torch.Tensor, 
            positive_range_self: int = 0,
            positive_range_tgt: int = 0,
            return_emb: bool = False
        ):
        """
        Calculate the Noise Contrastive Estimation (NCE) loss for the given features.

        Args:
            feature1 (torch.Tensor): The first feature tensor of shape (*batch, length, dim_input1).
            feature2 (torch.Tensor): The second feature tensor of shape (*batch, length, dim_input2).
            positive_range_self (int): The range for self-positive samples.
            positive_range_tgt (int): The range for target-positive samples.
            return_emb (bool): Whether to return the embeddings or not.
        """
        if not positive_range_self and not positive_range_tgt:
            return self.calc_nce(feature1, feature2, return_emb)

        feature1 = feature1.view(-1, feature1.shape[-2], self.dim_input1)
        feature2 = feature2.view(-1, feature2.shape[-2], self.dim_input2)
        batch, length, _ = feature1.shape

        loss_dict = {}
        emb_1 = self.eval_func(feature1)
        emb_2 = self.eval_func2(feature2)

        positive = torch.sum(emb_1.flatten(0,1) * emb_2.flatten(0,1), dim=-1) # (batch*length)
        loss_dict["positive"] = positive.detach().clone().mean()

        if positive_range_self > 0:
            self_positive_1 = []
            self_positive_2 = []
            for i in range(batch):
                
                self_pos_1 = torch.stack(
                        [torch.mm(emb_1[i], emb_1[i].T)[j, j-positive_range_self:j+positive_range_self+1].mean() for j in range(length)]
                ) # (length)
                self_positive_1.append(self_pos_1)
                self_pos_2 = torch.stack(
                        [torch.mm(emb_2[i], emb_2[i].T)[j, j-positive_range_self:j+positive_range_self+1].mean() for j in range(length)]
                ) # (length)
                self_positive_2.append(self_pos_2)
               
            self_positive_1 = torch.stack(self_positive_1) # (batch, length)
            self_positive_2 = torch.stack(self_positive_2) # (batch, length)
            positive += self_positive_1.flatten(0,1) + self_positive_2.flatten(0,1)
            loss_dict["self_positive_1"] = self_positive_1.detach().clone().mean()
            loss_dict["self_positive_2"] = self_positive_2.detach().clone().mean()

        if positive_range_tgt > 0:
            tgt_positive = []
            for i in range(batch):
                tgt_pos = torch.stack(
                        [torch.mm(emb_1[i], emb_2[i].T)[j, j-positive_range_tgt:j+positive_range_tgt+1].mean() for j in range(length)]
                ) # (length)
                tgt_positive.append(tgt_pos)
            tgt_positive = torch.stack(tgt_positive) # (batch, length)
            positive += tgt_positive.flatten(0,1)
            loss_dict["tgt_positive"] = tgt_positive.detach().clone().mean()

        sim_matrix = torch.mm(emb_1.flatten(0,1), emb_2.flatten(0,1).T)

        negative = torch.logsumexp(
            sim_matrix, dim=-1) - np.log(len(sim_matrix))
        loss_dict["negative"] = negative.detach().clone().mean()
        nce_loss = -positive + negative
        nce_loss = nce_loss.mean()
        loss_dict["nce"] = nce_loss
        
        if return_emb:
            return loss_dict, (emb_1, emb_2)
        else:
            return loss_dict

    def calc_nce(
            self, 
            feature1: torch.Tensor, 
            feature2: torch.Tensor, 
            return_emb: bool = False
        ):
        """
        Calculate the Noise Contrastive Estimation (NCE) loss for the given features.
        Args:
            feature1 (torch.Tensor): The first feature tensor of shape (*, dim_input1).
            feature2 (torch.Tensor): The second feature tensor of shape (*, dim_input2).
            return_emb (bool): Whether to return the embeddings or not.
        Returns:
            loss_dict (dict): A dictionary containing the loss values.
            (optional) embeddings (tuple): A tuple containing the embeddings of feature1 and feature2.
        """

        loss_dict = {}
        emb_1 = self.eval_func(feature1.view(-1, self.dim_input1))
        emb_2 = self.eval_func2(feature2.view(-1, self.dim_input2))
        if self.is_ce_like:

            labels = (
                torch.Tensor(list(range(len(emb_1))))
                .long()
                .to(emb_1.device)
            )

            sim_matrix = torch.mm(emb_1, emb_2.T)

            nce_loss = F.cross_entropy(sim_matrix, labels, reduction="mean") - np.log(
                len(sim_matrix)
            )
            loss_dict["nce"] = nce_loss

        else:
            positive = torch.sum(emb_1 * emb_2, dim=-1)
            loss_dict["positive"] = positive.detach().clone().mean()

            sim_matrix = torch.mm(emb_1, emb_2.T)

            negative = torch.logsumexp(
                sim_matrix, dim=-1) - np.log(len(sim_matrix))
            loss_dict["negative"] = negative.detach().clone().mean()
            nce_loss = -positive + negative
            nce_loss = nce_loss.mean()
            loss_dict["nce"] = nce_loss
        
        if return_emb:
            return loss_dict, (emb_1, emb_2)
        else:
            return loss_dict
    




