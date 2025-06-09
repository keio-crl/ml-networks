from __future__ import annotations
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchgeometry.contrib.spatial_soft_argmax2d import create_meshgrid, spatial_soft_argmax2d
from einops.layers.torch import Rearrange
from einops import rearrange

from ml_networks.layers import (
    ConvNormActivation,
)
from ml_networks.config import ConvConfig, UNetConfig

class ConditionalUnet(nn.Module):
    """条件付きUNetモデル。

    Args:
        feature_dim (int): 条件付き特徴量の次元数
        obs_shape (tuple[int, int, int]): 観測データの形状 (チャンネル数, 高さ, 幅)
        cfg (UNetConfig): UNetの設定

    Examples
    --------
    >>> from ml_networks.config import UNetConfig, ConvConfig, MLPConfig, LinearConfig
    >>> cfg = UNetConfig(
    ...     channels=[64, 128, 256],
    ...     conv_cfg=ConvConfig(
    ...         kernel_size=3,
    ...         padding=1,
    ...         stride=1,
    ...         groups=1,
    ...         activation="ReLU",
    ...         dropout=0.0
    ...     ),
    ...     cond_cfg=MLPConfig(
    ...         hidden_dim=128,
    ...         n_layers=2,
    ...         output_activation="ReLU",
    ...         linear_cfg=LinearConfig(
    ...             activation="ReLU",
    ...             dropout=0.0,
    ...         )
    ...     ),
    ...     has_attn=True,
    ...     nhead=8,
    ...     cond_pred_scale=True
    ... )
    >>> net = ConditionalUnet(feature_dim=32, obs_shape=(3, 64, 64), cfg=cfg)
    >>> x = torch.randn(2, 3, 64, 64)
    >>> cond = torch.randn(2, 32)
    >>> out = net(x, cond)
    >>> out.shape
    torch.Size([2, 3, 64, 64])
    """
    def __init__(self, 
        feature_dim: int,
        obs_shape: tuple[int, int, int],
        cfg: UNetConfig,
    ):
        super().__init__()
        all_dims = [obs_shape[0]] + list(cfg.channels)
        start_dim = cfg.channels[0]
        self.obs_shape = obs_shape

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=feature_dim,
                conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale
            ),
            Attention2d(mid_dim, cfg.nhead) if cfg.has_attn else nn.Identity(),
            ConditionalResidualBlock2D(
                mid_dim, mid_dim, cond_dim=feature_dim,
                conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    dim_in, dim_out, cond_dim=feature_dim, 
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale),
                Attention2d(dim_out, cfg.nhead) if cfg.has_attn else nn.Identity(),
                ConditionalResidualBlock2D(
                    dim_out, dim_out, cond_dim=feature_dim, 
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale),
                Downsample2d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock2D(
                    dim_out*2, dim_in, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale),
                Attention2d(dim_in, cfg.nhead) if cfg.has_attn else nn.Identity(),
                ConditionalResidualBlock2D(
                    dim_in, dim_in, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale),
                Upsample2d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            ConvNormActivation(start_dim, start_dim, cfg.conv_cfg),
            nn.Conv2d(start_dim, obs_shape[0], 1),
        )

        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv


    def forward(
            self, 
            base: torch.Tensor, 
            cond: torch.Tensor,
            ):
        """
        x: (B,T,input_dim)
        t: (B,) or int, diffusion step
        cond: (B,cond_dim)
        output: (B,T,input_dim)
        """
        batch_shape = base.shape[:-3]
        base = base.reshape(-1, *self.obs_shape) 

            
        global_feature = cond.reshape(-1, cond.shape[-1])
        
        
        x = base
        h = []
        for (resnet, attn, resnet2, downsample) in self.down_modules:
            x = resnet(x, global_feature)
            x = attn(x)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for (resnet, attn, resnet2, upsample) in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = attn(x)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = x.reshape(*batch_shape, *self.obs_shape)
        return x

class ConditionalResidualBlock2D(nn.Module):
    """条件付き2D残差ブロック。

    Args:
        in_channels (int): 入力チャンネル数
        out_channels (int): 出力チャンネル数
        cond_dim (int): 条件付き特徴量の次元数
        conv_cfg (ConvConfig): 畳み込み層の設定
        cond_predict_scale (bool): 条件付きスケール予測を行うかどうか

    Examples
    --------
    >>> from ml_networks.config import ConvConfig
    >>> cfg = ConvConfig(
    ...     kernel_size=3,
    ...     padding=1,
    ...     stride=1,
    ...     groups=1,
    ...     norm="group",
    ...     norm_cfg=dict(num_groups=8),
    ...     activation="ReLU",
    ...     dropout=0.0
    ... )
    >>> block = ConditionalResidualBlock2D(64, 128, cond_dim=32, conv_cfg=cfg)
    >>> x = torch.randn(2, 64, 32, 32)
    >>> cond = torch.randn(2, 32)
    >>> out = block(x, cond)
    >>> out.shape
    torch.Size([2, 128, 32, 32])
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        cond_dim,
        conv_cfg: ConvConfig,
        cond_predict_scale=False
    ):
        super().__init__()
        self.first_conv = ConvNormActivation(
            in_channels, out_channels, conv_cfg)
        self.last_conv = ConvNormActivation(   
            out_channels, out_channels, conv_cfg)

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch c -> batch c 1 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.first_conv(x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.last_conv(out)
        out = out + self.residual_conv(x)
        return out

class Attention2d(nn.Module):
    """2D自己注意機構。

    Args:
        channels (int): 入力・出力チャンネル数
        nhead (int): 注意ヘッドの数

    Examples
    --------
    >>> attn = Attention2d(channels=64, nhead=8)
    >>> x = torch.randn(2, 64, 32, 32)
    >>> out = attn(x)
    >>> out.shape
    torch.Size([2, 64, 32, 32])
    """
    def __init__(
        self,
        channels: int,
        nhead: int,
    ):
        super().__init__()
        self.channels = channels
        assert (
            channels % nhead == 0
        ), f"q,k,v channels {channels} is not divisible by num_head_channels {nhead}"
        self.n_heads = nhead
        cfg = ConvConfig(
            kernel_size=1,
            padding=0,
            stride=1,
            activation="Identity",
            dropout=0.0,
        )
        first_cfg = cfg
        first_cfg.norm_first = True
        self.qkv = ConvNormActivation(channels, channels * 3, first_cfg)

        self.proj_out = ConvNormActivation(
            channels, channels, cfg
        )

    def qkv_attn(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (Heads * 3 * C) x H x W] tensor of query, key, value. 
        :return: an [N x (C * Head) x H x W] tensor of attended values.
        """
        bs, channels, height, width = qkv.shape
        assert channels % (3 * self.n_heads) == 0
        ch = channels // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, height*width).split(ch, dim=1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight - torch.max(weight, dim=-1, keepdim=True)[0], dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, height, width)

    def forward(self, x, *args):
        b, c, *spatial = x.shape
        qkv = self.qkv(x)
        h = self.qkv_attn(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class Downsample2d(nn.Module):
    """2Dダウンサンプリング層。

    Args:
        dim (int): 入力・出力チャンネル数

    Examples
    --------
    >>> downsample = Downsample2d(dim=64)
    >>> x = torch.randn(2, 64, 32, 32)
    >>> out = downsample(x)
    >>> out.shape
    torch.Size([2, 64, 16, 16])
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample2d(nn.Module):
    """2Dアップサンプリング層。

    Args:
        dim (int): 入力・出力チャンネル数

    Examples
    --------
    >>> upsample = Upsample2d(dim=64)
    >>> x = torch.randn(2, 64, 16, 16)
    >>> out = upsample(x)
    >>> out.shape
    torch.Size([2, 64, 32, 32])
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        
    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
