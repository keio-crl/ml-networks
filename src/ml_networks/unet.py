from __future__ import annotations
from typing import Any, Dict, List, Literal, Tuple, Union, Optional

import numpy as np
import torch
import torch.func as tf
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange

from ml_networks.layers import (
    ConvNormActivation, ConvNormActivation1d,
    HorizonUnShuffle, HorizonShuffle,
)
from ml_networks.config import ConvConfig, UNetConfig, MLPConfig
from ml_networks.hypernetworks import HyperNet

class ConditionalUnet2d(nn.Module):
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
    ...     has_attn=True,
    ...     nhead=8,
    ...     cond_pred_scale=True
    ... )
    >>> net = ConditionalUnet2d(feature_dim=32, obs_shape=(3, 64, 64), cfg=cfg)
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
            ConditionalResidualBlock2d(
                mid_dim, mid_dim, cond_dim=feature_dim,
                conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale
            ),
            Attention2d(mid_dim, cfg.nhead) if cfg.has_attn else nn.Identity(),
            ConditionalResidualBlock2d(
                mid_dim, mid_dim, cond_dim=feature_dim,
                conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock2d(
                    dim_in, dim_out, cond_dim=feature_dim, 
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale),
                Attention2d(dim_out, cfg.nhead) if cfg.has_attn else nn.Identity(),
                ConditionalResidualBlock2d(
                    dim_out, dim_out, cond_dim=feature_dim, 
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale),
                Downsample2d(dim_out, cfg.use_shuffle) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock2d(
                    dim_out*2, dim_in, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale),
                Attention2d(dim_in, cfg.nhead) if cfg.has_attn else nn.Identity(),
                ConditionalResidualBlock2d(
                    dim_in, dim_in, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale),
                Upsample2d(dim_in, cfg.use_shuffle) if not is_last else nn.Identity()
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
        assert base.shape[-3:] == self.obs_shape, \
            f"Input shape {base.shape[-3:]} does not match expected shape {self.obs_shape}"
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

class ConditionalUnet1d(nn.Module):
    """条件付き1D UNetモデル。

    Args:
        feature_dim (int): 条件付き特徴量の次元数
        obs_shape (tuple[int, int]): 観測データの形状 (チャンネル数, 長さ)
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
    ...     has_attn=True,
    ...     nhead=8,
    ...     cond_pred_scale=True
    ... )
    >>> net = ConditionalUnet1d(feature_dim=32, obs_shape=(3, 64), cfg=cfg)
    >>> x = torch.randn(2, 3, 64)
    >>> cond = torch.randn(2, 32)
    >>> out = net(x, cond)
    >>> out.shape
    torch.Size([2, 3, 64])
    """

    def __init__(self, 
        feature_dim: int,
        obs_shape: tuple[int, int],
        cfg: UNetConfig,
    ):
        super().__init__()
        all_dims = [obs_shape[0]] + list(cfg.channels)
        start_dim = cfg.channels[0]
        self.obs_shape = obs_shape

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1d(
                mid_dim, mid_dim, cond_dim=feature_dim,
                conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale
            ) if not cfg.use_hypernet else HyperConditionalResidualBlock1d(
                mid_dim, mid_dim, cond_dim=feature_dim,
                conv_cfg=cfg.conv_cfg,
                hyper_mlp_cfg=cfg.hyper_mlp_cfg
            ),
            Attention1d(mid_dim, cfg.nhead) if cfg.has_attn else nn.Identity(),
            ConditionalResidualBlock1d(
                mid_dim, mid_dim, cond_dim=feature_dim,
                conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale
            ) if not cfg.use_hypernet else HyperConditionalResidualBlock1d(
                mid_dim, mid_dim, cond_dim=feature_dim,
                conv_cfg=cfg.conv_cfg,
                hyper_mlp_cfg=cfg.hyper_mlp_cfg
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1d(
                    dim_in, dim_out, cond_dim=feature_dim, 
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale) \
                if not cfg.use_hypernet else HyperConditionalResidualBlock1d(
                    dim_in, dim_out, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    hyper_mlp_cfg=cfg.hyper_mlp_cfg
                ),
                Attention1d(dim_out, cfg.nhead) if cfg.has_attn else nn.Identity(),
                ConditionalResidualBlock1d(
                    dim_out, dim_out, cond_dim=feature_dim, 
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale) \
                if not cfg.use_hypernet else HyperConditionalResidualBlock1d(
                    dim_out, dim_out, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    hyper_mlp_cfg=cfg.hyper_mlp_cfg
                ),
                Downsample1d(dim_out, cfg.use_shuffle) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1d(
                    dim_out*2, dim_in, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale) \
                if not cfg.use_hypernet else HyperConditionalResidualBlock1d(
                    dim_out*2, dim_in, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    hyper_mlp_cfg=cfg.hyper_mlp_cfg
                ),
                Attention1d(dim_in, cfg.nhead) if cfg.has_attn else nn.Identity(),
                ConditionalResidualBlock1d(
                    dim_in, dim_in, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale) \
                if not cfg.use_hypernet else HyperConditionalResidualBlock1d(
                    dim_in, dim_in, cond_dim=feature_dim,
                    conv_cfg=cfg.conv_cfg,
                    hyper_mlp_cfg=cfg.hyper_mlp_cfg
                ),
                Upsample1d(dim_in, cfg.use_shuffle) if not is_last else nn.Identity()
            ]))
        final_conv = nn.Sequential(
            ConvNormActivation1d(start_dim, start_dim, cfg.conv_cfg),
            nn.Conv1d(start_dim, obs_shape[0], 1),
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
        x: (B,input_dim, T)
        t: (B,) or int, diffusion step
        cond: (B,cond_dim)
        output: (B,T,input_dim)
        """
        batch_shape = base.shape[:-2]
        assert base.shape[-2:] == self.obs_shape, \
            f"Input shape {base.shape[-2:]} does not match expected shape {self.obs_shape}"
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
            if isinstance(mid_module, nn.Identity):
                x = mid_module(x)
            else:
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

class ConditionalResidualBlock2d(nn.Module):
    """条件付き2d残差ブロック。

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
    >>> block = ConditionalResidualBlock2d(64, 128, cond_dim=32, conv_cfg=cfg)
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

class HyperConditionalResidualBlock2d(ConditionalResidualBlock2d):
    """条件付き2d残差ブロック（ハイパーネットワーク版）。
    Args:
        in_channels (int): 入力チャンネル数
        out_channels (int): 出力チャンネル数
        cond_dim (int): 条件付き特徴量の次元数
        conv_cfg (ConvConfig): 畳み込み層の設定
        hyper_mlp_cfg (Optional[MLPConfig]): ハイパーネットワークのMLP設定
    Examples
    --------
    >>> from ml_networks.config import ConvConfig, MLPConfig, LinearConfig
    >>> cfg = ConvConfig(
    ...     kernel_size=3,
    ...     padding=1,
    ...     stride=1,
    ...     groups=1,
    ...     activation="ReLU",
    ...     dropout=0.0
    ... )
    >>> hyper_mlp_cfg = MLPConfig(
    ...     hidden_dim=128,
    ...     n_layers=2,
    ...     output_activation="Identity",
    ...     linear_cfg=LinearConfig(
    ...         activation="ReLU",
    ...         dropout=0.0,
    ...     )
    ... )
    >>> block = HyperConditionalResidualBlock2d(64, 128, cond_dim=32, conv_cfg=cfg, hyper_mlp_cfg=hyper_mlp_cfg)
    >>> x = torch.randn(2, 64, 32, 32)
    >>> cond = torch.randn(2, 32)
    >>> out = block(x, cond)
    >>> out.shape
    torch.Size([2, 128, 32, 32])
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_dim: int,
        conv_cfg: ConvConfig,
        hyper_mlp_cfg: Optional[MLPConfig] = None
    ):
        super().__init__(in_channels, out_channels, cond_dim, conv_cfg)

        self.conv_layer = nn.Sequential(
                self.first_conv, self.last_conv)

        self.cond_encoder = HyperNet(
            cond_dim, self.conv_layer.state_dict(), hyper_mlp_cfg)

    def functional_call(self, param: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        out = tf.functional_call(self.conv_layer, param, x)
        return out


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        param = self.cond_encoder(cond)
        out = torch.vmap(self.functional_call)(param, x)
        out = out + self.residual_conv(x)
        return out

class ConditionalResidualBlock1d(nn.Module):
    """条件付き1d残差ブロック。

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
    >>> block = ConditionalResidualBlock1d(64, 128, cond_dim=32, conv_cfg=cfg)
    >>> x = torch.randn(2, 64, 32)
    >>> cond = torch.randn(2, 32)
    >>> out = block(x, cond)
    >>> out.shape
    torch.Size([2, 128, 32])
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_dim: int,
        conv_cfg: ConvConfig,
        cond_predict_scale=False
    ):
        super().__init__()
        self.first_conv = ConvNormActivation1d(
            in_channels, out_channels, conv_cfg)
        self.last_conv = ConvNormActivation1d(   
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
            Rearrange('batch c -> batch c 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
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
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.last_conv(out)
        out = out + self.residual_conv(x)
        return out

class HyperConditionalResidualBlock1d(ConditionalResidualBlock1d):
    """条件付き1d残差ブロック（ハイパーネットワーク版）。
    Args:
        in_channels (int): 入力チャンネル数
        out_channels (int): 出力チャンネル数
        cond_dim (int): 条件付き特徴量の次元数
        conv_cfg (ConvConfig): 畳み込み層の設定
        hyper_mlp_cfg (Optional[MLPConfig]): ハイパーネットワークのMLP設定
    Examples
    --------
    >>> from ml_networks.config import ConvConfig, MLPConfig, LinearConfig
    >>> cfg = ConvConfig(
    ...     kernel_size=3,
    ...     padding=1,
    ...     stride=1,
    ...     groups=1,
    ...     activation="ReLU",
    ...     dropout=0.0
    ... )
    >>> hyper_mlp_cfg = MLPConfig(
    ...     hidden_dim=128,
    ...     n_layers=2,
    ...     output_activation="Identity",
    ...     linear_cfg=LinearConfig(
    ...         activation="ReLU",
    ...         dropout=0.0,
    ...     )
    ... )
    >>> block = HyperConditionalResidualBlock1d(64, 128, cond_dim=32, conv_cfg=cfg, hyper_mlp_cfg=hyper_mlp_cfg)
    >>> x = torch.randn(2, 64, 32)
    >>> cond = torch.randn(2, 32)
    >>> out = block(x, cond)
    >>> out.shape
    torch.Size([2, 128, 32])
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cond_dim: int,
        conv_cfg: ConvConfig,
        hyper_mlp_cfg: Optional[MLPConfig] = None
    ):
        super().__init__(in_channels, out_channels, cond_dim, conv_cfg)

        self.conv_layer = nn.Sequential(
                self.first_conv, self.last_conv)

        self.cond_encoder = HyperNet(
            cond_dim, self.conv_layer.state_dict(), hyper_mlp_cfg)

    def functional_call(self, param: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        out = tf.functional_call(self.conv_layer, param, x)
        return out

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        param = self.cond_encoder(cond)
        out = torch.vmap(self.functional_call)(param, x)
        out = out + self.residual_conv(x)
        return out

class Attention2d(nn.Module):
    """2d自己注意機構。

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

class Attention1d(nn.Module):
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
        self.qkv = ConvNormActivation1d(channels, channels * 3, first_cfg)

        self.proj_out = ConvNormActivation1d(
            channels, channels, cfg
        )

    def qkv_attention(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight - torch.max(weight, dim=-1, keepdim=True)[0], dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    def forward(self, x, *args):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(x)
        h = self.qkv_attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class Downsample2d(nn.Module):
    """2dダウンサンプリング層。

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
    def __init__(self, dim: int, use_shuffle: bool = False):
        super().__init__()
        if use_shuffle:
            self.conv = nn.Sequential(
                nn.PixelUnshuffle(2),
                nn.Conv2d(dim * 4, dim, 3, 1, 1)  # Reduce channels after unshuffle
            )
        else:
            # Conv2d is used for downsampling
            # with stride 2 and padding 1
            # to ensure the output size is half the input size
            self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample1d(nn.Module):
    """1Dダウンサンプリング層。

    Args:
        dim (int): 入力・出力チャンネル数

    Examples
    --------
    >>> downsample = Downsample1d(dim=64)
    >>> x = torch.randn(2, 64, 32)
    >>> out = downsample(x)
    >>> out.shape
    torch.Size([2, 64, 16])
    """
    def __init__(self, dim: int, use_shuffle: bool = False):
        super().__init__()
        if use_shuffle:
            self.conv = nn.Sequential(
                HorizonUnShuffle(2),
                nn.Conv1d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample2d(nn.Module):
    """2dアップサンプリング層。

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
    def __init__(self, dim: int, use_shuffle: bool = False):
        super().__init__()
        if use_shuffle:
            self.conv = nn.Sequential(
                    nn.Conv2d(dim, dim * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
            )
        else:
            # ConvTranspose2d is used for upsampling
            # with stride 2 and padding 1
            # to ensure the output size is double the input size
            self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        
    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    """1Dアップサンプリング層。

    Args:
        dim (int): 入力・出力チャンネル数

    Examples
    --------
    >>> upsample = Upsample1d(dim=64)
    >>> x = torch.randn(2, 64, 16)
    >>> out = upsample(x)
    >>> out.shape
    torch.Size([2, 64, 32])
    """
    def __init__(self, dim: int, use_shuffle: bool = False):
        super().__init__()
        if use_shuffle:
            self.conv = nn.Sequential(
                nn.Conv1d(dim, dim * 2, kernel_size=3, stride=1, padding=1),
                HorizonShuffle(2),
            )
        else:
            self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
