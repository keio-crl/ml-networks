import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from .activations import Activation
from .config import (MLPConfig, ConvConfig, 
    TransformerConfig, LinearConfig, SpatialSoftmaxConfig)
from typing import Tuple
from einops import rearrange
from typing import Literal
from omegaconf import DictConfig
from kornia.geometry.subpix import spatial_expectation2d, spatial_softmax2d
from torchgeometry.contrib import spatial_soft_argmax2d


def get_norm(norm: Literal["layer", "rms", "group", "batch", "none"], **kwargs):
    if norm == "layer":
        return nn.LayerNorm(**kwargs)
    elif norm == "rms":
        return nn.RMSNorm(**kwargs)
    elif norm == "group":
        return nn.GroupNorm(**kwargs)
    elif norm == "batch":
        return nn.BatchNorm2d(**kwargs)
    else:
        return nn.Identity()

class LinearNormActivation(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        cfg: LinearConfig
        ):
        super().__init__()
        self.linear = nn.Linear(
            input_dim, 
            output_dim*2 if "glu" in cfg.activation.lower() else output_dim, 
            bias=cfg.bias
        )
        self.norm = get_norm(cfg.norm, **cfg.norm_cfg)
        self.activation = Activation(cfg.activation)
        if cfg.dropout > 0:
            self.dropout = nn.Dropout(cfg.dropout)
        else:
            self.dropout = nn.Identity()
        self.norm_first = cfg.norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = self.norm(x)
            x = self.linear(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x = self.linear(x)
            x = self.norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class MLPLayer(pl.LightningModule):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 cfg: MLPConfig
                 ):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.dense = self._build_dense()

    def forward(self, x: torch.Tensor):
        return self.dense(x)

    def _build_dense(self):
        layers = []
        layers += [LinearNormActivation(self.input_dim, self.hidden_dim, self.cfg.linear_cfg)]
        for _ in range(self.n_layers - 1):
            layers += [LinearNormActivation(self.hidden_dim, self.hidden_dim, self.cfg.linear_cfg)]
        last_cfg = self.cfg.linear_cfg
        last_cfg.activation = self.cfg.output_activation
        layers += [LinearNormActivation(self.hidden_dim, self.output_dim, last_cfg)]
        return nn.Sequential(*layers)

class ConvNormActivation(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cfg: ConvConfig
        ):
        super().__init__()
        kernel_size: int = cfg.kernel_size
        stride: int = cfg.stride
        padding: int = cfg.padding
        dilation: int = cfg.dilation
        groups: int = cfg.groups
        activation: str = cfg.activation
        bias: bool = cfg.bias
        dropout: float = cfg.dropout
        norm: Literal["batch", "group", "none"] = cfg.norm
        norm_cfg: DictConfig = cfg.norm_cfg
        scale_factor: int = cfg.scale_factor
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels*2 if "glu" in activation.lower() else out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups, 
            bias=bias
        )
        if norm != "none":
            norm_cfg["num_channels"] = out_channels
        self.norm = get_norm(norm, **norm_cfg)
        if scale_factor > 0:
            self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        elif scale_factor < 0:
            self.pixel_shuffle = nn.PixelUnshuffle(-scale_factor)
        else:
            self.pixel_shuffle = nn.Identity()
        self.activation = Activation(activation, dim=-3)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        kernel_size: int,
        activation: str = "ReLU",
        norm: Literal["batch", "group", "none"] = "none",
        norm_cfg: DictConfig = DictConfig({}),
        dropout: float = 0.0
        ):

        super(ResidualBlock, self).__init__()
        first_cfg = ConvConfig(
            activation=activation, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=(kernel_size-1)//2, 
            dilation=1, 
            groups=1, 
            bias=True, 
            dropout=dropout, 
            norm=norm, 
            norm_cfg=norm_cfg
        )
        second_cfg = ConvConfig(
            activation="Identity", 
            kernel_size=kernel_size, 
            stride=1, 
            padding=(kernel_size-1)//2, 
            dilation=1, 
            groups=1, 
            bias=True, 
            dropout=dropout, 
            norm=norm, 
            norm_cfg=norm_cfg
        )
        self.conv_block = nn.Sequential(
            ConvNormActivation(
                in_features, 
                in_features, 
                first_cfg
            ),
            ConvNormActivation(
                in_features, 
                in_features*2 if "glu" in activation.lower() else in_features, 
                second_cfg
            )
        )
        self.activation = Activation(activation, dim=-3)

    def forward(self, x):
        return self.activation(x + self.conv_block(x))

class ConvTransposeNormActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig
        ):
        super().__init__()
        kernel_size: int = cfg.kernel_size
        stride: int = cfg.stride
        padding: int = cfg.padding
        output_padding: int = cfg.output_padding
        dilation: int = cfg.dilation
        groups: int = cfg.groups
        activation: str = cfg.activation
        bias: bool = cfg.bias
        dropout: float = cfg.dropout
        norm: Literal["batch", "group", "none"] = cfg.norm
        norm_cfg: DictConfig = cfg.norm_cfg

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels*2 if "glu" in activation.lower() else out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias=bias,
            dilation=dilation
        )
        if norm != "none":
            norm_cfg["num_channels"] = out_channels
        self.norm = get_norm(norm, **norm_cfg)
        self.activation = Activation(activation, dim=-3)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x



class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: TransformerConfig
    ):
        super().__init__()
        self.d_model = cfg.d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = cfg.nhead
        self.dim_feedforward = cfg.dim_ff
        self.n_layers = cfg.n_layers
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            Activation(cfg.hidden_activation)
        ) if input_dim != cfg.d_model else nn.Identity()
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.d_model, output_dim),
            Activation(cfg.output_activation)
        ) if output_dim else Activation(cfg.output_activation)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward,
            activation=cfg.hidden_activation.lower(), 
            dropout=cfg.dropout, 
            batch_first=True
            )

        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, 
            num_layers=self.n_layers, 
            enable_nested_tensor=True
        )

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x = self.transformer(x)
        x = self.out_proj(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, sequence_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatchEmbed(nn.Module):
    def __init__(
        self, emb_dim: int, patch_size: int, obs_shape: Tuple[int, int, int]
    ):
        """
        引数:
            in_channels: 入力画像のチャンネル数
            emb_dim: 埋め込み後のベクトルの長さ
            num_patch_row: 高さ方向のパッチの数。例は2x2であるため、2をデフォルト値とした
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
        """
        super(PatchEmbed, self).__init__()
        self.emb_dim = emb_dim
        # パッチの数

        self.obs_shape = obs_shape

        # パッチの大きさ
        self.patch_size = patch_size
        self.patch_num = (obs_shape[1] // patch_size) * \
            (obs_shape[2] // patch_size)
        assert (
            self.patch_size * self.patch_size * self.patch_num
            == self.obs_shape[1] * self.obs_shape[2]
        ), "patch_num is not correct"

        # 入力画像のパッチへの分割 & パッチの埋め込みを一気に行う層
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.obs_shape[0],
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: 入力画像。形状は、(B, C, H, W)。[式(1)]
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        返り値:
            z_0: ViTへの入力。形状は、(B, N, D)。
                B:バッチサイズ、N:トークン数、D:埋め込みベクトルの長さ
        """
        # パッチの埋め込み & flatten [式(3)]
        # パッチの埋め込み (B, C, H, W) -> (B, D, H/P, W/P)
        # ここで、Pはパッチ1辺の大きさ
        x = self.patch_emb_layer(x)

        # パッチのflatten (B, D, H/P, W/P) -> (B, D, Np)
        # ここで、Npはパッチの数(=H*W/Pˆ2)
        x = x.flatten(2)

        # 軸の入れ替え (B, D, Np) -> (B, Np, D)
        x = x.transpose(1, 2)

        return x


class SpatialSoftmaxFlatten(nn.Module):
    def __init__(self, cfg: SpatialSoftmaxConfig):
        super(SpatialSoftmaxFlatten, self).__init__()
        temperature = cfg.temperature
        assert temperature >= 0.0, "temperature must be non-negative"
        if temperature > 0.0:
            self.spatial_softmax = spatial_soft_argmax2d
        else:
            self.spatial_softmax = self.spatial_softmax_expectation2d
        self.register_buffer("temperature", torch.tensor(temperature).float())

    def spatial_softmax_expectation2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: 入力特徴量。形状は、(B, N, H, W)
                B: バッチサイズ、N: トークン数、H: 高さ、W: 幅
        返り値:
            x: Spatial Softmaxを適用した特徴量。形状は、(B, N, H, W)。
        """
        x = spatial_softmax2d(x, self.temperature)
        x = spatial_expectation2d(x)
        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: 入力特徴量。形状は、(B, N, H, W)
                B: バッチサイズ、N: トークン数、H: 高さ、W: 幅
        返り値:
            x: Spatial Softmaxを適用した特徴量。形状は、(B, N, D)。
        """
        x = self.spatial_softmax(x)
        x = x.flatten(1)
        return x
