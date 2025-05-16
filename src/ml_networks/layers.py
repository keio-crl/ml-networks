from __future__ import annotations
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchgeometry.contrib.spatial_soft_argmax2d import create_meshgrid, spatial_soft_argmax2d

from ml_networks.activations import Activation
from ml_networks.config import ConvConfig, LinearConfig, MLPConfig, SpatialSoftmaxConfig, TransformerConfig


def get_norm(
    norm: Literal["layer", "rms", "group", "batch", "none"],
    **kwargs: Any,
) -> nn.Module:
    """
    Get normalization layer.

    Parameters
    ----------
    norm : Literal["layer", "rms", "group", "batch", "none"]
        Normalization layer. If it's set to "none", normalization is not applied.
    kwargs : dict
        Normalization layer configuration.

    Returns
    -------
    nn.Module
        Normalization layer.

    Examples
    --------
    >>> cfg = {"normalized_shape": 1, "eps": 1e-05, "elementwise_affine": True, "bias": True}
    >>> norm = get_norm("layer", **cfg)
    >>> norm
    LayerNorm((1,), eps=1e-05, elementwise_affine=True)

    >>> cfg = {"normalized_shape": 1, "eps": 1e-05, "elementwise_affine": True}
    >>> norm = get_norm("rms", **cfg)
    >>> norm
    RMSNorm((1,), eps=1e-05, elementwise_affine=True)

    >>> cfg = {"num_groups": 1, "num_channels": 12, "eps": 1e-05, "affine": True}
    >>> norm = get_norm("group", **cfg)
    >>> norm
    GroupNorm(1, 12, eps=1e-05, affine=True)

    >>> cfg = {"num_features": 1, "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True}
    >>> norm = get_norm("batch", **cfg)
    >>> norm
    BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    >>> norm = get_norm("none")
    >>> norm
    Identity()

    """
    if norm == "layer":
        return nn.LayerNorm(**kwargs)
    if norm == "rms":
        return nn.RMSNorm(**kwargs)
    if norm == "group":
        return nn.GroupNorm(**kwargs)
    if norm == "batch":
        return nn.BatchNorm2d(**kwargs)
    return nn.Identity()


class LinearNormActivation(nn.Module):
    """
    Linear layer with normalization and activation, and dropouts.

    References
    ----------
    LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    RMSNorm: https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
    Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    Dropout: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    cfg : LinearConfig
        Linear layer configuration.

    Examples
    --------
    >>> cfg = LinearConfig(
    ...     activation="ReLU",
    ...     norm="layer",
    ...     norm_cfg={"eps": 1e-05, "elementwise_affine": True, "bias": True},
    ...     dropout=0.1,
    ...     norm_first=False,
    ...     bias=True
    ... )
    >>> linear = LinearNormActivation(32, 16, cfg)
    >>> linear
    LinearNormActivation(
      (linear): Linear(in_features=32, out_features=16, bias=True)
      (norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
      (activation): Activation(
        (activation): ReLU()
      )
      (dropout): Dropout(p=0.1, inplace=False)
    )
    >>> x = torch.randn(1, 32)
    >>> output = linear(x)
    >>> output.shape
    torch.Size([1, 16])

    >>> cfg = LinearConfig(
    ...     activation="SiGLU",
    ...     norm="none",
    ...     norm_cfg={},
    ...     dropout=0.0,
    ...     norm_first=True,
    ...     bias=True
    ... )
    >>> linear = LinearNormActivation(32, 16, cfg)
    >>> # If actication includes "glu", linear output_dim is doubled to adjust actual output_dim.
    >>> linear
    LinearNormActivation(
      (linear): Linear(in_features=32, out_features=32, bias=True)
      (norm): Identity()
      (activation): Activation(
        (activation): SiGLU()
      )
      (dropout): Identity()
    )
    >>> x = torch.randn(1, 32)
    >>> output = linear(x)
    >>> output.shape
    torch.Size([1, 16])


    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: LinearConfig,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            input_dim,
            output_dim * 2 if "glu" in cfg.activation.lower() else output_dim,
            bias=cfg.bias,
        )
        cfg.norm_cfg["normalized_shape"] = output_dim * 2 if "glu" in cfg.activation.lower() else output_dim
        self.norm = get_norm(cfg.norm, **cfg.norm_cfg)
        self.activation = Activation(cfg.activation)
        if cfg.dropout > 0:
            self.dropout = nn.Dropout(cfg.dropout)
        else:
            self.dropout = nn.Identity()
        self.norm_first = cfg.norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (*, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (*, output_dim)
        """
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
    """
    Multi-layer perceptron layer.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    cfg : MLPConfig

    Examples
    --------
    >>> cfg = MLPConfig(
    ...     hidden_dim=16,
    ...     n_layers=3,
    ...     output_activation="ReLU",
    ...     linear_cfg=LinearConfig(
    ...         activation="ReLU",
    ...         norm="layer",
    ...         norm_cfg={"eps": 1e-05, "elementwise_affine": True, "bias": True},
    ...         dropout=0.1,
    ...         norm_first=False,
    ...         bias=True
    ...     )
    ... )
    >>> mlp = MLPLayer(32, 16, cfg)
    >>> x = torch.randn(1, 32)
    >>> output = mlp(x)
    >>> output.shape
    torch.Size([1, 16])

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: MLPConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.dense = self._build_dense()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (*, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (*, output_dim)

        """
        return self.dense(x)

    def _build_dense(self) -> nn.Module:
        """
        Build dense layers.

        Returns
        -------
        nn.Sequential
            Dense layers

        Examples
        --------
        >>> cfg = MLPConfig(
        ...     hidden_dim=64,
        ...     n_layers=2,
        ...     output_activation="ReLU",
        ...     linear_cfg=LinearConfig(
        ...         activation="ReLU",
        ...         norm="layer",
        ...         norm_cfg={"eps": 1e-05, "elementwise_affine": True, "bias": True},
        ...         dropout=0.1,
        ...         norm_first=False,
        ...         bias=True
        ...     )
        ... )
        >>> mlp = MLPLayer(32, 16, cfg)
        >>> mlp._build_dense()
        Sequential(
          (0): LinearNormActivation(
            (linear): Linear(in_features=32, out_features=64, bias=True)
            (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): LinearNormActivation(
            (linear): Linear(in_features=64, out_features=64, bias=True)
            (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): LinearNormActivation(
            (linear): Linear(in_features=64, out_features=16, bias=True)
            (norm): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        """
        layers = []
        layers += [LinearNormActivation(self.input_dim, self.hidden_dim, self.cfg.linear_cfg)]
        for _ in range(self.n_layers - 1):
            layers += [LinearNormActivation(self.hidden_dim, self.hidden_dim, self.cfg.linear_cfg)]
        last_cfg = self.cfg.linear_cfg
        last_cfg.activation = self.cfg.output_activation
        layers += [LinearNormActivation(self.hidden_dim, self.output_dim, last_cfg)]
        return nn.Sequential(*layers)


class ConvNormActivation(nn.Module):
    """
    Convolutional layer with normalization and activation, and dropouts.

    References
    ----------
    PixelShuffle: https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
    PixelUnshuffle: https://pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html
    BatchNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    GroupNorm: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
    LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    InstanceNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
    Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Dropout: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    cfg : ConvConfig
        Convolutional layer configuration.

    Examples
    --------
    >>> cfg = ConvConfig(
    ...     activation="ReLU",
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1,
    ...     dilation=1,
    ...     groups=1,
    ...     bias=True,
    ...     dropout=0.1,
    ...     norm="batch",
    ...     norm_cfg={"affine": True, "track_running_stats": True},
    ...     scale_factor=0
    ... )
    >>> conv = ConvNormActivation(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])

    >>> cfg = ConvConfig(
    ...     activation="SiGLU",
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1,
    ...     dilation=1,
    ...     groups=1,
    ...     bias=True,
    ...     dropout=0.0,
    ...     norm="none",
    ...     norm_cfg={},
    ...     scale_factor=2
    ... )
    >>> conv = ConvNormActivation(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 64, 64])

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig,
    ) -> None:
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
        norm_cfg = cfg.norm_cfg
        scale_factor: int = cfg.scale_factor

        out_channels_ = out_channels
        if "glu" in activation.lower():
            out_channels_ *= 2
        if scale_factor > 0:
            out_channels_ *= abs(scale_factor) ** 2
        elif scale_factor < 0:
            out_channels_ //= abs(scale_factor) ** 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels_,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm != "none" and norm != "group":
            norm_cfg["num_features"] = out_channels_
        elif norm == "group":
            norm_cfg["num_channels"] = out_channels_

        self.norm = get_norm(norm, **norm_cfg)
        if scale_factor > 0:
            self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        elif scale_factor < 0:
            self.pixel_shuffle = nn.PixelUnshuffle(abs(scale_factor))
        else:
            self.pixel_shuffle = nn.Identity()
        self.activation = Activation(activation, dim=-3)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W) or (in_channels, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, H', W') or (out_channels, H', W')
        H' and W' are calculated as follows:
        H' = (H + 2*padding - dilation * (kernel_size - 1) - 1) // stride + 1
        H' = H' * scale_factor if scale_factor > 0 else H' // abs(scale_factor) if scale_factor < 0 else H'
        W' = (W + 2*padding - dilation * (kernel_size - 1) - 1) // stride + 1
        W' = W' * scale_factor if scale_factor > 0 else W' // abs(scale_factor) if scale_factor < 0 else W'

        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """
    Residual block.

    Parameters
    ----------
    in_features : int
        Input features.
    kernel_size : int
        Kernel size.
    activation : str
        Activation function.
    norm : Literal["batch", "group", "none"]
        Normalization layer. If it's set to "none", normalization is not applied. Default is "none".
    norm_cfg : DictConfig
        Normalization layer configuration. Default is {}.
    dropout : float
        Dropout rate. If it's set to 0.0, dropout is not applied. Default is 0.0.

    Examples
    --------
    >>> resblock = ResidualBlock(16, 3, "ReLU", "batch", {}, 0.1)
    >>> x = torch.randn(1, 16, 32, 32)
    >>> output = resblock(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])
    """

    def __init__(
        self,
        in_features: int,
        kernel_size: int,
        activation: str = "ReLU",
        norm: Literal["batch", "group", "none"] = "none",
        norm_cfg: dict[str, Any] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        first_cfg = ConvConfig(
            activation=activation,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            groups=1,
            bias=True,
            dropout=dropout,
            norm=norm,
            norm_cfg=norm_cfg or {},
        )
        second_cfg = ConvConfig(
            activation="Identity",
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            dilation=1,
            groups=1,
            bias=True,
            dropout=dropout,
            norm=norm,
            norm_cfg=norm_cfg or {},
        )
        self.conv_block = nn.Sequential(
            ConvNormActivation(
                in_features,
                in_features,
                first_cfg,
            ),
            ConvNormActivation(
                in_features,
                in_features * 2 if "glu" in activation.lower() else in_features,
                second_cfg,
            ),
        )
        self.activation = Activation(activation, dim=-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_features, H, W) or (in_features, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, in_features, H, W) or (in_features, H, W)
        """
        return self.activation(x + self.conv_block(x))


class ConvTransposeNormActivation(nn.Module):
    """
    Transposed convolutional layer with normalization and activation, and dropouts.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    cfg : ConvConfig
        Convolutional layer configuration.

    Examples
    --------
    >>> cfg = ConvConfig(
    ...     activation="ReLU",
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1,
    ...     output_padding=0,
    ...     dilation=1,
    ...     groups=1,
    ...     bias=True,
    ...     dropout=0.1,
    ...     norm="batch",
    ...     norm_cfg={"affine": True, "track_running_stats": True}
    ... )
    >>> conv = ConvTransposeNormActivation(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])

    >>> cfg = ConvConfig(
    ...     activation="SiGLU",
    ...     kernel_size=3,
    ...     stride=1,
    ...     padding=1,
    ...     output_padding=0,
    ...     dilation=1,
    ...     groups=1,
    ...     bias=True,
    ...     dropout=0.0,
    ...     norm="none",
    ...     norm_cfg={}
    ... )
    >>> conv = ConvTransposeNormActivation(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig,
    ) -> None:
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels * 2 if "glu" in cfg.activation.lower() else out_channels,
            cfg.kernel_size,
            cfg.stride,
            cfg.padding,
            cfg.output_padding,
            cfg.groups,
            bias=cfg.bias,
            dilation=cfg.dilation,
        )
        if cfg.norm not in {"none", "group"}:
            cfg.norm_cfg["num_features"] = out_channels * 2 if "glu" in cfg.activation.lower() else out_channels
        elif cfg.norm == "group":
            cfg.norm_cfg["num_channels"] = out_channels * 2 if "glu" in cfg.activation.lower() else out_channels
        self.norm = get_norm(cfg.norm, **cfg.norm_cfg)
        self.activation = Activation(cfg.activation, dim=-3)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W) or (in_channels, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, H', W') or (out_channels, H', W')
        H' and W' are calculated as follows:
        H' = (H - 1) * stride - 2 * padding + kernel_size + output_padding
        W' = (W - 1) * stride - 2 * padding + kernel_size + output_padding
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)


class TransformerLayer(nn.Module):
    """
    Transformer layer.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    cfg : TransformerConfig
        Transformer layer configuration.

    Examples
    --------
    >>> cfg = TransformerConfig(
    ...     d_model=16,
    ...     nhead=4,
    ...     dim_ff=64,
    ...     n_layers=3,
    ...     dropout=0.1,
    ...     hidden_activation="ReLU",
    ...     output_activation="ReLU"
    ... )
    >>> transformer = TransformerLayer(32, 16, cfg)
    >>> x = torch.randn(1, 8, 32)
    >>> output = transformer(x)
    >>> output.shape
    torch.Size([1, 8, 16])

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = cfg.nhead
        self.dim_feedforward = cfg.dim_ff
        self.n_layers = cfg.n_layers
        self.in_proj = (
            nn.Sequential(
                nn.Linear(input_dim, cfg.d_model),
                nn.LayerNorm(cfg.d_model),
                Activation(cfg.hidden_activation),
            )
            if input_dim != cfg.d_model
            else nn.Identity()
        )
        self.out_proj = (
            nn.Sequential(
                nn.Linear(cfg.d_model, output_dim),
                Activation(cfg.output_activation),
            )
            if output_dim
            else Activation(cfg.output_activation)
        )

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            activation=cfg.hidden_activation.lower(),
            dropout=cfg.dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=self.n_layers,
            enable_nested_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, output_dim)
        """
        x = self.in_proj(x)
        x = self.transformer(x)
        return self.out_proj(x)


class PositionalEncoding(nn.Module):
    """
    Positional encoding.

    Parameters
    ----------
    d_model : int
        Dimension of model.
    dropout : float
        Dropout rate. If it's set to 0.0, dropout is not applied.
    max_len : int
        Maximum length.

    Examples
    --------
    >>> pos_enc = PositionalEncoding(16, 0.1, 100)
    >>> x = torch.randn(1, 8, 16)
    >>> output = pos_enc(x)
    >>> output.shape
    torch.Size([1, 8, 16])

    >>> x = torch.randn(1, 100, 16)
    >>> output = pos_enc(x)
    >>> output.shape
    torch.Size([1, 100, 16])

    """

    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, D)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, L, D)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PatchEmbed(nn.Module):
    """
    Patch embedding layer.

    Parameters
    ----------
    emb_dim : int
        Embedding dimension.
    patch_size : int
        Patch size.
    obs_shape : Tuple[int, int, int]
        Observation shape.

    Examples
    --------
    >>> patch_embed = PatchEmbed(16, 4, (3, 32, 32))
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = patch_embed(x)
    >>> output.shape
    torch.Size([1, 64, 16])
    """

    def __init__(
        self,
        emb_dim: int,
        patch_size: int,
        obs_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.obs_shape = obs_shape

        # パッチの大きさ
        self.patch_size = patch_size
        self.patch_num = (obs_shape[1] // patch_size) * (obs_shape[2] // patch_size)
        assert self.patch_size * self.patch_size * self.patch_num == self.obs_shape[1] * self.obs_shape[2], (
            "patch_num is not correct"
        )

        # 入力画像のパッチへの分割 & パッチの埋め込みを一気に行う層
        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.obs_shape[0],
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, Np, D)

        Np is the number of patches.
            Np = H*W/P^2
        D is the embedding dimension.

        """
        # パッチの埋め込み && flatten[式(3)]
        # パッチの埋め込み (B, C, H, W) -> (B, D, H/P, W/P)
        # ここで、Pはパッチ1辺の大きさ
        x = self.patch_emb_layer(x)

        # パッチのflatten (B, D, H/P, W/P) -> (B, D, Np)
        # ここで、Np はパッチの数( = H*W/P^2)
        x = x.flatten(2)

        # 軸の入れ替え (B, D, Np) -> (B, Np, D)
        return x.transpose(1, 2)


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax and Flatten layer.

    Parameters
    ----------
    cfg : SpatialSoftmaxConfig
        Spatial softmax configuration.

    Examples
    --------
    >>> cfg = SpatialSoftmaxConfig(temperature=1.0, is_argmax=True)
    >>> spatial_softmax = SpatialSoftmax(cfg)
    >>> x = torch.randn(1, 64, 16, 16)
    >>> output = spatial_softmax(x)
    >>> output.shape
    torch.Size([1, 64, 2])

    """

    def __init__(self, cfg: SpatialSoftmaxConfig) -> None:
        super().__init__()
        self.temperature = cfg.temperature
        self.eps = cfg.eps
        assert self.temperature > 0.0, "temperature must be non-negative"
        if cfg.is_argmax:
            self.spatial_softmax = self.spatial_argmax2d
        elif cfg.is_straight_through:
            self.spatial_softmax = self.spatial_softmax_straight_through
        else:
            self.spatial_softmax = spatial_soft_argmax2d

    def spatial_softmax_straight_through(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial Softmax and Argmax layer.

        Parameters
        ----------
        x : torch.Tensor
            入力特徴量。形状は、(B, N, H, W)

        Returns
        -------
        torch.Tensor
            Spatial Softmaxを適用した特徴量。形状は、(B, N, 2)。
        """
        if not torch.is_tensor(x):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = x.shape
        pos_y, pos_x = create_meshgrid(x, normalized_coordinates=True)
        x = x.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)
        softmax_x = exp_x * exp_x_sum

        # straight-through trick
        softmax_x = softmax_x + x - x.detach()

        # create coordinates grid
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y = torch.sum(
            pos_y * softmax_x, dim=-1, keepdim=True)
        expected_x = torch.sum(
            pos_x * softmax_x, dim=-1, keepdim=True)
        output = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2


    def spatial_argmax2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial Softmax and Argmax layer.

        Parameters
        ----------
        x : torch.Tensor
            入力特徴量。形状は、(B, N, H, W)

        Returns
        -------
        torch.Tensor
            Spatial Softmaxを適用した特徴量。形状は、(B, N, 2)。
        """
        if not torch.is_tensor(x):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = x.shape
        pos_y, pos_x = create_meshgrid(x, normalized_coordinates=True)
        x = x.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)
        softmax_x = exp_x * exp_x_sum

        # straight-through trick
        argmax_x = torch.argmax(x, dim=-1, keepdim=True)
        argmax_x = F.one_hot(argmax_x, num_classes=x.shape[-1]).squeeze(2)
        argmax_x = argmax_x + softmax_x - softmax_x.detach()

        # create coordinates grid
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y = torch.sum(
            pos_y * argmax_x, dim=-1, keepdim=True)
        expected_x = torch.sum(
            pos_x * argmax_x, dim=-1, keepdim=True)
        output = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播.

        Parameters
        ----------
        x: 入力特徴量。形状は、(B, N, H, W)
            B: バッチサイズ、N: トークン数、H: 高さ、W: 幅

        Returns
        -------
            Spatial Softmaxを適用した特徴量。形状は、(B, N, D)。
        """
        x = self.spatial_softmax(x/self.temperature)
        return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
