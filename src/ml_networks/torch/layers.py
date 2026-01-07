"""レイヤーを扱うモジュール."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchgeometry.contrib.spatial_soft_argmax2d import (  # type: ignore[import-untyped]
    create_meshgrid,
    spatial_soft_argmax2d,
)

from ml_networks.config import (
    AttentionConfig,
    ConvConfig,
    LinearConfig,
    MLPConfig,
    SpatialSoftmaxConfig,
    TransformerConfig,
)
from ml_networks.torch.activations import Activation


def get_norm(
    norm: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"],
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
    >>> norm = get_norm("batch2d", **cfg)
    >>> norm
    BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    >>> cfg = {"num_features": 1, "eps": 1e-05, "momentum": 0.1, "affine": True, "track_running_stats": True}
    >>> norm = get_norm("batch1d", **cfg)
    >>> norm
    BatchNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

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
    if norm == "batch2d":
        return nn.BatchNorm2d(**kwargs)
    if norm == "batch1d":
        return nn.BatchNorm1d(**kwargs)
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
    >>> # If activation includes "glu", linear output_dim is doubled to adjust actual output_dim.
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
        if cfg.norm_first:
            normalized_shape = input_dim
        else:
            normalized_shape = output_dim * 2 if "glu" in cfg.activation.lower() else output_dim

        cfg.norm_cfg["normalized_shape"] = normalized_shape
        self.norm = get_norm(cfg.norm, **cfg.norm_cfg)
        self.activation = Activation(cfg.activation)
        self.dropout: nn.Module
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
        self.cfg = deepcopy(cfg)
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

        out_channels_ = out_channels
        if "glu" in cfg.activation.lower():
            out_channels_ *= 2
        if cfg.scale_factor > 0:
            out_channels_ *= abs(cfg.scale_factor) ** 2
        elif cfg.scale_factor < 0:
            out_channels_ //= abs(cfg.scale_factor) ** 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels_,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            dilation=cfg.dilation,
            groups=cfg.groups,
            bias=cfg.bias,
            padding_mode=cfg.padding_mode,
        )
        if cfg.norm != "none" and cfg.norm != "group":
            cfg.norm_cfg["num_features"] = out_channels_
        elif cfg.norm == "group":
            cfg.norm_cfg["num_channels"] = in_channels if cfg.norm_first else out_channels_

        norm_type: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"] = (
            "batch2d" if cfg.norm == "batch" else cfg.norm
        )  # type: ignore[assignment]
        self.norm = get_norm(norm_type, **cfg.norm_cfg)
        self.pixel_shuffle: nn.Module
        if cfg.scale_factor > 0:
            self.pixel_shuffle = nn.PixelShuffle(cfg.scale_factor)
        elif cfg.scale_factor < 0:
            self.pixel_shuffle = nn.PixelUnshuffle(abs(cfg.scale_factor))
        else:
            self.pixel_shuffle = nn.Identity()
        self.activation = Activation(cfg.activation, dim=-3)
        self.dropout: nn.Module = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.norm_first = cfg.norm_first

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
        if self.norm_first:
            x = self.norm(x)
            x = self.conv(x)
            x = self.pixel_shuffle(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.pixel_shuffle(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class ConvNormActivation1d(nn.Module):
    """
    1D Convolutional layer with normalization and activation, and dropouts.

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
    ...     padding_mode="zeros"
    ... )
    >>> conv = ConvNormActivation1d(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 32])
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
    ...     padding_mode="zeros"
    ... )
    >>> conv = ConvNormActivation1d(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig,
    ) -> None:
        super().__init__()

        out_channels_ = out_channels
        if "glu" in cfg.activation.lower():
            out_channels_ *= 2
        if cfg.scale_factor > 0:
            out_channels_ *= abs(cfg.scale_factor)
        elif cfg.scale_factor < 0:
            out_channels_ //= abs(cfg.scale_factor)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels_,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            dilation=cfg.dilation,
            groups=cfg.groups,
            bias=cfg.bias,
            padding_mode=cfg.padding_mode,
        )
        if cfg.norm != "none" and cfg.norm != "group":
            cfg.norm_cfg["num_features"] = out_channels_
        elif cfg.norm == "group":
            cfg.norm_cfg["num_channels"] = in_channels if cfg.norm_first else out_channels_

        if cfg.scale_factor > 0:
            self.horizontal_shuffle: nn.Module = HorizonShuffle(cfg.scale_factor)
        elif cfg.scale_factor < 0:
            self.horizontal_shuffle = HorizonUnShuffle(abs(cfg.scale_factor))
        else:
            self.horizontal_shuffle = nn.Identity()

        norm_type: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"] = (
            "batch1d" if cfg.norm == "batch" else cfg.norm
        )  # type: ignore[assignment]
        self.norm = get_norm(norm_type, **cfg.norm_cfg)
        self.activation = Activation(cfg.activation, dim=-2)
        self.dropout: nn.Module = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.norm_first = cfg.norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, L) or (in_channels, L)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, L') or (out_channels, L')
        L' is calculated as follows:
        L' = (L + 2*padding - dilation * (kernel_size - 1) - 1) // stride + 1

        """
        if self.norm_first:
            x = self.norm(x)
            x = self.conv(x)
            x = self.horizontal_shuffle(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x = self.conv(x)
            x = self.horizontal_shuffle(x)
            x = self.norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


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
    norm_cfg : dictConfig
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
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
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
            padding_mode=padding_mode,
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
            padding_mode=padding_mode,
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
        norm_type: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"] = (
            "batch2d" if cfg.norm == "batch" else cfg.norm
        )  # type: ignore[assignment]
        self.norm = get_norm(norm_type, **cfg.norm_cfg)
        self.activation = Activation(cfg.activation, dim=-3)
        self.dropout: nn.Module = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

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


class ConvTransposeNormActivation1d(nn.Module):
    """
    1D Transposed convolutional layer with normalization and activation, and dropouts.

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
    ... )
    >>> conv = ConvTransposeNormActivation1d(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 32])

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
    >>> conv = ConvTransposeNormActivation1d(3, 16, cfg)
    >>> x = torch.randn(1, 3, 32)
    >>> output = conv(x)
    >>> output.shape
    torch.Size([1, 16, 32])

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig,
    ) -> None:
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels * 2 if "glu" in cfg.activation.lower() else out_channels,
            kernel_size=cfg.kernel_size,
            stride=cfg.stride,
            padding=cfg.padding,
            output_padding=cfg.output_padding,
            groups=cfg.groups,
            bias=cfg.bias,
            dilation=cfg.dilation,
        )
        if cfg.norm not in {"none", "group"}:
            cfg.norm_cfg["num_features"] = out_channels * 2 if "glu" in cfg.activation.lower() else out_channels
        elif cfg.norm == "group":
            cfg.norm_cfg["num_channels"] = out_channels * 2 if "glu" in cfg.activation.lower() else out_channels
        norm_type: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"] = (
            "batch1d" if cfg.norm == "batch" else cfg.norm
        )  # type: ignore[assignment]
        self.norm = get_norm(norm_type, **cfg.norm_cfg)
        self.activation = Activation(cfg.activation, dim=-2)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, L) or (in_channels, L)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, L') or (out_channels, L')
        L' is calculated as follows:
        L' = (L - 1) * stride - 2 * padding + kernel_size + output_padding
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
        self.pe: torch.Tensor = pe

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


class Attention2d(nn.Module):
    """2d自己注意機構.

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
        nhead: int | None = None,
        patch_size: int = 1,
        attn_cfg: AttentionConfig | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        if nhead is None or patch_size is None:
            assert attn_cfg is not None, "Either nhead and patch_size or attn_cfg must be provided"
            self.n_heads = attn_cfg.nhead
            self.patch_size = attn_cfg.patch_size
        else:
            self.n_heads = nhead
            self.patch_size = patch_size
        assert channels % self.n_heads == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {nhead}"
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
            channels,
            channels,
            cfg,
        )

    def qkv_attn(self, qkv: torch.Tensor) -> torch.Tensor:
        """Apply QKV attention.

        Parameters
        ----------
        qkv : torch.Tensor
            An [N x (Heads * 3 * C) x H x W] tensor of query, key, value.

        Returns
        -------
        torch.Tensor
            An [N x (C * Head) x H x W] tensor of attended values.
        """
        bs, channels, height, width = qkv.shape
        assert channels % (3 * self.n_heads) == 0
        ch = channels // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, height * width).split(ch, dim=1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            q * scale,
            k * scale,
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight - torch.max(weight, dim=-1, keepdim=True)[0], dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=self.patch_size, p2=self.patch_size)
        h = self.qkv_attn(qkv)
        h = rearrange(h, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=self.patch_size, p2=self.patch_size)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Attention1d(nn.Module):
    """1d自己注意機構."""

    def __init__(
        self,
        channels: int,
        nhead: int | None = None,
        patch_size: int = 1,
        attn_cfg: AttentionConfig | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        if nhead is None:
            assert attn_cfg is not None, "Either nhead or attn_cfg must be provided"
            self.n_heads = attn_cfg.nhead
            self.patch_size = attn_cfg.patch_size
        else:
            self.n_heads = nhead
            self.patch_size = patch_size
        assert channels % self.n_heads == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {nhead}"
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
            channels,
            channels,
            cfg,
        )

    def qkv_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        """Apply QKV attention.

        Parameters
        ----------
        qkv : torch.Tensor
            An [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.

        Returns
        -------
        torch.Tensor
            An [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            q * scale,
            k * scale,
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight - torch.max(weight, dim=-1, keepdim=True)[0], dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b c (t p) -> b (c p) t", p=self.patch_size)
        h = self.qkv_attention(qkv)
        h = rearrange(h, "b (c p) t -> b c (t p)", p=self.patch_size)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


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
        """Spatial Softmax and Argmax layer.

        Parameters
        ----------
        x : torch.Tensor
            入力特徴量。形状は、(B, N, H, W)

        Returns
        -------
        torch.Tensor
            Spatial Softmaxを適用した特徴量。形状は、(B, N, 2)。

        Raises
        ------
        TypeError
            If input is not a torch.Tensor.
        ValueError
            If input shape is not 4D.
        """
        if not torch.is_tensor(x):
            msg = f"Input input type is not a torch.Tensor. Got {type(x)}"
            raise TypeError(msg)
        if len(x.shape) != 4:
            msg = f"Invalid input shape, we expect BxCxHxW. Got: {x.shape}"
            raise ValueError(msg)
        # unpack shapes and create view from input tensor
        batch_size, channels, _height, _width = x.shape
        pos_y, pos_x = create_meshgrid(x, normalized_coordinates=True)
        x = x.view(batch_size, channels, -1)

        # compute softmax with max subtraction  trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)
        softmax_x = exp_x * exp_x_sum

        # straight-through trick
        softmax_x = softmax_x + x - x.detach()

        # create coordinates grid
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y = torch.sum(pos_y * softmax_x, dim=-1, keepdim=True)
        expected_x = torch.sum(pos_x * softmax_x, dim=-1, keepdim=True)
        output = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2

    def spatial_argmax2d(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial Softmax and Argmax layer.

        Parameters
        ----------
        x : torch.Tensor
            入力特徴量。形状は、(B, N, H, W)

        Returns
        -------
        torch.Tensor
            Spatial Softmaxを適用した特徴量。形状は、(B, N, 2)。

        Raises
        ------
        TypeError
            If input is not a torch.Tensor.
        ValueError
            If input shape is not 4D.
        """
        if not torch.is_tensor(x):
            msg = f"Input input type is not a torch.Tensor. Got {type(x)}"
            raise TypeError(msg)
        if len(x.shape) != 4:
            msg = f"Invalid input shape, we expect BxCxHxW. Got: {x.shape}"
            raise ValueError(msg)
        # unpack shapes and create view from input tensor
        batch_size, channels, _height, _width = x.shape
        pos_y, pos_x = create_meshgrid(x, normalized_coordinates=True)
        x = x.view(batch_size, channels, -1)

        # compute softmax with max subtraction  trick
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
        expected_y = torch.sum(pos_y * argmax_x, dim=-1, keepdim=True)
        expected_x = torch.sum(pos_x * argmax_x, dim=-1, keepdim=True)
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
        return self.spatial_softmax(x / self.temperature)


class HorizonShuffle(nn.Module):
    """Horizon Shuffle Layer.

    Args:
        dim (int): 入力・出力チャンネル数

    Examples
    --------
    >>> shuffle = HorizonShuffle(2)
    >>> x = torch.randn(2, 64, 50)
    >>> out = shuffle(x)
    >>> out.shape
    torch.Size([2, 32, 100])
    """

    def __init__(self, upscale_factor: int = 2) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Horizon Shuffle Layer.

        Args:
            x (torch.Tensor): 入力テンソル (B, C, T)

        Returns
        -------
            torch.Tensor: 出力テンソル (B, C // upscale_factor, T * upscale_factor)
        """
        _b, c, _t = x.shape
        assert c % self.upscale_factor == 0, "Input length must be divisible by upscale_factor."
        return rearrange(x, "b (c u) t -> b c (t u)", u=self.upscale_factor)


class HorizonUnShuffle(nn.Module):
    """Horizon UnShuffle Layer.

    Args:
        upscale_factor (int): アップスケールファクター

    Examples
    --------
    >>> unshuffle = HorizonUnShuffle(2)
    >>> x = torch.randn(2, 32, 100)
    >>> out = unshuffle(x)
    >>> out.shape
    torch.Size([2, 64, 50])
    """

    def __init__(self, downscale_factor: int = 2) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Horizon UnShuffle Layer.

        Args:
            x (torch.Tensor): 入力テンソル (B, C // upscale_factor, T * upscale_factor)

        Returns
        -------
            torch.Tensor: 出力テンソル (B, C, T)
        """
        _b, _c, t = x.shape
        assert t % self.downscale_factor == 0, "Input length must be divisible by upscale_factor."
        return rearrange(x, "b c (t u) -> b (c u) t", u=self.downscale_factor)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
