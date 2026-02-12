"""レイヤーを扱うモジュール."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from ml_networks.config import (
    AttentionConfig,
    ConvConfig,
    LinearConfig,
    MLPConfig,
    SpatialSoftmaxConfig,
    TransformerConfig,
)
from ml_networks.jax.activations import Activation


class Identity(nnx.Module):
    """Identity module that passes input through unchanged."""

    def __call__(self, x: jax.Array, *args: Any, **kwargs: Any) -> jax.Array:
        return x


def _translate_norm_kwargs(
    norm: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Translate PyTorch norm kwargs to Flax NNX kwargs.

    Parameters
    ----------
    norm : str
        Normalization type.
    kwargs : dict
        PyTorch-style normalization kwargs.

    Returns
    -------
    dict[str, Any]
        Flax NNX-style normalization kwargs.
    """
    result: dict[str, Any] = {}

    if norm in {"layer", "rms"}:
        if "normalized_shape" in kwargs:
            result["num_features"] = kwargs["normalized_shape"]
        if "eps" in kwargs:
            result["epsilon"] = kwargs["eps"]
        if "elementwise_affine" in kwargs:
            result["use_scale"] = kwargs["elementwise_affine"]
            result["use_bias"] = kwargs.get("bias", kwargs["elementwise_affine"])
        elif "bias" in kwargs:
            result["use_bias"] = kwargs["bias"]
    elif norm == "group":
        if "num_groups" in kwargs:
            result["num_groups"] = kwargs["num_groups"]
        if "num_channels" in kwargs:
            result["num_features"] = kwargs["num_channels"]
        if "eps" in kwargs:
            result["epsilon"] = kwargs["eps"]
        if "affine" in kwargs:
            result["use_scale"] = kwargs["affine"]
            result["use_bias"] = kwargs["affine"]
    elif norm in {"batch2d", "batch1d", "batch"}:
        if "num_features" in kwargs:
            result["num_features"] = kwargs["num_features"]
        if "eps" in kwargs:
            result["epsilon"] = kwargs["eps"]
        if "momentum" in kwargs:
            result["momentum"] = kwargs["momentum"]
        if "affine" in kwargs:
            result["use_scale"] = kwargs["affine"]
            result["use_bias"] = kwargs["affine"]

    return result


def get_norm(
    norm: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"],
    *,
    rngs: nnx.Rngs,
    **kwargs: Any,
) -> nnx.Module:
    """
    Get normalization layer.

    Parameters
    ----------
    norm : Literal["layer", "rms", "group", "batch2d", "batch1d", "none"]
        Normalization layer. If it's set to "none", normalization is not applied.
    rngs : nnx.Rngs
        Random number generators.
    kwargs : dict
        Normalization layer configuration (PyTorch-style kwargs, will be translated).

    Returns
    -------
    nnx.Module
        Normalization layer.
    """
    flax_kwargs = _translate_norm_kwargs(norm, **kwargs)

    if norm == "layer":
        return nnx.LayerNorm(**flax_kwargs, rngs=rngs)
    if norm == "rms":
        return nnx.RMSNorm(**flax_kwargs, rngs=rngs)
    if norm == "group":
        return nnx.GroupNorm(**flax_kwargs, rngs=rngs)
    if norm in {"batch2d", "batch1d", "batch"}:
        return nnx.BatchNorm(**flax_kwargs, rngs=rngs)
    return Identity()


def _padding_to_flax(padding: int, dilation: int = 1) -> tuple[tuple[int, int], ...]:
    """Convert integer padding to Flax padding format for 2D conv.

    Parameters
    ----------
    padding : int
        Padding size.
    dilation : int
        Dilation factor.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Flax-style padding.
    """
    effective_padding = padding * dilation if dilation > 1 else padding
    return ((effective_padding, effective_padding),)


def _pad_input(
    x: jax.Array,
    padding: int,
    padding_mode: str,
    n_spatial_dims: int,
) -> jax.Array:
    """Manually pad input for non-zero padding modes.

    Parameters
    ----------
    x : jax.Array
        Input tensor in channels-last format.
    padding : int
        Padding size.
    padding_mode : str
        One of "zeros", "reflect", "replicate", "circular".
    n_spatial_dims : int
        Number of spatial dimensions (1 or 2).

    Returns
    -------
    jax.Array
        Padded tensor.
    """
    if padding == 0 or padding_mode == "zeros":
        return x

    mode_map = {
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }
    jnp_mode = mode_map.get(padding_mode, "constant")

    # Build pad_width: (batch, spatial..., channels) for channels-last
    pad_width = [(0, 0)]  # batch
    for _ in range(n_spatial_dims):
        pad_width.append((padding, padding))
    pad_width.append((0, 0))  # channels

    return jnp.pad(x, pad_width, mode=jnp_mode)


def pixel_shuffle_2d(x: jax.Array, factor: int) -> jax.Array:
    """Depth-to-space for NHWC format (equivalent to PyTorch PixelShuffle).

    Parameters
    ----------
    x : jax.Array
        Input tensor of shape (B, H, W, C * factor^2).
    factor : int
        Upscale factor.

    Returns
    -------
    jax.Array
        Output tensor of shape (B, H * factor, W * factor, C).
    """
    b, h, w, c = x.shape
    oc = c // (factor * factor)
    x = x.reshape(b, h, w, oc, factor, factor)
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(b, h * factor, w * factor, oc)


def pixel_unshuffle_2d(x: jax.Array, factor: int) -> jax.Array:
    """Space-to-depth for NHWC format (equivalent to PyTorch PixelUnshuffle).

    Parameters
    ----------
    x : jax.Array
        Input tensor of shape (B, H * factor, W * factor, C).
    factor : int
        Downscale factor.

    Returns
    -------
    jax.Array
        Output tensor of shape (B, H, W, C * factor^2).
    """
    b, h, w, c = x.shape
    oh, ow = h // factor, w // factor
    x = x.reshape(b, oh, factor, ow, factor, c)
    x = jnp.transpose(x, (0, 1, 3, 5, 2, 4))
    return x.reshape(b, oh, ow, c * factor * factor)


class LinearNormActivation(nnx.Module):
    """
    Linear layer with normalization and activation, and dropouts.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    cfg : LinearConfig
        Linear layer configuration.
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
    >>> cfg = LinearConfig(
    ...     activation="ReLU",
    ...     norm="layer",
    ...     norm_cfg={"eps": 1e-05, "elementwise_affine": True, "bias": True},
    ...     dropout=0.1,
    ...     norm_first=False,
    ...     bias=True
    ... )
    >>> linear = LinearNormActivation(32, 16, cfg, rngs=rngs)
    >>> x = jnp.ones((1, 32))
    >>> output = linear(x)
    >>> output.shape
    (1, 16)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: LinearConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        out_features = output_dim * 2 if "glu" in cfg.activation.lower() else output_dim

        self.linear = nnx.Linear(input_dim, out_features, use_bias=cfg.bias, rngs=rngs)

        if cfg.norm_first:
            normalized_shape = input_dim
        else:
            normalized_shape = out_features

        norm_cfg = dict(cfg.norm_cfg)
        norm_cfg["normalized_shape"] = normalized_shape
        self.norm = get_norm(cfg.norm, rngs=rngs, **norm_cfg)
        self.activation = Activation(cfg.activation)
        self.dropout: nnx.Module
        if cfg.dropout > 0:
            self.dropout = nnx.Dropout(rate=cfg.dropout, rngs=rngs)
        else:
            self.dropout = Identity()
        self.norm_first = cfg.norm_first

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (*, input_dim)

        Returns
        -------
        jax.Array
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


class MLPLayer(nnx.Module):
    """
    Multi-layer perceptron layer.

    Parameters
    ----------
    input_dim : int
        Input dimension.
    output_dim : int
        Output dimension.
    cfg : MLPConfig
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
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
    >>> mlp = MLPLayer(32, 16, cfg, rngs=rngs)
    >>> x = jnp.ones((1, 32))
    >>> output = mlp(x)
    >>> output.shape
    (1, 16)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: MLPConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = deepcopy(cfg)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layers = cfg.n_layers
        self.layers = nnx.List(self._build_dense(rngs=rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (*, input_dim)

        Returns
        -------
        jax.Array
            Output tensor of shape (*, output_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def _build_dense(self, *, rngs: nnx.Rngs) -> list:
        """Build dense layers."""
        layers = []
        layers.append(LinearNormActivation(self.input_dim, self.hidden_dim, self.cfg.linear_cfg, rngs=rngs))
        for _ in range(self.n_layers - 1):
            layers.append(LinearNormActivation(self.hidden_dim, self.hidden_dim, self.cfg.linear_cfg, rngs=rngs))
        last_cfg = deepcopy(self.cfg.linear_cfg)
        last_cfg.activation = self.cfg.output_activation
        layers.append(LinearNormActivation(self.hidden_dim, self.output_dim, last_cfg, rngs=rngs))
        return layers


class ConvNormActivation(nnx.Module):
    """
    Convolutional layer with normalization and activation, and dropouts.

    Uses NHWC (channels-last) format.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    cfg : ConvConfig
        Convolutional layer configuration.
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
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
    >>> conv = ConvNormActivation(3, 16, cfg, rngs=rngs)
    >>> x = jnp.ones((1, 32, 32, 3))
    >>> output = conv(x)
    >>> output.shape
    (1, 32, 32, 16)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        out_channels_ = out_channels
        if "glu" in cfg.activation.lower():
            out_channels_ *= 2
        if cfg.scale_factor > 0:
            out_channels_ *= abs(cfg.scale_factor) ** 2
        elif cfg.scale_factor < 0:
            out_channels_ //= abs(cfg.scale_factor) ** 2

        # Handle padding mode
        self.padding_mode = cfg.padding_mode
        self.manual_padding = cfg.padding if cfg.padding_mode != "zeros" else 0
        conv_padding: Any
        if cfg.padding_mode != "zeros":
            conv_padding = "VALID"
        else:
            conv_padding = ((cfg.padding, cfg.padding), (cfg.padding, cfg.padding))

        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels_,
            kernel_size=(cfg.kernel_size, cfg.kernel_size),
            strides=(cfg.stride, cfg.stride),
            padding=conv_padding,
            kernel_dilation=(cfg.dilation, cfg.dilation),
            feature_group_count=cfg.groups,
            use_bias=cfg.bias,
            rngs=rngs,
        )

        norm_cfg = dict(cfg.norm_cfg) if cfg.norm_cfg else {}
        if cfg.norm not in {"none", "group"}:
            norm_cfg["num_features"] = out_channels_
        elif cfg.norm == "group":
            norm_cfg["num_channels"] = in_channels if cfg.norm_first else out_channels_

        norm_type: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"] = (
            "batch2d" if cfg.norm == "batch" else cfg.norm
        )  # type: ignore[assignment]
        self.norm = get_norm(norm_type, rngs=rngs, **norm_cfg)

        self.scale_factor = cfg.scale_factor
        self.activation = Activation(cfg.activation, dim=-1)
        self.dropout: nnx.Module = nnx.Dropout(rate=cfg.dropout, rngs=rngs) if cfg.dropout > 0 else Identity()
        self.norm_first = cfg.norm_first

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, in_channels) or (H, W, in_channels)

        Returns
        -------
        jax.Array
            Output tensor of shape (B, H', W', out_channels) or (H', W', out_channels)
        """
        if self.norm_first:
            x = self.norm(x)
            x = _pad_input(x, self.manual_padding, self.padding_mode, n_spatial_dims=2)
            x = self.conv(x)
            x = self._apply_shuffle(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x = _pad_input(x, self.manual_padding, self.padding_mode, n_spatial_dims=2)
            x = self.conv(x)
            x = self.norm(x)
            x = self._apply_shuffle(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x

    def _apply_shuffle(self, x: jax.Array) -> jax.Array:
        if self.scale_factor > 0:
            return pixel_shuffle_2d(x, self.scale_factor)
        elif self.scale_factor < 0:
            return pixel_unshuffle_2d(x, abs(self.scale_factor))
        return x


class ConvNormActivation1d(nnx.Module):
    """
    1D Convolutional layer with normalization and activation, and dropouts.

    Uses NLC (channels-last) format.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    cfg : ConvConfig
        Convolutional layer configuration.
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
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
    >>> conv = ConvNormActivation1d(3, 16, cfg, rngs=rngs)
    >>> x = jnp.ones((1, 32, 3))
    >>> output = conv(x)
    >>> output.shape
    (1, 32, 16)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        out_channels_ = out_channels
        if "glu" in cfg.activation.lower():
            out_channels_ *= 2
        if cfg.scale_factor > 0:
            out_channels_ *= abs(cfg.scale_factor)
        elif cfg.scale_factor < 0:
            out_channels_ //= abs(cfg.scale_factor)

        # Handle padding mode
        self.padding_mode = cfg.padding_mode
        self.manual_padding = cfg.padding if cfg.padding_mode != "zeros" else 0
        conv_padding: Any
        if cfg.padding_mode != "zeros":
            conv_padding = "VALID"
        else:
            conv_padding = ((cfg.padding, cfg.padding),)

        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels_,
            kernel_size=(cfg.kernel_size,),
            strides=(cfg.stride,),
            padding=conv_padding,
            kernel_dilation=(cfg.dilation,),
            feature_group_count=cfg.groups,
            use_bias=cfg.bias,
            rngs=rngs,
        )

        norm_cfg = dict(cfg.norm_cfg) if cfg.norm_cfg else {}
        if cfg.norm not in {"none", "group"}:
            norm_cfg["num_features"] = out_channels_
        elif cfg.norm == "group":
            norm_cfg["num_channels"] = in_channels if cfg.norm_first else out_channels_

        norm_type: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"] = (
            "batch1d" if cfg.norm == "batch" else cfg.norm
        )  # type: ignore[assignment]
        self.norm = get_norm(norm_type, rngs=rngs, **norm_cfg)

        self.scale_factor = cfg.scale_factor
        self.activation = Activation(cfg.activation, dim=-1)
        self.dropout: nnx.Module = nnx.Dropout(rate=cfg.dropout, rngs=rngs) if cfg.dropout > 0 else Identity()
        self.norm_first = cfg.norm_first

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, L, in_channels) or (L, in_channels)

        Returns
        -------
        jax.Array
            Output tensor of shape (B, L', out_channels) or (L', out_channels)
        """
        if self.norm_first:
            x = self.norm(x)
            x = _pad_input(x, self.manual_padding, self.padding_mode, n_spatial_dims=1)
            x = self.conv(x)
            x = self._apply_shuffle(x)
            x = self.activation(x)
            x = self.dropout(x)
        else:
            x = _pad_input(x, self.manual_padding, self.padding_mode, n_spatial_dims=1)
            x = self.conv(x)
            x = self._apply_shuffle(x)
            x = self.norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x

    def _apply_shuffle(self, x: jax.Array) -> jax.Array:
        """Apply HorizonShuffle or HorizonUnShuffle."""
        if self.scale_factor > 0:
            # NLC format: (B, L, C) -> (B, L*factor, C//factor)
            return rearrange(x, "b l (c u) -> b (l u) c", u=self.scale_factor)
        elif self.scale_factor < 0:
            factor = abs(self.scale_factor)
            # NLC format: (B, L, C) -> (B, L//factor, C*factor)
            return rearrange(x, "b (l u) c -> b l (c u)", u=factor)
        return x


class ResidualBlock(nnx.Module):
    """
    Residual block.

    Uses NHWC (channels-last) format.

    Parameters
    ----------
    in_features : int
        Input features.
    kernel_size : int
        Kernel size.
    activation : str
        Activation function.
    norm : Literal["batch", "group", "none"]
        Normalization layer.
    norm_cfg : dict
        Normalization layer configuration.
    dropout : float
        Dropout rate.
    padding_mode : str
        Padding mode.
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
    >>> resblock = ResidualBlock(16, 3, "ReLU", "batch", {}, 0.1, rngs=rngs)
    >>> x = jnp.ones((1, 32, 32, 16))
    >>> output = resblock(x)
    >>> output.shape
    (1, 32, 32, 16)
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
        *,
        rngs: nnx.Rngs,
    ) -> None:
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
        self.conv1 = ConvNormActivation(
            in_features,
            in_features,
            first_cfg,
            rngs=rngs,
        )
        self.conv2 = ConvNormActivation(
            in_features,
            in_features * 2 if "glu" in activation.lower() else in_features,
            second_cfg,
            rngs=rngs,
        )
        self.activation = Activation(activation, dim=-1)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, in_features) or (H, W, in_features)

        Returns
        -------
        jax.Array
            Output tensor of shape (B, H, W, in_features) or (H, W, in_features)
        """
        h = self.conv1(x)
        h = self.conv2(h)
        return self.activation(x + h)


class ConvTransposeNormActivation(nnx.Module):
    """
    Transposed convolutional layer with normalization and activation, and dropouts.

    Uses NHWC (channels-last) format.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    cfg : ConvConfig
        Convolutional layer configuration.
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
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
    >>> conv = ConvTransposeNormActivation(3, 16, cfg, rngs=rngs)
    >>> x = jnp.ones((1, 32, 32, 3))
    >>> output = conv(x)
    >>> output.shape
    (1, 32, 32, 16)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        out_features = out_channels * 2 if "glu" in cfg.activation.lower() else out_channels

        self.conv = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_features,
            kernel_size=(cfg.kernel_size, cfg.kernel_size),
            strides=(cfg.stride, cfg.stride),
            padding=((cfg.padding, cfg.padding + cfg.output_padding), (cfg.padding, cfg.padding + cfg.output_padding)),
            kernel_dilation=(cfg.dilation, cfg.dilation),
            use_bias=cfg.bias,
            rngs=rngs,
        )

        norm_cfg = dict(cfg.norm_cfg) if cfg.norm_cfg else {}
        if cfg.norm not in {"none", "group"}:
            norm_cfg["num_features"] = out_features
        elif cfg.norm == "group":
            norm_cfg["num_channels"] = out_features
        norm_type: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"] = (
            "batch2d" if cfg.norm == "batch" else cfg.norm
        )  # type: ignore[assignment]
        self.norm = get_norm(norm_type, rngs=rngs, **norm_cfg)
        self.activation = Activation(cfg.activation, dim=-1)
        self.dropout: nnx.Module = nnx.Dropout(rate=cfg.dropout, rngs=rngs) if cfg.dropout > 0 else Identity()

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, in_channels) or (H, W, in_channels)

        Returns
        -------
        jax.Array
            Output tensor of shape (B, H', W', out_channels) or (H', W', out_channels)
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)


class ConvTransposeNormActivation1d(nnx.Module):
    """
    1D Transposed convolutional layer with normalization and activation, and dropouts.

    Uses NLC (channels-last) format.

    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels.
    cfg : ConvConfig
        Convolutional layer configuration.
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
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
    >>> conv = ConvTransposeNormActivation1d(3, 16, cfg, rngs=rngs)
    >>> x = jnp.ones((1, 32, 3))
    >>> output = conv(x)
    >>> output.shape
    (1, 32, 16)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cfg: ConvConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        out_features = out_channels * 2 if "glu" in cfg.activation.lower() else out_channels

        self.conv = nnx.ConvTranspose(
            in_features=in_channels,
            out_features=out_features,
            kernel_size=(cfg.kernel_size,),
            strides=(cfg.stride,),
            padding=((cfg.padding, cfg.padding + cfg.output_padding),),
            kernel_dilation=(cfg.dilation,),
            use_bias=cfg.bias,
            rngs=rngs,
        )

        norm_cfg = dict(cfg.norm_cfg) if cfg.norm_cfg else {}
        if cfg.norm not in {"none", "group"}:
            norm_cfg["num_features"] = out_features
        elif cfg.norm == "group":
            norm_cfg["num_channels"] = out_features
        norm_type: Literal["layer", "rms", "group", "batch2d", "batch1d", "none"] = (
            "batch1d" if cfg.norm == "batch" else cfg.norm
        )  # type: ignore[assignment]
        self.norm = get_norm(norm_type, rngs=rngs, **norm_cfg)
        self.activation = Activation(cfg.activation, dim=-1)
        self.dropout: nnx.Module = nnx.Dropout(rate=cfg.dropout, rngs=rngs) if cfg.dropout > 0 else Identity()

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, L, in_channels) or (L, in_channels)

        Returns
        -------
        jax.Array
            Output tensor of shape (B, L', out_channels) or (L', out_channels)
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)


class TransformerLayer(nnx.Module):
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
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
    >>> cfg = TransformerConfig(
    ...     d_model=16,
    ...     nhead=4,
    ...     dim_ff=64,
    ...     n_layers=3,
    ...     dropout=0.1,
    ...     hidden_activation="ReLU",
    ...     output_activation="ReLU"
    ... )
    >>> transformer = TransformerLayer(32, 16, cfg, rngs=rngs)
    >>> x = jnp.ones((1, 8, 32))
    >>> output = transformer(x)
    >>> output.shape
    (1, 8, 16)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: TransformerConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.d_model = cfg.d_model
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection
        if input_dim != cfg.d_model:
            self.in_linear = nnx.Linear(input_dim, cfg.d_model, rngs=rngs)
            self.in_norm = nnx.LayerNorm(num_features=cfg.d_model, rngs=rngs)
            self.in_activation = Activation(cfg.hidden_activation)
            self._has_in_proj = True
        else:
            self._has_in_proj = False

        # Output projection
        if output_dim:
            self.out_linear = nnx.Linear(cfg.d_model, output_dim, rngs=rngs)
            self._has_out_proj = True
        else:
            self._has_out_proj = False
        self.out_activation = Activation(cfg.output_activation)

        # Transformer encoder layers
        layers = []
        for _ in range(cfg.n_layers):
            layers.append(
                _TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.nhead,
                    dim_feedforward=cfg.dim_ff,
                    dropout=cfg.dropout,
                    activation=cfg.hidden_activation,
                    rngs=rngs,
                )
            )
        self.encoder_layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, L, input_dim)

        Returns
        -------
        jax.Array
            Output tensor of shape (B, L, output_dim)
        """
        if self._has_in_proj:
            x = self.in_linear(x)
            x = self.in_norm(x)
            x = self.in_activation(x)

        for layer in self.encoder_layers:
            x = layer(x)

        if self._has_out_proj:
            x = self.out_linear(x)
        x = self.out_activation(x)
        return x


class _TransformerEncoderLayer(nnx.Module):
    """Single transformer encoder layer (pre-norm style)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=nhead,
            in_features=d_model,
            decode=False,
            rngs=rngs,
        )
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout, rngs=rngs) if dropout > 0 else Identity()
        self.dropout2 = nnx.Dropout(rate=dropout, rngs=rngs) if dropout > 0 else Identity()
        self.activation = Activation(activation)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = self.dropout1(x)
        x = residual + x

        # Feedforward with residual
        residual = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = residual + x

        return x


class PositionalEncoding(nnx.Module):
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
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
    >>> pos_enc = PositionalEncoding(16, 0.1, 100, rngs=rngs)
    >>> x = jnp.ones((1, 8, 16))
    >>> output = pos_enc(x)
    >>> output.shape
    (1, 8, 16)
    """

    def __init__(self, d_model: int, dropout: float, max_len: int, *, rngs: nnx.Rngs) -> None:
        self.dropout: nnx.Module = nnx.Dropout(rate=dropout, rngs=rngs) if dropout > 0 else Identity()

        position = jnp.arange(max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = jnp.zeros((1, max_len, d_model))
        pe = pe.at[0, :, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[0, :, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe  # Non-trainable buffer

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, L, D)

        Returns
        -------
        jax.Array
            Output tensor of shape (B, L, D)
        """
        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)


class PatchEmbed(nnx.Module):
    """
    Patch embedding layer.

    Uses NHWC (channels-last) format.

    Parameters
    ----------
    emb_dim : int
        Embedding dimension.
    patch_size : int
        Patch size.
    obs_shape : tuple[int, int, int]
        Observation shape in (H, W, C) format.
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
    >>> patch_embed = PatchEmbed(16, 4, (32, 32, 3), rngs=rngs)
    >>> x = jnp.ones((1, 32, 32, 3))
    >>> output = patch_embed(x)
    >>> output.shape
    (1, 64, 16)
    """

    def __init__(
        self,
        emb_dim: int,
        patch_size: int,
        obs_shape: tuple[int, int, int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.emb_dim = emb_dim
        self.obs_shape = obs_shape  # (H, W, C)

        self.patch_size = patch_size
        self.patch_num = (obs_shape[0] // patch_size) * (obs_shape[1] // patch_size)
        assert self.patch_size * self.patch_size * self.patch_num == self.obs_shape[0] * self.obs_shape[1], (
            "patch_num is not correct"
        )

        self.patch_emb_layer = nnx.Conv(
            in_features=obs_shape[2],  # C is last in NHWC
            out_features=emb_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, C)

        Returns
        -------
        jax.Array
            Output tensor of shape (B, Np, D)

        Np is the number of patches.
            Np = H*W/P^2
        D is the embedding dimension.
        """
        # (B, H, W, C) -> (B, H/P, W/P, D)
        x = self.patch_emb_layer(x)
        b = x.shape[0]
        # (B, H/P, W/P, D) -> (B, Np, D)
        return x.reshape(b, -1, self.emb_dim)


def _create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Create a 2D meshgrid for spatial coordinate computation.

    Parameters
    ----------
    height : int
        Grid height.
    width : int
        Grid width.
    normalized_coordinates : bool
        Whether to normalize coordinates to [-1, 1].

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (pos_y, pos_x) coordinate grids, each of shape (height, width).
    """
    if normalized_coordinates:
        xs = jnp.linspace(-1, 1, width)
        ys = jnp.linspace(-1, 1, height)
    else:
        xs = jnp.arange(width, dtype=jnp.float32)
        ys = jnp.arange(height, dtype=jnp.float32)
    pos_x, pos_y = jnp.meshgrid(xs, ys)
    return pos_y, pos_x


class Attention2d(nnx.Module):
    """2D self-attention mechanism.

    Uses NHWC (channels-last) format.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    nhead : int | None
        Number of attention heads.
    patch_size : int
        Patch size for local attention.
    attn_cfg : AttentionConfig | None
        Attention configuration.
    rngs : nnx.Rngs
        Random number generators.

    Examples
    --------
    >>> rngs = nnx.Rngs(0)
    >>> attn = Attention2d(64, nhead=8, rngs=rngs)
    >>> x = jnp.ones((2, 32, 32, 64))
    >>> out = attn(x)
    >>> out.shape
    (2, 32, 32, 64)
    """

    def __init__(
        self,
        channels: int,
        nhead: int | None = None,
        patch_size: int = 1,
        attn_cfg: AttentionConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
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
        first_cfg = deepcopy(cfg)
        first_cfg.norm_first = True
        self.qkv = ConvNormActivation(channels, channels * 3, first_cfg, rngs=rngs)
        self.proj_out = ConvNormActivation(channels, channels, cfg, rngs=rngs)

    def qkv_attn(self, qkv: jax.Array) -> jax.Array:
        """Apply QKV attention.

        Parameters
        ----------
        qkv : jax.Array
            An [N x H x W x (Heads * 3 * C)] tensor of query, key, value.

        Returns
        -------
        jax.Array
            An [N x H x W x (C * Head)] tensor of attended values.
        """
        bs, height, width, channels = qkv.shape
        assert channels % (3 * self.n_heads) == 0
        ch = channels // (3 * self.n_heads)
        qkv_r = qkv.reshape(bs * self.n_heads, height * width, ch * 3)
        q, k, v = jnp.split(qkv_r, 3, axis=-1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = jnp.einsum("btc,bsc->bts", q * scale, k * scale)
        weight = jax.nn.softmax(weight - jnp.max(weight, axis=-1, keepdims=True), axis=-1)
        a = jnp.einsum("bts,bsc->btc", weight, v)
        return a.reshape(bs, height, width, -1)

    def __call__(self, x: jax.Array) -> jax.Array:
        b, h, w, c = x.shape
        qkv = self.qkv(x)
        # NHWC patch rearrange
        qkv = rearrange(qkv, "b (h p1) (w p2) c -> b h w (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
        a = self.qkv_attn(qkv)
        a = rearrange(a, "b h w (c p1 p2) -> b (h p1) (w p2) c", p1=self.patch_size, p2=self.patch_size)
        a = self.proj_out(a)
        return (x + a).reshape(b, h, w, c)


class Attention1d(nnx.Module):
    """1D self-attention mechanism.

    Uses NLC (channels-last) format.

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    nhead : int | None
        Number of attention heads.
    patch_size : int
        Patch size.
    attn_cfg : AttentionConfig | None
        Attention configuration.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        channels: int,
        nhead: int | None = None,
        patch_size: int = 1,
        attn_cfg: AttentionConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
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
        first_cfg = deepcopy(cfg)
        first_cfg.norm_first = True
        self.qkv = ConvNormActivation1d(channels, channels * 3, first_cfg, rngs=rngs)
        self.proj_out = ConvNormActivation1d(channels, channels, cfg, rngs=rngs)

    def qkv_attention(self, qkv: jax.Array) -> jax.Array:
        """Apply QKV attention.

        Parameters
        ----------
        qkv : jax.Array
            An [N x T x (H * 3 * C)] tensor (NLC format).

        Returns
        -------
        jax.Array
            An [N x T x (H * C)] tensor after attention.
        """
        bs, length, channels = qkv.shape
        assert channels % (3 * self.n_heads) == 0
        ch = channels // (3 * self.n_heads)
        qkv_r = qkv.reshape(bs * self.n_heads, length, ch * 3)
        q, k, v = jnp.split(qkv_r, 3, axis=-1)
        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = jnp.einsum("btc,bsc->bts", q * scale, k * scale)
        weight = jax.nn.softmax(weight - jnp.max(weight, axis=-1, keepdims=True), axis=-1)
        a = jnp.einsum("bts,bsc->btc", weight, v)
        return a.reshape(bs, length, -1)

    def __call__(self, x: jax.Array) -> jax.Array:
        b, *spatial, c = x.shape
        x_flat = x.reshape(b, -1, c)
        qkv = self.qkv(x_flat)
        # NLC patch rearrange
        qkv = rearrange(qkv, "b (t p) c -> b t (c p)", p=self.patch_size)
        h = self.qkv_attention(qkv)
        h = rearrange(h, "b t (c p) -> b (t p) c", p=self.patch_size)
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, *spatial, c)


class SpatialSoftmax(nnx.Module):
    """
    Spatial Softmax and Flatten layer.

    Uses NHWC (channels-last) format: input (B, H, W, C).

    Parameters
    ----------
    cfg : SpatialSoftmaxConfig
        Spatial softmax configuration.

    Examples
    --------
    >>> cfg = SpatialSoftmaxConfig(temperature=1.0, is_argmax=True)
    >>> spatial_softmax = SpatialSoftmax(cfg)
    >>> x = jnp.ones((1, 16, 16, 64))
    >>> output = spatial_softmax(x)
    >>> output.shape
    (1, 64, 2)
    """

    def __init__(self, cfg: SpatialSoftmaxConfig) -> None:
        self.temperature = cfg.temperature
        self.eps = cfg.eps
        assert self.temperature > 0.0, "temperature must be non-negative"
        if cfg.is_argmax:
            self._mode = "argmax"
        elif cfg.is_straight_through:
            self._mode = "straight_through"
        else:
            self._mode = "soft_argmax"

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, C)

        Returns
        -------
        jax.Array
            Output tensor of shape (B, C, 2)
        """
        x = x / self.temperature
        if self._mode == "argmax":
            return self._spatial_argmax2d(x)
        elif self._mode == "straight_through":
            return self._spatial_softmax_straight_through(x)
        else:
            return self._spatial_soft_argmax2d(x)

    def _spatial_soft_argmax2d(self, x: jax.Array) -> jax.Array:
        """Standard spatial soft-argmax (differentiable)."""
        batch_size, height, width, channels = x.shape
        pos_y, pos_x = _create_meshgrid(height, width, normalized_coordinates=True)

        # (B, H, W, C) -> (B, C, H*W)
        x_flat = x.reshape(batch_size, height * width, channels)
        x_flat = jnp.transpose(x_flat, (0, 2, 1))

        softmax_x = jax.nn.softmax(x_flat, axis=-1)

        pos_x_flat = pos_x.reshape(-1)
        pos_y_flat = pos_y.reshape(-1)

        expected_x = jnp.sum(pos_x_flat * softmax_x, axis=-1, keepdims=True)
        expected_y = jnp.sum(pos_y_flat * softmax_x, axis=-1, keepdims=True)
        return jnp.concatenate([expected_x, expected_y], axis=-1).reshape(batch_size, channels, 2)

    def _spatial_softmax_straight_through(self, x: jax.Array) -> jax.Array:
        """Spatial softmax with straight-through estimator."""
        batch_size, height, width, channels = x.shape
        pos_y, pos_x = _create_meshgrid(height, width, normalized_coordinates=True)

        # (B, H, W, C) -> (B, C, H*W)
        x_flat = x.reshape(batch_size, height * width, channels)
        x_flat = jnp.transpose(x_flat, (0, 2, 1))

        exp_x = jnp.exp(x_flat - jnp.max(x_flat, axis=-1, keepdims=True))
        exp_x_sum = 1.0 / (exp_x.sum(axis=-1, keepdims=True) + self.eps)
        softmax_x = exp_x * exp_x_sum

        # straight-through trick
        softmax_x = softmax_x + x_flat - jax.lax.stop_gradient(x_flat)

        pos_x_flat = pos_x.reshape(-1)
        pos_y_flat = pos_y.reshape(-1)

        expected_y = jnp.sum(pos_y_flat * softmax_x, axis=-1, keepdims=True)
        expected_x = jnp.sum(pos_x_flat * softmax_x, axis=-1, keepdims=True)
        return jnp.concatenate([expected_x, expected_y], axis=-1).reshape(batch_size, channels, 2)

    def _spatial_argmax2d(self, x: jax.Array) -> jax.Array:
        """Spatial argmax with straight-through estimator."""
        batch_size, height, width, channels = x.shape
        pos_y, pos_x = _create_meshgrid(height, width, normalized_coordinates=True)

        # (B, H, W, C) -> (B, C, H*W)
        x_flat = x.reshape(batch_size, height * width, channels)
        x_flat = jnp.transpose(x_flat, (0, 2, 1))

        exp_x = jnp.exp(x_flat - jnp.max(x_flat, axis=-1, keepdims=True))
        exp_x_sum = 1.0 / (exp_x.sum(axis=-1, keepdims=True) + self.eps)
        softmax_x = exp_x * exp_x_sum

        argmax_x = jnp.argmax(x_flat, axis=-1)
        argmax_x = jax.nn.one_hot(argmax_x, num_classes=x_flat.shape[-1])
        argmax_x = argmax_x + softmax_x - jax.lax.stop_gradient(softmax_x)

        pos_x_flat = pos_x.reshape(-1)
        pos_y_flat = pos_y.reshape(-1)

        expected_y = jnp.sum(pos_y_flat * argmax_x, axis=-1, keepdims=True)
        expected_x = jnp.sum(pos_x_flat * argmax_x, axis=-1, keepdims=True)
        return jnp.concatenate([expected_x, expected_y], axis=-1).reshape(batch_size, channels, 2)


class HorizonShuffle(nnx.Module):
    """Horizon Shuffle Layer.

    Uses NLC (channels-last) format.

    Parameters
    ----------
    upscale_factor : int
        Upscale factor.

    Examples
    --------
    >>> shuffle = HorizonShuffle(2)
    >>> x = jnp.ones((2, 50, 64))
    >>> out = shuffle(x)
    >>> out.shape
    (2, 100, 32)
    """

    def __init__(self, upscale_factor: int = 2) -> None:
        self.upscale_factor = upscale_factor

    def __call__(self, x: jax.Array) -> jax.Array:
        """Horizon Shuffle Layer.

        Parameters
        ----------
        x : jax.Array
            Input tensor (B, L, C)

        Returns
        -------
        jax.Array
            Output tensor (B, L * upscale_factor, C // upscale_factor)
        """
        _b, _l, c = x.shape
        assert c % self.upscale_factor == 0, "Channel dimension must be divisible by upscale_factor."
        return rearrange(x, "b l (c u) -> b (l u) c", u=self.upscale_factor)


class HorizonUnShuffle(nnx.Module):
    """Horizon UnShuffle Layer.

    Uses NLC (channels-last) format.

    Parameters
    ----------
    downscale_factor : int
        Downscale factor.

    Examples
    --------
    >>> unshuffle = HorizonUnShuffle(2)
    >>> x = jnp.ones((2, 100, 32))
    >>> out = unshuffle(x)
    >>> out.shape
    (2, 50, 64)
    """

    def __init__(self, downscale_factor: int = 2) -> None:
        self.downscale_factor = downscale_factor

    def __call__(self, x: jax.Array) -> jax.Array:
        """Horizon UnShuffle Layer.

        Parameters
        ----------
        x : jax.Array
            Input tensor (B, L * downscale_factor, C)

        Returns
        -------
        jax.Array
            Output tensor (B, L, C * downscale_factor)
        """
        _b, l, _c = x.shape
        assert l % self.downscale_factor == 0, "Length dimension must be divisible by downscale_factor."
        return rearrange(x, "b (l u) c -> b l (c u)", u=self.downscale_factor)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
