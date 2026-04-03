"""Vision モデルを扱うモジュール."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx

from ml_networks.config import (
    AdaptiveAveragePoolingConfig,
    ConvConfig,
    ConvNetConfig,
    LinearConfig,
    MLPConfig,
    ResNetConfig,
    SpatialSoftmaxConfig,
    ViTConfig,
)
from ml_networks.jax.layers import (
    Attention2d,
    ConvNormActivation,
    ConvTransposeNormActivation,
    Identity,
    LinearNormActivation,
    MLPLayer,
    PatchEmbed,
    PositionalEncoding,
    ResidualBlock,
    SpatialSoftmax,
    TransformerLayer,
)
from ml_networks.utils import conv_out_shape, conv_transpose_in_shape


class Encoder(nnx.Module):
    """
    Image encoder module (NHWC format).

    Parameters
    ----------
    feature_dim : int | tuple[int, int, int]
        Output feature dimension.
        If int, a fully-connected layer flattens and projects the backbone output.
        If tuple, the backbone output is returned directly (fc is identity).
    obs_shape : tuple[int, int, int]
        Observation shape in (H, W, C) format.
    backbone_cfg : ViTConfig | ConvNetConfig | ResNetConfig
        Backbone configuration.
    fc_cfg : MLPConfig | LinearConfig | SpatialSoftmaxConfig | AdaptiveAveragePoolingConfig | None
        Fully-connected layer configuration. Required when ``feature_dim`` is int.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        feature_dim: int | tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        backbone_cfg: ViTConfig | ConvNetConfig | ResNetConfig,
        fc_cfg: MLPConfig | LinearConfig | SpatialSoftmaxConfig | AdaptiveAveragePoolingConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        self.encoder: nnx.Module
        if isinstance(backbone_cfg, ViTConfig):
            self.encoder = ViT(obs_shape, backbone_cfg, rngs=rngs)
            self.last_channel: int = self.encoder.last_channel
            self.conved_size: int = cast("int", self.encoder.conved_size)
            self.conved_shape: tuple[int, ...] = cast("tuple[int, ...]", self.encoder.conved_shape)
        elif isinstance(backbone_cfg, ConvNetConfig):
            self.encoder = ConvNet(obs_shape, backbone_cfg, rngs=rngs)
            self.last_channel = self.encoder.last_channel
            self.conved_size = cast("int", self.encoder.conved_size)
            self.conved_shape = cast("tuple[int, ...]", self.encoder.conved_shape)
        elif isinstance(backbone_cfg, ResNetConfig):
            self.encoder = ResNetPixUnshuffle(obs_shape, backbone_cfg, rngs=rngs)
            self.last_channel = self.encoder.last_channel
            self.conved_size = cast("int", self.encoder.conved_size)
            self.conved_shape = cast("tuple[int, ...]", self.encoder.conved_shape)
        else:
            msg = f"{type(backbone_cfg)} is not implemented"
            raise NotImplementedError(msg)

        if isinstance(feature_dim, int):
            assert fc_cfg is not None, "fc_cfg must be provided if feature_dim is int"
        else:
            assert feature_dim == (self.last_channel, *self.conved_shape), (
                f"{feature_dim} != {(self.last_channel, *self.conved_shape)}"
            )

        self.fc: nnx.Module
        if isinstance(fc_cfg, MLPConfig):
            assert isinstance(feature_dim, int)
            self.fc = MLPLayer(self.conved_size, feature_dim, fc_cfg, rngs=rngs)
        elif isinstance(fc_cfg, LinearConfig):
            assert isinstance(feature_dim, int)
            self.fc = LinearNormActivation(self.conved_size, feature_dim, fc_cfg, rngs=rngs)
        elif isinstance(fc_cfg, AdaptiveAveragePoolingConfig):
            assert isinstance(feature_dim, int)
            output_size = fc_cfg.output_size
            pooled_size = int(self.last_channel * np.prod(output_size))
            if isinstance(fc_cfg.additional_layer, LinearConfig):
                self.fc = LinearNormActivation(pooled_size, feature_dim, fc_cfg.additional_layer, rngs=rngs)
            elif isinstance(fc_cfg.additional_layer, MLPConfig):
                self.fc = MLPLayer(pooled_size, feature_dim, fc_cfg.additional_layer, rngs=rngs)
            else:
                self.fc = Identity()
            if fc_cfg.additional_layer is None:
                self.feature_dim = pooled_size
            self._adaptive_pool_output_size = output_size
        elif isinstance(fc_cfg, SpatialSoftmaxConfig):
            assert isinstance(feature_dim, int)
            if isinstance(fc_cfg.additional_layer, LinearConfig):
                self.fc = LinearNormActivation(
                    self.last_channel * 2,
                    feature_dim,
                    fc_cfg.additional_layer,
                    rngs=rngs,
                )
            elif isinstance(fc_cfg.additional_layer, MLPConfig):
                self.fc = MLPLayer(self.last_channel * 2, feature_dim, fc_cfg.additional_layer, rngs=rngs)
            else:
                self.fc = Identity()
            if fc_cfg.additional_layer is None:
                self.feature_dim = self.last_channel * 2
            self._spatial_softmax = SpatialSoftmax(fc_cfg)
        else:
            self.fc = Identity()
        self._fc_cfg = fc_cfg

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (*, H, W, C) in NHWC format.

        Returns
        -------
        jax.Array
            Encoded tensor of shape (*, feature_dim).
        """
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *self.obs_shape)
        x = self.encoder(x)
        if isinstance(self._fc_cfg, AdaptiveAveragePoolingConfig):
            # NHWC adaptive average pooling: (B, H, W, C) -> (B, oh, ow, C)
            pool_size = self._adaptive_pool_output_size
            assert isinstance(pool_size, tuple)
            oh, ow = pool_size
            b, h, w, c = x.shape
            # Reshape to windows and average
            x = x.reshape(b, oh, h // oh, ow, w // ow, c)
            x = x.mean(axis=(2, 4))
            x = x.reshape(b, -1)
            x = self.fc(x)
        elif isinstance(self._fc_cfg, SpatialSoftmaxConfig):
            x = self._spatial_softmax(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc(x)
        else:
            x = x.reshape(x.shape[0], -1)
            x = self.fc(x)
        return x.reshape(*batch_shape, *x.shape[1:])


class Decoder(nnx.Module):
    """
    Image decoder module (NHWC format).

    Parameters
    ----------
    feature_dim : int | tuple[int, int, int]
        Input feature dimension.
        If int, a fully-connected layer projects and reshapes input before the backbone.
        If tuple, input is passed directly to the backbone.
    obs_shape : tuple[int, int, int]
        Output observation shape in (H, W, C) format.
    backbone_cfg : ConvNetConfig | ViTConfig | ResNetConfig
        Backbone configuration.
    fc_cfg : MLPConfig | LinearConfig | None
        Fully-connected layer configuration. Required when ``feature_dim`` is int.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        feature_dim: int | tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        backbone_cfg: ConvNetConfig | ViTConfig | ResNetConfig,
        fc_cfg: MLPConfig | LinearConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        self.input_shape: tuple[int, int, int]
        if isinstance(backbone_cfg, ViTConfig):
            self.input_shape = ViT.get_input_shape(obs_shape, backbone_cfg)
        elif isinstance(backbone_cfg, ConvNetConfig):
            self.input_shape = cast(
                "tuple[int, int, int]",
                ConvTranspose.get_input_shape(obs_shape, backbone_cfg),
            )
        elif isinstance(backbone_cfg, ResNetConfig):
            self.input_shape = ResNetPixShuffle.get_input_shape(obs_shape, backbone_cfg)
        else:
            msg = f"{type(backbone_cfg)} is not implemented"
            raise NotImplementedError(msg)

        if isinstance(feature_dim, int):
            assert fc_cfg is not None, "fc_cfg must be provided if feature_dim is int"
            self.has_fc = True
        else:
            assert feature_dim == self.input_shape, f"{feature_dim} != {self.input_shape}"
            self.has_fc = False

        input_size = int(np.prod(self.input_shape))
        self.fc: nnx.Module
        if isinstance(fc_cfg, MLPConfig):
            assert isinstance(feature_dim, int)
            self.fc = MLPLayer(feature_dim, input_size, fc_cfg, rngs=rngs)
        elif isinstance(fc_cfg, LinearConfig):
            assert isinstance(feature_dim, int)
            self.fc = LinearNormActivation(feature_dim, input_size, fc_cfg, rngs=rngs)
        else:
            self.fc = Identity()

        if isinstance(backbone_cfg, ViTConfig):
            self.decoder: nnx.Module = ViT(
                in_shape=self.input_shape,
                cfg=backbone_cfg,
                obs_shape=obs_shape,
                rngs=rngs,
            )
        elif isinstance(backbone_cfg, ConvNetConfig):
            self.decoder = ConvTranspose(
                in_shape=self.input_shape,
                obs_shape=obs_shape,
                cfg=backbone_cfg,
                rngs=rngs,
            )
        elif isinstance(backbone_cfg, ResNetConfig):
            self.decoder = ResNetPixShuffle(
                in_shape=self.input_shape,
                obs_shape=obs_shape,
                cfg=backbone_cfg,
                rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (*, feature_dim).

        Returns
        -------
        jax.Array
            Decoded tensor of shape (*, H, W, C) in NHWC format.
        """
        if self.has_fc:
            batch_shape, data_shape = x.shape[:-1], x.shape[-1:]
        else:
            batch_shape, data_shape = x.shape[:-3], x.shape[-3:]
        x = x.reshape(-1, *data_shape)
        x = self.fc(x)
        x = x.reshape(-1, *self.input_shape)
        x = self.decoder(x)
        return x.reshape(*batch_shape, *self.obs_shape)


class ViT(nnx.Module):
    """
    Vision Transformer for Encoder and Decoder (NHWC format).

    Parameters
    ----------
    in_shape : tuple[int, int, int]
        Input shape in (H, W, C) format.
    cfg : ViTConfig
        ViT configuration.
    obs_shape : tuple[int, int, int] | None
        Output shape in (H, W, C) format. If None, acts as encoder.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        cfg: ViTConfig,
        obs_shape: tuple[int, int, int] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = cfg
        self.in_shape = in_shape
        self.obs_shape = obs_shape if obs_shape is not None else in_shape
        self.patch_size = cfg.patch_size

        t_cfg = cfg.transformer_cfg
        self.transformer_cfg = t_cfg
        # NHWC: (H, W, C) -> patch_dim = patch_size^2 * C
        self.in_patch_dim = self.get_patch_dim(in_shape)
        self.out_patch_dim = self.get_patch_dim(obs_shape) if obs_shape is not None else t_cfg.d_model

        self.positional_encoding = PositionalEncoding(
            self.in_patch_dim,
            t_cfg.dropout,
            max_len=self.get_n_patches(in_shape),
            rngs=rngs,
        )
        self.vit = TransformerLayer(self.in_patch_dim, self.out_patch_dim, t_cfg, rngs=rngs)

        self.is_encoder = obs_shape is None
        if self.is_encoder:
            self.n_patches = self.get_n_patches(in_shape)
            self.patch_embed = PatchEmbed(
                emb_dim=self.in_patch_dim,
                patch_size=self.patch_size,
                obs_shape=in_shape,
                rngs=rngs,
            )

        self.should_unpatchify = cfg.unpatchify
        if cfg.cls_token:
            self._cls_token = nnx.Param(jax.random.normal(rngs(), (1, 1, self.in_patch_dim)) * 0.02)
        self.last_channel = self.get_n_patches(in_shape)
        self.output_dim = self.out_patch_dim

    def __call__(self, x: jax.Array, *, return_cls_token: bool = False) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, C) in NHWC format.
        return_cls_token : bool
            Whether to return CLS token only. Default is False.

        Returns
        -------
        jax.Array
            Output tensor.
        """
        x = self.patch_embed(x) if self.is_encoder else self.patchify(x)
        x = self.positional_encoding(x)
        if hasattr(self, "_cls_token"):
            cls_token = jnp.broadcast_to(self._cls_token.value, (x.shape[0], 1, x.shape[-1]))
            x = jnp.concatenate([cls_token, x], axis=1)
        x = self.vit(x)
        if hasattr(self, "_cls_token"):
            cls_token = x[:, 0]
            x = x[:, 1:]
        if self.should_unpatchify:
            x = self.unpatchify(x)
        if return_cls_token and hasattr(self, "_cls_token"):
            return cls_token
        return x

    def patchify(self, imgs: jax.Array) -> jax.Array:
        """Split images into patches.

        Parameters
        ----------
        imgs : jax.Array
            Input images of shape (N, H, W, C) in NHWC format.

        Returns
        -------
        jax.Array
            Patchified images of shape (N, L, patch_size**2 * C).
        """
        p = self.patch_size
        return rearrange(imgs, "n (h p1) (w p2) c -> n (h w) (p1 p2 c)", p1=p, p2=p)

    def unpatchify(self, x: jax.Array) -> jax.Array:
        """Reconstruct images from patches.

        Parameters
        ----------
        x : jax.Array
            Input of shape (N, L, patch_size**2 * C).

        Returns
        -------
        jax.Array
            Images of shape (N, H, W, C) in NHWC format.
        """
        p = self.patch_size
        h = self.obs_shape[0] // p
        w = self.obs_shape[1] // p
        assert h * w == x.shape[1], (
            f"{h * w} != {x.shape[1]}, please check the shape {x.shape} and obs_shape {self.obs_shape}"
        )
        return rearrange(x, "n (h w) (p1 p2 c) -> n (h p1) (w p2) c", h=h, w=w, p1=p, p2=p)

    @property
    def conved_size(self) -> int:
        """Get the flattened output size."""
        return self.out_patch_dim * self.get_n_patches(self.in_shape)

    @property
    def conved_shape(self) -> tuple[int, int]:
        """Get the output shape after transformer."""
        return (self.out_patch_dim, self.out_patch_dim)

    def get_n_patches(self, obs_shape: tuple[int, int, int]) -> int:
        """Get number of patches for a given shape (NHWC: H, W, C)."""
        return (obs_shape[0] // self.patch_size) * (obs_shape[1] // self.patch_size)

    def get_patch_dim(self, obs_shape: tuple[int, int, int]) -> int:
        """Get patch dimension for a given shape (NHWC: H, W, C)."""
        return self.patch_size**2 * obs_shape[2]

    @staticmethod
    def get_input_shape(obs_shape: tuple[int, int, int], cfg: ViTConfig) -> tuple[int, int, int]:
        """Get the required input shape (NHWC: H, W, C)."""
        return (obs_shape[0], obs_shape[1], cfg.init_channel)


class _ViTBlock(nnx.Module):
    """Single ViT transformer block (pre-norm)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm1 = nnx.LayerNorm(num_features=dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=dim,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(num_features=dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, mlp_hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_hidden_dim, dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Self-attention with residual
        h = self.norm1(x)
        h = self.attn(h)
        x = x + h
        # MLP with residual
        h = self.norm2(x)
        h = self.fc1(h)
        h = jax.nn.gelu(h)
        h = self.fc2(h)
        return x + h


class ConvNet(nnx.Module):
    """
    Convolutional network (NHWC format).

    Parameters
    ----------
    obs_shape : tuple[int, int, int]
        Observation shape in (H, W, C) format.
    cfg : ConvNetConfig
        Configuration.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        cfg: ConvNetConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = cfg
        self.obs_shape = obs_shape
        in_channels = obs_shape[2]  # NHWC
        self.channels = [in_channels, *cfg.channels]

        layers: list[nnx.Module] = []
        attn_layers: list[nnx.Module] = []
        spatial_shape: tuple[int, ...] = (obs_shape[0], obs_shape[1])

        for ch, conv_cfg_i in zip(cfg.channels, cfg.conv_cfgs, strict=True):
            layers.append(ConvNormActivation(in_channels, ch, conv_cfg_i, rngs=rngs))

            if cfg.attention is not None:
                attn_layers.append(
                    Attention2d(ch, nhead=None, attn_cfg=cfg.attention, rngs=rngs),
                )
            else:
                attn_layers.append(Identity())

            spatial_shape = conv_out_shape(
                spatial_shape,
                padding=conv_cfg_i.padding,
                kernel_size=conv_cfg_i.kernel_size,
                stride=conv_cfg_i.stride,
                dilation=conv_cfg_i.dilation,
            )
            in_channels = ch

        self.conv_layers = nnx.List(layers)
        self.attn_layers = nnx.List(attn_layers)
        self.output_spatial_shape = spatial_shape
        self.output_channels = in_channels
        self.last_channel = in_channels
        self.output_dim = in_channels * int(np.prod(spatial_shape))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, C) in NHWC format.

        Returns
        -------
        jax.Array
            Flattened tensor of shape (B, output_dim).
        """
        for conv, attn in zip(self.conv_layers, self.attn_layers, strict=False):
            x = conv(x)
            x = attn(x)
        return x.reshape(x.shape[0], -1)

    @property
    def conved_shape(self) -> tuple[int, ...]:
        """Get the spatial shape of the output after convolutional layers."""
        spatial: tuple[int, ...] = (self.obs_shape[0], self.obs_shape[1])
        for conv_cfg_i in self.cfg.conv_cfgs:
            spatial = conv_out_shape(
                spatial,
                padding=conv_cfg_i.padding,
                kernel_size=conv_cfg_i.kernel_size,
                stride=conv_cfg_i.stride,
                dilation=conv_cfg_i.dilation,
            )
        return spatial

    @property
    def conved_size(self) -> int:
        """Get the flattened size of the output after convolutional layers."""
        return self.output_channels * int(np.prod(self.conved_shape))


class ConvTranspose(nnx.Module):
    """
    Transposed convolutional network (NHWC format).

    Parameters
    ----------
    in_shape : tuple[int, int, int]
        Input shape in (H, W, C) format.
    obs_shape : tuple[int, int, int]
        Output observation shape in (H, W, C) format.
    cfg : ConvNetConfig
        Configuration (channels are in decode order).
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        cfg: ConvNetConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_shape = in_shape
        self.obs_shape = obs_shape
        self.cfg = cfg
        # channels: cfg.channels -> obs_shape[2] (output channels)
        self.channels = [*cfg.channels, obs_shape[2]]

        assert len(cfg.channels) == len(cfg.conv_cfgs)

        # first_conv if input channels != first cfg channel
        self.have_first_conv = in_shape[2] != cfg.channels[0]
        if self.have_first_conv:
            first_conv_cfg = ConvConfig(
                activation="Identity",
                kernel_size=1,
                stride=1,
                padding=0,
                norm="none",
            )
            self.first_conv = ConvNormActivation(
                in_shape[2],
                cfg.channels[0],
                first_conv_cfg,
                rngs=rngs,
            )

        layers: list[nnx.Module] = []
        for i, conv_cfg_i in enumerate(cfg.conv_cfgs):
            if cfg.attention is not None:
                layers.append(
                    Attention2d(self.channels[i], nhead=None, attn_cfg=cfg.attention, rngs=rngs),
                )
            layers.append(
                ConvTransposeNormActivation(self.channels[i], self.channels[i + 1], conv_cfg_i, rngs=rngs),
            )

        self.conv_layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, C) in NHWC format.

        Returns
        -------
        jax.Array
            Output tensor of shape (B, H', W', out_C) in NHWC format.
        """
        if self.have_first_conv:
            x = self.first_conv(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x

    @staticmethod
    def get_input_shape(obs_shape: tuple[int, int, int], cfg: ConvNetConfig) -> tuple[int, ...]:
        """Get the required input shape for a given output shape and config.

        Parameters
        ----------
        obs_shape : tuple[int, int, int]
            Output shape in (H, W, C) format.
        cfg : ConvNetConfig
            Configuration.

        Returns
        -------
        tuple[int, ...]
            Required input shape in (H, W, C) format.
        """
        # NHWC: spatial dims are [0], [1]
        in_spatial: tuple[int, ...] = obs_shape[:2]
        for conv_cfg_i in reversed(cfg.conv_cfgs):
            in_spatial = conv_transpose_in_shape(
                in_spatial,
                padding=conv_cfg_i.padding,
                kernel_size=conv_cfg_i.kernel_size,
                stride=conv_cfg_i.stride,
                dilation=conv_cfg_i.dilation,
            )
        return (*in_spatial, cfg.init_channel)


class ResNetPixShuffle(nnx.Module):
    """
    ResNet with PixelShuffle upsampling (NHWC format).

    Parameters
    ----------
    in_shape : tuple[int, int, int]
        Input shape in (H, W, C) format.
    obs_shape : tuple[int, int, int]
        Output observation shape in (H, W, C) format.
    cfg : ResNetConfig
        Configuration.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        cfg: ResNetConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = cfg
        self.in_shape = in_shape
        self.obs_shape = obs_shape
        out_channels = obs_shape[2]  # NHWC

        conv_cfg = ConvConfig(
            activation=cfg.conv_activation,
            kernel_size=cfg.conv_kernel,
            stride=1,
            padding=cfg.conv_kernel // 2,
            dilation=1,
            groups=1,
            bias=True,
            dropout=cfg.dropout,
            norm=cfg.norm,
            norm_cfg=cfg.norm_cfg,
            padding_mode=cfg.padding_mode,
        )

        # First layer
        self.conv1 = ConvNormActivation(in_shape[2], cfg.conv_channel, conv_cfg, rngs=rngs)

        # Residual blocks
        res_blocks: list[nnx.Module] = []
        for _ in range(cfg.n_res_blocks):
            res_blocks.append(
                ResidualBlock(
                    cfg.conv_channel,
                    cfg.conv_kernel,
                    cfg.conv_activation,
                    cfg.norm,
                    cfg.norm_cfg,
                    cfg.dropout,
                    cfg.padding_mode,
                    rngs=rngs,
                ),
            )
            if cfg.attention is not None:
                res_blocks.append(
                    Attention2d(cfg.conv_channel, nhead=None, attn_cfg=cfg.attention, rngs=rngs),
                )
        self.res_blocks = nnx.List(res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = ConvNormActivation(cfg.conv_channel, cfg.conv_channel, conv_cfg, rngs=rngs)

        # Upsampling layers
        upscale_cfg = deepcopy(conv_cfg)
        upscale_cfg.scale_factor = cfg.scale_factor
        upsample_layers = [
            ConvNormActivation(cfg.conv_channel, cfg.conv_channel, upscale_cfg, rngs=rngs) for _ in range(cfg.n_scaling)
        ]
        self.upsampling = nnx.List(upsample_layers)

        # Final output layer
        final_cfg = ConvConfig(
            activation=cfg.out_activation,
            kernel_size=cfg.f_kernel,
            stride=1,
            padding=cfg.f_kernel // 2,
            norm="none",
            norm_cfg={},
            dropout=0.0,
        )
        self.conv3 = ConvNormActivation(cfg.conv_channel, out_channels, final_cfg, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, C) in NHWC format.

        Returns
        -------
        jax.Array
            Upsampled output of shape (B, H', W', C').
        """
        out1 = self.conv1(x)
        out = out1
        for layer in self.res_blocks:
            out = layer(out)
        out2 = self.conv2(out)
        out = out1 + out2
        for layer in self.upsampling:
            out = layer(out)
        return self.conv3(out)

    @staticmethod
    def get_input_shape(obs_shape: tuple[int, int, int], cfg: ResNetConfig) -> tuple[int, int, int]:
        """Get the required input shape for a given output shape and config."""
        scaling = cfg.scale_factor**cfg.n_scaling
        return (
            obs_shape[0] // scaling,
            obs_shape[1] // scaling,
            cfg.init_channel,
        )


class ResNetPixUnshuffle(nnx.Module):
    """
    ResNet with PixelUnshuffle downsampling (NHWC format).

    Parameters
    ----------
    obs_shape : tuple[int, int, int]
        Input observation shape in (H, W, C) format.
    cfg : ResNetConfig
        Configuration.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        cfg: ResNetConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = cfg
        self.obs_shape = obs_shape

        first_cfg = ConvConfig(
            activation=cfg.conv_activation,
            kernel_size=cfg.f_kernel,
            stride=1,
            padding=cfg.f_kernel // 2,
            dilation=1,
            groups=1,
            bias=True,
            dropout=cfg.dropout,
            norm=cfg.norm,
            norm_cfg=cfg.norm_cfg,
            padding_mode=cfg.padding_mode,
        )
        # First layer: input channels -> conv_channel
        self.conv1 = ConvNormActivation(obs_shape[2], cfg.conv_channel, first_cfg, rngs=rngs)

        # Downsampling layers
        downsample_cfg = deepcopy(first_cfg)
        downsample_cfg.kernel_size = cfg.conv_kernel
        downsample_cfg.padding = cfg.conv_kernel // 2
        downsample_cfg.scale_factor = -cfg.scale_factor
        downsample_layers = [
            ConvNormActivation(cfg.conv_channel, cfg.conv_channel, downsample_cfg, rngs=rngs)
            for _ in range(cfg.n_scaling)
        ]
        self.downsample = nnx.List(downsample_layers)

        # Residual blocks
        res_blocks: list[nnx.Module] = []
        for _ in range(cfg.n_res_blocks):
            res_blocks.append(
                ResidualBlock(
                    cfg.conv_channel,
                    cfg.conv_kernel,
                    cfg.conv_activation,
                    cfg.norm,
                    cfg.norm_cfg,
                    cfg.dropout,
                    cfg.padding_mode,
                    rngs=rngs,
                ),
            )
            if cfg.attention is not None:
                res_blocks.append(
                    Attention2d(cfg.conv_channel, nhead=None, attn_cfg=cfg.attention, rngs=rngs),
                )
        self.res_blocks = nnx.List(res_blocks)

        # Post-residual conv
        conv_cfg = deepcopy(first_cfg)
        conv_cfg.kernel_size = cfg.conv_kernel
        conv_cfg.padding = cfg.conv_kernel // 2
        conv_cfg.scale_factor = 0
        self.conv2 = ConvNormActivation(cfg.conv_channel, cfg.conv_channel, conv_cfg, rngs=rngs)

        # Final conv
        self.conv3 = ConvNormActivation(cfg.conv_channel, cfg.conv_channel, conv_cfg, rngs=rngs)
        self.last_channel = cfg.conv_channel

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, C) in NHWC format.

        Returns
        -------
        jax.Array
            Downsampled output.
        """
        out = self.conv1(x)
        for layer in self.downsample:
            out = layer(out)
        out1 = out
        for layer in self.res_blocks:
            out = layer(out)
        out2 = self.conv2(out)
        out = out1 + out2
        return self.conv3(out)

    @property
    def conved_shape(self) -> tuple[int, int]:
        """Get the spatial shape after downsampling."""
        scaling = self.cfg.scale_factor**self.cfg.n_scaling
        return (
            self.obs_shape[0] // scaling,
            self.obs_shape[1] // scaling,
        )

    @property
    def conved_size(self) -> int:
        """Get the flattened size after downsampling."""
        return self.last_channel * int(np.prod(self.conved_shape))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
