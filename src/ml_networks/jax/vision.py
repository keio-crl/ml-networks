"""Vision モデルを扱うモジュール."""

from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
from flax import nnx

from ml_networks.config import (
    ConvConfig,
    ConvNetConfig,
    DecoderConfig,
    EncoderConfig,
    ResNetConfig,
    ViTConfig,
)
from ml_networks.jax.layers import (
    Attention2d,
    ConvNormActivation,
    ConvTransposeNormActivation,
    Identity,
    PatchEmbed,
    ResidualBlock,
    SpatialSoftmax,
)
from ml_networks.utils import conv_out_shape


class Encoder(nnx.Module):
    """
    Image encoder module (NHWC format).

    Parameters
    ----------
    obs_shape : tuple[int, int, int]
        Observation shape in (H, W, C) format.
    cfg : EncoderConfig
        Encoder configuration.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        cfg: EncoderConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = cfg
        self.obs_shape = obs_shape
        in_channels = obs_shape[2]  # NHWC: C is last

        layers: list[nnx.Module] = []
        spatial_shape: tuple[int, int] = (obs_shape[0], obs_shape[1])

        for _i, (n_channels, kernel_size, stride) in enumerate(
            zip(cfg.channels, cfg.kernel_sizes, cfg.strides, strict=False),
        ):
            layers.append(
                ConvNormActivation(
                    in_channels,
                    n_channels,
                    ConvConfig(
                        activation=cfg.activation,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=(kernel_size - 1) // 2,
                        dilation=1,
                        groups=1,
                        bias=True,
                        dropout=0.0,
                        norm=cfg.norm,
                        norm_cfg=cfg.norm_cfg,
                    ),
                    rngs=rngs,
                ),
            )
            spatial_shape = conv_out_shape(
                spatial_shape,
                padding=(kernel_size - 1) // 2,
                kernel_size=kernel_size,
                stride=stride,
            )
            in_channels = n_channels

        self.conv_layers = nnx.List(layers)
        self.output_spatial_shape = spatial_shape
        self.output_channels = in_channels
        self.output_dim = in_channels * spatial_shape[0] * spatial_shape[1]

        if cfg.use_spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(cfg.spatial_softmax_cfg)
            self.output_dim = self.output_channels * 2
        else:
            self.spatial_softmax = None

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
            Encoded tensor. Shape depends on spatial_softmax setting.
        """
        for layer in self.conv_layers:
            x = layer(x)
        if self.spatial_softmax is not None:
            return self.spatial_softmax(x)
        return x.reshape(x.shape[0], -1)


class Decoder(nnx.Module):
    """
    Image decoder module (NHWC format).

    Parameters
    ----------
    obs_shape : tuple[int, int, int]
        Observation shape in (H, W, C) format.
    cfg : DecoderConfig
        Decoder configuration.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        cfg: DecoderConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = cfg
        self.obs_shape = obs_shape
        out_channels = obs_shape[2]  # NHWC: C is last

        channels = list(cfg.channels)
        kernel_sizes = list(cfg.kernel_sizes)
        strides = list(cfg.strides)

        layers: list[nnx.Module] = []
        for i in range(len(channels) - 1):
            conv_cfg = ConvConfig(
                activation=cfg.activation,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=(kernel_sizes[i] - 1) // 2,
                output_padding=strides[i] - 1,
                dilation=1,
                groups=1,
                bias=True,
                dropout=0.0,
                norm=cfg.norm,
                norm_cfg=cfg.norm_cfg,
            )
            layers.append(
                ConvTransposeNormActivation(
                    channels[i],
                    channels[i + 1],
                    conv_cfg,
                    rngs=rngs,
                ),
            )

        # Final layer with Identity activation
        final_cfg = ConvConfig(
            activation="Identity",
            kernel_size=kernel_sizes[-1],
            stride=strides[-1],
            padding=(kernel_sizes[-1] - 1) // 2,
            output_padding=strides[-1] - 1,
            dilation=1,
            groups=1,
            bias=True,
            dropout=0.0,
            norm="none",
        )
        layers.append(
            ConvTransposeNormActivation(
                channels[-1],
                out_channels,
                final_cfg,
                rngs=rngs,
            ),
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
            Decoded tensor of shape (B, H', W', C) in NHWC format.
        """
        for layer in self.conv_layers:
            x = layer(x)
        return x


class ViT(nnx.Module):
    """
    Vision Transformer (NHWC format).

    Parameters
    ----------
    obs_shape : tuple[int, int, int]
        Observation shape in (H, W, C) format.
    cfg : ViTConfig
        ViT configuration.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        cfg: ViTConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = cfg
        self.obs_shape = obs_shape  # (H, W, C)

        self.patch_embed = PatchEmbed(cfg.embed_dim, cfg.patch_size, obs_shape, rngs=rngs)
        num_patches = self.patch_embed.patch_num

        # CLS token
        self.cls_token = nnx.Param(jax.random.normal(rngs(), (1, 1, cfg.embed_dim)) * 0.02)

        # Position embedding
        self.pos_embed = nnx.Param(jax.random.normal(rngs(), (1, num_patches + 1, cfg.embed_dim)) * 0.02)

        # Transformer blocks (simple self-attention blocks)
        blocks = [_ViTBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, rngs=rngs) for _ in range(cfg.depth)]
        self.blocks = nnx.List(blocks)

        self.norm = nnx.LayerNorm(num_features=cfg.embed_dim, rngs=rngs)
        self.output_dim = cfg.embed_dim

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
            CLS token output of shape (B, embed_dim).
        """
        # (B, H, W, C) -> (B, Np, D)
        x = self.patch_embed(x)
        b = x.shape[0]

        # Prepend CLS token
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (b, 1, self.cfg.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # Add position embedding
        x = x + self.pos_embed.value

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token


class _ViTBlock(nnx.Module):
    """Single ViT transformer block (pre-norm)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
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
        mlp_hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, mlp_hidden, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_hidden, dim, rngs=rngs)

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

        layers: list[nnx.Module] = []
        attn_layers: list[nnx.Module] = []
        spatial_shape: tuple[int, int] = (obs_shape[0], obs_shape[1])

        for _i, (ch, ks, st) in enumerate(
            zip(
                cfg.channels,
                cfg.conv_cfg.kernel_size
                if isinstance(cfg.conv_cfg.kernel_size, list)
                else [cfg.conv_cfg.kernel_size] * len(cfg.channels),
                cfg.conv_cfg.stride
                if isinstance(cfg.conv_cfg.stride, list)
                else [cfg.conv_cfg.stride] * len(cfg.channels),
                strict=False,
            ),
        ):
            conv_cfg_i = deepcopy(cfg.conv_cfg)
            conv_cfg_i.kernel_size = ks
            conv_cfg_i.stride = st
            conv_cfg_i.padding = (ks - 1) // 2
            layers.append(ConvNormActivation(in_channels, ch, conv_cfg_i, rngs=rngs))

            if cfg.has_attn and cfg.nhead is not None:
                attn_layers.append(Attention2d(ch, cfg.nhead, rngs=rngs))
            else:
                attn_layers.append(Identity())

            spatial_shape = conv_out_shape(
                spatial_shape,
                padding=(ks - 1) // 2,
                kernel_size=ks,
                stride=st,
            )
            in_channels = ch

        self.conv_layers = nnx.List(layers)
        self.attn_layers = nnx.List(attn_layers)
        self.output_spatial_shape = spatial_shape
        self.output_channels = in_channels
        self.output_dim = in_channels * spatial_shape[0] * spatial_shape[1]

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


class ConvTranspose(nnx.Module):
    """
    Transposed convolutional network (NHWC format).

    Parameters
    ----------
    obs_shape : tuple[int, int, int]
        Output observation shape in (H, W, C) format.
    cfg : ConvNetConfig
        Configuration (channels are in reverse).
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
        out_channels = obs_shape[2]  # NHWC

        channels = list(cfg.channels)

        layers: list[nnx.Module] = []
        for i in range(len(channels) - 1):
            conv_cfg_i = deepcopy(cfg.conv_cfg)
            layers.append(
                ConvTransposeNormActivation(channels[i], channels[i + 1], conv_cfg_i, rngs=rngs),
            )

        # Final layer
        final_cfg = deepcopy(cfg.conv_cfg)
        final_cfg.activation = "Identity"
        final_cfg.norm = "none"
        layers.append(
            ConvTransposeNormActivation(channels[-1], out_channels, final_cfg, rngs=rngs),
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
        for layer in self.conv_layers:
            x = layer(x)
        return x


class ResNetPixShuffle(nnx.Module):
    """
    ResNet with PixelShuffle upsampling (NHWC format).

    Parameters
    ----------
    obs_shape : tuple[int, int, int]
        Output observation shape in (H, W, C) format.
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
        out_channels = obs_shape[2]

        channels = list(cfg.channels)

        layers: list[nnx.Module] = []
        for i in range(len(channels) - 1):
            conv_cfg_i = deepcopy(cfg.conv_cfg)
            conv_cfg_i.scale_factor = cfg.scale_factor
            layers.extend((
                ConvNormActivation(channels[i], channels[i + 1], conv_cfg_i, rngs=rngs),
                ResidualBlock(
                    channels[i + 1],
                    kernel_size=cfg.conv_cfg.kernel_size,
                    activation=cfg.conv_cfg.activation,
                    norm=cfg.conv_cfg.norm,
                    norm_cfg=cfg.conv_cfg.norm_cfg or None,
                    dropout=cfg.conv_cfg.dropout,
                    rngs=rngs,
                ),
            ))

        # Final conv to output channels
        final_cfg = ConvConfig(
            activation="Identity",
            kernel_size=3,
            stride=1,
            padding=1,
            norm="none",
        )
        layers.append(ConvNormActivation(channels[-1], out_channels, final_cfg, rngs=rngs))

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
            Upsampled output.
        """
        for layer in self.conv_layers:
            x = layer(x)
        return x


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
        in_channels = obs_shape[2]

        channels = list(cfg.channels)

        layers: list[nnx.Module] = []
        # Initial conv from input channels
        init_cfg = ConvConfig(
            activation=cfg.conv_cfg.activation,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=cfg.conv_cfg.norm,
            norm_cfg=cfg.conv_cfg.norm_cfg,
        )
        layers.append(ConvNormActivation(in_channels, channels[0], init_cfg, rngs=rngs))

        for i in range(len(channels) - 1):
            layers.append(
                ResidualBlock(
                    channels[i],
                    kernel_size=cfg.conv_cfg.kernel_size,
                    activation=cfg.conv_cfg.activation,
                    norm=cfg.conv_cfg.norm,
                    norm_cfg=cfg.conv_cfg.norm_cfg or None,
                    dropout=cfg.conv_cfg.dropout,
                    rngs=rngs,
                ),
            )
            conv_cfg_i = deepcopy(cfg.conv_cfg)
            conv_cfg_i.scale_factor = -cfg.scale_factor
            layers.append(
                ConvNormActivation(channels[i], channels[i + 1], conv_cfg_i, rngs=rngs),
            )

        self.conv_layers = nnx.List(layers)
        self.output_channels = channels[-1]

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
        for layer in self.conv_layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
