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
from ml_networks.jax.activations import Activation
from ml_networks.jax.layers import (
    Attention2d,
    ConvNormActivation,
    ConvTransposeNormActivation,
    Identity,
    LinearNormActivation,
    MLPLayer,
    PatchEmbed,
    ResidualBlock,
    SpatialSoftmax,
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
        self._is_vit = isinstance(backbone_cfg, ViTConfig)

        self.encoder: nnx.Module
        if isinstance(backbone_cfg, ViTConfig):
            self.encoder = ViT(obs_shape, backbone_cfg, rngs=rngs)
            d_model = backbone_cfg.transformer_cfg.d_model
            self.last_channel: int = d_model
            self.conved_size: int = d_model
            self.conved_shape: tuple[int, ...] = (1, 1)
            assert isinstance(feature_dim, int), "feature_dim must be int when using ViTConfig backbone"
            self.fc: nnx.Module
            if isinstance(fc_cfg, MLPConfig):
                self.fc = MLPLayer(d_model, feature_dim, fc_cfg, rngs=rngs)
            elif isinstance(fc_cfg, LinearConfig):
                self.fc = LinearNormActivation(d_model, feature_dim, fc_cfg, rngs=rngs)
            elif fc_cfg is None:
                assert d_model == feature_dim, (
                    f"feature_dim must equal transformer d_model when fc_cfg is None, got {feature_dim} vs {d_model}"
                )
                self.fc = Identity()
            else:
                msg = f"fc_cfg type {type(fc_cfg)} is not supported with ViTConfig backbone"
                raise NotImplementedError(msg)
            self._fc_cfg = fc_cfg
            return
        if isinstance(backbone_cfg, ConvNetConfig):
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
        if self._is_vit:
            # ViT encoder returns the CLS token (B, d_model); fc projects to feature_dim.
            x = self.fc(x)
            return x.reshape(*batch_shape, *x.shape[1:])
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
        self._is_vit = isinstance(backbone_cfg, ViTConfig)

        if isinstance(backbone_cfg, ViTConfig):
            d_model = backbone_cfg.transformer_cfg.d_model
            assert isinstance(feature_dim, int), "feature_dim must be int when using ViTConfig backbone"
            self.fc: nnx.Module
            if isinstance(fc_cfg, MLPConfig):
                self.fc = MLPLayer(feature_dim, d_model, fc_cfg, rngs=rngs)
            elif isinstance(fc_cfg, LinearConfig):
                self.fc = LinearNormActivation(feature_dim, d_model, fc_cfg, rngs=rngs)
            elif fc_cfg is None:
                assert feature_dim == d_model, (
                    f"feature_dim must equal transformer d_model when fc_cfg is None, got {feature_dim} vs {d_model}"
                )
                self.fc = Identity()
            else:
                msg = f"fc_cfg type {type(fc_cfg)} is not supported with ViTConfig backbone"
                raise NotImplementedError(msg)
            self.input_shape = (d_model,)  # type: ignore[assignment]
            self.has_fc = True
            self.decoder: nnx.Module = ViT(
                in_shape=(d_model,),
                cfg=backbone_cfg,
                obs_shape=obs_shape,
                rngs=rngs,
            )
            return

        self.input_shape: tuple[int, int, int]
        if isinstance(backbone_cfg, ConvNetConfig):
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
        if isinstance(fc_cfg, MLPConfig):
            assert isinstance(feature_dim, int)
            self.fc = MLPLayer(feature_dim, input_size, fc_cfg, rngs=rngs)
        elif isinstance(fc_cfg, LinearConfig):
            assert isinstance(feature_dim, int)
            self.fc = LinearNormActivation(feature_dim, input_size, fc_cfg, rngs=rngs)
        else:
            self.fc = Identity()

        if isinstance(backbone_cfg, ConvNetConfig):
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
        if self._is_vit:
            batch_shape, data_shape = x.shape[:-1], x.shape[-1:]
            x = x.reshape(-1, *data_shape)
            x = self.fc(x)  # (B, d_model)
            x = self.decoder(x)  # (B, H, W, C)
            return x.reshape(*batch_shape, *self.obs_shape)
        if self.has_fc:
            batch_shape, data_shape = x.shape[:-1], x.shape[-1:]
        else:
            batch_shape, data_shape = x.shape[:-3], x.shape[-3:]
        x = x.reshape(-1, *data_shape)
        x = self.fc(x)
        x = x.reshape(-1, *self.input_shape)
        x = self.decoder(x)
        return x.reshape(*batch_shape, *self.obs_shape)


class _ViTEncoderBlock(nnx.Module):
    """ViT encoder block (pre-norm). Positional embedding is added to query and key only.

    Follows the DETR convention (https://github.com/gokul-pv/DetectionTransformer): at every
    layer the spatial positional embedding is added to the query and key tensors of the
    self-attention, while the value tensor is left unmodified.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float,
        activation: str,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=nhead,
            in_features=d_model,
            dropout_rate=dropout,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.linear1 = nnx.Linear(d_model, dim_ff, rngs=rngs)
        self.activation = Activation(activation)
        self.linear2 = nnx.Linear(dim_ff, d_model, rngs=rngs)
        self.dropout: nnx.Module = nnx.Dropout(rate=dropout, rngs=rngs) if dropout > 0 else Identity()

    def __call__(self, x: jax.Array, pos: jax.Array) -> jax.Array:
        h = self.norm1(x)
        q = h + pos
        k = h + pos
        v = h
        attn_out = self.attn(q, k, v)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + self.dropout(h)


class _ViTDecoderBlock(nnx.Module):
    """ViT decoder block: cross-attention from learnable queries to the projected CLS token.

    Performs (pre-norm) cross-attention where the queries are the learnable patch tokens and
    the key/value come from the projected CLS representation, followed by a residual MLP block.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float,
        activation: str,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.norm_q = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=nhead,
            in_features=d_model,
            dropout_rate=dropout,
            decode=False,
            rngs=rngs,
        )
        self.norm_mlp = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.linear1 = nnx.Linear(d_model, dim_ff, rngs=rngs)
        self.activation = Activation(activation)
        self.linear2 = nnx.Linear(dim_ff, d_model, rngs=rngs)
        self.dropout: nnx.Module = nnx.Dropout(rate=dropout, rngs=rngs) if dropout > 0 else Identity()

    def __call__(self, queries: jax.Array, memory: jax.Array) -> jax.Array:
        q = self.norm_q(queries)
        kv = self.norm_kv(memory)
        attn_out = self.cross_attn(q, kv, kv)
        queries = queries + self.dropout(attn_out)
        h = self.norm_mlp(queries)
        h = self.linear1(h)
        h = self.activation(h)
        h = self.linear2(h)
        return queries + self.dropout(h)


class ViT(nnx.Module):
    """
    Vision Transformer for Encoder and Decoder (NHWC format).

    The encoder mode (``obs_shape is None``) follows the DETR convention: a learnable per-patch
    positional embedding is added to the query and key tensors of every self-attention layer
    (rather than being added once at the input). When ``cfg.cls_token`` is True, a CLS token
    with its own learnable positional embedding is prepended and the forward pass returns the
    CLS token of shape ``(B, d_model)``.

    The decoder mode (``obs_shape is not None``) takes a CLS token of shape ``(B, d_model)`` or
    ``(B, 1, d_model)`` and reconstructs an image. The CLS token is projected to a hidden
    dimension and used as the key/value of cross-attention. A fixed set of
    ``P = (H // p) * (W // p)`` learnable query tokens interacts with this representation
    through several cross-attention layers with residual MLP blocks. Each query is then
    linearly projected to ``p * p * C`` pixels and rearranged into a ``(B, H, W, C)`` image.

    Parameters
    ----------
    in_shape : tuple[int, ...]
        Input shape. Encoder mode: ``(H, W, C)`` (NHWC). Decoder mode: ``(d_model,)`` — the CLS
        token is the input.
    cfg : ViTConfig
        ViT configuration.
    obs_shape : tuple[int, int, int] | None
        Output shape in (H, W, C) format. If ``None``, acts as encoder.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_shape: tuple[int, ...],
        cfg: ViTConfig,
        obs_shape: tuple[int, int, int] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.cfg = cfg
        self.in_shape = in_shape
        self.patch_size = cfg.patch_size
        self.transformer_cfg = cfg.transformer_cfg
        self.is_encoder = obs_shape is None
        self.obs_shape: tuple[int, int, int] = (
            obs_shape if obs_shape is not None else cast("tuple[int, int, int]", in_shape)
        )

        d_model = self.transformer_cfg.d_model
        self.d_model = d_model

        if self.is_encoder:
            self._build_encoder(rngs=rngs)
            self.last_channel = d_model
            self.out_patch_dim = d_model
        else:
            self._build_decoder(rngs=rngs)
            self.out_patch_dim = self.patch_size**2 * self.obs_shape[2]
            self.last_channel = self.out_patch_dim
        self.output_dim = self.last_channel

    def _build_encoder(self, *, rngs: nnx.Rngs) -> None:
        cfg = self.cfg
        t_cfg = self.transformer_cfg
        d_model = self.d_model
        assert len(self.in_shape) == 3, "Encoder mode requires in_shape=(H, W, C)"
        in_shape3 = cast("tuple[int, int, int]", self.in_shape)
        n_patches = self.get_n_patches(in_shape3)

        self.patch_embed = PatchEmbed(
            emb_dim=d_model,
            patch_size=self.patch_size,
            obs_shape=in_shape3,
            rngs=rngs,
        )
        # Learnable positional embedding added to query and key at every encoder layer.
        self._pos_emb = nnx.Param(jax.random.normal(rngs(), (1, n_patches, d_model)) * 0.02)

        if cfg.cls_token:
            self._cls_token = nnx.Param(jax.random.normal(rngs(), (1, 1, d_model)) * 0.02)
            self._cls_pos_emb = nnx.Param(jax.random.normal(rngs(), (1, 1, d_model)) * 0.02)

        self.encoder_blocks = [
            _ViTEncoderBlock(
                d_model=d_model,
                nhead=t_cfg.nhead,
                dim_ff=t_cfg.dim_ff,
                dropout=t_cfg.dropout,
                activation=t_cfg.hidden_activation,
                rngs=rngs,
            )
            for _ in range(t_cfg.n_layers)
        ]
        self.encoder_norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.n_patches = n_patches

    def _build_decoder(self, *, rngs: nnx.Rngs) -> None:
        t_cfg = self.transformer_cfg
        d_model = self.d_model
        n_patches = self.get_n_patches(self.obs_shape)

        # Learnable patch queries (1, P, d_model). One query per output patch.
        self._queries = nnx.Param(jax.random.normal(rngs(), (1, n_patches, d_model)) * 0.02)
        # Project the CLS token to the hidden dimension used as K/V in cross-attention.
        self.kv_norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.kv_proj = nnx.Linear(d_model, d_model, rngs=rngs)

        self.decoder_blocks = [
            _ViTDecoderBlock(
                d_model=d_model,
                nhead=t_cfg.nhead,
                dim_ff=t_cfg.dim_ff,
                dropout=t_cfg.dropout,
                activation=t_cfg.hidden_activation,
                rngs=rngs,
            )
            for _ in range(t_cfg.n_layers)
        ]
        self.decoder_norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        out_patch_dim = self.patch_size**2 * self.obs_shape[2]
        self.out_proj = nnx.Linear(d_model, out_patch_dim, rngs=rngs)
        self.n_patches = n_patches

    def __call__(self, x: jax.Array, *, return_cls_token: bool = False) -> jax.Array:
        """
        Forward pass.

        Parameters
        ----------
        x : jax.Array
            Encoder mode: image tensor of shape ``(B, H, W, C)`` in NHWC format.
            Decoder mode: CLS token of shape ``(B, d_model)`` or ``(B, 1, d_model)``.
        return_cls_token : bool
            Retained for backward compatibility; the encoder always returns the CLS token.

        Returns
        -------
        jax.Array
            Encoder mode: CLS token of shape ``(B, d_model)``.
            Decoder mode: reconstructed image of shape ``(B, H, W, C)``.
        """
        del return_cls_token
        if self.is_encoder:
            return self._forward_encoder(x)
        return self._forward_decoder(x)

    def _forward_encoder(self, x: jax.Array) -> jax.Array:
        x = self.patch_embed(x)  # (B, N, d_model)
        pos = jnp.broadcast_to(self._pos_emb.value, (x.shape[0], x.shape[1], x.shape[2]))
        if hasattr(self, "_cls_token"):
            cls_token = jnp.broadcast_to(self._cls_token.value, (x.shape[0], 1, x.shape[-1]))
            cls_pos = jnp.broadcast_to(self._cls_pos_emb.value, (x.shape[0], 1, x.shape[-1]))
            x = jnp.concatenate([cls_token, x], axis=1)
            pos = jnp.concatenate([cls_pos, pos], axis=1)
        for block in self.encoder_blocks:
            x = block(x, pos)
        x = self.encoder_norm(x)
        if hasattr(self, "_cls_token"):
            return x[:, 0]
        return x.mean(axis=1)

    def _forward_decoder(self, cls_token: jax.Array) -> jax.Array:
        if cls_token.ndim == 2:
            cls_token = cls_token[:, None, :]
        memory = self.kv_proj(self.kv_norm(cls_token))  # (B, 1, d_model)
        queries = jnp.broadcast_to(
            self._queries.value,
            (cls_token.shape[0], self.n_patches, self.d_model),
        )
        for block in self.decoder_blocks:
            queries = block(queries, memory)
        queries = self.decoder_norm(queries)
        patches = self.out_proj(queries)  # (B, P, p*p*C)
        return self.unpatchify(patches)

    def patchify(self, imgs: jax.Array) -> jax.Array:
        """Split images into patches (NHWC)."""
        p = self.patch_size
        return rearrange(imgs, "n (h p1) (w p2) c -> n (h w) (p1 p2 c)", p1=p, p2=p)

    def unpatchify(self, x: jax.Array) -> jax.Array:
        """Reconstruct images from patches (NHWC)."""
        p = self.patch_size
        h = self.obs_shape[0] // p
        w = self.obs_shape[1] // p
        assert h * w == x.shape[1], (
            f"{h * w} != {x.shape[1]}, please check the shape {x.shape} and obs_shape {self.obs_shape}"
        )
        return rearrange(x, "n (h w) (p1 p2 c) -> n (h p1) (w p2) c", h=h, w=w, p1=p, p2=p)

    @property
    def conved_size(self) -> int:
        """Get the CLS-token output size."""
        return self.d_model

    @property
    def conved_shape(self) -> tuple[int, int]:
        return (1, 1)

    def get_n_patches(self, obs_shape: tuple[int, int, int]) -> int:
        """Get number of patches for a given shape (NHWC: H, W, C)."""
        return (obs_shape[0] // self.patch_size) * (obs_shape[1] // self.patch_size)

    def get_patch_dim(self, obs_shape: tuple[int, int, int]) -> int:
        """Get patch dimension for a given shape (NHWC: H, W, C)."""
        return self.patch_size**2 * obs_shape[2]

    @staticmethod
    def get_input_shape(obs_shape: tuple[int, int, int], cfg: ViTConfig) -> tuple[int, ...]:
        """Input shape consumed by the ViT decoder: the CLS token has dimension ``d_model``."""
        del obs_shape
        return (cfg.transformer_cfg.d_model,)


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
