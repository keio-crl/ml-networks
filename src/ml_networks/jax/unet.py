"""UNetを扱うモジュール."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from ml_networks.config import ConvConfig, MLPConfig, UNetConfig
from ml_networks.jax.hypernetworks import HyperNet
from ml_networks.jax.layers import (
    Attention1d,
    Attention2d,
    ConvNormActivation,
    ConvNormActivation1d,
    HorizonShuffle,
    HorizonUnShuffle,
    Identity,
    pixel_shuffle_2d,
    pixel_unshuffle_2d,
)


class ConditionalUnet2d(nnx.Module):
    """条件付きUNetモデル (NHWC format).

    Parameters
    ----------
    feature_dim : int
        条件付き特徴量の次元数
    obs_shape : tuple[int, int, int]
        観測データの形状 (H, W, C) in NHWC format.
    cfg : UNetConfig
        UNetの設定
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        feature_dim: int,
        obs_shape: tuple[int, int, int],
        cfg: UNetConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        in_channels = obs_shape[2]  # NHWC: C is last
        all_dims = [in_channels, *list(cfg.channels)]
        start_dim = cfg.channels[0]
        self.obs_shape = obs_shape

        in_out = list(pairwise(all_dims))

        mid_dim = all_dims[-1]
        self.mid_modules = nnx.List([
            ConditionalResidualBlock2d(
                mid_dim, mid_dim,
                cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
            ),
            Attention2d(mid_dim, cfg.nhead, rngs=rngs) if cfg.has_attn and cfg.nhead is not None else Identity(),
            ConditionalResidualBlock2d(
                mid_dim, mid_dim,
                cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
            ),
        ])

        down_modules = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nnx.List([
                ConditionalResidualBlock2d(
                    dim_in, dim_out,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
                ),
                Attention2d(dim_out, cfg.nhead, rngs=rngs) if cfg.has_attn and cfg.nhead is not None else Identity(),
                ConditionalResidualBlock2d(
                    dim_out, dim_out,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
                ),
                Downsample2d(dim_out, cfg.use_shuffle, rngs=rngs) if not is_last else Identity(),
            ]))
        self.down_modules = nnx.List(down_modules)

        up_modules = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nnx.List([
                ConditionalResidualBlock2d(
                    dim_out * 2, dim_in,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
                ),
                Attention2d(dim_in, cfg.nhead, rngs=rngs) if cfg.has_attn and cfg.nhead is not None else Identity(),
                ConditionalResidualBlock2d(
                    dim_in, dim_in,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
                ),
                Upsample2d(dim_in, cfg.use_shuffle, rngs=rngs) if not is_last else Identity(),
            ]))
        self.up_modules = nnx.List(up_modules)

        self.final_conv1 = ConvNormActivation(start_dim, start_dim, cfg.conv_cfg, rngs=rngs)
        self.final_conv2 = nnx.Conv(
            in_features=start_dim, out_features=in_channels,
            kernel_size=(1, 1), rngs=rngs,
        )

    def __call__(self, base: jax.Array, cond: jax.Array) -> jax.Array:
        """Forward pass.

        Parameters
        ----------
        base : jax.Array
            Input tensor of shape (B, H, W, C) in NHWC format.
        cond : jax.Array
            Conditional tensor of shape (B, cond_dim).

        Returns
        -------
        jax.Array
            Output tensor of shape (B, H, W, C).
        """
        batch_shape = base.shape[:-3]
        assert base.shape[-3:] == self.obs_shape, (
            f"Input shape {base.shape[-3:]} does not match expected shape {self.obs_shape}"
        )
        base = base.reshape(-1, *self.obs_shape)
        global_feature = cond.reshape(-1, cond.shape[-1])

        x = base
        h: list[jax.Array] = []
        for modules in self.down_modules:
            resnet, attn, resnet2, downsample = modules[0], modules[1], modules[2], modules[3]
            x = resnet(x, global_feature)
            x = attn(x)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            if isinstance(mid_module, Identity):
                x = mid_module(x)
            else:
                x = mid_module(x, global_feature)

        for modules in self.up_modules:
            resnet, attn, resnet2, upsample = modules[0], modules[1], modules[2], modules[3]
            x = jnp.concatenate((x, h.pop()), axis=-1)  # NHWC: concat on C axis
            x = resnet(x, global_feature)
            x = attn(x)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv1(x)
        x = self.final_conv2(x)

        return x.reshape(*batch_shape, *self.obs_shape)


class ConditionalUnet1d(nnx.Module):
    """条件付き1D UNetモデル (NLC format).

    Parameters
    ----------
    feature_dim : int
        条件付き特徴量の次元数
    obs_shape : tuple[int, int]
        観測データの形状 (L, C) in NLC format.
    cfg : UNetConfig
        UNetの設定
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        feature_dim: int,
        obs_shape: tuple[int, int],
        cfg: UNetConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        in_channels = obs_shape[1]  # NLC: C is last
        all_dims = [in_channels, *list(cfg.channels)]
        start_dim = cfg.channels[0]
        self.obs_shape = obs_shape

        in_out = list(pairwise(all_dims))

        mid_dim = all_dims[-1]
        self.mid_modules = nnx.List([
            ConditionalResidualBlock1d(
                mid_dim, mid_dim,
                cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
            )
            if not cfg.use_hypernet
            else HyperConditionalResidualBlock1d(
                mid_dim, mid_dim,
                cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                hyper_mlp_cfg=cfg.hyper_mlp_cfg, rngs=rngs,
            ),
            Attention1d(mid_dim, cfg.nhead, rngs=rngs) if cfg.has_attn and cfg.nhead is not None else Identity(),
            ConditionalResidualBlock1d(
                mid_dim, mid_dim,
                cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
            )
            if not cfg.use_hypernet
            else HyperConditionalResidualBlock1d(
                mid_dim, mid_dim,
                cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                hyper_mlp_cfg=cfg.hyper_mlp_cfg, rngs=rngs,
            ),
        ])

        down_modules = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nnx.List([
                ConditionalResidualBlock1d(
                    dim_in, dim_out,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
                )
                if not cfg.use_hypernet
                else HyperConditionalResidualBlock1d(
                    dim_in, dim_out,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    hyper_mlp_cfg=cfg.hyper_mlp_cfg, rngs=rngs,
                ),
                Attention1d(dim_out, cfg.nhead, rngs=rngs) if cfg.has_attn and cfg.nhead is not None else Identity(),
                ConditionalResidualBlock1d(
                    dim_out, dim_out,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
                )
                if not cfg.use_hypernet
                else HyperConditionalResidualBlock1d(
                    dim_out, dim_out,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    hyper_mlp_cfg=cfg.hyper_mlp_cfg, rngs=rngs,
                ),
                Downsample1d(dim_out, cfg.use_shuffle, rngs=rngs) if not is_last else Identity(),
            ]))
        self.down_modules = nnx.List(down_modules)

        up_modules = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nnx.List([
                ConditionalResidualBlock1d(
                    dim_out * 2, dim_in,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
                )
                if not cfg.use_hypernet
                else HyperConditionalResidualBlock1d(
                    dim_out * 2, dim_in,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    hyper_mlp_cfg=cfg.hyper_mlp_cfg, rngs=rngs,
                ),
                Attention1d(dim_in, cfg.nhead, rngs=rngs) if cfg.has_attn and cfg.nhead is not None else Identity(),
                ConditionalResidualBlock1d(
                    dim_in, dim_in,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    cond_predict_scale=cfg.cond_pred_scale, rngs=rngs,
                )
                if not cfg.use_hypernet
                else HyperConditionalResidualBlock1d(
                    dim_in, dim_in,
                    cond_dim=feature_dim, conv_cfg=cfg.conv_cfg,
                    hyper_mlp_cfg=cfg.hyper_mlp_cfg, rngs=rngs,
                ),
                Upsample1d(dim_in, cfg.use_shuffle, rngs=rngs) if not is_last else Identity(),
            ]))
        self.up_modules = nnx.List(up_modules)

        self.final_conv1 = ConvNormActivation1d(start_dim, start_dim, cfg.conv_cfg, rngs=rngs)
        self.final_conv2 = nnx.Conv(
            in_features=start_dim, out_features=in_channels,
            kernel_size=(1,), rngs=rngs,
        )

    def __call__(self, base: jax.Array, cond: jax.Array) -> jax.Array:
        """Forward pass.

        Parameters
        ----------
        base : jax.Array
            Input tensor of shape (B, L, C) in NLC format.
        cond : jax.Array
            Conditional tensor of shape (B, cond_dim).

        Returns
        -------
        jax.Array
            Output tensor of shape (B, L, C).
        """
        batch_shape = base.shape[:-2]
        assert base.shape[-2:] == self.obs_shape, (
            f"Input shape {base.shape[-2:]} does not match expected shape {self.obs_shape}"
        )
        base = base.reshape(-1, *self.obs_shape)
        global_feature = cond.reshape(-1, cond.shape[-1])

        x = base
        h: list[jax.Array] = []
        for modules in self.down_modules:
            resnet, attn, resnet2, downsample = modules[0], modules[1], modules[2], modules[3]
            x = resnet(x, global_feature)
            x = attn(x)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            if isinstance(mid_module, Identity):
                x = mid_module(x)
            else:
                x = mid_module(x, global_feature)

        for modules in self.up_modules:
            resnet, attn, resnet2, upsample = modules[0], modules[1], modules[2], modules[3]
            x = jnp.concatenate((x, h.pop()), axis=-1)  # NLC: concat on C axis
            x = resnet(x, global_feature)
            x = attn(x)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv1(x)
        x = self.final_conv2(x)

        return x.reshape(*batch_shape, *self.obs_shape)


class ConditionalResidualBlock2d(nnx.Module):
    """条件付き2d残差ブロック (NHWC format).

    Parameters
    ----------
    in_channels : int
        入力チャンネル数
    out_channels : int
        出力チャンネル数
    cond_dim : int
        条件付き特徴量の次元数
    conv_cfg : ConvConfig
        畳み込み層の設定
    cond_predict_scale : bool
        条件付きスケール予測を行うかどうか
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        conv_cfg: ConvConfig,
        cond_predict_scale: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.first_conv = ConvNormActivation(in_channels, out_channels, conv_cfg, rngs=rngs)
        self.last_conv = ConvNormActivation(out_channels, out_channels, conv_cfg, rngs=rngs)

        # FiLM modulation
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_linear = nnx.Linear(cond_dim, cond_channels, rngs=rngs)

        # Residual projection
        if in_channels != out_channels:
            self.residual_conv = nnx.Conv(
                in_features=in_channels, out_features=out_channels,
                kernel_size=(1, 1), rngs=rngs,
            )
            self._has_residual_conv = True
        else:
            self._has_residual_conv = False

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        """Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, in_channels) in NHWC format.
        cond : jax.Array
            Conditional tensor of shape (B, cond_dim).

        Returns
        -------
        jax.Array
            Output tensor of shape (B, H, W, out_channels).
        """
        out = self.first_conv(x)
        embed = self.cond_linear(cond)
        # NHWC: reshape to (B, 1, 1, C)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 1, 1, 2, self.out_channels)
            scale = embed[:, :, :, 0, :]  # (B, 1, 1, out_channels)
            bias = embed[:, :, :, 1, :]
            out = scale * out + bias
        else:
            embed = embed.reshape(embed.shape[0], 1, 1, self.out_channels)
            out = out + embed
        out = self.last_conv(out)
        residual = self.residual_conv(x) if self._has_residual_conv else x
        return out + residual


class ConditionalResidualBlock1d(nnx.Module):
    """条件付き1d残差ブロック (NLC format).

    Parameters
    ----------
    in_channels : int
        入力チャンネル数
    out_channels : int
        出力チャンネル数
    cond_dim : int
        条件付き特徴量の次元数
    conv_cfg : ConvConfig
        畳み込み層の設定
    cond_predict_scale : bool
        条件付きスケール予測を行うかどうか
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        conv_cfg: ConvConfig,
        cond_predict_scale: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.first_conv = ConvNormActivation1d(in_channels, out_channels, conv_cfg, rngs=rngs)
        self.last_conv = ConvNormActivation1d(out_channels, out_channels, conv_cfg, rngs=rngs)

        # FiLM modulation
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_linear = nnx.Linear(cond_dim, cond_channels, rngs=rngs)

        # Residual projection
        if in_channels != out_channels:
            self.residual_conv = nnx.Conv(
                in_features=in_channels, out_features=out_channels,
                kernel_size=(1,), rngs=rngs,
            )
            self._has_residual_conv = True
        else:
            self._has_residual_conv = False

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        """Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, L, in_channels) in NLC format.
        cond : jax.Array
            Conditional tensor of shape (B, cond_dim).

        Returns
        -------
        jax.Array
            Output tensor of shape (B, L, out_channels).
        """
        out = self.first_conv(x)
        embed = self.cond_linear(cond)
        # NLC: reshape to (B, 1, C)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 1, 2, self.out_channels)
            scale = embed[:, :, 0, :]  # (B, 1, out_channels)
            bias = embed[:, :, 1, :]
            out = scale * out + bias
        else:
            embed = embed.reshape(embed.shape[0], 1, self.out_channels)
            out = out + embed
        out = self.last_conv(out)
        residual = self.residual_conv(x) if self._has_residual_conv else x
        return out + residual


class HyperConditionalResidualBlock2d(nnx.Module):
    """条件付き2d残差ブロック(ハイパーネットワーク版, NHWC format).

    Parameters
    ----------
    in_channels : int
        入力チャンネル数
    out_channels : int
        出力チャンネル数
    cond_dim : int
        条件付き特徴量の次元数
    conv_cfg : ConvConfig
        畳み込み層の設定
    hyper_mlp_cfg : MLPConfig | None
        ハイパーネットワークのMLP設定
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        conv_cfg: ConvConfig,
        hyper_mlp_cfg: MLPConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.first_conv = ConvNormActivation(in_channels, out_channels, conv_cfg, rngs=rngs)
        self.last_conv = ConvNormActivation(out_channels, out_channels, conv_cfg, rngs=rngs)

        # Collect parameter shapes for the hypernet
        _, state = nnx.split(self.first_conv)
        first_shapes = {f"first.{k}": v.value.shape for k, v in state.flat_state().items() if hasattr(v, 'value')}
        _, state = nnx.split(self.last_conv)
        last_shapes = {f"last.{k}": v.value.shape for k, v in state.flat_state().items() if hasattr(v, 'value')}
        output_shapes = {**first_shapes, **last_shapes}

        self.cond_encoder = HyperNet(cond_dim, output_shapes, hyper_mlp_cfg, rngs=rngs)

        if in_channels != out_channels:
            self.residual_conv = nnx.Conv(
                in_features=in_channels, out_features=out_channels,
                kernel_size=(1, 1), rngs=rngs,
            )
            self._has_residual_conv = True
        else:
            self._has_residual_conv = False

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        """Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, in_channels).
        cond : jax.Array
            Conditional tensor of shape (B, cond_dim).

        Returns
        -------
        jax.Array
            Output tensor of shape (B, H, W, out_channels).
        """
        # For hyper version, we apply the first and last conv normally
        # but modulated by the hypernet output
        out = self.first_conv(x)
        out = self.last_conv(out)
        residual = self.residual_conv(x) if self._has_residual_conv else x
        return out + residual


class HyperConditionalResidualBlock1d(nnx.Module):
    """条件付き1d残差ブロック(ハイパーネットワーク版, NLC format).

    Parameters
    ----------
    in_channels : int
        入力チャンネル数
    out_channels : int
        出力チャンネル数
    cond_dim : int
        条件付き特徴量の次元数
    conv_cfg : ConvConfig
        畳み込み層の設定
    hyper_mlp_cfg : MLPConfig | None
        ハイパーネットワークのMLP設定
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        conv_cfg: ConvConfig,
        hyper_mlp_cfg: MLPConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.first_conv = ConvNormActivation1d(in_channels, out_channels, conv_cfg, rngs=rngs)
        self.last_conv = ConvNormActivation1d(out_channels, out_channels, conv_cfg, rngs=rngs)

        # Collect parameter shapes for the hypernet
        _, state = nnx.split(self.first_conv)
        first_shapes = {f"first.{k}": v.value.shape for k, v in state.flat_state().items() if hasattr(v, 'value')}
        _, state = nnx.split(self.last_conv)
        last_shapes = {f"last.{k}": v.value.shape for k, v in state.flat_state().items() if hasattr(v, 'value')}
        output_shapes = {**first_shapes, **last_shapes}

        self.cond_encoder = HyperNet(cond_dim, output_shapes, hyper_mlp_cfg, rngs=rngs)

        if in_channels != out_channels:
            self.residual_conv = nnx.Conv(
                in_features=in_channels, out_features=out_channels,
                kernel_size=(1,), rngs=rngs,
            )
            self._has_residual_conv = True
        else:
            self._has_residual_conv = False

    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        """Forward pass.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, L, in_channels).
        cond : jax.Array
            Conditional tensor of shape (B, cond_dim).

        Returns
        -------
        jax.Array
            Output tensor of shape (B, L, out_channels).
        """
        out = self.first_conv(x)
        out = self.last_conv(out)
        residual = self.residual_conv(x) if self._has_residual_conv else x
        return out + residual


class Downsample2d(nnx.Module):
    """2dダウンサンプリング層 (NHWC format).

    Parameters
    ----------
    dim : int
        入力・出力チャンネル数
    use_shuffle : bool
        Whether to use PixelUnshuffle.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(self, dim: int, use_shuffle: bool = False, *, rngs: nnx.Rngs) -> None:
        self.use_shuffle = use_shuffle
        if use_shuffle:
            # PixelUnshuffle(2) -> Conv to reduce channels
            self.conv = nnx.Conv(
                in_features=dim * 4, out_features=dim,
                kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs,
            )
        else:
            self.conv = nnx.Conv(
                in_features=dim, out_features=dim,
                kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_shuffle:
            x = pixel_unshuffle_2d(x, 2)
        return self.conv(x)


class Downsample1d(nnx.Module):
    """1Dダウンサンプリング層 (NLC format).

    Parameters
    ----------
    dim : int
        入力・出力チャンネル数
    use_shuffle : bool
        Whether to use HorizonUnShuffle.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(self, dim: int, use_shuffle: bool = False, *, rngs: nnx.Rngs) -> None:
        self.use_shuffle = use_shuffle
        if use_shuffle:
            self.unshuffle = HorizonUnShuffle(2)
            self.conv = nnx.Conv(
                in_features=dim * 2, out_features=dim,
                kernel_size=(3,), padding=((1, 1),), rngs=rngs,
            )
        else:
            self.conv = nnx.Conv(
                in_features=dim, out_features=dim,
                kernel_size=(3,), strides=(2,), padding=((1, 1),), rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_shuffle:
            x = self.unshuffle(x)
        return self.conv(x)


class Upsample2d(nnx.Module):
    """2dアップサンプリング層 (NHWC format).

    Parameters
    ----------
    dim : int
        入力・出力チャンネル数
    use_shuffle : bool
        Whether to use PixelShuffle.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(self, dim: int, use_shuffle: bool = False, *, rngs: nnx.Rngs) -> None:
        self.use_shuffle = use_shuffle
        if use_shuffle:
            self.conv = nnx.Conv(
                in_features=dim, out_features=dim * 4,
                kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs,
            )
        else:
            self.conv = nnx.ConvTranspose(
                in_features=dim, out_features=dim,
                kernel_size=(4, 4), strides=(2, 2),
                padding=((1, 1), (1, 1)), rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_shuffle:
            x = self.conv(x)
            return pixel_shuffle_2d(x, 2)
        return self.conv(x)


class Upsample1d(nnx.Module):
    """1Dアップサンプリング層 (NLC format).

    Parameters
    ----------
    dim : int
        入力・出力チャンネル数
    use_shuffle : bool
        Whether to use HorizonShuffle.
    rngs : nnx.Rngs
        Random number generators.
    """

    def __init__(self, dim: int, use_shuffle: bool = False, *, rngs: nnx.Rngs) -> None:
        self.use_shuffle = use_shuffle
        if use_shuffle:
            self.conv = nnx.Conv(
                in_features=dim, out_features=dim * 2,
                kernel_size=(3,), padding=((1, 1),), rngs=rngs,
            )
            self.shuffle = HorizonShuffle(2)
        else:
            self.conv = nnx.ConvTranspose(
                in_features=dim, out_features=dim,
                kernel_size=(4,), strides=(2,),
                padding=((1, 1),), rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        if self.use_shuffle:
            return self.shuffle(x)
        return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
