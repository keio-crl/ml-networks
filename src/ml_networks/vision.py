from typing import Union

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from mnetworks.layers import (PatchEmbed, TransformerLayer, 
        ConvTransposeNormActivation, ResidualBlock, 
        ConvNormActivation, PositionalEncoding,
        MLPLayer, LinearNormActivation, SpatialSoftmaxFlatten)
from mnetworks.config import ConvNetConfig, ViTConfig, ResNetConfig, MLPConfig, LinearConfig, SpatialSoftmaxConfig, ConvConfig
from mnetworks.utils import conv_out_shape, conv_transpose_in_shape
from einops import rearrange


class Encoder(pl.LightningModule):
    """ Encoder + reparameterization

    ただのEncoderよりモデルを軽く
    self.sequenceでは
        (C, W, H) -> (batch_size, W/2 * H/2 * 256)
        -> (batch_size, middle_layer_dim)
    self.fc_for_mu/ligvarで
        (batch_size, middle_layer_dim) -> (batch_size, latent_dim)
    になる

    """

    def __init__(
        self,
        feature_dim: Union[int, tuple[int, int, int]],
        obs_shape: tuple[int, int, int],
        backbone_cfg: Union[ViTConfig, ConvNetConfig, ResNetConfig],
        fc_cfg: Union[MLPConfig, LinearConfig, SpatialSoftmaxConfig] = None,
    ):
        super().__init__()

        self.obs_shape = obs_shape

        if isinstance(backbone_cfg, ViTConfig):
            self.encoder = ViT(obs_shape, backbone_cfg)
            self.encoder_type = "ViT"
        elif isinstance(backbone_cfg, ConvNetConfig):
            self.encoder = ConvNet(obs_shape, backbone_cfg)
            self.encoder_type = "CNN"
        elif isinstance(backbone_cfg, ResNetConfig):
            self.encoder = ResNetPixUnshuffle(obs_shape, backbone_cfg)
            self.encoder_type = "ResNet"
        else:
            raise NotImplementedError(
                f"{type(backbone_cfg)} is not implemented")

        self.feature_dim = feature_dim
        self.conved_size = self.encoder.conved_size
        self.conved_shape = self.encoder.conved_shape
        self.last_channel = self.encoder.last_channel

        if isinstance(feature_dim, int):
            assert fc_cfg is not None, "fc_cfg must be provided if feature_dim is provided"
        else:
            assert feature_dim == (self.encoder.last_channel, *self.encoder.conved_shape), f"{feature_shape} != {(self.encoder.last_channel, *self.encoder.conved_shape)}"
        if isinstance(fc_cfg, MLPConfig):
            self.fc = nn.Sequential(
                nn.Flatten(),
                MLPLayer(self.encoder.conved_size, feature_dim, fc_cfg)
            )
        elif isinstance(fc_cfg, LinearConfig):
            self.fc = nn.Sequential(
                nn.Flatten(),
                LinearNormActivation(self.encoder.conved_size, feature_dim, fc_cfg)
            )
        elif isinstance(fc_cfg, SpatialSoftmaxConfig):
            self.fc = nn.Sequential(
                SpatialSoftmaxFlatten(fc_cfg)
            )
            self.feature_dim = self.last_channel * 2
        else:
            self.fc = nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-3]

        x = x.reshape([-1, *self.obs_shape])
        x = self.encoder(x)
        x = x.view(-1, self.conved_size)
        x = self.fc(x)

        x = x.reshape([*batch_shape, *x.shape[1:]])
        return x


class Decoder(pl.LightningModule):
    def __init__(
        self,
        feature_dim: Union[int, tuple[int, int, int]],
        obs_shape: tuple[int, int, int],
        backbone_cfg: Union[ConvNetConfig, ViTConfig, ResNetConfig],
        fc_cfg: Union[MLPConfig, LinearConfig] = None,
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        if isinstance(backbone_cfg, ViTConfig):
            decoder = ViT
        elif isinstance(backbone_cfg, ConvNetConfig):
            decoder = ConvTranspose
        elif isinstance(backbone_cfg, ResNetConfig):
            decoder = ResNetPixShuffle
        else:
            raise NotImplementedError(
                f"{type(backbone_cfg)} is not implemented")

        self.input_shape = decoder.get_input_shape(obs_shape, backbone_cfg)
        if isinstance(feature_dim, int):
            assert fc_cfg is not None, "fc_cfg must be provided if feature_dim is provided"
            self.has_fc = True
        else:
            assert feature_dim == self.input_shape, f"{feature_dim} != {self.input_shape}"
            self.has_fc = False

        if isinstance(fc_cfg, MLPConfig):
            self.fc = MLPLayer(feature_dim, np.prod(self.input_shape), fc_cfg)
        elif isinstance(fc_cfg, LinearConfig):
            self.fc = LinearNormActivation(feature_dim, np.prod(self.input_shape), fc_cfg)
        else:
            self.fc = nn.Identity()

        self.decoder = decoder(in_shape=self.input_shape, obs_shape=obs_shape, cfg=backbone_cfg)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_fc:
            batch_shape, data_shape = x.shape[:-1], x.shape[-1:]
        else:
            batch_shape, data_shape = x.shape[:-3], x.shape[-3:]
        x = x.reshape([-1, *data_shape])
        x = self.fc(x)
        x = x.reshape([-1, *self.input_shape])
        x = self.decoder(x)

        return x.reshape([*batch_shape, *self.obs_shape])

class ViT(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int, int],
        cfg: ViTConfig,
        obs_shape: tuple[int, int, int]=None,
    ):
        super().__init__()

        self.in_shape = in_shape
        self.cfg = cfg
        self.obs_shape = obs_shape
        self.patch_size = cfg.patch_size

        self.transformer_cfg = cfg.transformer_cfg
        self.in_patch_dim = self.get_patch_dim(in_shape)
        self.out_patch_dim = self.get_patch_dim(obs_shape) if obs_shape is not None else self.transformer_cfg.d_model
        self.positional_embedding = PositionalEncoding(
            self.in_patch_dim, self.transformer_cfg.dropout, max_len=self.get_n_patches(in_shape))
        self.vit = TransformerLayer(
            self.in_patch_dim, self.out_patch_dim, self.transformer_cfg
        )
        self.is_encoder = obs_shape is None
        if self.is_encoder:
            self.n_patches = self.get_n_patches(in_shape)
            self.patch_embed = PatchEmbed(
                emb_dim=self.in_patch_dim,
                patch_size=self.patch_size,
                obs_shape=in_shape
            )
        if cfg.cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.in_patch_dim))
        self.last_channel = self.out_patch_dim // (self.patch_size ** 2)

    def forward(self, x: torch.Tensor, return_cls_token: bool = False):
        if self.is_encoder:
            x = self.patch_embed(x)
        else:
            x = self.patchify(x)
        x = self.positional_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        x = self.vit(x)
        if hasattr(self, "cls_token"):
            cls_token = x[:, 0]
            x = x[:, 1:]
        x = self.unpatchify(x)
        if return_cls_token and hasattr(self, "cls_token"):
            return x, cls_token
        else:
            return x

    def patchify(self, imgs: torch.Tensor):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[-1] % p == 0 and imgs.shape[-2] % p == 0

        x = rearrange(imgs, "n c (h p1) (w p2) -> n (h w) (p1 p2 c)", p1=p, p2=p)
        return x

    def unpatchify(self, x: torch.Tensor):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = self.obs_shape[1] // p
        w = self.obs_shape[2] // p
        assert h * \
            w == x.shape[1], f"{h*w} != {x.shape[1]}, please check the shape {x.shape} and obs_shape {self.obs_shape}"

        imgs = rearrange(x, "n (h w) (p1 p2 c) -> n c (h p1) (w p2)", h=h, w=w, p1=p, p2=p)
        return imgs

    @property
    def conved_size(self):
        return self.out_patch_dim * self.get_n_patches(self.in_shape)

    @property
    def conved_shape(self):
        return (self.in_shape[1] // self.patch_size, self.in_shape[2] // self.patch_size)

    def get_n_patches(self, obs_shape: tuple[int, int, int]):
        return (obs_shape[1] // self.patch_size) * (obs_shape[2] // self.patch_size)

    def get_patch_dim(self, obs_shape: tuple[int, int, int]):
        return self.patch_size ** 2 * obs_shape[0]

    @staticmethod
    def get_input_shape(obs_shape: tuple[int, int, int], cfg: ViTConfig):
        return (cfg.init_channel, obs_shape[1], obs_shape[2])

class ResNetPixUnshuffle(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        cfg: ResNetConfig,
        ):
        super().__init__() 

        self.obs_shape = obs_shape
        self.cfg = cfg

        first_cfg = ConvConfig(
            activation=cfg.conv_activation,
            kernel_size=cfg.f_kernel,
            stride=1,
            padding=cfg.conv_kernel//2,
            dilation=1,
            groups=1,
            bias=True,
            dropout=cfg.dropout,
            norm=cfg.norm,
            norm_cfg=cfg.norm_cfg,
        )
        # First layer
        self.conv1 = ConvNormActivation(self.obs_shape[0], cfg.conv_channel, first_cfg)

        # downsampling
        downsample = []
        downsample_cfg = first_cfg
        downsample_cfg.kernel_size = cfg.conv_kernel
        downsample_cfg.padding = cfg.conv_kernel//2
        downsample_cfg.scale_factor = -cfg.scale_factor
        for i in range(cfg.n_scaling):
            downsample += [
                ConvNormActivation(cfg.conv_channel, cfg.conv_channel, downsample_cfg)
            ]
        self.downsample = nn.Sequential(*downsample)


        # Residual blocks
        res_blocks = []
        for _ in range(cfg.n_res_blocks):
            res_blocks.append(ResidualBlock(
                cfg.conv_channel, 
                cfg.conv_kernel, 
                cfg.conv_activation, 
                cfg.norm, 
                cfg.norm_cfg,
                cfg.dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        cov2_cfg = first_cfg
        cov2_cfg.kernel_size = cfg.conv_kernel
        cov2_cfg.padding = cfg.conv_kernel//2

        # Second conv layer post residual blocks
        self.conv2 = ConvNormActivation(cfg.conv_channel, cfg.conv_channel, cov2_cfg)

        # Final output layer
        final_cfg = first_cfg
        final_cfg.kernel_size = cfg.conv_kernel
        final_cfg.padding = cfg.conv_kernel//2

        self.conv3 = ConvNormActivation(cfg.conv_channel, cfg.conv_channel, final_cfg)
        self.last_channel = cfg.conv_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out1 = self.downsample(out)
        out_res = self.res_blocks(out1)
        out2 = self.conv2(out_res)
        out = torch.add(out1, out2)
        out = self.conv3(out)

        return out

    @property
    def conved_shape(self):
        height = self.obs_shape[1] // (self.cfg.scale_factor ** self.cfg.n_scaling)
        width = self.obs_shape[2] // (self.cfg.scale_factor ** self.cfg.n_scaling)
        return (height, width)

    @property
    def conved_size(self):
        return self.last_channel * np.prod(self.conved_shape).item()

class ConvNet(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        cfg: ConvNetConfig,
    ):
        super().__init__()

        self.obs_shape = obs_shape
        self.channels = [obs_shape[0], *cfg.channels]
        self.cfg = cfg

        self.conv = self._build_conv()

        self.last_channel = self.channels[-1]

    def _build_conv(self):
        convs = []
        for i in range(len(self.channels)-1):
            convs += [ConvNormActivation(
                self.channels[i], self.channels[i+1], self.cfg.conv_cfgs[i])]

        return nn.Sequential(*convs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)

        return x

    @property
    def conved_shape(self):
        conv_shape = self.obs_shape[1:]
        for i in range(len(self.channels)-1):
            padding, kernel, stride, dilation \
                = self.cfg.conv_cfgs[i].padding, self.cfg.conv_cfgs[i].kernel_size, self.cfg.conv_cfgs[i].stride, self.cfg.conv_cfgs[i].dilation
            conv_shape = conv_out_shape(
                conv_shape, padding, kernel, stride, dilation)

        return conv_shape

    @property
    def conved_size(self):
        return self.last_channel * np.prod(self.conved_shape).item()


class ResNetPixShuffle(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        cfg: ResNetConfig,
    ):
        super().__init__()

        self.in_shape = in_shape
        self.obs_shape = obs_shape
        self.conv_channel = cfg.conv_channel
        self.conv_kernel = cfg.conv_kernel
        self.final_kernel = cfg.f_kernel
        self.conv_activation = cfg.conv_activation
        self.out_activation = cfg.out_activation
        self.n_res_blocks = cfg.n_res_blocks
        self.upscale_factor = cfg.scale_factor
        self.n_upsampling = cfg.n_scaling
        self.norm = cfg.norm
        self.norm_cfg = cfg.norm_cfg
        self.dropout = cfg.dropout

        self._scaling_factor = self.upscale_factor ** self.n_upsampling


        height = obs_shape[1]
        width = obs_shape[2]

        out_channels = obs_shape[0]
        self.input_height, self.input_width = height//self._scaling_factor, width//self._scaling_factor
        assert self.input_height == in_shape[1] and self.input_width == in_shape[2], f"{self.input_height} != {in_shape[1]} and {self.input_width} != {in_shape[2]}"

        conv_cfg = ConvConfig(
            activation=self.conv_activation,
            kernel_size=self.conv_kernel,
            stride=1,
            padding=self.conv_kernel//2,
            dilation=1,
            groups=1,
            bias=True,
            dropout=self.dropout,
            norm=cfg.norm,
            norm_cfg=cfg.norm_cfg,
        )


        # First layer
        self.conv1 = ConvNormActivation(in_shape[0], self.conv_channel, conv_cfg)

        # Residual blocks
        res_blocks = []
        for _ in range(self.n_res_blocks):
            res_blocks.append(ResidualBlock(
                self.conv_channel, 
                self.conv_kernel, 
                self.conv_activation, 
                self.norm, 
                self.norm_cfg,
                self.dropout))
        self.res_blocks = nn.Sequential(*res_blocks)



        # Second conv layer post residual blocks
        self.conv2 = ConvNormActivation(self.conv_channel, self.conv_channel, conv_cfg)

        upscale_cfg = conv_cfg
        upscale_cfg.scale_factor = self.upscale_factor

        # Upsampling layers
        upsampling = []
        for _ in range(self.n_upsampling):
            upsampling += [
                ConvNormActivation(self.conv_channel, self.conv_channel, upscale_cfg)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        final_cfg = conv_cfg
        final_cfg.kernel_size = self.final_kernel
        final_cfg.padding = self.final_kernel//2
        final_cfg.activation = self.out_activation
        # Final output layer
        self.conv3 = ConvNormActivation(self.conv_channel, out_channels, final_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        return out

    def get_input_shape(self, obs_shape: tuple[int, int, int], cfg: ResNetConfig):
        return (cfg.init_channel, obs_shape[1]//(cfg.scale_factor**cfg.n_scaling), obs_shape[2]//(cfg.scale_factor**cfg.n_scaling))

class ConvTranspose(nn.Module):
    """

    Vision Decoder

    self.sequenceで
        (batch_size, latent_dim) -> (batch_size, middle_layer_dim)
        -> (batch_size, W/2 * H/2 * 256) -> (batch_size, C, W, H))
    になる

    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        cfg: ConvNetConfig,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.obs_shape = obs_shape
        self.conv_out_shapes = []
        self.cfg = cfg
        self.channels = [*cfg.channels, obs_shape[0]]
        assert len(cfg.channels) == len(cfg.conv_cfgs)
        if self.in_shape[0] != cfg.channels[0]:
            self.first_conv = nn.Conv2d(in_shape[0], cfg.channels[0], kernel_size=1, stride=1, padding=0)
            self.init_channel = cfg.channels[0]
            self.have_first_conv = True
        else:
            self.init_channel = in_shape[0]
            self.have_first_conv = False

        prev_shape = in_shape
        for conv_cfg in cfg.conv_cfgs:
            padding, kernel, stride, dilation \
            = conv_cfg.padding, conv_cfg.kernel_size, conv_cfg.stride, conv_cfg.dilation
            prev_shape = conv_out_shape(prev_shape, padding, kernel, stride, dilation)
            self.conv_out_shapes += [prev_shape]
        assert self.conv_out_shapes[-1] == obs_shape[1:], f"{self.conv_out_shapes[-1]} != {obs_shape[1:]}"

        self.conv = self._build_conv()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.have_first_conv:
            z = self.first_conv(z)
        reconstruct = self.conv(z)

        return reconstruct

    def _build_conv(self):
        convs = []
        for i, cfg in enumerate(self.cfg.conv_cfgs):
            convs += [ConvTransposeNormActivation(
                self.channels[i], self.channels[i+1], cfg)]

        return nn.Sequential(*convs)

    @property
    def conved_size(self):
        conved_size = self.init_channel * np.prod(self.conv_shapes[-1]).item()
        print(f"conved_size: {conved_size}")
        return conved_size

    def get_input_shape(self, obs_shape: tuple[int, int, int], cfg: ConvNetConfig):
        in_shape = obs_shape[1:]
        for conv_cfg in reversed(cfg.conv_cfgs):
            padding, kernel, stride, dilation \
            = conv_cfg.padding, conv_cfg.kernel_size, conv_cfg.stride, conv_cfg.dilation
            in_shape = conv_transpose_in_shape(
                in_shape, padding, kernel, stride, dilation)
        return (cfg.init_channel, *in_shape)
