
from __future__ import annotations
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import nn

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
from ml_networks.layers import (
    ConvNormActivation,
    ConvTransposeNormActivation,
    LinearNormActivation,
    MLPLayer,
    PatchEmbed,
    PositionalEncoding,
    ResidualBlock,
    SpatialSoftmax,
    TransformerLayer,
)
from ml_networks.utils import conv_out_shape, conv_transpose_in_shape, conv_transpose_out_shape


class Encoder(pl.LightningModule):
    """
    Encoder with various architectures.

    Parameters
    ----------
    feature_dim: Union[int, tuple[int, int, int]]
        Dimension of the feature tensor.
        If int, Encoder includes full connection layer to downsample the feature tensor.
        Otherwise, Encoder does not include full connection layer and directly process with backbone network.
    obs_shape: tuple[int, int, int]
        shape of the input tensor
    backbone_cfg: Union[ViTConfig, ConvNetConfig, ResNetConfig]
        configuration of the network
    fc_cfg: Union[MLPConfig, LinearConfig, SpatialSoftmaxConfig]
        configuration of the full connection layer. If feature_dim is tuple, fc_cfg is ignored.
        If feature_dim is int, fc_cfg must be provided. Default is None.


    Examples
    --------
    >>> feature_dim = 128
    >>> obs_shape = (3, 64, 64)
    >>> cfg = ConvNetConfig(
    ...     channels=[16, 32, 64],
    ...     conv_cfgs=[
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...     ]
    ... )
    >>> fc_cfg = LinearConfig(
    ...     activation="ReLU",
    ...     bias=True
    ... )
    >>> encoder = Encoder(feature_dim, obs_shape, cfg, fc_cfg)
    >>> x = torch.randn(2, *obs_shape)
    >>> y = encoder(x)
    >>> y.shape
    torch.Size([2, 128])

    >>> encoder
    Encoder(
      (encoder): ConvNet(
        (conv): Sequential(
          (0): ConvNormActivation(
            (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pixel_shuffle): Identity()
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
          (1): ConvNormActivation(
            (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pixel_shuffle): Identity()
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
          (2): ConvNormActivation(
            (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pixel_shuffle): Identity()
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
        )
      )
      (fc): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): LinearNormActivation(
          (linear): Linear(in_features=4096, out_features=128, bias=True)
          (norm): Identity()
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
      )
    )

    """

    def __init__(
        self,
        feature_dim: int | tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        backbone_cfg: ViTConfig | ConvNetConfig | ResNetConfig,
        fc_cfg: MLPConfig | LinearConfig | SpatialSoftmaxConfig | None = None,
    ) -> None:
        super().__init__()

        self.obs_shape = obs_shape

        if isinstance(backbone_cfg, ViTConfig):
            self.encoder = ViT(obs_shape, backbone_cfg)
        elif isinstance(backbone_cfg, ConvNetConfig):
            self.encoder = ConvNet(obs_shape, backbone_cfg)
        elif isinstance(backbone_cfg, ResNetConfig):
            self.encoder = ResNetPixUnshuffle(obs_shape, backbone_cfg)
        else:
            msg = f"{type(backbone_cfg)} is not implemented"
            raise NotImplementedError(msg)

        self.feature_dim = feature_dim
        self.conved_size = self.encoder.conved_size
        self.conved_shape = self.encoder.conved_shape
        self.last_channel = self.encoder.last_channel

        if isinstance(feature_dim, int):
            assert fc_cfg is not None, "fc_cfg must be provided if feature_dim is provided"
        else:
            assert feature_dim == (self.encoder.last_channel, *self.encoder.conved_shape), (
                f"{feature_dim} != {(self.encoder.last_channel, *self.encoder.conved_shape)}"
            )
        if isinstance(fc_cfg, MLPConfig):
            self.fc = nn.Sequential(
                nn.Flatten(),
                MLPLayer(self.conved_size, feature_dim, fc_cfg),
            )
        elif isinstance(fc_cfg, LinearConfig):
            self.fc = nn.Sequential(
                nn.Flatten(),
                LinearNormActivation(self.conved_size, feature_dim, fc_cfg),
            )
        elif isinstance(fc_cfg, AdaptiveAveragePoolingConfig):
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(fc_cfg.output_size),
                nn.Flatten(),
                LinearNormActivation(
                    int(self.last_channel * np.prod(fc_cfg.output_size)),
                    feature_dim,
                    fc_cfg.additional_layer
                ) if isinstance(
                        fc_cfg.additional_layer, LinearConfig
                ) else MLPLayer(
                    int(self.last_channel * np.prod(fc_cfg.output_size)),
                    feature_dim,
                    fc_cfg.additional_layer
                ) if isinstance(
                        fc_cfg.additional_layer, MLPConfig
                ) else nn.Identity(),
            )
            if fc_cfg.additional_layer is None:
                self.feature_dim = self.last_channel * np.prod(fc_cfg.output_size)
        
        elif isinstance(fc_cfg, SpatialSoftmaxConfig):
            self.fc = nn.Sequential(
                SpatialSoftmax(fc_cfg),
                nn.Flatten(),
                LinearNormActivation(
                    self.last_channel * 2,
                    self.feature_dim,
                    fc_cfg.additional_layer
                ) if isinstance(
                        fc_cfg.additional_layer, LinearConfig
                ) else MLPLayer(
                    self.last_channel * 2,
                    self.feature_dim,
                    fc_cfg.additional_layer
                ) if isinstance(
                        fc_cfg.additional_layer, MLPConfig
                ) else nn.Identity(),
            )
            if fc_cfg.additional_layer is None:
                self.feature_dim = self.last_channel * 2
        else:
            self.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            input tensor of shape (batch_size, *obs_shape)

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, *feature_dim)
        """
        batch_shape = x.shape[:-3]

        x = x.reshape([-1, *self.obs_shape])
        x = self.encoder(x)
        x = x.view(-1, self.last_channel, *self.conved_shape)
        x = self.fc(x)
        return x.reshape([*batch_shape, *x.shape[1:]])


class Decoder(pl.LightningModule):
    """
    Decoder with various architectures.

    Parameters
    ----------
    feature_dim: Union[int, tuple[int, int, int]]
        dimension of the feature tensor, if int, Decoder includes full connection layer to upsample the feature tensor.
        Otherwise, Decoder does not include full connection layer and directly process with backbone network.
    obs_shape: tuple[int, int, int]
        shape of the output tensor
    backbone_cfg: Union[ConvNetConfig, ViTConfig, ResNetConfig]
        configuration of the network
    fc_cfg: Union[MLPConfig, LinearConfig]
        configuration of the full connection layer. If feature_dim is tuple, fc_cfg is ignored.
        If feature_dim is int, fc_cfg must be provided. Default is None.

    Examples
    --------
    >>> feature_dim = 128
    >>> obs_shape = (3, 64, 64)
    >>> cfg = ConvNetConfig(
    ...     channels=[64, 32, 16],
    ...     conv_cfgs=[
    ...         ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...     ]
    ... )
    >>> fc_cfg = MLPConfig(
    ...     hidden_dim=256,
    ...     n_layers=2,
    ...     output_activation= "ReLU",
    ...     linear_cfg= LinearConfig(
    ...         activation= "ReLU",
    ...         bias= True
    ...     )
    ... )

    >>> decoder = Decoder(feature_dim, obs_shape, cfg, fc_cfg)
    >>> x = torch.randn(2, feature_dim)
    >>> y = decoder(x)
    >>> y.shape
    torch.Size([2, 3, 64, 64])

    >>> decoder
    Decoder(
      (fc): MLPLayer(
        (dense): Sequential(
          (0): LinearNormActivation(
            (linear): Linear(in_features=128, out_features=256, bias=True)
            (norm): Identity()
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
          (1): LinearNormActivation(
            (linear): Linear(in_features=256, out_features=256, bias=True)
            (norm): Identity()
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
          (2): LinearNormActivation(
            (linear): Linear(in_features=256, out_features=1024, bias=True)
            (norm): Identity()
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
        )
      )
      (decoder): ConvTranspose(
        (first_conv): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
        (conv): Sequential(
          (0): ConvTransposeNormActivation(
            (conv): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
          (1): ConvTransposeNormActivation(
            (conv): ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
          (2): ConvTransposeNormActivation(
            (conv): ConvTranspose2d(16, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (norm): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (activation): Activation(
              (activation): ReLU()
            )
            (dropout): Identity()
          )
        )
      )
    )
    """

    def __init__(
        self,
        feature_dim: int | tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        backbone_cfg: ConvNetConfig | ViTConfig | ResNetConfig,
        fc_cfg: MLPConfig | LinearConfig | None = None,
    ) -> None:
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
            msg = f"{type(backbone_cfg)} is not implemented"
            raise NotImplementedError(msg)

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
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            input tensor of shape (batch_size, *feature_dim)

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, *obs_shape)

        """
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
    """
    Vision Transformer for Encoder and Decoder.

    Parameters
    ----------
    in_shape: tuple[int, int, int]
        shape of input tensor
    cfg: ViTConfig
        configuration of the network
    obs_shape: tuple[int, int, int]
        shape of output tensor. If None, it is considered as Encoder. Default is None.

    Examples
    --------
    >>> from ml_networks.layers import TransformerConfig
    >>> in_shape = (3, 64, 64)
    >>> cfg = ViTConfig(
    ...     patch_size=8,
    ...     cls_token=True,
    ...     transformer_cfg=TransformerConfig(
    ...         d_model=64,
    ...         nhead=8,
    ...         dim_ff=256,
    ...         n_layers=3,
    ...         dropout=0.0,
    ...         hidden_activation="ReLU",
    ...         output_activation="ReLU"
    ...     ),
    ...     init_channel=3
    ... )
    >>> encoder = ViT(in_shape, cfg)
    >>> x = torch.randn(2, *in_shape)
    >>> y = encoder(x)
    >>> y.shape
    torch.Size([2, 1, 64, 64])
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        cfg: ViTConfig,
        obs_shape: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()

        self.in_shape = in_shape
        self.cfg = cfg
        self.obs_shape = obs_shape if obs_shape is not None else in_shape
        self.patch_size = cfg.patch_size

        self.transformer_cfg = cfg.transformer_cfg
        self.in_patch_dim = self.get_patch_dim(in_shape)
        self.out_patch_dim = self.get_patch_dim(obs_shape) if obs_shape is not None else self.transformer_cfg.d_model
        self.positional_embedding = PositionalEncoding(
            self.in_patch_dim,
            self.transformer_cfg.dropout,
            max_len=self.get_n_patches(in_shape),
        )
        self.vit = TransformerLayer(
            self.in_patch_dim,
            self.out_patch_dim,
            self.transformer_cfg,
        )
        self.is_encoder = obs_shape is None
        if self.is_encoder:
            self.n_patches = self.get_n_patches(in_shape)
            self.patch_embed = PatchEmbed(
                emb_dim=self.in_patch_dim,
                patch_size=self.patch_size,
                obs_shape=in_shape,
            )
        if cfg.cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.in_patch_dim))
        self.last_channel = self.out_patch_dim // (self.patch_size**2)

    def forward(
        self,
        x: torch.Tensor,
        return_cls_token: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            input tensor of shape (batch_size, *in_shape)
        return_cls_token: bool
            whether to return cls_token. Default is False.

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, *obs_shape)
        torch.Tensor
            cls_token of shape (batch_size, self.out_patch_dim) if return_cls_token

        """
        x = self.patch_embed(x) if self.is_encoder else self.patchfy(x)
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
        return x

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        画像をパッチに分割する.

        Paramters
        ---------
        imgs: torch.Tensor
            入力画像. (N, C, H, W)

        Returns
        -------
        torch.Tensor
            パッチ化した画像. (N, L, patch_size**2 * D)
        """
        p = self.patch_size
        assert imgs.shape[-1] % p == 0
        assert imgs.shape[-2] % p == 0
        return rearrange(imgs, "n c (h p1) (w p2) -> n (h w) (p1 p2 c)", p1=p, p2=p)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        パッチを画像に戻す.

        Parameters
        ----------
        x : torch.Tensor
            入力. (N, L, patch_size**2 * D)

        Returns
        -------
            画像. (N, C, H, W)
        """
        p = self.patch_size
        h = self.obs_shape[1] // p
        w = self.obs_shape[2] // p
        assert h * w == x.shape[1], (
            f"{h * w} != {x.shape[1]}, please check the shape {x.shape} and obs_shape {self.obs_shape}"
        )
        return rearrange(x, "n (h w) (p1 p2 c) -> n c (h p1) (w p2)", h=h, w=w, p1=p, p2=p)

    @property
    def conved_size(self) -> int:
        return self.out_patch_dim * self.get_n_patches(self.in_shape)

    @property
    def conved_shape(self) -> tuple[int, int]:
        return (self.in_shape[1] // self.patch_size, self.in_shape[2] // self.patch_size)

    def get_n_patches(self, obs_shape: tuple[int, int, int]) -> int:
        return (obs_shape[1] // self.patch_size) * (obs_shape[2] // self.patch_size)

    def get_patch_dim(self, obs_shape: tuple[int, int, int]) -> int:
        return self.patch_size**2 * obs_shape[0]

    @staticmethod
    def get_input_shape(obs_shape: tuple[int, int, int], cfg: ViTConfig) -> tuple[int, int, int]:
        return (cfg.init_channel, obs_shape[1], obs_shape[2])


class ResNetPixUnshuffle(nn.Module):
    """
    ResNet with PixelUnshuffle for Encoder.

    Parameters
    ----------
    obs_shape: tuple[int, int, int]
        shape of input tensor
    cfg: ResNetConfig
        configuration of the network

    Examples
    --------
    >>> obs_shape = (3, 64, 64)
    >>> cfg = ResNetConfig(
    ...     conv_channel=64,
    ...     conv_kernel=3,
    ...     f_kernel=3,
    ...     conv_activation="ReLU",
    ...     out_activation="ReLU",
    ...     n_res_blocks=2,
    ...     scale_factor=2,
    ...     n_scaling=3,
    ...     norm="batch",
    ...     norm_cfg={},
    ...     dropout=0.0
    ... )
    >>> encoder = ResNetPixUnshuffle(obs_shape, cfg)
    >>> x = torch.randn(2, *obs_shape)
    >>> y = encoder(x)
    >>> y.shape
    torch.Size([2, 64, 8, 8])
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        cfg: ResNetConfig,
    ) -> None:
        super().__init__()

        self.obs_shape = obs_shape
        self.cfg = cfg

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
            padding_mode=cfg.padding_mode
        )
        # First layer
        self.conv1 = ConvNormActivation(self.obs_shape[0], cfg.conv_channel, first_cfg)

        # downsampling
        downsample = []
        downsample_cfg = first_cfg
        downsample_cfg.kernel_size = cfg.conv_kernel
        downsample_cfg.padding = cfg.conv_kernel // 2
        downsample_cfg.scale_factor = -cfg.scale_factor
        for _ in range(cfg.n_scaling):
            downsample += [
                ConvNormActivation(cfg.conv_channel, cfg.conv_channel, downsample_cfg),
            ]
        self.downsample = nn.Sequential(*downsample)

        # Residual blocks
        res_blocks = [
            ResidualBlock(
                cfg.conv_channel,
                cfg.conv_kernel,
                cfg.conv_activation,
                cfg.norm,
                cfg.norm_cfg,
                cfg.dropout,
                cfg.padding_mode
            )
            for _ in range(cfg.n_res_blocks)
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        cov2_cfg = first_cfg
        cov2_cfg.kernel_size = cfg.conv_kernel
        cov2_cfg.padding = cfg.conv_kernel // 2
        cov2_cfg.scale_factor = 0

        # Second conv layer post residual blocks
        self.conv2 = ConvNormActivation(cfg.conv_channel, cfg.conv_channel, cov2_cfg)

        # Final output layer
        final_cfg = first_cfg
        final_cfg.kernel_size = cfg.conv_kernel
        final_cfg.padding = cfg.conv_kernel // 2

        self.conv3 = ConvNormActivation(cfg.conv_channel, cfg.conv_channel, final_cfg)
        self.last_channel = cfg.conv_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            input tensor of shape (batch_size, *obs_shape)

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, self.last_channel, *self.conved_shape)

        """
        out = self.conv1(x)
        out1 = self.downsample(out)
        out_res = self.res_blocks(out1)
        out2 = self.conv2(out_res)
        out = torch.add(out1, out2)
        return self.conv3(out)

    @property
    def conved_shape(self) -> tuple[int, int]:
        """
        Get the shape of the output tensor after convolutional layers.

        Returns
        -------
        tuple[int, int]
            shape of the output tensor
        """
        height = self.obs_shape[1] // (self.cfg.scale_factor**self.cfg.n_scaling)
        width = self.obs_shape[2] // (self.cfg.scale_factor**self.cfg.n_scaling)
        return (height, width)

    @property
    def conved_size(self) -> int:
        """
        Get the size of the output tensor after convolutional layers.

        Returns
        -------
        int
            size of the output tensor
        """
        return self.last_channel * np.prod(self.conved_shape).item()


class ConvNet(nn.Module):
    """
    Convolutional Neural Network for Encoder.

    Parameters
    ----------
    obs_shape: tuple[int, int, int]
        shape of input tensor
    cfg: ConvNetConfig
        configuration of the network

    Examples
    --------
    >>> obs_shape = (3, 64, 64)
    >>> cfg = ConvNetConfig(
    ...     channels=[16, 32, 64],
    ...     conv_cfgs=[
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...     ]
    ... )
    >>> encoder = ConvNet(obs_shape, cfg)
    >>> encoder
    ConvNet(
      (conv): Sequential(
        (0): ConvNormActivation(
          (conv): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pixel_shuffle): Identity()
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
        (1): ConvNormActivation(
          (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pixel_shuffle): Identity()
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
        (2): ConvNormActivation(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pixel_shuffle): Identity()
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
      )
    )
    >>> x = torch.randn(2, *obs_shape)
    >>> y = encoder(x)
    >>> y.shape
    torch.Size([2, 64, 8, 8])
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        cfg: ConvNetConfig,
    ) -> None:
        super().__init__()

        self.obs_shape = obs_shape
        self.channels = [obs_shape[0], *cfg.channels]
        self.cfg = cfg

        self.conv = self._build_conv()

        self.last_channel = self.channels[-1]

    def _build_conv(self) -> nn.Module:
        convs = []
        for i in range(len(self.channels) - 1):
            convs += [ConvNormActivation(self.channels[i], self.channels[i + 1], self.cfg.conv_cfgs[i])]

        return nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            input tensor of shape (batch_size, *obs_shape)

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, self.last_channel, *self.conved_shape)

        """
        return self.conv(x)

    @property
    def conved_shape(self) -> tuple[int, int]:
        """
        Get the shape of the output tensor after convolutional layers.

        Returns
        -------
        tuple[int, int]
            shape of the output tensor

        Examples
        --------
        >>> obs_shape = (3, 64, 64)
        >>> cfg = ConvNetConfig(
        ...     channels=[64, 32, 16],
        ...     conv_cfgs=[
        ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
        ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
        ...         ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
        ...     ]
        ... )
        >>> encoder = ConvNet(obs_shape, cfg)
        >>> encoder.conved_shape
        (8, 8)

        """
        conv_shape = self.obs_shape[1:]
        for i in range(len(self.channels) - 1):
            padding, kernel, stride, dilation = (
                self.cfg.conv_cfgs[i].padding,
                self.cfg.conv_cfgs[i].kernel_size,
                self.cfg.conv_cfgs[i].stride,
                self.cfg.conv_cfgs[i].dilation,
            )
            conv_shape = conv_out_shape(conv_shape, padding, kernel, stride, dilation)

        return conv_shape

    @property
    def conved_size(self) -> int:
        """
        Get the size of the output tensor after convolutional layers.

        Returns
        -------
        int
            size of the output tensor

        """
        return self.last_channel * np.prod(self.conved_shape).item()


class ResNetPixShuffle(nn.Module):
    """
    ResNet with PixelShuffle.

    Parameters
    ----------
    in_shape: tuple[int, int, int]
        shape of input tensor
    obs_shape: tuple[int, int, int]
        shape of output tensor
    cfg: ResNetConfig
        configuration of the network

    Examples
    --------
    >>> in_shape = (128, 16, 16)
    >>> obs_shape = (3, 64, 64)
    >>> cfg = ResNetConfig(
    ...     conv_channel=64,
    ...     conv_kernel=3,
    ...     f_kernel=3,
    ...     conv_activation="ReLU",
    ...     out_activation="ReLU",
    ...     n_res_blocks=2,
    ...     scale_factor=2,
    ...     n_scaling=2,
    ...     norm="batch",
    ...     norm_cfg={},
    ...     dropout=0.0
    ... )
    >>> decoder = ResNetPixShuffle(in_shape, obs_shape, cfg)
    >>> x = torch.randn(2, *in_shape)
    >>> y = decoder(x)
    >>> y.shape
    torch.Size([2, 3, 64, 64])

    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        cfg: ResNetConfig,
    ) -> None:
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

        self._scaling_factor = self.upscale_factor**self.n_upsampling

        height = obs_shape[1]
        width = obs_shape[2]

        out_channels = obs_shape[0]
        self.input_height, self.input_width = height // self._scaling_factor, width // self._scaling_factor
        assert self.input_height == in_shape[1], f"{self.input_height} != {in_shape[1]}"
        assert self.input_width == in_shape[2], f"{self.input_width} != {in_shape[2]}"

        conv_cfg = ConvConfig(
            activation=self.conv_activation,
            kernel_size=self.conv_kernel,
            stride=1,
            padding=self.conv_kernel // 2,
            dilation=1,
            groups=1,
            bias=True,
            dropout=self.dropout,
            norm=cfg.norm,
            norm_cfg=cfg.norm_cfg,
            padding_mode=cfg.padding_mode,
        )

        # First layer
        self.conv1 = ConvNormActivation(in_shape[0], self.conv_channel, conv_cfg)

        # Residual blocks
        res_blocks = [
            ResidualBlock(
                self.conv_channel,
                self.conv_kernel,
                self.conv_activation,
                self.norm,
                self.norm_cfg,
                self.dropout,
                cfg.padding_mode
            )
            for _ in range(self.n_res_blocks)
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = ConvNormActivation(self.conv_channel, self.conv_channel, conv_cfg)

        upscale_cfg = conv_cfg
        upscale_cfg.scale_factor = self.upscale_factor

        # Upsampling layers
        upsampling = []
        for _ in range(self.n_upsampling):
            upsampling += [
                ConvNormActivation(self.conv_channel, self.conv_channel, upscale_cfg),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        final_cfg = conv_cfg
        final_cfg.kernel_size = self.final_kernel
        final_cfg.padding = self.final_kernel // 2
        final_cfg.activation = self.out_activation
        final_cfg.norm = "none"
        final_cfg.norm_cfg = {}
        final_cfg.dropout = 0.0
        final_cfg.scale_factor = 0
        # Final output layer
        self.conv3 = ConvNormActivation(self.conv_channel, out_channels, final_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            input tensor of shape (batch_size, *in_shape)

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, *obs_shape)

        """
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        return self.conv3(out)

    @staticmethod
    def get_input_shape(obs_shape: tuple[int, int, int], cfg: ResNetConfig) -> tuple[int, int, int]:
        """
        Get input shape of the decoder.

        Parameters
        ----------
        obs_shape: tuple[int, int, int]
            shape of the output tensor
        cfg: ConvNetConfig
            configuration of the network

        Returns
        -------
        tuple[int, int, int]
            shape of the input tensor

        """
        return (
            cfg.init_channel,
            obs_shape[1] // (cfg.scale_factor**cfg.n_scaling),
            obs_shape[2] // (cfg.scale_factor**cfg.n_scaling),
        )


class ConvTranspose(nn.Module):
    """
    Convolutional Transpose Network for Decoder.

    Parameters
    ----------
    in_shape: tuple[int, int, int]
        shape of input tensor
    obs_shape: tuple[int, int, int]
        shape of output tensor
    cfg: ConvNetConfig
        configuration of the network

    Examples
    --------
    >>> in_shape = (128, 8, 8)
    >>> obs_shape = (3, 64, 64)
    >>> cfg = ConvNetConfig(
    ...     channels=[64, 32, 16],
    ...     conv_cfgs=[
    ...         ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...         ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
    ...     ]
    ... )
    >>> decoder = ConvTranspose(in_shape, obs_shape, cfg)
    >>> decoder
    ConvTranspose(
      (first_conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
      (conv): Sequential(
        (0): ConvTransposeNormActivation(
          (conv): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
          (norm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
        (1): ConvTransposeNormActivation(
          (conv): ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
          (norm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
        (2): ConvTransposeNormActivation(
          (conv): ConvTranspose2d(16, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
          (norm): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activation): Activation(
            (activation): ReLU()
          )
          (dropout): Identity()
        )
      )
    )
    >>> x = torch.randn(2, *in_shape)
    >>> y = decoder(x)
    >>> y.shape
    torch.Size([2, 3, 64, 64])
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        obs_shape: tuple[int, int, int],
        cfg: ConvNetConfig,
    ) -> None:
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

        prev_shape = in_shape[1:]
        for conv_cfg in cfg.conv_cfgs:
            padding, kernel, stride, dilation = (
                conv_cfg.padding,
                conv_cfg.kernel_size,
                conv_cfg.stride,
                conv_cfg.dilation,
            )
            prev_shape = conv_transpose_out_shape(prev_shape, padding, kernel, stride, dilation)
            self.conv_out_shapes += [prev_shape]
        assert self.conv_out_shapes[-1] == obs_shape[1:], f"{self.conv_out_shapes[-1]} != {obs_shape[1:]}"

        self.conv = self._build_conv()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        z: torch.Tensor
            input tensor of shape (batch_size, *in_shape)

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, *obs_shape)


        """
        if self.have_first_conv:
            z = self.first_conv(z)
        return self.conv(z)

    def _build_conv(self) -> nn.Module:
        convs = []
        for i, cfg in enumerate(self.cfg.conv_cfgs):
            convs += [ConvTransposeNormActivation(self.channels[i], self.channels[i + 1], cfg)]
        return nn.Sequential(*convs)

    @staticmethod
    def get_input_shape(obs_shape: tuple[int, int, int], cfg: ConvNetConfig) -> tuple[int, ...]:
        """
        Get input shape of the decoder.

        Parameters
        ----------
        obs_shape: tuple[int, int, int]
            shape of the output tensor
        cfg: ConvNetConfig
            configuration of the network

        Returns
        -------
        tuple[int, int, int]
            shape of the input tensor

        Examples
        --------
        >>> obs_shape = (3, 64, 64)
        >>> cfg = ConvNetConfig(
        ...     channels=[64, 32, 16],
        ...     conv_cfgs=[
        ...         ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
        ...         ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
        ...         ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU", norm="batch", dropout=0.0),
        ...     ]
        ... )
        >>> ConvTranspose.get_input_shape(obs_shape, cfg)
        (16, 8, 8)
        """
        in_shape = obs_shape[1:]
        for conv_cfg in reversed(cfg.conv_cfgs):
            padding, kernel, stride, dilation = (
                conv_cfg.padding,
                conv_cfg.kernel_size,
                conv_cfg.stride,
                conv_cfg.dilation,
            )
            in_shape = conv_transpose_in_shape(in_shape, padding, kernel, stride, dilation)
        return (cfg.init_channel, *in_shape)
