from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Union, Tuple

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback


def load_config(path: str) -> DictConfig:
    """Convert model config `.yaml` to `Dictconfig` with custom resolvers."""
    path = Path(path)

    config = OmegaConf.load(path)

    return config


@dataclass
class ConvConfig:
    activation: str
    kernel_size: int
    stride: int
    padding: int
    output_padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    dropout: float = 0.0
    norm: Literal["batch", "layer", "batch", "group", "none"] = "none"
    norm_cfg: dict = field(default_factory=dict)
    scale_factor: int = 0

    def __post_init__(self):
        if self.norm == "none":
            self.norm_cfg = dict()
        else:
            self.norm_cfg = dict(**self.norm_cfg)

@dataclass
class ConvNetConfig:
    channels: Tuple[int, ...]
    conv_cfgs: Tuple[ConvConfig, ...]
    init_channel: int

    def __post_init__(self):
        self.conv_cfgs = tuple(self.conv_cfgs)
        self.channels = tuple(self.channels)



@dataclass
class ResNetConfig:
    conv_channel: int
    conv_kernel: int
    f_kernel: int
    conv_activation: str
    out_activation: str
    n_res_blocks: int
    scale_factor: int = 2
    n_scaling: int = 2
    norm: Literal["batch", "group", "none"] = "none"
    norm_cfg: dict = field(default_factory=dict)
    dropout: float = 0.0
    init_channel: int = 16

@dataclass
class MLPConfig:
    hidden_dim: int
    n_layers: int
    output_activation: str
    linear_cfg: LinearConfig

@dataclass
class LinearConfig:
    activation: str
    norm: Literal["layer", "rms", "none"] = "none"
    norm_cfg: dict = field(default_factory=dict)
    dropout: float = 0.0
    norm_first: bool = False
    bias: bool = True

@dataclass
class TransformerConfig:
    d_model: int
    nhead: int
    dim_ff: int
    n_layers: int
    dropout: float = 0.1
    hidden_activation: Literal["ReLU", "GELU"] = "GELU"
    output_activation: str = "GeLU"

@dataclass
class ViTConfig:
    patch_size: int
    transformer_cfg: TransformerConfig
    cls_token: bool = True  
    init_channel: int = 16

@dataclass
class SpatialSoftmaxConfig:
    temperature: float = 1.0

