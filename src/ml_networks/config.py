from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(path: str) -> DictConfig | ListConfig:
    """
    Convert model config `.yaml` to `Dictconfig` with custom resolvers.

    Returns
    -------
    DictConfig
    """
    return OmegaConf.load(Path(path))

def convert_dictconfig_to_dict(obj):
    """Recursively convert DictConfig to dict in any iterable object."""
    if isinstance(obj, DictConfig):
        return dict(obj)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_dictconfig_to_dict(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: convert_dictconfig_to_dict(v) for k, v in obj.items()}
    return obj


@dataclass
class ConvConfig:
    """
    A convolutional layer configuration.

    Attributes
    ----------
    activation : str
        Activation function.
    kernel_size : int
        Kernel size.
    stride : int
        Stride.
    padding : int
        Padding.
    output_padding : int
        Output padding, especially for transposed convolution. Default is 0.
    dilation : int
        Dilation.
    groups : int
        Number of groups. Default is 1. See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.
    bias : bool
        Whether to use bias. Default is True.
    dropout : float
        Dropout rate. If it's set to 0.0, dropout is not applied. Default is 0.0.
    norm : Literal["batch", "group", "none"]
        Normalization layer. If it's set to "none", normalization is not applied. Default is "none".
    norm_cfg : dict
        Normalization layer configuration. If you want to use Instance, Layer, or Group normalization,
        set norm to "group" and set norm_cfg with "num_groups=$in_channel, 1, or any value". Default is {}.
    scale_factor : int
        Scale factor for upsample, especially for PixelShuffle or PixelUnshuffle.
        If it's set to >0, upsample is applied. If it's set to <0 downsample is applied.
        Otherwise, no upsample or downsample is applied. Default is 0.


    """

    activation: str
    kernel_size: int
    stride: int
    padding: int
    output_padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    dropout: float = 0.0
    norm: Literal["batch", "group", "none"] = "none"
    norm_cfg: dict[str, Any] = field(default_factory=dict)
    scale_factor: int = 0

    def __post_init__(self) -> None:
        """Set `.norm_cfg`."""
        if self.norm == "none":
            self.norm_cfg = {}
        else:
            self.norm_cfg = dict(**self.norm_cfg)

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `ConvConfig`.

        Returns
        -------
        dict
            Dictionary representation of ConvConfig.
        """
        self.norm_cfg = dict(self.norm_cfg)

        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))
        


@dataclass
class ConvNetConfig:
    """
    Convolutional neural network layers configuration.

    Attributes
    ----------
    channels : Tuple[int, ...]
        Number of channels for each layer.
    conv_cfgs : Tuple[ConvConfig, ...]
        Convolutional layer configurations.
        The length of conv_cfgs should be the same as the length of channels.
    init_channel : int
        Initial number of channels, especially for transposed convolution.
    """

    channels: tuple[int, ...]
    conv_cfgs: tuple[ConvConfig, ...]
    init_channel: int = 16

    def __post_init__(self) -> None:
        """Set `channels` and `conv_cfgs` as tuple."""
        self.conv_cfgs = tuple(self.conv_cfgs)
        self.channels = tuple(self.channels)

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `ConvNetConfig`.

        Returns
        -------
        dict
            Dictionary representation of ConvNetConfig.
        """
        self.channels = tuple(self.channels)
        conv_cfgs = []
        for conv_cfg in self.conv_cfgs:
            if isinstance(conv_cfg, DictConfig):
                conv_cfg = ConvConfig(**conv_cfg)
            conv_cfg.dictcfg2dict()
            conv_cfgs.append(conv_cfg)
        self.conv_cfgs = tuple(conv_cfgs)
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))


@dataclass
class ResNetConfig:
    """
    Residual neural network layers configuration.

    Attributes
    ----------
    conv_channel : int
        Number of channels for convolutional layer.
        In ResNet, common number of channels is used for all layers.
    conv_kernel : int
        Kernel size for convolutional layer.
        In ResNet, common kernel size is used for all layers.
    f_kernel : int
        Kernel size for final or first convolutional layer.
        This depends on whether PixelShuffle or PixelUnshuffle is used.
    conv_activation : str
        Activation function for convolutional layer.
    out_activation : str
        Activation function for output layer.
    n_res_blocks : int
        Number of residual blocks.
    scale_factor : int
        Scale factor for upsample, especially for PixelShuffle or PixelUnshuffle.
    n_scaling : int
        Number of upsample or downsample layers. The image size is scaled by scalefactor^n_scaling.
    norm : Literal["batch", "group", "none"]
        Normalization layer. If it's set to "none", normalization is not applied. Default is "none".
    norm_cfg : dict
        Normalization layer configuration. If you want to use Instance, Layer, or Group normalization,
        set norm to "group" and set norm_cfg with "num_groups=$in_channel, 1, or any value". Default is {}.
    dropout : float
        Dropout rate. If it's set to 0.0, dropout is not applied. Default is 0.0.
    init_channel : int
        Initial number of channels, especially for decoder.
    """

    conv_channel: int
    conv_kernel: int
    f_kernel: int
    conv_activation: str
    out_activation: str
    n_res_blocks: int
    scale_factor: int = 2
    n_scaling: int = 2
    norm: Literal["batch", "group", "none"] = "none"
    norm_cfg: dict[str, Any] = field(default_factory=dict)
    dropout: float = 0.0
    init_channel: int = 16

    def __post_init__(self) -> None:
        """Set `norm_cfg`."""
        if self.norm == "none":
            self.norm_cfg = {}
        else:
            self.norm_cfg = dict(**self.norm_cfg)

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `ResNetConfig`.

        Returns
        -------
        dict
            Dictionary representation of ResNetConfig.
        """
        self.norm_cfg = dict(self.norm_cfg)
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))

@dataclass
class LinearConfig:
    """
    A linear layer configuration.

    Attributes
    ----------
    activation : str
        Activation function.
    norm : Literal["layer", "rms", "none"]
        Normalization layer. If it's set to "none", normalization is not applied. Default is "none".
    norm_cfg : dict
        Normalization layer configuration. Default is {}.
    dropout : float
        Dropout rate. If it's set to 0.0, dropout is not applied. Default is 0.0.
    norm_first : bool
        Whether to apply normalization before linear layer. Default is False.
    bias : bool
        Whether to use bias. Default is True.
    """

    activation: str
    norm: Literal["layer", "rms", "none"] = "none"
    norm_cfg: dict[str, Any] = field(default_factory=dict)
    dropout: float = 0.0
    norm_first: bool = False
    bias: bool = True

    def __post_init__(self) -> None:
        """Set `norm_cfg`."""
        if self.norm == "none":
            self.norm_cfg = {}
        else:
            self.norm_cfg = dict(**self.norm_cfg)

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `LinearConfig`.

        Returns
        -------
        dict
            Dictionary representation of LinearConfig.
        """
        self.norm_cfg = dict(self.norm_cfg)
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))

@dataclass
class MLPConfig:
    """
    Multi-layer perceptron configuration.

    Attributes
    ----------
    hidden_dim : int
        Number of hidden units.
    n_layers : int
        Number of layers.
    output_activation : str
        Activation function for output layer.
    linear_cfg : LinearConfig
        Linear layer configuration.

    """

    hidden_dim: int
    n_layers: int
    output_activation: str
    linear_cfg: LinearConfig

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `MLPConfig`.

        Returns
        -------
        dict
            Dictionary representation of MLPConfig.
        """
        self.linear_cfg.dictcfg2dict()
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))


@dataclass
class TransformerConfig:
    """
    Transformer configuration.

    Attributes
    ----------
    d_model : int
        Dimension of model.
    nhead : int
        Number of heads.
    dim_ff : int
        Dimension of feedforward network.
    n_layers : int
        Number of layers.
    dropout : float
        Dropout rate. Default is 0.1.
    hidden_activation : Literal["ReLU", "GELU"]
        Activation function for hidden layer. Default is "GELU".
    output_activation : str
        Activation function for output layer. Default is "GeLU".

    """

    d_model: int
    nhead: int
    dim_ff: int
    n_layers: int
    dropout: float = 0.1
    hidden_activation: Literal["ReLU", "GELU"] = "GELU"
    output_activation: str = "GeLU"


@dataclass
class ViTConfig:
    """
    Vision Transformer configuration.

    Attributes
    ----------
    patch_size : int
        Patch size.
    transformer_cfg : TransformerConfig
        Transformer configuration.
    cls_token : bool
        Whether to use class token. Default is True.
    init_channel : int
        Initial number of channels. Default is 16.
    """

    patch_size: int
    transformer_cfg: TransformerConfig
    cls_token: bool = True
    init_channel: int = 16


@dataclass
class SpatialSoftmaxConfig:
    """
    Spatial softmax configuration.

    Attributes
    ----------
    temperature : float
        Softmax temperature. If it's set to 0.0, the layer outputs the coordinates of the maximum value.
        Otherwise, the layer outputs the expectation of the coordinates with softmax function.
        Default is 0.0.
    """

    temperature: float = 1.0
    eps: float = 1e-6
    is_argmax: bool = False
    is_straight_through: bool = False

@dataclass 
class EncoderConfig:
    """
    Encoder configuration.

    Attributes
    ----------
    backbone: Union[ConvNetConfig, ResNetConfig]
        Backbone configuration.
    full_connection: Union[MLPConfig, LinearConfig, SpatialSoftmaxConfig]
        Full connection configuration.
    """

    backbone: Union[ConvNetConfig, ResNetConfig]
    full_connection: Union[MLPConfig, LinearConfig, SpatialSoftmaxConfig]

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `EncoderConfig`.

        Returns
        -------
        dict
            Dictionary representation of EncoderConfig.
        """
        self.backbone.dictcfg2dict()
        if hasattr(self.full_connection, "dictcfg2dict"):
            self.full_connection.dictcfg2dict()
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))


@dataclass
class DecoderConfig:
    """
    Decoder configuration.

    Attributes
    ----------
    backbone: Union[ConvNetConfig, ResNetConfig]
        Backbone configuration.
    full_connection: Union[MLPConfig, LinearConfig, SpatialSoftmaxConfig]
        Full connection configuration.
    """
    
    backbone: Union[ConvNetConfig, ResNetConfig]
    full_connection: Union[MLPConfig, LinearConfig]

    def dictcfg2dict(self):
        """
        Convert DictConfig to dict for `DecoderConfig`.

        Returns
        -------
        dict
            Dictionary representation of DecoderConfig.
        """
        self.backbone.dictcfg2dict()
        self.full_connection.dictcfg2dict()
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, ListConfig):
                setattr(self, key, convert_dictconfig_to_dict(value))
            elif isinstance(value, (list, tuple, dict)):
                setattr(self, key, convert_dictconfig_to_dict(value))


