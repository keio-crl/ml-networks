"""設定を扱うモジュール."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from omegaconf import ListConfig, OmegaConf, dictConfig


def load_config(path: str) -> dictConfig | ListConfig:
    """
    Convert model config `.yaml` to `dictconfig` with custom resolvers.

    Returns
    -------
    dictConfig
    """
    return OmegaConf.load(Path(path))


def convert_dictconfig_to_dict(obj: Any) -> Any:
    """Recursively convert dictConfig to dict in any iterable object.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        Converted object.
    """
    if isinstance(obj, dictConfig):
        return dict(obj)
    if isinstance(obj, list | tuple):
        return type(obj)(convert_dictconfig_to_dict(item) for item in obj)
    if isinstance(obj, dict):
        return {k: convert_dictconfig_to_dict(v) for k, v in obj.items()}
    return obj


@dataclass
class ContrastiveLearningConfig:
    """Contrastive learning configuration."""

    dim_feature: int
    eval_func: MLPConfig
    dim_input2: int | None = None
    cross_entropy_like: bool = False


@dataclass
class AttentionConfig:
    """
    Attention configuration.

    Attributes
    ----------
    nhead : int
        Number of heads.
    patch_size : int
        Patch size.
    """

    nhead: int
    patch_size: int


@dataclass
class SoftmaxTransConfig:
    """
    Softmax transformation configuration.

    Attributes
    ----------
    vector : int
        Vector size.
    sigma : float
        Sigma value.
    n_ignore : int
        Number of ignored elements. Default is 0.
    max : float
        Maximum value. Default is 1.0.
    min : float
        Minimum value. Default is -1.0.
    """

    vector: int
    sigma: float
    n_ignore: int = 0
    max: float = 1.0
    min: float = -1.0


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
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"
    dropout: float = 0.0
    norm: Literal["batch", "group", "none"] = "none"
    norm_cfg: dict[str, Any] = field(default_factory=dict)
    norm_first: bool = False
    scale_factor: int = 0

    def __post_init__(self) -> None:
        """Set `.norm_cfg`."""
        if self.norm == "none":
            self.norm_cfg = {}
        else:
            self.norm_cfg = dict(**self.norm_cfg)

    def dictcfg2dict(self) -> None:
        """Convert dictConfig to dict for `ConvConfig`."""
        self.norm_cfg = dict(self.norm_cfg)

        for key, value in self.__dict__.items():
            if isinstance(value, dictConfig | ListConfig | list | tuple | dict):
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
    attention: AttentionConfig | None = None
    init_channel: int = 16

    def __post_init__(self) -> None:
        """Set `channels` and `conv_cfgs` as tuple."""
        self.conv_cfgs = tuple(self.conv_cfgs)
        self.channels = tuple(self.channels)

    def dictcfg2dict(self) -> None:
        """Convert dictConfig to dict for `ConvNetConfig`."""
        self.channels = tuple(self.channels)
        conv_cfgs = []
        for cfg_item in self.conv_cfgs:
            conv_cfg = cfg_item
            if isinstance(conv_cfg, dictConfig):
                conv_cfg = ConvConfig(**conv_cfg)
            conv_cfg.dictcfg2dict()
            conv_cfgs.append(conv_cfg)
        self.conv_cfgs = tuple(conv_cfgs)
        for key, value in self.__dict__.items():
            if isinstance(value, dictConfig | ListConfig | list | tuple | dict):
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
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros"
    attention: AttentionConfig | None = None

    def __post_init__(self) -> None:
        """Set `norm_cfg`."""
        if self.norm == "none":
            self.norm_cfg = {}
        else:
            self.norm_cfg = dict(**self.norm_cfg)

    def dictcfg2dict(self) -> None:
        """Convert dictConfig to dict for `ResNetConfig`."""
        self.norm_cfg = dict(self.norm_cfg)
        for key, value in self.__dict__.items():
            if isinstance(value, dictConfig | ListConfig | list | tuple | dict):
                setattr(self, key, convert_dictconfig_to_dict(value))


@dataclass
class UNetConfig:
    """
    UNet configuration.

    Attributes
    ----------
    channels : Tuple[int, ...]
        Number of channels for each layer.
    conv_cfg : ConvConfig
        Convolutional layer configuration.
    cond_cfg : MLPConfig
        Conditional configuration for UNet.
    cond_pred_scale : bool
        Whether to scale the conditional prediction. Default is False.
    nhead : Optional[int]
        Number of heads for attention mechanism. If it's set to None, attention is not applied.
        Default is None.
    has_attn : bool
        Whether to use attention mechanism. Default is False.
    use_shuffle : bool
        Whether to use PixelShuffle or PixelUnshuffle. Default is False.
    use_hypernet : bool
        Whether to use hypernetwork. Default is False.
    hyper_mlp_cfg : Optional[MLPConfig]
        Hypernetwork configuration. If it's set to None, hypernetwork is not used.
        Default is None.
    """

    channels: tuple[int, ...]
    conv_cfg: ConvConfig
    cond_pred_scale: bool = False
    nhead: int | None = None
    has_attn: bool = False
    use_shuffle: bool = False
    use_hypernet: bool = False
    hyper_mlp_cfg: MLPConfig | None = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.has_attn:
            assert self.nhead is not None, "nhead must be specified when has_attn is True."
        if isinstance(self.channels, list | ListConfig):
            self.channels = tuple(self.channels)


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

    def dictcfg2dict(self) -> None:
        """Convert dictConfig to dict for `LinearConfig`."""
        self.norm_cfg = dict(self.norm_cfg)
        for key, value in self.__dict__.items():
            if isinstance(value, dictConfig | ListConfig | list | tuple | dict):
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

    def dictcfg2dict(self) -> None:
        """Convert dictConfig to dict for `MLPConfig`."""
        self.linear_cfg.dictcfg2dict()
        for key, value in self.__dict__.items():
            if isinstance(value, dictConfig | ListConfig | list | tuple | dict):
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
    unpatchify: bool = False


@dataclass
class AdaptiveAveragePoolingConfig:
    """
    Adaptive average pooling configuration.

    Attributes
    ----------
    output_size : Union[int, Tuple[int, ...]]
        Output size of the pooling layer. If it's an integer, it will be used for both height and width.
        If it's a tuple, it should contain two integers for height and width.
    """

    output_size: int | tuple[int, ...] = (1, 1)
    additional_layer: (MLPConfig | LinearConfig) | None = None

    def __post_init__(self) -> None:
        """Ensure output_size is a tuple."""
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        elif isinstance(self.output_size, list | ListConfig):
            self.output_size = tuple(self.output_size)


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
    eps : float
        Epsilon value for numerical stability in softmax. Default is 1e-6.
    is_argmax : bool
        Whether to use argmax instead of softmax. Default is False.
    is_straight_through : bool
        Whether to use straight-through estimator for backpropagation. Default is False.
    additional_layer : Optional[Union[MLPConfig, LinearConfig]]
        Additional layer configuration. If it's set to None, no additional layer is applied.
        Default is None.
    """

    temperature: float = 1.0
    eps: float = 1e-6
    is_argmax: bool = False
    is_straight_through: bool = False
    additional_layer: (MLPConfig | LinearConfig) | None = None


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

    backbone: ConvNetConfig | ResNetConfig
    full_connection: MLPConfig | LinearConfig | SpatialSoftmaxConfig

    def dictcfg2dict(self) -> None:
        """Convert dictConfig to dict for `EncoderConfig`."""
        if hasattr(self.backbone, "dictcfg2dict"):
            self.backbone.dictcfg2dict()
        if hasattr(self.full_connection, "dictcfg2dict"):
            self.full_connection.dictcfg2dict()
        for key, value in self.__dict__.items():
            if isinstance(value, dictConfig | ListConfig | list | tuple | dict):
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

    backbone: ConvNetConfig | ResNetConfig
    full_connection: MLPConfig | LinearConfig

    def dictcfg2dict(self) -> None:
        """Convert dictConfig to dict for `DecoderConfig`."""
        self.backbone.dictcfg2dict()
        if hasattr(self.full_connection, "dictcfg2dict"):
            self.full_connection.dictcfg2dict()
        for key, value in self.__dict__.items():
            if isinstance(value, dictConfig | ListConfig | list | tuple | dict):
                setattr(self, key, convert_dictconfig_to_dict(value))
