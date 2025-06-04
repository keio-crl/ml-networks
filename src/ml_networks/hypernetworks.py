import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from ml_networks.layers import MLPLayer
from ml_networks.config import MLPConfig, LinearConfig
from torch.nn import init

Shape = Tuple[int, ...]

InputMode = Literal[None, "cos|sin", "z|1-z"]

encoding_multiplier: Dict[InputMode, int] = {
    None: 1,
    "cos|sin": 2,
    "z|1-z": 2,
}


def _same_keys(
    required: Dict[str, Any], provided: Dict[str, Any], name: str = ""
) -> None:
    """
    Checks if two dictionaries have the same keys.

    Args:
        required: A dictionary with required keys.
        provided: A dictionary with provided keys.
        name: A string name to identify the dictionaries.

    Raises:
        ValueError: If the dictionaries have missing or unexpected keys.

    Examples
    --------
    >>> required = {"a": 1, "b": 2}
    >>> provided = {"a": 1, "b": 2}
    >>> _same_keys(required, provided)
    >>> provided = {"a": 1}
    >>> _same_keys(required, provided)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Missing : ['b']
    """
    missing = required.keys() - provided.keys()
    if len(missing) > 0:
        raise ValueError(f"Missing {name}: {list(missing)}")

    unexpected = provided.keys() - required.keys()
    if len(unexpected) > 0:
        raise ValueError(f"Unexpected {name}: {list(missing)}")


class HyperNetMixin:

    input_shapes: Dict[str, Shape]
    output_shapes: Dict[str, Shape]
    encoding: InputMode

    def encode_inputs(self, inputs: Dict[str, Shape]) -> Dict[str, Shape]:
        """Encodes the inputs using the specified `encoding` mode.

        Args:
            inputs: A dictionary of input names and shapes.

        Returns:
            Dict[str, Shape]: A dictionary of encoded input names and shapes.

        Examples
        --------
        >>> class TestHyperNet(HyperNetMixin):
        ...     def __init__(self):
        ...         self.input_shapes = {"x": (2,)}
        ...         self.encoding = "cos|sin"
        >>> net = TestHyperNet()
        >>> inputs = {"x": torch.tensor([[0.5, 0.3]])}
        >>> encoded = net.encode_inputs(inputs)
        >>> encoded["x"].shape
        torch.Size([1, 4])
        """
        return {k: encode_input(x, mode=self.encoding) for k, x in inputs.items()}

    def flat_input_size(self) -> int:
        """Returns the flat input size after encoding.

        Returns:
            int: The flat input size.

        Examples
        --------
        >>> class TestHyperNet(HyperNetMixin):
        ...     def __init__(self):
        ...         self.input_shapes = {"x": (2,), "y": (3,)}
        ...         self.encoding = "cos|sin"
        >>> net = TestHyperNet()
        >>> net.flat_input_size()
        10
        """
        flat_input_size = sum(int(np.prod(v))
                              for v in self.input_shapes.values())
        flat_input_size *= encoding_multiplier[self.encoding]
        return flat_input_size

    def flat_output_size(self) -> int:
        """Returns the flat output size.

        Returns:
            int: The flat output size.

        Examples
        --------
        >>> class TestHyperNet(HyperNetMixin):
        ...     def __init__(self):
        ...         self.output_shapes = {"x": (2,), "y": (3,)}
        >>> net = TestHyperNet()
        >>> net.flat_output_size()
        5
        """
        flat_output_size = sum(int(np.prod(v))
                               for v in self.output_shapes.values())
        return flat_output_size

    def output_offsets(self) -> Dict[str, Tuple[int, int]]:
        """Returns the output offsets.

        Returns:
            Dict[str, Tuple[int, int]]: A dictionary of output names and 
            their corresponding offsets and sizes.

        Examples
        --------
        >>> class TestHyperNet(HyperNetMixin):
        ...     def __init__(self):
        ...         self.output_shapes = {"x": (2,), "y": (3,)}
        >>> net = TestHyperNet()
        >>> offsets = net.output_offsets()
        >>> offsets["x"]
        (0, 2)
        >>> offsets["y"]
        (2, 3)
        """
        offset = 0
        offsets = {}
        for name, shape in self.output_shapes.items():
            size = int(np.prod(shape))
            offsets[name] = (offset, size)
            offset += size
        return offsets

    def _validate_inputs(self, inputs: Dict[str, torch.Tensor]) -> int:
        """
        Validates the inputs.

        Args:
            inputs (Dict[str, Tensor]): A dictionary of input names and their corresponding tensors

        Returns:
            int: The batch dimension.

        Raises:
            ValueError: If the input shapes are not as expected or if different batch dimensions are found.

        Examples
        --------
        >>> class TestHyperNet(HyperNetMixin):
        ...     def __init__(self):
        ...         self.input_shapes = {"x": (2,), "y": (3,)}
        >>> net = TestHyperNet()
        >>> inputs = {"x": torch.randn(1, 2), "y": torch.randn(1, 3)}
        >>> net._validate_inputs(inputs)
        1
        >>> inputs = {"x": torch.randn(2,), "y": torch.randn(3,)}
        >>> net._validate_inputs(inputs)
        1
        """
        _same_keys(self.input_shapes, inputs, name="inputs")

        for name, required_shape in self.input_shapes.items():

            shape = inputs[name].shape

            # If batch dimension is already provided
            if shape[1:] == required_shape:
                continue

            # If no batch dimension is provided, but shape is correct
            if shape == required_shape:
                # Add batch dimension
                inputs[name] = inputs[name][None]
                continue

            error_msg = f"Wrong shape for {name}, expected {required_shape} or {('B',*required_shape)}, got {shape}"
            raise ValueError(error_msg)

        # All batch dims must match
        batch_dims = [x.shape[0] for x in inputs.values()]
        if len(set(batch_dims)) > 1:
            raise ValueError(f"Multiple batch dimensions found: {batch_dims}")

        return batch_dims[0]

    def flatten_inputs(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flattens the input tensors into a single tensor along the last dimension.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary of input tensor names and tensor values.

        Returns:
            Tensor: The flattened input tensor.

        Examples
        --------
        >>> class TestHyperNet(HyperNetMixin):
        ...     def __init__(self):
        ...         self.input_shapes = {"x": (2,), "y": (3,)}
        >>> net = TestHyperNet()
        >>> inputs = {"x": torch.randn(1, 2), "y": torch.randn(1, 3)}
        >>> flat = net.flatten_inputs(inputs)
        >>> flat.shape
        torch.Size([1, 5])
        """
        batch_dim = self._validate_inputs(inputs)

        flat_input = torch.cat(
            [inputs[hp].view(batch_dim, -1) for hp in sorted(self.input_shapes)], dim=-1
        )
        return flat_input

    def unflatten_output(self, flat_output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Unflattens the output tensor into a dictionary of named tensors.

        Args:
            flat_output (Tensor): The flattened output tensor.

        Returns:
            Dict[str, Tensor]: A dictionary of named tensors.

        Examples
        --------
        >>> class TestHyperNet(HyperNetMixin):
        ...     def __init__(self):
        ...         self.output_shapes = {"x": (2,), "y": (3,)}
        ...         self._output_offsets = {"x": (0, 2), "y": (2, 3)}
        >>> net = TestHyperNet()
        >>> flat_output = torch.randn(1, 5)
        >>> outputs = net.unflatten_output(flat_output)
        >>> outputs["x"].shape
        torch.Size([1, 2])
        >>> outputs["y"].shape
        torch.Size([1, 3])
        """
        outputs = {}
        batch_dim = flat_output.shape[0]
        for name, shape in self.output_shapes.items():
            shape = (batch_dim, *shape)
            outputs[name] = flat_output.narrow(1, *self._output_offsets[name]).view(shape)
        return outputs


class HyperNet(pl.LightningModule, HyperNetMixin):
    """A hypernetwork that generates weights for a target network. Shape is Dict[str, Tuple[int, ...]]

    Args:
        input_dim: The dimension of the input.
        output_shapes: The shapes of the primary network weights being predicted.
        mlp_cfg: Configuration for the MLP backbone.
        encoding: The input encoding mode. Defaults to None.

    Examples
    --------
    >>> from ml_networks.config import MLPConfig, LinearConfig
    >>> input_dim = 10
    >>> output_shapes = {"weight": (5, 10), "bias": (5,)}
    >>> mlp_cfg = MLPConfig(
    ...     hidden_dim=64,
    ...     n_layers=2,
    ...     output_activation="ReLU",
    ...     linear_cfg=LinearConfig(
    ...         activation="ReLU",
    ...         norm="none",
    ...         dropout=0.0,
    ...         bias=True
    ...     )
    ... )
    >>> net = HyperNet(input_dim, output_shapes, mlp_cfg)
    >>> x = torch.randn(2, input_dim)
    >>> outputs = net(x)
    >>> outputs["weight"].shape
    torch.Size([2, 5, 10])
    >>> outputs["bias"].shape
    torch.Size([2, 5])
    """

    def __init__(
        self,
        input_dim: int,
        output_shapes: Dict[str, Shape],
        mlp_cfg: MLPConfig,
        encoding=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_shapes = output_shapes
        self.encoding = encoding

        # Cache this property to avoid recomputation
        self._output_offsets = self.output_offsets()

        self.backbone = MLPLayer(
            self.input_dim,
            self.flat_output_size(),
            mlp_cfg
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Performs a forward pass of the neural network.

        Args:
            inputs: The input tensors.

        Returns:
            A dictionary of output tensors.

        Examples
        --------
        >>> from ml_networks.config import MLPConfig, LinearConfig
        >>> input_dim = 10
        >>> output_shapes = {"weight": (5, 10), "bias": (5,)}
        >>> mlp_cfg = MLPConfig(
        ...     hidden_dim=64,
        ...     n_layers=2,
        ...     output_activation="ReLU",
        ...     linear_cfg=LinearConfig(
        ...         activation="ReLU",
        ...         norm="none",
        ...         dropout=0.0,
        ...         bias=True
        ...     )
        ... )
        >>> net = HyperNet(input_dim, output_shapes, mlp_cfg)
        >>> x = torch.randn(2, input_dim)
        >>> outputs = net(x)
        >>> outputs["weight"].shape
        torch.Size([2, 5, 10])
        >>> outputs["bias"].shape
        torch.Size([2, 5])
        """

        inputs = encode_input(inputs, self.encoding)

        flat_output = self.backbone(inputs)
        outputs = self.unflatten_output(flat_output)
        return outputs


def encode_input(input: torch.Tensor, mode: str = "cos|sin") -> torch.Tensor:
    """
    Encodes the input tensor based on the specified mode.

    Args:
        input (Tensor): The input tensor.
        mode (InputMode): The encoding mode.

    Returns:
        Tensor: The encoded tensor.

    Raises:
        UserWarning: If the input tensor is outside the specified range in cos|sin mode.

    Examples
    --------
    >>> x = torch.tensor([[0.5, 0.3], [0.7, 0.9]])
    >>> encoded = encode_input(x, "cos|sin")
    >>> encoded.shape
    torch.Size([2, 4])
    >>> x = torch.tensor([[0.5, 0.3], [0.7, 0.9]])
    >>> encoded = encode_input(x, "z|1-z")
    >>> encoded.shape
    torch.Size([2, 4])
    >>> x = torch.tensor([[0.5, 0.3], [0.7, 0.9]])
    >>> encoded = encode_input(x, None)
    >>> encoded.shape
    torch.Size([2, 2])
    """
    z = input

    if mode is None:
        return input

    if mode == "cos|sin":
        _check_tensor_range(input, low=0, high=1)
        scaled_z = torch.pi * z / 2

        return torch.cat([torch.cos(scaled_z), torch.sin(scaled_z)], dim=-1)

    if mode == "z|1-z":
        return torch.cat([z, 1 - z], dim=-1)


def _check_tensor_range(tensor: torch.Tensor, low: float, high: float) -> None:
    """
    Check if the input tensor is within specified range.

    Args:
        tensor (Tensor): The input tensor.
        low (float): The lower bound of the range.
        high (float): The upper bound of the range.

    Raises:
        UserWarning: If the input tensor is outside the specified range.

    Examples
    --------
    >>> x = torch.tensor([[0.5, 0.3], [0.7, 0.9]])
    >>> _check_tensor_range(x, 0, 1)  # No warning
    >>> x = torch.tensor([[0.5, 0.3], [0.7, 1.1]])
    """
    if (tensor < low).any() or (tensor > high).any():
        warnings.warn(
            f"Input tensor has dimensions outside of [{low},{high}].")


def initialize_weight(
    weight: torch.Tensor,
    distribution: Optional[str],
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """
    Initialize weight tensor using the specified distribution and nonlinearity function.

    Args:
        weight (torch.Tensor): Tensor to be initialized.
        distribution (str, optional): Distribution to use for initialization.
        nonlinearity (str, optional): Nonlinearity function to use. Defaults to "LeakyReLU".

    Raises:
        ValueError: When the specified distribution is not supported.

    Examples
    --------
    >>> weight = torch.empty(2, 3)
    >>> initialize_weight(weight, "kaiming_normal")
    >>> weight = torch.empty(2, 3)
    >>> initialize_weight(weight, "zeros")
    >>> weight = torch.empty(2, 3)
    >>> initialize_weight(weight, "invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Unsupported weight distribution 'invalid'
    """
    if distribution is None:
        return

    if nonlinearity:
        nonlinearity = nonlinearity.lower()
        if nonlinearity == "leakyrelu":
            nonlinearity = "leaky_relu"

    if nonlinearity is None:
        nonlinearity = "linear"

    if nonlinearity in ("silu", "gelu", "mish", "tanhexp", "elu"):
        nonlinearity = "leaky_relu"

    gain = 1 if nonlinearity is None else init.calculate_gain(nonlinearity)

    if distribution == "zeros":
        init.zeros_(weight)
    elif distribution == "kaiming_normal":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_uniform":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_normal_fanout":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "kaiming_uniform_fanout":
        init.kaiming_uniform_(
            weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "glorot_normal":
        init.xavier_normal_(weight, gain=gain)
    elif distribution == "glorot_uniform":
        init.xavier_uniform_(weight, gain)
    elif distribution == "orthogonal":
        init.orthogonal_(weight, gain)
    else:
        raise ValueError(f"Unsupported weight distribution '{distribution}'")


def initialize_bias(bias: torch.Tensor, distribution: Optional[float] = 0.0) -> None:
    """
    Initializes the bias tensor of a layer using the given distribution.

    Args:
        bias (nn.Parameter): the bias tensor to be initialized
        distribution (float): the distribution to use for initialization, default is 0 (constant)

    Raises:
        ValueError: When the specified distribution is not supported.

    Examples
    --------
    >>> bias = torch.empty(3)
    >>> initialize_bias(bias, 0.1)
    >>> bias = torch.empty(3)
    >>> initialize_bias(bias, "invalid")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Unsupported bias distribution 'invalid'
    """
    if distribution is None:
        return

    if isinstance(distribution, (int, float)):
        init.constant_(bias, distribution)
        return

    raise ValueError(f"Unsupported bias distribution '{distribution}'")


def initialize_layer(
    layer: nn.Module,
    distribution: Optional[str] = "kaiming_normal",
    init_bias: Optional[float] = 0.0,
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """
    Initializes the weight and bias tensors of a linear or convolutional layer using the given distribution.

    Args:
    - layer (nn.Module): the linear or convolutional layer to be initialized
    - distribution (str): the distribution to use for initialization, default is 'kaiming_normal'
    - init_bias (float): the initial value of the bias tensor, default is 0
    - nonlinearity (str): the nonlinearity function to use for initialization, default is "LeakyReLU"

    Returns:
    - None

    Examples
    --------
    >>> layer = nn.Linear(2, 3)
    >>> initialize_layer(layer, "kaiming_normal", 0.1)
    >>> layer = nn.Conv2d(2, 3, 3)
    >>> initialize_layer(layer, "kaiming_normal", 0.1)
    >>> layer = nn.BatchNorm2d(3)  
    >>> initialize_layer(layer, "kaiming_normal", 0.1)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    AssertionError: Can only be applied to linear and conv layers, given BatchNorm2d
    """
    assert isinstance(
        layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ), f"Can only be applied to linear and conv layers, given {layer.__class__.__name__}"

    initialize_weight(layer.weight, distribution, nonlinearity)
    if layer.bias is not None:
        initialize_bias(layer.bias, init_bias)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
