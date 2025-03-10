import random
from collections.abc import Callable, Iterator

import numpy as np
import pytorch_lightning as pl
import pytorch_optimizer
import schedulefree
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


def conv_out(h_in, padding, kernel_size, stride, dilation=1):
    """
    Calculate the output size of convolutional layer.

    Args:
    -----
    h_in : int
        Input size.
    padding : int
        Padding size.
    kernel_size : int
        Kernel size.
    stride : int
        Stride size.
    dilation : int
        Dilation size. Default is 1.

    Returns
    -------
    int
        Output size.

    Examples
    --------
    >>> conv_out(32, 1, 3, 1)
    32
    >>> conv_out(32, 1, 3, 2)
    16
    >>> conv_out(32, 1, 3, 1, 2)
    30
    """
    return int((h_in + 2.0 * padding - dilation * (kernel_size - 1.0) - 1.0) / stride + 1.0)


def conv_transpose_out(h_in, padding, kernel_size, stride, dilation=1, output_padding=0):
    """
    Calculate the output size of transposed convolutional layer.

    Args:
    -----
    h_in : int
        Input size.
    padding : int
        Padding size.
    kernel_size : int
        Kernel size.
    stride : int
        Stride size.
    dilation : int
        Dilation size. Default is 1.
    output_padding : int
        Output padding size. Default is 0.

    Returns
    -------
    int
        Output size.

    Examples
    --------
    >>> conv_transpose_out(32, 1, 3, 1)
    32
    >>> conv_transpose_out(32, 1, 3, 2)
    63
    >>> conv_transpose_out(32, 1, 3, 1, 2)
    34
    >>> conv_transpose_out(32, 1, 3, 1, 1, 1)
    33


    """
    return (h_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


def conv_transpose_in(h_out, padding, kernel_size, stride, dilation=1, output_padding=0):
    """
    Calculate the input size of transposed convolutional layer.

    Args:
    -----
    h_out : int
        Output size.
    padding : int
        Padding size.
    kernel_size : int
        Kernel size.
    stride : int
        Stride size.
    dilation : int
        Dilation size. Default is 1.
    output_padding : int
        Output padding size. Default is 0.

    Returns
    -------
    int
        Input size.

    Examples
    --------
    >>> conv_transpose_in(32, 1, 3, 1)
    32
    >>> conv_transpose_in(32, 1, 3, 2)
    16
    >>> conv_transpose_in(32, 1, 3, 1, 2)
    30
    >>> conv_transpose_in(32, 1, 3, 1, 1, 1)
    31

    """
    return int((h_out - output_padding - 1 + 2 * padding - dilation * (kernel_size - 1)) / stride + 1)


def output_padding(h_in, h_out, padding, kernel_size, stride, dilation=1):
    """
    Calculate the output padding size of transposed convolutional layer.

    Args:
    -----
    h_in : int
        Input size.
    h_out : int
        Output size.
    padding : int
        Padding size.
    kernel_size : int
        Kernel size.
    stride : int
        Stride size.
    dilation : int
        Dilation size. Default is 1.

    Returns
    -------
    int
        Output padding size.

    Examples
    --------
    >>> output_padding(32, 32, 1, 3, 1)
    0
    >>> output_padding(32, 16, 1, 3, 2)
    1
    >>> output_padding(32, 30, 1, 3, 1, 2)
    0

    """
    return h_in - (h_out - 1) * stride + 2 * padding - dilation * (kernel_size - 1) - 1


def conv_out_shape(shape: tuple[int, ...], padding: int, kernel_size: int, stride: int, dilation: int = 1):
    """
    Calculate the output size of convolutional layer.

    Args:
    -----
    shape : tuple[int, ...]
        Input shape.
    padding : int
        Padding size.
    kernel_size : int
        Kernel size.
    stride : int
        Stride size.
    dilation : int
        Dilation size. Default is 1.

    Returns
    -------
    tuple[int, ...]
        Output shape.

    Examples
    --------
    >>> conv_out_shape((32, 32), 1, 3, 1)
    (32, 32)
    >>> conv_out_shape((32, 32), 1, 3, 2)
    (16, 16)


    """
    return tuple(conv_out(x, padding, kernel_size, stride, dilation) for x in shape)


def conv_transpose_out_shape(
    in_shape: tuple[int, ...], padding: int, kernel_size: int, stride: int, dilation: int = 1, output_padding: int = 0,
):
    """
    Calculate the output size of transposed convolutional layer.

    Args:
    -----
    in_shape : tuple[int, ...]
        Input shape.
    padding : int
        Padding size.
    kernel_size : int
        Kernel size.
    stride : int
        Stride size.
    dilation : int
        Dilation size. Default is 1.
    output_padding : int
        Output padding size. Default is 0.

    Returns
    -------
    tuple[int, ...]

    Examples
    --------
    >>> conv_transpose_out_shape((32, 32), 1, 3, 1)
    (32, 32)

    >>> conv_transpose_out_shape((32, 32), 1, 3, 2)
    (63, 63)

    >>> conv_transpose_out_shape((32, 32), 1, 3, 1, 2)
    (34, 34)

    >>> conv_transpose_out_shape((32, 32), 1, 3, 1, 1, 1)
    (33, 33)
    """
    return tuple(
        conv_transpose_out(in_shape[i], padding, kernel_size, stride, dilation, output_padding)
        for i in range(len(in_shape))
    )


def conv_transpose_in_shape(
    out_shape: tuple[int, ...],
    padding: int,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
    output_padding: int = 0,
) -> tuple[int, ...]:
    """
    Calculate the input size of transposed convolutional layer.

    Args:
    -----
    out_shape : tuple[int, ...]
        Output shape.
    padding : int
        Padding size.
    kernel_size : int
        Kernel size.
    stride : int
        Stride size.
    dilation : int
        Dilation size. Default is 1.
    output_padding : int
        Output padding size. Default is 0.

    Returns
    -------
    tuple[int, ...]

    Examples
    --------
    >>> conv_transpose_in_shape((32, 32), 1, 3, 1)
    (32, 32)

    >>> conv_transpose_in_shape((32, 32), 1, 3, 2)
    (16, 16)

    >>> conv_transpose_in_shape((32, 32), 1, 3, 1, 2)
    (30, 30)


    >>> conv_transpose_in_shape((32, 32), 1, 3, 1, 1, 1)
    (31, 31)
    """
    return tuple(
        conv_transpose_in(out_shape[i], padding, kernel_size, stride, dilation, output_padding)
        for i in range(len(out_shape))
    )


def output_padding_shape(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    padding: int,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
) -> tuple[int, ...]:
    """
    Calculate the output padding size of transposed convolutional layer.

    Args:
    -----
    in_shape : tuple[int, ...]
        Input shape.
    out_shape : tuple[int, ...]
        Output shape.
    padding : int
        Padding size.
    kernel_size : int
        Kernel size.
    stride : int
        Stride size.
    dilation : int
        Dilation size. Default is 1.

    Returns
    -------
    tuple[int, ...]

    Examples
    --------
    >>> output_padding_shape((32, 32), (32, 32), 1, 3, 1)
    (0, 0)

    >>> output_padding_shape((32, 32), (16, 16), 1, 3, 2)
    (1, 1)

    >>> output_padding_shape((32, 32), (30, 30), 1, 3, 1, 2)
    (0, 0)
    """
    return tuple(
        output_padding(in_shape[i], out_shape[i], padding, kernel_size, stride, dilation) for i in range(len(in_shape))
    )


def get_optimizer(
    param: Iterator[nn.Parameter],
    name: str,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Get optimizer from torch.optim or pytorch_optimizer.

    Args:
    -----
    param : Iterator[nn.Parameter]
        Parameters of models to optimize.
    name : str
        Optimizer name.
    kwargs : dict
        Optimizer arguments(settings).

    Returns
    -------
    torch.optim.Optimizer

    Examples
    --------
    >>> get_optimizer([nn.Parameter(torch.randn(1, 3))], "Adam", lr=0.01)
    Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        capturable: False
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 0.01
        maximize: False
        weight_decay: 0
    )
    """
    if hasattr(schedulefree, name):
        optimizer = getattr(schedulefree, name)
    elif hasattr(torch.optim, name):
        optimizer = getattr(torch.optim, name)
    elif hasattr(pytorch_optimizer, name):
        optimizer = getattr(pytorch_optimizer, name)
    else:
        msg = f"Optimizer {name} is not implemented in torch.optim or pytorch_optimizer, schedulefree. "
        msg += "Please check the name and capitalization."
        raise NotImplementedError(msg)
    return optimizer(param, **kwargs)


class mytorch:
    @staticmethod
    @torch.jit.script
    def softmax(
        inputs: torch.Tensor,
        dim: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Softmax function with temperature. This prevents overflow and underflow.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.
        dim : int
            Dimension to apply softmax.
        temperature : float
            Temperature. Default is 1.0.

        Returns
        -------
        torch.Tensor
            Softmaxed tensor.
        """
        x = inputs - torch.max(inputs.detach(), dim=-1, keepdim=True)[0]
        x = x / temperature
        x = torch.softmax(x, dim=dim)
        if torch.isinf(x).any() or torch.isnan(x).any():
            raise ValueError("softmax is inf or nan")
        return x

    @staticmethod
    def gumbel_softmax(
        inputs: torch.Tensor,
        dim: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Gumbel softmax function with temperature. This prevents overflow and underflow.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.
        dim : int
            Dimension to apply softmax.
        temperature : float
            Temperature. Default is 1.0.

        Returns
        -------
        torch.Tensor
            Gumbel softmaxed tensor.
        """
        x = inputs - torch.max(inputs.detach(), dim=-1, keepdim=True)[0]
        x = F.gumbel_softmax(x, dim=dim, tau=temperature, hard=True)
        if torch.isinf(x).any() or torch.isnan(x).any():
            raise ValueError("gumbel_softmax is inf or nan")
        return x


def determine_loader(
    data: Dataset,
    seed: int,
    batch_size: int,
    shuffle: bool = True,
    collate_fn: Callable | None = None,
) -> DataLoader:
    """
    Determine DataLoader with fixed seed.

    Args:
    -----
    data : Dataset
        Dataset to load.
    seed : int
        Random seed.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle data. Default is True.
    collate_fn : callable
        Collate function. Default is None.

    Returns
    -------
    DataLoader
        DataLoader with fixed seed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_fn,
    )


def torch_fix_seed(seed: int = 42) -> None:
    """
    乱数を固定する関数.

    References
    ----------
    - https://qiita.com/north_redwing/items/1e153139125d37829d2d
    """
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id: int) -> None:
    """
    DataLoaderのworkerの固定.

    Dataloaderの乱数固定にはgeneratorの固定も必要らしい
    """
    worker_seed = torch.initial_seed() % 2**32
    pl.seed_everything(worker_seed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    # print
