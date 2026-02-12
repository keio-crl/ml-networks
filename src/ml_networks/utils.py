"""PyTorch 非依存 (の一部も含む) ユーティリティ関数を扱うモジュール."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import blosc2  # type: ignore[import-not-found]
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
import pytorch_lightning as pl  # type: ignore[import-untyped]
import torch
from torch.utils.data import DataLoader, Dataset


def save_blosc2(path: str, x: np.ndarray) -> None:
    """Save numpy array with blosc2 compression.

    Args:
    -----
    path : str
        Path to save.
    x : np.ndarray
        Numpy array to save.

    Examples
    --------
    >>> save_blosc2("test.blosc2", np.random.randn(10, 10))

    """
    Path(path).write_bytes(blosc2.pack_array2(x))


def load_blosc2(path: str) -> np.ndarray:
    """Load numpy array with blosc2 compression.

    Args:
    -----
    path : str
        Path to load.

    Returns
    -------
    np.ndarray
        Numpy array.

    Examples
    --------
    >>> data = load_blosc2("test.blosc2")
    >>> type(data)
    <class 'numpy.ndarray'>
    """
    return blosc2.unpack_array2(Path(path).read_bytes())


def conv_out(h_in: int, padding: int, kernel_size: int, stride: int, dilation: int = 1) -> int:
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


def conv_transpose_out(
    h_in: int,
    padding: int,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
    output_padding: int = 0,
) -> int:
    """
    Calculate the output size of transposed convolutional layer.

    Parameters
    ----------
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


def conv_transpose_in(
    h_out: int,
    padding: int,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
    output_padding: int = 0,
) -> int:
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


def output_padding(h_in: int, h_out: int, padding: int, kernel_size: int, stride: int, dilation: int = 1) -> int:
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


def conv_out_shape(
    shape: tuple[int, ...],
    padding: int,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
) -> tuple[int, ...]:
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
    in_shape: tuple[int, ...],
    padding: int,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
    output_padding: int = 0,
) -> tuple[int, ...]:
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
    g = torch.Generator()  # type: ignore[attr-defined]
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
