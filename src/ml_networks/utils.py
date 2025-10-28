from __future__ import annotations
import random
from collections.abc import Callable, Iterator

import pytorch_lightning as pl
import pytorch_optimizer
import schedulefree
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import blosc2
import numpy as np
from torchvision import transforms
from ml_networks.config import SoftmaxTransConfig
from einops import rearrange

class MinMaxNormalize(transforms.Normalize):
    def __init__(self, min: float, max: float, old_min: float = 0.0, old_max: float = 1.0) -> None:
        """
        MinMaxNormalizeの初期化.

        Args:
        -----
        min : float
            最小値.
        max : float
            最大値.
        old_min : float
            元の最小値.
        old_max : float
            元の最大値.


        """
        scale = (max - min) / (old_max - old_min)
        shift = min - old_min * scale          # new = scale·x + shift

        #   Normalize does (x − mean)/std  ⇒  x · (1/std)  − mean/std
        mean = -shift / scale                      # invert the affine form
        std  = 1.0 / scale

        super().__init__(mean=[mean]*3, std=[std]*3)
        self.min = min
        self.max = max



def save_blosc2(path: str, x: np.ndarray) -> None:
    """
    Save numpy array with blosc2 compression.

    Args:
    -----
    path : str
        Path to save.
    x : np.ndarray
        Numpy array to save.

    Returns
    -------
    None

    Examples
    --------
    >>> save_blosc2("test.blosc2", np.random.randn(10, 10))

    """
    with open(path, "wb") as f:
        f.write(blosc2.pack_array2(x))


def load_blosc2(path: str) -> np.ndarray:
    """
    Load numpy array with blosc2 compression.

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
    with open(path, "rb") as f:
        return blosc2.unpack_array2(f.read())

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


def get_optimizer(
    param: Iterator[nn.Parameter],
    name: str,
    **kwargs: float | str | bool,
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

    Raises
    ------
    ValueError
        If the softmax is inf or nan.
    """
    x = inputs / temperature
    x = torch.exp(F.log_softmax(x, dim=dim))
    if torch.isinf(x).any() or torch.isnan(x).any():
        msg = "softmax is inf or nan"
        raise ValueError(msg)
    return x


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

    Raises
    ------
    ValueError
        If the gumbel_softmax is inf or nan.
    """
    x = inputs - torch.max(inputs.detach(), dim=-1, keepdim=True)[0]
    x = F.gumbel_softmax(x, dim=dim, tau=temperature, hard=True)
    if torch.isinf(x).any() or torch.isnan(x).any():
        msg = "gumbel_softmax is inf or nan"
        raise ValueError(msg)
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
    pl.seed_everything(seed, workers=True)
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


class SoftmaxTransformation:
    def __init__(
        self, 
        cfg: SoftmaxTransConfig
        ):
        """
        SoftmaxTransformationの初期化.

        Args:
        -----
        cfg : SoftmaxTransConfig
            SoftmaxTransformationの設定.
        """

        super(SoftmaxTransformation, self).__init__()
        self.vector = cfg.vector
        self.sigma = cfg.sigma
        self.n_ignore = cfg.n_ignore
        self.max = cfg.max
        self.min = cfg.min
        self.k = torch.linspace(self.min, self.max, self.vector)


    def __call__(self, x: torch.Tensor):
        return self.transform(x)

    def get_transformed_dim(self, dim: int):
        return (dim - self.n_ignore) * self.vector + self.n_ignore

    def transform(self, x: torch.Tensor):
        """
        SoftmaxTransformationの実行.

        Args:
        -----
        x : torch.Tensor
            入力テンソル.

        Returns
        -------
        torch.Tensor
            出力テンソル.

        Examples
        --------
        >>> trans = SoftmaxTransformation(SoftmaxTransConfig(vector=16, sigma=0.01, n_ignore=1, min=-1.0, max=1.0))
        >>> x = torch.randn(2, 3, 4)
        >>> transformed = trans(x)
        >>> transformed.shape
        torch.Size([2, 3, 49])

        >>> trans = SoftmaxTransformation(SoftmaxTransConfig(vector=11, sigma=0.05, n_ignore=0, min=-1.0, max=1.0))
        >>> x = torch.randn(2, 3, 4)
        >>> transformed = trans(x)
        >>> transformed.shape
        torch.Size([2, 3, 44])

        """
        *batch, dim = x.shape
        x = x.reshape(-1, dim)
        if self.n_ignore:
            data, ignored = x[:, :-self.n_ignore], x[:, -self.n_ignore:]
        else:
            data = x

        negative = torch.stack(
            [torch.exp((-(data-self.k[v])**2)/self.sigma) for v in range(self.vector)])
        negative_sum = negative.sum(dim=0)
        
        transformed = negative/(negative_sum+1e-8)
        transformed = rearrange(transformed, 'v b d -> b (d v)')

        if self.n_ignore:
            transformed = torch.cat([transformed, ignored], dim=-1)
        else:
            transformed = transformed
        return transformed.reshape(*batch, self.get_transformed_dim(dim))

    def inverse(self, x: torch.Tensor):
        """
        SoftmaxTransformationの逆変換.

        Args:
        -----
        x : torch.Tensor
            入力テンソル.

        Returns
        -------
        torch.Tensor
            出力テンソル.

        """

        *batch, dim = x.shape
        x = x.reshape(-1, dim)
        if self.n_ignore:
            data, ignored = x[:, :-self.n_ignore], x[:, -self.n_ignore:]
        else:
            data = x

        data = data.reshape([len(data), -1, self.vector])

        data = rearrange(data, 'b d v -> v b d')

        data = torch.stack([data[v]*self.k[v] for v in range(self.vector)]).sum(dim=0)

        if self.n_ignore:
            data = torch.cat([data, ignored], dim=-1)
        else:
            data = data
        return data.reshape(*batch, -1)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
