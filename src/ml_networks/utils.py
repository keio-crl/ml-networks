import random
from typing import Iterator

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import pytorch_optimizer
import schedulefree
from torch.utils.data import DataLoader, Dataset


def conv_out(h_in, padding, kernel_size, stride, dilation=1):
    return int((h_in + 2.0 * padding - dilation*(kernel_size - 1.0) - 1.0) / stride + 1.0)

def conv_transpose_out(h_in, padding, kernel_size, stride, dilation=1, output_padding=0):
    return (h_in - 1) * stride - 2 * padding + dilation*(kernel_size - 1) + output_padding + 1

def conv_transpose_in(h_out, padding, kernel_size, stride, dilation=1, output_padding=0):
    return (h_out - output_padding - 1 + 2 * padding - dilation*(kernel_size - 1)) / stride + 1

def output_padding(h_in, conv_out, padding, kernel_size, stride, dilation=1):
    return h_in - (conv_out - 1) * stride + 2 * padding - dilation*(kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride, dilation=1):
    return tuple(conv_out(x, padding, kernel_size, stride, dilation) for x in h_in)

def conv_transpose_out_shape(h_in, padding, kernel_size, stride, dilation=1, output_padding=0):
    return tuple(conv_transpose_out(h_in[i], padding, kernel_size, stride, dilation, output_padding)
                 for i in range(len(h_in)))

def conv_transpose_in_shape(h_out, padding, kernel_size, stride, dilation=1, output_padding=0):
    return tuple(conv_transpose_in(h_out[i], padding, kernel_size, stride, dilation, output_padding)
                 for i in range(len(h_out)))

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride, dilation=1):
    return tuple(
        output_padding(h_in[i], conv_out[i], padding, kernel_size, stride, dilation)
        for i in range(len(h_in))
    )


def get_optimizer(
    param: Iterator[nn.Parameter], name: str, **kwargs
):
    if hasattr(schedulefree, name):
        optimizer = getattr(schedulefree, name)
    elif hasattr(torch.optim, name):
        optimizer = getattr(torch.optim, name)
    elif hasattr(pytorch_optimizer, name):
        optimizer = getattr(pytorch_optimizer, name)
    else:
        raise NotImplementedError(f"Optimizer {name} is not implemented in torch.optim or pytorch_optimizer, schedulefree. Please check the name and capitalization.")
    return optimizer(param, **kwargs)


class mytorch:
    @staticmethod
    def concat(tensor_list: list, dim=0):
        if None in tensor_list:
            return None
        return torch.cat(tensor_list, dim=dim)

    @staticmethod
    def stack(tensor_list: list, dim=0):
        if None in tensor_list:
            return None
        return torch.stack(tensor_list, dim=dim)

    @staticmethod
    @torch.jit.script
    def softmax(
        inputs: torch.Tensor, dim: int, temperature: torch.Tensor = torch.tensor(1.0)
    ):

        x = inputs - torch.max(inputs.detach(), dim=-1, keepdim=True)[0]
        x = x / temperature

        x = torch.softmax(x, dim=dim)

        if torch.isinf(x).any() or torch.isnan(x).any():
            print("inputs", inputs)
            print("result", x)
            raise ValueError("softmax is inf or nan")

        return x


# 明るさを変更する関数
def change_brightness(img: torch.Tensor, std: float = 0.1, max: float = 1.0, min: float = -1.0, clamp: bool = True):
    """

    画像(tensor)明るさを変更する関数
    M1のコードそのまま

    Args:
        img(np.ndarray): 元画像
        alpha(float): コントラスト
        beta(float): 明るさ

    Returns:
        torch.Tensor: 明るさを変えた画像

    """
    alpha = torch.normal(1.0, std, size=(1,))

    beta = torch.normal(0.0, std, size=(1,))

    bright_img = alpha * img + beta

    if clamp:
        bright_img = torch.clamp(bright_img, min, max)

    return bright_img


@torch.jit.script
def add_noise(
    input_data: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.1,
    max: float = 0.95,
    min: float = -0.95,
    clamp: bool = True,
):
    """関節角度にノイズを加える

    Args:
        angles(np.ndarray): 元関節データ
        alpha(float): ノイズ
        beta(float): バイアス

    Returns:
        torch.Tensor: ノイズを加えた関節データ

    """

    noise = torch.normal(mean, std, size=input_data.shape)

    data = input_data + noise
    if clamp:

        clip_data = torch.clamp(data, min, max)

    else:
        clip_data = data
    return clip_data


def determine_loader(
    data: Dataset, seed: int, batch_size: int, shuffle: bool = True, collate_fn=None
):
    g = torch.Generator()
    g.manual_seed(seed)
    if collate_fn is not None:
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=2,
            pin_memory=False,
            collate_fn=collate_fn,
        )
    else:
        loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=2,
            pin_memory=False,
        )
    return loader


def torch_fix_seed(seed=42):
    """乱数を固定する関数

    各行でやっていることは
        https://qiita.com/north_redwing/items/1e153139125d37829d2d
    などに詳細あり

    """

    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms = True


def seed_worker(worker_id):
    """

    DataLoaderのworkerの固定
    Dataloaderの乱数固定にはgeneratorの固定も必要らしい

    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


