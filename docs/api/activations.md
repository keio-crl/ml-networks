# 活性化関数

カスタム活性化関数を提供します。

`ml_networks.torch.activations`（PyTorch）と`ml_networks.jax.activations`（JAX）の両方で提供されています。

PyTorchに実装されている活性化関数に加えて、以下のカスタム活性化関数が使えます。

## Activation

::: ml_networks.torch.activations.Activation

## REReLU

Reparametrized ReLU: 逆伝播がGELU等になるReLU。See [paper](https://openreview.net/forum?id=lNCnZwcH5Z)

::: ml_networks.torch.activations.REReLU

## SiGLU

SiLU + GLU: SiLU(Swish)とGLUを組み合わせた活性化関数。See [paper](https://arxiv.org/abs/2102.11972v2)

::: ml_networks.torch.activations.SiGLU

## CRReLU

Correction Regularized ReLU: 正則化されたReLU。See [paper](https://openreview.net/forum?id=7TZYM6Hm9p)

::: ml_networks.torch.activations.CRReLU

## TanhExp

Mishの改善版という位置付け。See [article](https://qiita.com/kuroitu/items/73cd401afd463a78115a)

::: ml_networks.torch.activations.TanhExp

## L2Norm

L2正規化レイヤー。特徴量を単位超球上に射影します。

::: ml_networks.torch.activations.L2Norm
