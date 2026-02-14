# UNetガイド

条件付きUNet（Conditional UNet）の使用方法を説明します。

!!! tip "ベストプラクティス"
    YAMLファイルから設定を読み込む方法については、[設定管理ガイド](config-management.md)を参照してください。

## 概要

`ml-networks`では、Diffusion Modelなどで広く使われる条件付きUNetを提供しています。2Dバージョン（画像用）と1Dバージョン（時系列用）の両方に対応しています。

主な特徴：

- **条件付き生成**: 特徴量ベクトルを条件として画像/時系列を生成
- **Attention機構**: セルフアテンションによる長距離依存性のモデリング
- **残差接続**: スキップ接続による情報の保持
- **HyperNetwork対応**: 重みの動的生成（オプション）

## ConditionalUnet2d（2D画像用）

### 基本的な使用方法

#### 方法1: YAMLファイルから読み込む（推奨）

設定ファイル `configs/unet2d_config.yaml` を作成します：

```yaml
_target_: ml_networks.torch.unet.ConditionalUnet2d
feature_dim: 32
obs_shape: [3, 64, 64]
cfg:
  _target_: ml_networks.config.UNetConfig
  channels: [64, 128, 256]
  conv_cfg:
    _target_: ml_networks.config.ConvConfig
    kernel_size: 3
    padding: 1
    stride: 1
    groups: 1
    activation: ReLU
    dropout: 0.0
  has_attn: true
  nhead: 8
  cond_pred_scale: true
```

Pythonコード：

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

cfg = OmegaConf.load("configs/unet2d_config.yaml")
net = instantiate(cfg)

# 入力: ノイズ画像 + 条件ベクトル
x = torch.randn(2, 3, 64, 64)      # (batch, channels, height, width)
cond = torch.randn(2, 32)            # (batch, feature_dim)
out = net(x, cond)
print(out.shape)  # torch.Size([2, 3, 64, 64])
```

#### 方法2: Pythonコードで直接設定する

```python
from ml_networks.torch import ConditionalUnet2d
from ml_networks import UNetConfig, ConvConfig
import torch

cfg = UNetConfig(
    channels=[64, 128, 256],       # 各解像度レベルのチャンネル数
    conv_cfg=ConvConfig(
        kernel_size=3,
        padding=1,
        stride=1,
        groups=1,
        activation="ReLU",
        dropout=0.0,
    ),
    has_attn=True,                 # Attention機構を使用
    nhead=8,                       # Attentionのヘッド数
    cond_pred_scale=True,          # 条件付きスケーリング
)

net = ConditionalUnet2d(
    feature_dim=32,                # 条件ベクトルの次元
    obs_shape=(3, 64, 64),         # 画像の形状 (C, H, W)
    cfg=cfg,
)

# 推論
x = torch.randn(2, 3, 64, 64)
cond = torch.randn(2, 32)
out = net(x, cond)
print(out.shape)  # torch.Size([2, 3, 64, 64])
```

## ConditionalUnet1d（1D時系列用）

### 基本的な使用方法

```python
from ml_networks.torch import ConditionalUnet1d
from ml_networks import UNetConfig, ConvConfig
import torch

cfg = UNetConfig(
    channels=[64, 128, 256],
    conv_cfg=ConvConfig(
        kernel_size=3,
        padding=1,
        stride=1,
        groups=1,
        activation="ReLU",
        dropout=0.0,
    ),
    has_attn=False,
    cond_pred_scale=True,
)

net = ConditionalUnet1d(
    feature_dim=32,
    obs_shape=(8, 128),            # (チャンネル数, シーケンス長)
    cfg=cfg,
)

# 推論
x = torch.randn(2, 8, 128)        # (batch, channels, length)
cond = torch.randn(2, 32)
out = net(x, cond)
print(out.shape)  # torch.Size([2, 8, 128])
```

## UNetConfig の設定パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|---|----------|------|
| `channels` | `tuple[int, ...]` | - | 各解像度レベルのチャンネル数 |
| `conv_cfg` | `ConvConfig` | - | 畳み込み層の設定 |
| `cond_pred_scale` | `bool` | `False` | 条件付きスケーリングを使用するか |
| `nhead` | `int \| None` | `None` | Attentionのヘッド数 |
| `has_attn` | `bool` | `False` | Attention機構を使用するか |
| `use_shuffle` | `bool` | `False` | PixelShuffle/Unshuffleを使用するか |
| `use_hypernet` | `bool` | `False` | HyperNetworkを使用するか |
| `hyper_mlp_cfg` | `MLPConfig \| None` | `None` | HyperNetworkのMLP設定 |

!!! warning "`has_attn`を`True`にする場合"
    `has_attn=True`の場合、`nhead`を必ず指定してください。指定しないとアサーションエラーが発生します。

## Diffusion Modelでの使用例

条件付きUNetは、Diffusion Model（DDPM、DDIMなど）のノイズ予測ネットワークとして典型的に使用されます：

```python
from ml_networks.torch import ConditionalUnet2d
from ml_networks import UNetConfig, ConvConfig
import torch

# UNetの設定
cfg = UNetConfig(
    channels=[64, 128, 256, 512],
    conv_cfg=ConvConfig(
        kernel_size=3,
        padding=1,
        stride=1,
        activation="ReLU",
    ),
    has_attn=True,
    nhead=8,
    cond_pred_scale=True,
)

net = ConditionalUnet2d(
    feature_dim=256,
    obs_shape=(3, 64, 64),
    cfg=cfg,
)

# Diffusionの1ステップ
noisy_image = torch.randn(4, 3, 64, 64)   # ノイズが加えられた画像
condition = torch.randn(4, 256)             # 条件（テキスト埋め込みなど）

# ノイズ予測
predicted_noise = net(noisy_image, condition)
print(predicted_noise.shape)  # torch.Size([4, 3, 64, 64])
```

## PixelShuffle/Unshuffleの使用

ダウンサンプリング/アップサンプリングにPixelShuffle/Unshuffleを使用できます：

```yaml
cfg:
  _target_: ml_networks.config.UNetConfig
  channels: [64, 128, 256]
  conv_cfg:
    _target_: ml_networks.config.ConvConfig
    kernel_size: 3
    padding: 1
    stride: 1
    activation: ReLU
  use_shuffle: true       # PixelShuffle/Unshuffleを有効化
  has_attn: false
```

## HyperNetworkとの組み合わせ

UNetの重みをHyperNetworkで動的に生成することも可能です：

```yaml
cfg:
  _target_: ml_networks.config.UNetConfig
  channels: [64, 128]
  conv_cfg:
    _target_: ml_networks.config.ConvConfig
    kernel_size: 3
    padding: 1
    stride: 1
    activation: ReLU
  use_hypernet: true
  hyper_mlp_cfg:
    _target_: ml_networks.config.MLPConfig
    hidden_dim: 256
    n_layers: 2
    output_activation: Identity
    linear_cfg:
      _target_: ml_networks.config.LinearConfig
      activation: ReLU
      bias: true
```
