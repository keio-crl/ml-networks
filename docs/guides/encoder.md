# Encoderガイド

Encoderの使用方法を説明します。

!!! tip "ベストプラクティス"
    YAMLファイルから設定を読み込む方法については、[設定管理ガイド](config-management.md)を参照してください。

## 概要

Encoderは画像などの入力を特徴量に変換するモジュールです。様々なバックボーン（ConvNet、ResNet、ViTなど）をサポートしています。

## 基本的な使用方法

### 方法1: YAMLファイルから読み込む（推奨）

設定ファイル `configs/encoder_config.yaml` を作成します：

```yaml
_target_: ml_networks.vision.Encoder
feature_dim: 64
obs_shape: [3, 64, 64]
encoder_cfg:
  _target_: ml_networks.config.ConvNetConfig
  channels: [16, 32, 64]
  conv_cfgs:
    - _target_: ml_networks.config.ConvConfig
      kernel_size: 3
      stride: 2
      padding: 1
      activation: ReLU
    - _target_: ml_networks.config.ConvConfig
      kernel_size: 3
      stride: 2
      padding: 1
      activation: ReLU
    - _target_: ml_networks.config.ConvConfig
      kernel_size: 3
      stride: 2
      padding: 1
      activation: ReLU
full_connection_cfg:
  _target_: ml_networks.config.LinearConfig
  activation: ReLU
  bias: true
```

Pythonコード：

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

# YAMLファイルから設定を読み込む
cfg = OmegaConf.load("configs/encoder_config.yaml")

# instantiateを使用してオブジェクトをインスタンス化
encoder = instantiate(cfg)

# 推論
obs = torch.randn(32, 3, 64, 64)
z = encoder(obs)
print(z.shape)  # torch.Size([32, 64])
```

### 方法2: Pythonコードで直接設定する

```python
from ml_networks import Encoder, ConvNetConfig, ConvConfig, LinearConfig
import torch

# Encoderの設定
encoder_cfg = ConvNetConfig(
    channels=[16, 32, 64],  # 各層のchannel数
    conv_cfgs=[
        ConvConfig(
            kernel_size=3,
            stride=2,
            padding=1,
            activation="ReLU",
        ),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
    ]
)

# 全結合層の設定
full_connection_cfg = LinearConfig(
    activation="ReLU",
    bias=True,
)

# Encoderの作成
obs_shape = (3, 64, 64)  # (C, H, W)
feature_dim = 64
encoder = Encoder(feature_dim, obs_shape, encoder_cfg, full_connection_cfg)

# 推論
obs = torch.randn(32, 3, 64, 64)
z = encoder(obs)
print(z.shape)  # torch.Size([32, 64])
```

## バックボーンの種類

### ConvNet

多層CNNのエンコーダ：

```python
encoder_cfg = ConvNetConfig(
    channels=[16, 32, 64],
    conv_cfgs=[
        ConvConfig(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            activation="ReLU",
            groups=1,
            bias=True,
            norm="none",
            norm_cfg={},
            dropout=0.0,
            scale_factor=0,
        ),
        # ... 他の層
    ]
)
```

### ResNet + PixelUnShuffle

ResNetとPixelUnShuffleを組み合わせたエンコーダ：

**YAMLファイル** (`configs/encoder_resnet.yaml`):

```yaml
_target_: ml_networks.vision.Encoder
feature_dim: 64
obs_shape: [3, 64, 64]
encoder_cfg:
  _target_: ml_networks.config.ResNetConfig
  conv_channel: 64
  conv_kernel: 3
  f_kernel: 3
  conv_activation: ReLU
  out_activation: ReLU
  n_res_blocks: 3
  scale_factor: 2  # PixelUnShuffleのスケールファクタ
  n_scaling: 3     # PixelUnShuffleの数
  norm: batch
  norm_cfg:
    affine: true
  dropout: 0.0
full_connection_cfg:
  _target_: ml_networks.config.LinearConfig
  activation: ReLU
  bias: true
```

**Pythonコード**:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/encoder_resnet.yaml")
encoder = instantiate(cfg)
```

### ViT（非推奨）

Vision Transformerも使用可能ですが、実装が不安定なため非推奨です。

## 全結合層の設定

### 単一の線形層

```python
full_connection_cfg = LinearConfig(
    activation="ReLU",
    bias=True,
)
```

### 多層MLP

```python
from ml_networks import MLPConfig

full_connection_cfg = MLPConfig(
    hidden_dim=128,
    n_layers=2,
    output_activation="Tanh",
    linear_cfg=LinearConfig(
        activation="ReLU",
        bias=True,
    )
)
```

### SpatialSoftmax

```python
from ml_networks import SpatialSoftmaxConfig

full_connection_cfg = SpatialSoftmaxConfig(
    temperature=1.0,
    eps=1e-6,
    is_argmax=False,
)
```

### AdaptiveAveragePooling

```python
from ml_networks import AdaptiveAveragePoolingConfig

full_connection_cfg = AdaptiveAveragePoolingConfig()
```

### 特徴マップをそのまま出力

```python
full_connection_cfg = None
# feature_dimはbackboneの出力特徴マップ次元と一致させる必要がある
```

## 独立したSpatialSoftmaxの使用

```python
from ml_networks import SpatialSoftmaxConfig, SpatialSoftmax

data = torch.randn(32, 3, 64, 64)
cfg = SpatialSoftmaxConfig(
    temperature=1.0,
    eps=1e-6,
    is_argmax=False,
)
spatial_softmax = SpatialSoftmax(cfg)
z = spatial_softmax(data)
```
