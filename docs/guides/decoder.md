# Decoderガイド

Decoderの使用方法を説明します。

!!! tip "ベストプラクティス"
    YAMLファイルから設定を読み込む方法については、[設定管理ガイド](config-management.md)を参照してください。

## 概要

Decoderは特徴量から画像などを再構成するモジュールです。Encoderと対になる構造を持ちます。

## 基本的な使用方法

### 方法1: YAMLファイルから読み込む（推奨）

設定ファイル `configs/decoder_config.yaml` を作成します：

```yaml
_target_: ml_networks.vision.Decoder
feature_dim: 64
obs_shape: [3, 64, 64]
decoder_cfg:
  _target_: ml_networks.config.ConvNetConfig
  channels: [64, 32, 16]
  conv_cfgs:
    - _target_: ml_networks.config.ConvConfig
      output_padding: 0
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
      activation: Tanh
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
cfg = OmegaConf.load("configs/decoder_config.yaml")

# instantiateを使用してオブジェクトをインスタンス化
decoder = instantiate(cfg)

# 推論
z = torch.randn(32, 64)
predicted_obs = decoder(z)
print(predicted_obs.shape)  # torch.Size([32, 3, 64, 64])
```

### 方法2: Pythonコードで直接設定する

```python
from ml_networks import Decoder, ConvNetConfig, ConvConfig, LinearConfig
import torch

# Decoderの設定
decoder_cfg = ConvNetConfig(
    channels=[64, 32, 16],  # 各層のchannel数
    conv_cfgs=[
        ConvConfig(
            output_padding=0,  # ConvTranspose2dのみに利用される
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="Tanh"),  # 最後の層
    ]
)

# 全結合層の設定
full_connection_cfg = LinearConfig(
    activation="ReLU",
    bias=True,
)

# Decoderの作成
obs_shape = (3, 64, 64)
feature_dim = 64
decoder = Decoder(feature_dim, obs_shape, decoder_cfg, full_connection_cfg)

# 推論
z = torch.randn(32, feature_dim)
predicted_obs = decoder(z)
print(predicted_obs.shape)  # torch.Size([32, 3, 64, 64])
```

## バックボーンの種類

### ConvTranspose

多層ConvTransposeのデコーダ：

```python
decoder_cfg = ConvNetConfig(
    channels=[64, 32, 16],
    conv_cfgs=[
        ConvConfig(
            output_padding=0,  # 出力パディング
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        # ... 他の層
    ]
)
```

### ResNet + PixelShuffle

ResNetとPixelShuffleを組み合わせたデコーダ：

**YAMLファイル** (`configs/decoder_resnet.yaml`):

```yaml
_target_: ml_networks.vision.Decoder
feature_dim: 64
obs_shape: [3, 64, 64]
decoder_cfg:
  _target_: ml_networks.config.ResNetConfig
  conv_channel: 64
  conv_kernel: 3
  f_kernel: 3
  conv_activation: ReLU
  output_activation: Tanh  # 出力層の活性化関数
  n_res_blocks: 3
  scale_factor: 2  # PixelShuffleのスケールファクタ
  n_scaling: 3     # PixelShuffleの数
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

cfg = OmegaConf.load("configs/decoder_resnet.yaml")
decoder = instantiate(cfg)
```

## 全結合層の設定

Encoderと同様に、以下の設定が可能です：

- `LinearConfig`: 単一の線形層
- `MLPConfig`: 多層MLP
- `None`: 特徴マップをそのまま入力

## 注意事項

- `output_padding`は`ConvTranspose2d`のみに利用されます
- 最後の層の活性化関数は出力に適したものを選択してください（例: "Tanh"）
- `feature_dim`はbackboneの入力特徴マップ次元と一致させる必要があります
