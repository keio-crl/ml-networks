# クイックスタート

このガイドでは、`ml-networks`の基本的な使用方法を説明します。

## インストール

まず、`ml-networks`をインストールします：

```bash
pip install https://github.com/keio-crl/ml-networks.git
```

## 基本的な使用例

!!! tip "ベストプラクティス: YAMLファイルから設定を読み込む"
    設定をPythonコードにベタ書きするのではなく、YAMLファイルから読み込むことを推奨します。
    詳細は[設定管理ガイド](guides/config-management.md)を参照してください。

### MLPの使用

最もシンプルな例として、MLP（多層パーセプトロン）を使用します。

#### 方法1: YAMLファイルから読み込む（推奨）

まず、設定ファイル `configs/mlp_config.yaml` を作成します：

```yaml
_target_: ml_networks.layers.MLPLayer
input_dim: 16
output_dim: 8
mlp_config:
  _target_: ml_networks.config.MLPConfig
  hidden_dim: 128
  n_layers: 2
  output_activation: Tanh
  linear_cfg:
    _target_: ml_networks.config.LinearConfig
    activation: ReLU
    bias: true
    norm: none
```

Pythonコード：

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

# YAMLファイルから設定を読み込む
cfg = OmegaConf.load("configs/mlp_config.yaml")

# instantiateを使用してオブジェクトをインスタンス化
mlp = instantiate(cfg)

# 推論
x = torch.randn(32, 16)
y = mlp(x)
print(y.shape)  # torch.Size([32, 8])
```

#### 方法2: Pythonコードで直接設定する

```python
from ml_networks import MLPLayer, MLPConfig, LinearConfig
import torch

# MLPの設定
mlp_config = MLPConfig(
    hidden_dim=128,  # 隠れ層の次元
    n_layers=2,      # 隠れ層の数
    output_activation="Tanh",  # 出力層の活性化関数
    linear_cfg=LinearConfig(
        activation="ReLU",  # 活性化関数
        bias=True,          # バイアスを使うかどうか
        norm="none",        # 正規化を行うかどうか
    )
)

# MLPの作成
input_dim = 16
output_dim = 8
mlp = MLPLayer(input_dim, output_dim, mlp_config)

# 推論
x = torch.randn(32, input_dim)
y = mlp(x)
print(y.shape)  # torch.Size([32, 8])
```

### Encoderの使用

画像を特徴量に変換するEncoderを使用します。

#### 方法1: YAMLファイルから読み込む（推奨）

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

#### 方法2: Pythonコードで直接設定する

```python
from ml_networks import Encoder, ConvNetConfig, ConvConfig, LinearConfig
import torch

# Encoderの設定
encoder_cfg = ConvNetConfig(
    channels=[16, 32, 64],
    conv_cfgs=[
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
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

### Decoderの使用

特徴量から画像を再構成するDecoderを使用します：

```python
from ml_networks import Decoder, ConvNetConfig, ConvConfig, LinearConfig
import torch

# Decoderの設定
decoder_cfg = ConvNetConfig(
    channels=[64, 32, 16],
    conv_cfgs=[
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="Tanh"),
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

### 分布の使用

特徴量を分布に変換します：

```python
from ml_networks import Distribution, Encoder, ConvNetConfig, ConvConfig, MLPConfig, LinearConfig
import torch

# Encoderの設定（分布のパラメータを出力するため、特徴量次元の2倍が必要）
encoder_cfg = ConvNetConfig(
    channels=[16, 32, 64],
    conv_cfgs=[
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
    ]
)

full_connection_cfg = MLPConfig(
    hidden_dim=128,
    n_layers=2,
    output_activation="Identity",
    linear_cfg=LinearConfig(activation="ReLU", bias=True)
)

feature_dim = 64
obs_shape = (3, 64, 64)

# 正規分布の場合、平均と標準偏差で特徴量次元の2倍が必要
encoder = Encoder(feature_dim * 2, obs_shape, encoder_cfg, full_connection_cfg)

# 分布の設定
dist = Distribution(
    in_dim=feature_dim,  # 分布の次元
    dist="normal",       # 分布の種類
    n_groups=1,          # 分布のグループ数
)

# 推論
obs = torch.randn(32, 3, 64, 64)
z = encoder(obs)
dist_z = dist(z)
print(dist_z)  # NormalStoch(mean: torch.Size([32, 64]), std: torch.Size([32, 64]), stoch: torch.Size([32, 64]))
```

## 次のステップ

- [設定管理ガイド](guides/config-management.md) - YAMLファイルから設定を読み込む方法（**推奨**）
- [MLPガイド](guides/mlp.md) - MLPの詳細な使用方法
- [Encoderガイド](guides/encoder.md) - Encoderの詳細な使用方法
- [Decoderガイド](guides/decoder.md) - Decoderの詳細な使用方法
- [分布ガイド](guides/distributions.md) - 分布の詳細な使用方法
- [API リファレンス](api/index.md) - 完全なAPIドキュメント
