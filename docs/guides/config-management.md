# 設定管理

このガイドでは、Hydraの`hydra.utils.instantiate`を使用してYAMLファイルから設定を読み込み、オブジェクトをインスタンス化する方法を説明します。これは、設定をベタ書きするのではなく、YAMLファイルから読み込むベストプラクティスです。

## 概要

`ml-networks`では、設定をPythonコードにベタ書きするのではなく、YAMLファイルから読み込むことを推奨しています。これにより、以下のメリットがあります：

## インストール

Hydraを使用するには、`hydra-core`をインストールする必要があります：

```bash
pip install hydra-core
```

または、`omegaconf`が既にインストールされている場合は、`hydra-core`も一緒にインストールされます。

- **設定の分離**: コードと設定を分離できる
- **再利用性**: 同じ設定を複数の実験で再利用できる
- **可読性**: YAMLファイルは構造化されており、読みやすい
- **柔軟性**: 設定を変更する際にコードを変更する必要がない

## 基本的な使用方法

### 1. YAMLファイルの作成

まず、設定をYAMLファイルに記述します。Hydraの`instantiate`を使用するには、`_target_`フィールドで対象クラスの完全なパスを指定します。

**例: `configs/mlp_config.yaml`**

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
    dropout: 0.0
```

**例: `configs/encoder_config.yaml`**

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
      bias: true
      norm: none
      dropout: 0.0
    - _target_: ml_networks.config.ConvConfig
      kernel_size: 3
      stride: 2
      padding: 1
      activation: ReLU
      bias: true
      norm: none
      dropout: 0.0
    - _target_: ml_networks.config.ConvConfig
      kernel_size: 3
      stride: 2
      padding: 1
      activation: ReLU
      bias: true
      norm: none
      dropout: 0.0
full_connection_cfg:
  _target_: ml_networks.config.LinearConfig
  activation: ReLU
  bias: true
```

### 2. Pythonコードでの使用

YAMLファイルから設定を読み込み、`hydra.utils.instantiate`を使用してオブジェクトをインスタンス化します。

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

# YAMLファイルから設定を読み込む
cfg = OmegaConf.load("configs/mlp_config.yaml")

# instantiateを使用してオブジェクトをインスタンス化
mlp = instantiate(cfg)

# 使用
import torch
x = torch.randn(32, 16)
y = mlp(x)
print(y.shape)  # torch.Size([32, 8])
```

### 3. ネストされた設定の例

複雑な設定でも、同様にYAMLファイルから読み込めます。

**例: `configs/distribution_config.yaml`**

```yaml
encoder:
  _target_: ml_networks.vision.Encoder
  feature_dim: 128  # 正規分布の場合、特徴量次元の2倍が必要
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
  full_connection_cfg:
    _target_: ml_networks.config.MLPConfig
    hidden_dim: 128
    n_layers: 2
    output_activation: Identity
    linear_cfg:
      _target_: ml_networks.config.LinearConfig
      activation: ReLU
      bias: true

distribution:
  _target_: ml_networks.distributions.Distribution
  in_dim: 64
  dist: normal
  n_groups: 1
  spherical: false
```

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

# 設定を読み込む
cfg = OmegaConf.load("configs/distribution_config.yaml")

# エンコーダと分布をインスタンス化
encoder = instantiate(cfg.encoder)
dist = instantiate(cfg.distribution)

# 使用
obs = torch.randn(32, 3, 64, 64)
z = encoder(obs)
dist_z = dist(z)
print(dist_z)
```

## 設定の部分的なインスタンス化

設定の一部だけをインスタンス化することもできます。

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/encoder_config.yaml")

# encoder_cfgだけをインスタンス化
encoder_cfg = instantiate(cfg.encoder_cfg)

# その後、手動でEncoderを作成
from ml_networks.vision import Encoder
encoder = Encoder(
    feature_dim=cfg.feature_dim,
    obs_shape=tuple(cfg.obs_shape),
    encoder_cfg=encoder_cfg,
    full_connection_cfg=instantiate(cfg.full_connection_cfg)
)
```

## 設定の検証

設定が正しいかどうかを検証するには、`OmegaConf`の機能を使用します。

```python
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/mlp_config.yaml")

# 設定の構造を確認
print(OmegaConf.to_yaml(cfg))

# 必須フィールドのチェック
if "input_dim" not in cfg:
    raise ValueError("input_dim is required")
```

## 設定の継承と合成

複数の設定ファイルを組み合わせることもできます。

**例: `configs/base_encoder.yaml`**

```yaml
encoder_cfg:
  _target_: ml_networks.config.ConvNetConfig
  channels: [16, 32, 64]
  conv_cfgs:
    - _target_: ml_networks.config.ConvConfig
      kernel_size: 3
      stride: 2
      padding: 1
      activation: ReLU
```

**例: `configs/custom_encoder.yaml`**

```yaml
defaults:
  - base_encoder

feature_dim: 128
obs_shape: [3, 128, 128]
```

## ベストプラクティス

1. **設定ファイルの命名規則**: `{component}_config.yaml`のような命名規則を使用する
2. **設定の階層化**: 共通の設定は別ファイルに分離し、必要に応じて継承する
3. **設定のバージョン管理**: 設定ファイルもGitで管理し、実験の再現性を確保する
4. **設定の検証**: 設定を読み込んだ後、必須フィールドの存在を確認する
5. **ドキュメント化**: 設定ファイルにコメントを追加して、各パラメータの意味を説明する

## 設定ファイルの例

### MLP設定

```yaml
# configs/mlp_simple.yaml
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
```

### Encoder設定（ResNet + PixelUnShuffle）

```yaml
# configs/encoder_resnet.yaml
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
  scale_factor: 2
  n_scaling: 3
  norm: batch
  norm_cfg:
    affine: true
  dropout: 0.0
full_connection_cfg:
  _target_: ml_networks.config.LinearConfig
  activation: ReLU
  bias: true
```

### Decoder設定

```yaml
# configs/decoder_config.yaml
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

## トラブルシューティング

### よくあるエラー

1. **`_target_`が見つからない**: クラスの完全なパスが正しいか確認してください
2. **型の不一致**: YAMLの値が期待される型と一致しているか確認してください（例: `true` vs `True`）
3. **必須パラメータの欠如**: 設定に必須パラメータが含まれているか確認してください

### デバッグのヒント

```python
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/mlp_config.yaml")

# 設定の構造を確認
print(OmegaConf.to_yaml(cfg))

# 特定の値を確認
print(cfg.mlp_config.hidden_dim)

# 設定を検証
OmegaConf.set_struct(cfg, True)  # 構造を固定して検証
```
