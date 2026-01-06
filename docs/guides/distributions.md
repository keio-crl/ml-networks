# 分布ガイド

分布の使用方法を説明します。

!!! tip "ベストプラクティス"
    YAMLファイルから設定を読み込む方法については、[設定管理ガイド](config-management.md)を参照してください。

## 概要

`ml-networks`は、特徴量を分布に変換する機能を提供します。正規分布、カテゴリカル分布、ベルヌーイ分布をサポートしています。

## 正規分布

### 基本的な使用方法

#### 方法1: YAMLファイルから読み込む（推奨）

設定ファイル `configs/distribution_config.yaml` を作成します：

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

Pythonコード：

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import torch.distributions as D

# 設定を読み込む
cfg = OmegaConf.load("configs/distribution_config.yaml")

# エンコーダと分布をインスタンス化
encoder = instantiate(cfg.encoder)
dist = instantiate(cfg.distribution)

# 使用
obs = torch.randn(32, 3, 64, 64)
z = encoder(obs)

# 自動的に分布のパラメータへの変換・再パラメータ化トリックが適用される
dist_z = dist(z)
print(dist_z)
# NormalStoch(mean: torch.Size([32, 64]), std: torch.Size([32, 64]), stoch: torch.Size([32, 64]))

# torch.distributions.Distributionに変換
torch_dist_z = dist_z.get_distribution(independent=1)

# KLDの計算
normal = D.Normal(0, 1)
kld = D.kl_divergence(torch_dist_z, normal).mean()
```

#### 方法2: Pythonコードで直接設定する

```python
from ml_networks import Distribution, Encoder, ConvNetConfig, ConvConfig, MLPConfig, LinearConfig
import torch
import torch.distributions as D

feature_dim = 64
obs_shape = (3, 64, 64)

# ガウス分布を使う場合は平均と標準偏差で特徴量次元の2倍の次元が必要
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
    output_activation="Identity",  # 分布に変換する場合何もかけないのがいい
    linear_cfg=LinearConfig(
        activation="ReLU",
        bias=True,
    )
)

encoder = Encoder(feature_dim * 2, obs_shape, encoder_cfg, full_connection_cfg)

dist = Distribution(
    in_dim=feature_dim,  # 分布の次元（平均・標準偏差の次元）
    dist="normal",        # 分布の種類
    n_groups=1,          # 分布のグループ数（ガウス分布の場合は意味ない）
)

obs = torch.randn(32, 3, 64, 64)
z = encoder(obs)

# 自動的に分布のパラメータへの変換・再パラメータ化トリックが適用される
dist_z = dist(z)
print(dist_z)
# NormalStoch(mean: torch.Size([32, 64]), std: torch.Size([32, 64]), stoch: torch.Size([32, 64]))

# torch.distributions.Distributionに変換
torch_dist_z = dist_z.get_distribution(independent=1)

# KLDの計算
normal = D.Normal(0, 1)
kld = D.kl_divergence(torch_dist_z, normal).mean()
```

## カテゴリカル分布

```python
encoder = Encoder(feature_dim, obs_shape, encoder_cfg, full_connection_cfg)

dist = Distribution(
    in_dim=feature_dim,
    dist="categorical",
    n_groups=8,  # feature_dimがn_groupsの倍数でないとエラーが出る
)

z = encoder(obs)
dist_z = dist(z)
print(dist_z)
# CategoricalStoch(logits: torch.Size([32, 8, 8]), probs: torch.Size([32, 8, 8]), stoch: torch.Size([32, 8, 8]))

flat_dist = D.OneHotCategorical(probs=torch.ones_like(dist_z.probs)/dist_z.probs.shape[-1])
kld = D.kl_divergence(dist_z.get_distribution(), flat_dist).mean()
```

## ベルヌーイ分布

```python
dist = Distribution(
    in_dim=feature_dim,
    dist="bernoulli",
    n_groups=2,      # 超球の数
    spherical=False,  # 超球にするかどうか
)
```

## 分布データの操作

### stack

```python
from ml_networks import stack_dist

dist_list = []
for batch in dataloader:
    obs = batch["obs"]
    z = encoder(obs)
    dist_z = dist(z)
    dist_list.append(dist_z)

# 分布データをstack
stacked_dist = stack_dist(dist_list, dim=0)
print(stacked_dist.shape)
# NormalShape(mean: torch.Size([100, 32, 64]), std: torch.Size([100, 32, 64]), stoch: torch.Size([100, 32, 64]))
```

### concatenate

```python
from ml_networks import cat_dist

# 分布データをconcatenate
concatenated_dist = cat_dist(dist_list, dim=-1)
print(concatenated_dist.shape)
# NormalShape(mean: torch.Size([32, 6400]), std: torch.Size([32, 6400]), stoch: torch.Size([32, 6400]))
```

## 分布データの保存

```python
dist = Distribution(
    in_dim=feature_dim,
    dist="normal",
    n_groups=1,
)

z = encoder(obs)
dist_z = dist(z)

# 分布データの保存
dist_z.save("reports")
# reportsの下にmean.blosc2, std.blosc2, stoch.blosc2が保存される
# 他の分布データも同様に保存される
```
