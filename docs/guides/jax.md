# JAXバックエンドガイド

JAX（Flax NNX）バックエンドの使用方法を説明します。

## 概要

`ml-networks`はPyTorchに加えて、JAX（Flax NNX）バックエンドを提供しています。PyTorchと同一の`Config`体系を使用するため、設定を変更することなくフレームワークを切り替えることができます。

!!! info "JAXバックエンドの特徴"
    - PyTorchと同一の`Config`クラスを使用
    - Flax NNX（`flax.nnx`）ベースの実装
    - optaxによる最適化
    - distraxによる確率分布

## インストール

JAXバックエンドを使用するには、追加の依存関係が必要です：

```bash
pip install jax flax optax distrax
```

### 要件

- JAX >= 0.4.30
- Flax >= 0.12.0（NNXモジュール）
- optax >= 0.2.0
- distrax >= 0.1.5

## インポート

PyTorchバックエンドとJAXバックエンドは、それぞれ別のサブモジュールからインポートします：

=== "PyTorch"

    ```python
    from ml_networks.torch import (
        MLPLayer, Encoder, Decoder,
        Distribution, ConditionalUnet2d,
        focal_loss, charbonnier,
        get_optimizer, torch_fix_seed,
    )
    ```

=== "JAX"

    ```python
    from ml_networks.jax import (
        MLPLayer, Encoder, Decoder,
        Distribution, ConditionalUnet2d,
        focal_loss, charbonnier,
        get_optimizer, jax_fix_seed,
    )
    ```

設定クラスは共通です：

```python
from ml_networks import (
    MLPConfig, LinearConfig, ConvConfig, ConvNetConfig,
    ResNetConfig, UNetConfig, ViTConfig,
)
```

## 基本的な使用方法

### MLP

```python
from ml_networks.jax import MLPLayer
from ml_networks import MLPConfig, LinearConfig
import jax
import jax.numpy as jnp

# 設定（PyTorchと同じ）
mlp_config = MLPConfig(
    hidden_dim=128,
    n_layers=2,
    output_activation="Tanh",
    linear_cfg=LinearConfig(activation="ReLU", bias=True)
)

# MLPの作成（rngsパラメータが必要）
rngs = jax.random.PRNGKey(0)
mlp = MLPLayer(input_dim=16, output_dim=8, mlp_config=mlp_config, rngs=rngs)

# 推論
x = jnp.ones((32, 16))
y = mlp(x)
print(y.shape)  # (32, 8)
```

!!! note "PyTorchとの違い: `rngs`パラメータ"
    JAXバックエンドでは、モデルの初期化時に乱数キー（`rngs`）を渡す必要があります。
    これはJAXの関数型プログラミングモデルに基づくもので、再現性を保証します。

### Encoder

```python
from ml_networks.jax import Encoder
from ml_networks import ConvNetConfig, ConvConfig, LinearConfig
import jax
import jax.numpy as jnp

# バックボーンの設定
encoder_cfg = ConvNetConfig(
    channels=[16, 32, 64],
    conv_cfgs=[
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
    ]
)

# 全結合層の設定
full_connection_cfg = LinearConfig(activation="ReLU", bias=True)

# Encoderの作成
obs_shape = (3, 64, 64)
feature_dim = 64
rngs = jax.random.PRNGKey(0)
encoder = Encoder(feature_dim, obs_shape, encoder_cfg, full_connection_cfg, rngs=rngs)

# 推論
obs = jnp.ones((32, 3, 64, 64))
z = encoder(obs)
print(z.shape)  # (32, 64)
```

### Decoder

```python
from ml_networks.jax import Decoder
from ml_networks import ConvNetConfig, ConvConfig, LinearConfig
import jax
import jax.numpy as jnp

# デコーダの設定
decoder_cfg = ConvNetConfig(
    channels=[64, 32, 16],
    conv_cfgs=[
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="Tanh"),
    ]
)

full_connection_cfg = LinearConfig(activation="ReLU", bias=True)

obs_shape = (3, 64, 64)
feature_dim = 64
rngs = jax.random.PRNGKey(0)
decoder = Decoder(feature_dim, obs_shape, decoder_cfg, full_connection_cfg, rngs=rngs)

# 推論
z = jnp.ones((32, feature_dim))
predicted_obs = decoder(z)
print(predicted_obs.shape)  # (32, 3, 64, 64)
```

### 分布

```python
from ml_networks.jax import Distribution, Encoder
from ml_networks import ConvNetConfig, ConvConfig, MLPConfig, LinearConfig
import jax
import jax.numpy as jnp

feature_dim = 64
obs_shape = (3, 64, 64)

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

rngs = jax.random.PRNGKey(0)

# 正規分布の場合、平均と標準偏差で特徴量次元の2倍が必要
encoder = Encoder(feature_dim * 2, obs_shape, encoder_cfg, full_connection_cfg, rngs=rngs)

dist = Distribution(
    in_dim=feature_dim,
    dist="normal",
    n_groups=1,
    rngs=rngs,
)

obs = jnp.ones((32, 3, 64, 64))
z = encoder(obs)
dist_z = dist(z)
```

## 最適化

JAXバックエンドでは、optaxベースの最適化を使用します：

```python
from ml_networks.jax import get_optimizer

# PyTorchスタイルの名前でoptaxオプティマイザを取得
optimizer = get_optimizer("Adam", learning_rate=1e-3)

# PyTorchの名前が自動的にoptaxの名前にマッピングされる
# "Adam" -> "adam", "SGD" -> "sgd", "AdamW" -> "adamw" など
```

## Seed固定

```python
from ml_networks.jax import jax_fix_seed

# JAX, numpy, randomのseedを固定
jax_fix_seed(42)
```

## 損失関数

```python
from ml_networks.jax import focal_loss, binary_focal_loss, charbonnier
import jax.numpy as jnp

# Focal Loss
logits = jnp.ones((32, 10))
labels = jnp.zeros((32,), dtype=jnp.int32)
loss = focal_loss(logits, labels, gamma=2.0)

# Charbonnier Loss
predicted = jnp.ones((32, 3, 64, 64))
target = jnp.zeros((32, 3, 64, 64))
loss = charbonnier(predicted, target)
```

## PyTorchとJAXの対応表

| 機能 | PyTorch (`ml_networks.torch`) | JAX (`ml_networks.jax`) |
|------|------|------|
| MLP | `MLPLayer` | `MLPLayer` |
| Encoder | `Encoder` | `Encoder` |
| Decoder | `Decoder` | `Decoder` |
| ConvNet | `ConvNet` | `ConvNet` |
| ResNet | `ResNetPixShuffle` / `ResNetPixUnshuffle` | `ResNetPixShuffle` / `ResNetPixUnshuffle` |
| ViT | `ViT` | `ViT` |
| UNet | `ConditionalUnet1d` / `ConditionalUnet2d` | `ConditionalUnet1d` / `ConditionalUnet2d` |
| 分布 | `Distribution` | `Distribution` |
| 活性化関数 | `Activation` | `Activation` |
| 損失関数 | `focal_loss`, `charbonnier` など | `focal_loss`, `charbonnier` など |
| 最適化 | `get_optimizer` (PyTorch/pytorch_optimizer) | `get_optimizer` (optax) |
| Seed固定 | `torch_fix_seed` | `jax_fix_seed` |
| HyperNet | `HyperNet` | `HyperNet` |
| 対照学習 | `ContrastiveLearningLoss` | `ContrastiveLearningLoss` |

## 注意事項

- JAXバックエンドは比較的新しい実装であり、PyTorchバックエンドに比べてテストが限られている場合があります
- `rngs`パラメータはFlax NNXの仕様であり、モデルの初期化時に必ず渡す必要があります
- JAXの制約により、動的な形状変更（例: `torch.view`の一部用法）は制限される場合があります
- GPUの使用にはJAXのGPU版をインストールする必要があります（`jax[cuda12]`）
