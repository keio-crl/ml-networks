# MLPガイド

MLP（多層パーセプトロン）の使用方法を説明します。

!!! tip "ベストプラクティス"
    YAMLファイルから設定を読み込む方法については、[設定管理ガイド](config-management.md)を参照してください。

## 基本的な使用方法

### 方法1: YAMLファイルから読み込む（推奨）

設定ファイル `configs/mlp_config.yaml` を作成します：

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
    norm_cfg: {}
    norm_first: false
    dropout: 0.0
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

### 方法2: Pythonコードで直接設定する

```python
from ml_networks import MLPLayer, MLPConfig, LinearConfig
import torch

mlp_config = MLPConfig(
    hidden_dim=128,  # 隠れ層の次元
    n_layers=2,      # 隠れ層の数
    output_activation="Tanh",  # 出力層の活性化関数
    linear_cfg=LinearConfig(
        activation="ReLU",  # 活性化関数
        bias=True,         # バイアスを使うかどうか
        norm="none",       # 正規化を行うかどうか
                         # "none"で正規化なし
                         # "layer"でLayerNorm
                         # "rms"でRMSNormが使える
        norm_cfg={},      # 正規化の設定
        norm_first=False, # 正規化をnn.Linearの前に行うかどうか
        dropout=0.0,      # ドロップアウト率
    )
)

input_dim = 16
output_dim = 8

mlp = MLPLayer(input_dim, output_dim, mlp_config)

x = torch.randn(32, input_dim)
y = mlp(x)
print(y.shape)  # torch.Size([32, 8])
```

## 設定オプション

### LinearConfig

`LinearConfig`は各線形層の設定を制御します：

- `activation`: 活性化関数（"ReLU", "Tanh", "Sigmoid"など）
- `bias`: バイアスを使うかどうか（デフォルト: `True`）
- `norm`: 正規化の種類（"none", "layer", "rms"）
- `norm_cfg`: 正規化の設定（辞書形式）
- `norm_first`: 正規化を線形層の前に行うかどうか（デフォルト: `False`）
- `dropout`: ドロップアウト率（デフォルト: `0.0`）

### MLPConfig

`MLPConfig`はMLP全体の設定を制御します：

- `hidden_dim`: 隠れ層の次元
- `n_layers`: 隠れ層の数
- `output_activation`: 出力層の活性化関数
- `linear_cfg`: `LinearConfig`のインスタンス

## 使用例

### ドロップアウト付きMLP

**YAMLファイル** (`configs/mlp_dropout.yaml`):

```yaml
_target_: ml_networks.layers.MLPLayer
input_dim: 16
output_dim: 8
mlp_config:
  _target_: ml_networks.config.MLPConfig
  hidden_dim: 128
  n_layers: 3
  output_activation: Tanh
  linear_cfg:
    _target_: ml_networks.config.LinearConfig
    activation: ReLU
    bias: true
    norm: layer
    dropout: 0.2  # 20%のドロップアウト
```

**Pythonコード**:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/mlp_dropout.yaml")
mlp = instantiate(cfg)
```

### LayerNorm付きMLP

**YAMLファイル** (`configs/mlp_layernorm.yaml`):

```yaml
_target_: ml_networks.layers.MLPLayer
input_dim: 16
output_dim: 8
mlp_config:
  _target_: ml_networks.config.MLPConfig
  hidden_dim: 256
  n_layers: 4
  output_activation: Identity
  linear_cfg:
    _target_: ml_networks.config.LinearConfig
    activation: ReLU
    bias: true
    norm: layer
    norm_cfg:
      eps: 1e-5
    norm_first: true  # 正規化を先に実行
```

**Pythonコード**:

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/mlp_layernorm.yaml")
mlp = instantiate(cfg)
```
