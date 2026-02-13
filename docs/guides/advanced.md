# 高度な機能

HyperNetwork、対照学習（Contrastive Learning）、Attention機構など、高度な機能の使用方法を説明します。

## HyperNetwork

HyperNetworkは、あるネットワーク（HyperNet）が別のネットワーク（TargetNet）の重みを動的に生成するメタ学習手法です。

### 基本的な使用方法

```python
from ml_networks.torch import HyperNet
from ml_networks import MLPConfig, LinearConfig
import torch

# ターゲットネットワークの定義
target_net = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 8),
)

# HyperNetの設定
mlp_cfg = MLPConfig(
    hidden_dim=256,
    n_layers=2,
    output_activation="Identity",
    linear_cfg=LinearConfig(activation="ReLU", bias=True)
)

# HyperNetの作成
# 条件ベクトルからターゲットネットワークの重みを生成
hypernet = HyperNet(
    target_net=target_net,
    input_dim=64,           # 条件ベクトルの次元
    mlp_cfg=mlp_cfg,
)

# 条件ベクトル
condition = torch.randn(4, 64)

# HyperNetで生成した重みでターゲットネットワークを実行
x = torch.randn(4, 16)
output = hypernet(x, condition)
```

### 入力エンコーディング

HyperNetは入力のエンコーディングモードをサポートしています：

- `None`: エンコーディングなし（そのまま入力）
- `"cos|sin"`: cos/sinエンコーディング（位置エンコーディングに類似）
- `"z|1-z"`: z, 1-zのペアにエンコーディング

```python
hypernet = HyperNet(
    target_net=target_net,
    input_dim=64,
    mlp_cfg=mlp_cfg,
    input_mode="cos|sin",   # cos/sinエンコーディングを使用
)
```

## 対照学習（Contrastive Learning）

対照学習は、類似したサンプルの表現を近づけ、異なるサンプルの表現を遠ざけるための手法です。

### 基本的な使用方法

```python
from ml_networks.torch import ContrastiveLearningLoss
from ml_networks import ContrastiveLearningConfig, MLPConfig, LinearConfig
import torch

# 対照学習の設定
cfg = ContrastiveLearningConfig(
    dim_feature=128,                    # 特徴量の次元
    eval_func=MLPConfig(
        hidden_dim=256,
        n_layers=2,
        output_activation="ReLU",
        linear_cfg=LinearConfig(
            activation="ReLU",
            norm="layer",
            norm_cfg={"eps": 1e-5, "elementwise_affine": True, "bias": True},
            dropout=0.1,
            bias=True,
        )
    ),
    cross_entropy_like=False,           # NCE損失を使用
)

# 対照学習モジュールの作成
model = ContrastiveLearningLoss(
    dim_input1=256,   # 入力1の次元
    dim_input2=256,   # 入力2の次元
    cfg=cfg,
)

# 2つのモダリティ/ビューの表現
x1 = torch.randn(32, 256)
x2 = torch.randn(32, 256)

# NCE損失の計算
output = model.calc_nce(x1, x2)
loss = output["nce"]

# 埋め込みも取得する場合
output, embeddings = model.calc_nce(x1, x2, return_emb=True)
emb1, emb2 = embeddings
print(emb1.shape)  # torch.Size([32, 128])
```

### YAMLファイルでの設定

```yaml
_target_: ml_networks.torch.contrastive.ContrastiveLearningLoss
dim_input1: 256
dim_input2: 256
cfg:
  _target_: ml_networks.config.ContrastiveLearningConfig
  dim_feature: 128
  eval_func:
    _target_: ml_networks.config.MLPConfig
    hidden_dim: 256
    n_layers: 2
    output_activation: ReLU
    linear_cfg:
      _target_: ml_networks.config.LinearConfig
      activation: ReLU
      norm: layer
      dropout: 0.1
      bias: true
  cross_entropy_like: false
```

## Attention機構

### Attention1d / Attention2d

1次元・2次元データに対するセルフアテンションレイヤーです。

```python
from ml_networks.torch import Attention1d, Attention2d
import torch

# 1Dセルフアテンション（時系列データ用）
attn1d = Attention1d(in_channels=64, nhead=8)
x = torch.randn(4, 64, 128)          # (batch, channels, length)
out = attn1d(x)
print(out.shape)  # torch.Size([4, 64, 128])

# 2Dセルフアテンション（画像データ用）
attn2d = Attention2d(in_channels=64, nhead=8)
x = torch.randn(4, 64, 32, 32)       # (batch, channels, height, width)
out = attn2d(x)
print(out.shape)  # torch.Size([4, 64, 32, 32])
```

### TransformerLayer

Transformerエンコーダブロックを使用できます。

```python
from ml_networks.torch import TransformerLayer
from ml_networks import TransformerConfig
import torch

cfg = TransformerConfig(
    d_model=256,
    nhead=8,
    dim_ff=512,
    n_layers=4,
    dropout=0.1,
    hidden_activation="GELU",
    output_activation="GELU",
)

# TransformerLayerは設定からインスタンス化
# （通常はEncoder/Decoderの一部として使用）
```

### PatchEmbed

Vision Transformer用のパッチ埋め込みレイヤーです。

```python
from ml_networks.torch import PatchEmbed
import torch

# 画像をパッチに分割して埋め込み
patch_embed = PatchEmbed(
    in_channels=3,
    patch_size=16,
    embed_dim=256,
)

x = torch.randn(4, 3, 64, 64)
patches = patch_embed(x)
print(patches.shape)  # (4, 16, 256) = (batch, num_patches, embed_dim)
```

### PositionalEncoding

Transformerの位置エンコーディングです。

```python
from ml_networks.torch import PositionalEncoding
import torch

pos_enc = PositionalEncoding(d_model=256, max_len=512)
x = torch.randn(4, 100, 256)         # (batch, seq_len, d_model)
out = pos_enc(x)
print(out.shape)  # torch.Size([4, 100, 256])
```

## ResidualBlock

残差接続を持つブロックです。ConvNet内部で使用されますが、独立しても使用可能です。

```python
from ml_networks.torch import ResidualBlock
from ml_networks import ConvConfig
import torch

cfg = ConvConfig(
    kernel_size=3,
    stride=1,
    padding=1,
    activation="ReLU",
    norm="batch",
)

block = ResidualBlock(in_channels=64, out_channels=64, conv_cfg=cfg)
x = torch.randn(4, 64, 32, 32)
out = block(x)
print(out.shape)  # torch.Size([4, 64, 32, 32])
```

## BSQCodebook（ベクトル量子化）

BSQ（Binary Spherical Quantization）コードブックは、ベクトル量子化による離散表現学習に使用されます。

```python
from ml_networks.torch import BSQCodebook
import torch

codebook = BSQCodebook(
    dim=64,           # 入力特徴量の次元
    codebook_dim=8,   # コードブックの次元
)

# 連続的な特徴量を離散的なコードに変換
features = torch.randn(32, 64)
quantized = codebook(features)
```

## L2Norm

L2正規化レイヤーです。特徴量を単位超球上に射影します。

```python
from ml_networks.torch import Activation
import torch

# L2Normは活性化関数として使用可能
l2_norm = Activation("L2Norm")
x = torch.randn(32, 64)
normalized = l2_norm(x)
# normalized の各行のL2ノルムは1になる
```

## BaseModule

`BaseModule`はPyTorch Lightningの`LightningModule`を拡張した基底クラスです。

```python
from ml_networks.torch import BaseModule

class MyModel(BaseModule):
    def __init__(self):
        super().__init__()
        # モデルの定義
```
