# ml-networks

村田研共通フレームワーク化計画の一環として、基本的な深層学習モデルのアーキテクチャを提供するPythonパッケージです。

## 概要

`ml-networks`は、**PyTorch**および**JAX（Flax NNX）**ベースの深層学習モデル構築を支援するライブラリです。以下の機能を提供します：

- **基本的なニューラルネットワークアーキテクチャ**: MLP、Encoder、Decoder、UNet、Vision Transformer（ViT）など
- **分布のサポート**: 正規分布、カテゴリカル分布、ベルヌーイ分布、BSQコードブック
- **損失関数**: Focal Loss、Charbonnier Loss、Focal Frequency Loss、KLダイバージェンスなど
- **便利なユーティリティ**: 活性化関数、最適化手法、データ保存・読み込み機能
- **高度な機能**: HyperNetwork、対照学習（Contrastive Learning）、条件付きUNet

## 特徴

- **マルチフレームワーク対応**: PyTorchとJAX（Flax NNX）の両方をサポート。同一のConfig体系で切り替え可能
- **使いやすい**: 直感的なAPI設計。YAMLファイルから設定を読み込み、`hydra.utils.instantiate`でインスタンス化
- **柔軟性**: 豊富な設定オプション。バックボーン、正規化、活性化関数を自由に組み合わせ
- **包括的**: 深層学習に必要な主要コンポーネントを網羅
- **実用的**: 型チェック（mypy）対応、CI/CD整備済み

## パッケージ構成

```
ml_networks/
├── config.py          # 共通設定クラス（PyTorch/JAX共通）
├── utils.py           # 共通ユーティリティ
├── callbacks.py       # PyTorch Lightning コールバック
├── torch/             # PyTorch実装
│   ├── layers.py      # MLP, Conv, Attention, Transformerなど
│   ├── vision.py      # Encoder, Decoder, ConvNet, ResNet, ViT
│   ├── unet.py        # ConditionalUnet1d, ConditionalUnet2d
│   ├── distributions.py  # 確率分布
│   ├── loss.py        # 損失関数
│   ├── activations.py # カスタム活性化関数
│   ├── hypernetworks.py  # HyperNetwork
│   ├── contrastive.py # 対照学習
│   └── torch_utils.py # PyTorch固有ユーティリティ
└── jax/               # JAX (Flax NNX) 実装
    ├── layers.py      # MLP, Conv, Attention, Transformerなど
    ├── vision.py      # Encoder, Decoder, ConvNet, ResNet, ViT
    ├── unet.py        # ConditionalUnet1d, ConditionalUnet2d
    ├── distributions.py  # 確率分布
    ├── loss.py        # 損失関数
    ├── activations.py # カスタム活性化関数
    ├── hypernetworks.py  # HyperNetwork
    ├── contrastive.py # 対照学習
    └── jax_utils.py   # JAX固有ユーティリティ
```

## インストール

### 要件

- Python >= 3.10
- PyTorch >= 2.0（PyTorchバックエンドを使用する場合）
- JAX >= 0.4.30 + Flax >= 0.12.0（JAXバックエンドを使用する場合）

### インストール方法

以下のいずれかの方法でインストールできます：

#### pipを使用する場合

```bash
pip install https://github.com/keio-crl/ml-networks.git
```

#### ryeを使用する場合

```bash
rye add ml-networks --git https://github.com/keio-crl/ml-networks.git
```

#### uvを使用する場合

```bash
uv add https://github.com/keio-crl/ml-networks.git
```

#### JAXサポートを追加する場合

```bash
pip install "ml-networks[jax] @ https://github.com/keio-crl/ml-networks.git"
```

**注意**: uvを使用する場合は、`<access token>`をGitHubのPersonal Access Tokenに置き換えてください。

## クイックスタート

=== "PyTorch"

    ```python
    from ml_networks.torch import MLPLayer
    from ml_networks import MLPConfig, LinearConfig
    import torch

    # MLPの設定
    mlp_config = MLPConfig(
        hidden_dim=128,
        n_layers=2,
        output_activation="Tanh",
        linear_cfg=LinearConfig(activation="ReLU", bias=True)
    )

    # MLPの作成と推論
    mlp = MLPLayer(input_dim=16, output_dim=8, mlp_config=mlp_config)
    x = torch.randn(32, 16)
    y = mlp(x)
    print(y.shape)  # torch.Size([32, 8])
    ```

=== "JAX (Flax NNX)"

    ```python
    from ml_networks.jax import MLPLayer
    from ml_networks import MLPConfig, LinearConfig
    import jax
    import jax.numpy as jnp

    # MLPの設定（PyTorchと同じConfigを使用）
    mlp_config = MLPConfig(
        hidden_dim=128,
        n_layers=2,
        output_activation="Tanh",
        linear_cfg=LinearConfig(activation="ReLU", bias=True)
    )

    # MLPの作成と推論
    mlp = MLPLayer(input_dim=16, output_dim=8, mlp_config=mlp_config, rngs=jax.random.PRNGKey(0))
    x = jnp.ones((32, 16))
    y = mlp(x)
    print(y.shape)  # (32, 8)
    ```

詳細は[クイックスタートガイド](getting-started.md)を参照してください。

## ドキュメント

- [クイックスタート](getting-started.md) - 基本的な使用方法
- [設定管理ガイド](guides/config-management.md) - YAMLファイルから設定を読み込む方法（**推奨**）
- **ガイド**:
    - [MLP](guides/mlp.md) - 多層パーセプトロン
    - [Encoder](guides/encoder.md) - 画像エンコーダ
    - [Decoder](guides/decoder.md) - 画像デコーダ
    - [UNet](guides/unet.md) - 条件付きUNet
    - [Distributions](guides/distributions.md) - 確率分布
    - [損失関数](guides/loss-functions.md) - 各種損失関数
    - [データの保存と読み込み](guides/data-io.md) - blosc2形式のデータI/O
    - [その他の便利な機能](guides/utilities.md) - 活性化関数、最適化、seed固定など
    - [高度な機能](guides/advanced.md) - HyperNetwork、対照学習、Attention
    - [JAXバックエンド](guides/jax.md) - JAX (Flax NNX) での使用方法
- [API リファレンス](api/index.md) - 完全なAPIドキュメント

## ライセンス

このプロジェクトのライセンス情報については、リポジトリのLICENSEファイルを参照してください。

## 作者

- oakwood-fujiken (oakwood.n14.4sp@keio.jp)
- nomutin (nomura0508@icloud.com)
