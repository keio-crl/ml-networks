# API リファレンス

`ml-networks`の完全なAPIリファレンスです。

## モジュール一覧

### 共通モジュール

- [設定](config.md) - 設定クラス（PyTorch/JAX共通）
- [ユーティリティ](utils.md) - 共通ユーティリティ関数

### PyTorch (`ml_networks.torch`)

- [レイヤー](layers.md) - 基本的なレイヤー（MLP、Conv、Attention、Transformerなど）
- [ビジョン](vision.md) - ビジョン関連のモジュール（Encoder、Decoder、ConvNet、ResNet、ViTなど）
- [分布](distributions.md) - 分布関連のクラスと関数
- [損失関数](loss.md) - 損失関数
- [活性化関数](activations.md) - カスタム活性化関数
- [UNet](unet.md) - 条件付きUNetクラス
- [その他](others.md) - HyperNet、ContrastiveLearning、BaseModule、ProgressBarCallback

### JAX (`ml_networks.jax`)

- [JAX API](jax.md) - JAX（Flax NNX）実装のAPIリファレンス

## 主要なクラスと関数

### レイヤー

::: ml_networks.torch.layers.MLPLayer
::: ml_networks.torch.layers.LinearNormActivation
::: ml_networks.torch.layers.ConvNormActivation
::: ml_networks.torch.layers.ConvTransposeNormActivation

### ビジョン

::: ml_networks.torch.vision.Encoder
::: ml_networks.torch.vision.Decoder
::: ml_networks.torch.vision.ConvNet
::: ml_networks.torch.vision.ResNetPixUnshuffle

### 分布

::: ml_networks.torch.distributions.Distribution
::: ml_networks.torch.distributions.NormalStoch
::: ml_networks.torch.distributions.CategoricalStoch

### 損失関数

::: ml_networks.torch.loss.focal_loss
::: ml_networks.torch.loss.charbonnier
::: ml_networks.torch.loss.FocalFrequencyLoss

### ユーティリティ

::: ml_networks.torch.torch_utils.get_optimizer
::: ml_networks.torch.torch_utils.torch_fix_seed
::: ml_networks.utils.save_blosc2
::: ml_networks.utils.load_blosc2
