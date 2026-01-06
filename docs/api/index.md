# API リファレンス

`ml-networks`の完全なAPIリファレンスです。

## モジュール一覧

- [レイヤー](layers.md) - 基本的なレイヤー（MLP、Conv、Linearなど）
- [ビジョン](vision.md) - ビジョン関連のモジュール（Encoder、Decoderなど）
- [分布](distributions.md) - 分布関連のクラスと関数
- [損失関数](loss.md) - 損失関数
- [設定](config.md) - 設定クラス
- [ユーティリティ](utils.md) - ユーティリティ関数
- [活性化関数](activations.md) - 活性化関数
- [UNet](unet.md) - UNet関連のクラス
- [その他](others.md) - その他のクラスと関数

## 主要なクラスと関数

### レイヤー

::: ml_networks.layers.MLPLayer
::: ml_networks.layers.LinearNormActivation
::: ml_networks.layers.ConvNormActivation
::: ml_networks.layers.ConvTransposeNormActivation

### ビジョン

::: ml_networks.vision.Encoder
::: ml_networks.vision.Decoder
::: ml_networks.vision.ConvNet
::: ml_networks.vision.ResNetPixUnshuffle

### 分布

::: ml_networks.distributions.Distribution
::: ml_networks.distributions.NormalStoch
::: ml_networks.distributions.CategoricalStoch

### 損失関数

::: ml_networks.loss.focal_loss
::: ml_networks.loss.charbonnier
::: ml_networks.loss.FocalFrequencyLoss

### ユーティリティ

::: ml_networks.utils.get_optimizer
::: ml_networks.utils.torch_fix_seed
::: ml_networks.utils.save_blosc2
::: ml_networks.utils.load_blosc2
