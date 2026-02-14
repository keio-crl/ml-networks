# JAX API リファレンス

JAX（Flax NNX）バックエンドのAPIリファレンスです。

PyTorchバックエンドと同一のインターフェースを提供しています。詳細は各PyTorch APIリファレンスページを参照してください。

## レイヤー (`ml_networks.jax.layers`)

::: ml_networks.jax.layers.MLPLayer
::: ml_networks.jax.layers.LinearNormActivation
::: ml_networks.jax.layers.ConvNormActivation
::: ml_networks.jax.layers.ConvTransposeNormActivation

## ビジョン (`ml_networks.jax.vision`)

::: ml_networks.jax.vision.Encoder
::: ml_networks.jax.vision.Decoder
::: ml_networks.jax.vision.ConvNet
::: ml_networks.jax.vision.ConvTranspose
::: ml_networks.jax.vision.ResNetPixUnshuffle
::: ml_networks.jax.vision.ResNetPixShuffle
::: ml_networks.jax.vision.ViT

## 分布 (`ml_networks.jax.distributions`)

::: ml_networks.jax.distributions.Distribution
::: ml_networks.jax.distributions.NormalStoch
::: ml_networks.jax.distributions.CategoricalStoch
::: ml_networks.jax.distributions.BernoulliStoch

## 損失関数 (`ml_networks.jax.loss`)

::: ml_networks.jax.loss.focal_loss
::: ml_networks.jax.loss.binary_focal_loss
::: ml_networks.jax.loss.charbonnier
::: ml_networks.jax.loss.kl_divergence

## 活性化関数 (`ml_networks.jax.activations`)

::: ml_networks.jax.activations.Activation
::: ml_networks.jax.activations.REReLU
::: ml_networks.jax.activations.SiGLU
::: ml_networks.jax.activations.CRReLU
::: ml_networks.jax.activations.TanhExp

## UNet (`ml_networks.jax.unet`)

::: ml_networks.jax.unet.ConditionalUnet2d
::: ml_networks.jax.unet.ConditionalUnet1d

## ユーティリティ (`ml_networks.jax.jax_utils`)

::: ml_networks.jax.jax_utils.get_optimizer
::: ml_networks.jax.jax_utils.jax_fix_seed
::: ml_networks.jax.jax_utils.MinMaxNormalize
::: ml_networks.jax.jax_utils.SoftmaxTransformation

## その他

::: ml_networks.jax.hypernetworks.HyperNet
::: ml_networks.jax.contrastive.ContrastiveLearningLoss
::: ml_networks.jax.base.BaseModule
