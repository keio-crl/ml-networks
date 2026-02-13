# UNet

条件付きUNet関連のクラスを提供します。

`ml_networks.torch.unet`（PyTorch）と`ml_networks.jax.unet`（JAX）の両方で提供されています。

## ConditionalUnet2d

2D画像データ用の条件付きUNet。Diffusion Modelのノイズ予測ネットワークとして典型的に使用されます。

::: ml_networks.torch.unet.ConditionalUnet2d

## ConditionalUnet1d

1D時系列データ用の条件付きUNet。

::: ml_networks.torch.unet.ConditionalUnet1d
