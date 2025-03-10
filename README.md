# ml-networks


こちらは 村田研共通フレームワーク化計画の一環として，基本的な深層学習モデルのアーキテクチャを提供するリポジトリです．

## Installation
以下のようなコマンドでインストールが可能です．  

```bash
pip install https://github.com/keio-crl/ml-networks.git
rye add ml-networks --git https://github.com/keio-crl/ml-networks.git
uv add "ml-networks @ git+https://github.com/keio-crl/ml-networks.git"
```

install時はアカウント名とパスワードが求められる場合があります．  
その場合は，アカウント名にはkeio-crlに登録されているユーザーネーム，パスワードにはトークンを入力してください．  
トークンの取得方法は[こちら](https://docs.github.com/ja/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)を参照してください．

## Usage
主要なものたちを以下に示します．  
詳細な説明は[こちら](https://github.com/keio-crl/ml-networks.git)(in coming).

### MLP
```python
from ml_networks import MLPLayer, MLPConfig, LinearConfig

mlp_config = MLPConfig(
        hidden_dim= 128,
        n_layers= 2,
        output_activation="Tanh",
        linaer_cfg=LinearConfig(
            activation="ReLU",
            bias=True,
        )
)
input_dim = 16
output_dim = 8

mlp = MLPLayer(input_dim, output_dim, mlp_config)

x = torch.randn(32, input_dim)
y = mlp(x)
print(y.shape)
>>> torch.Size([32, 8])

```

### Encoder
そんなに強いエンコーダがいらない時は，以下のように使うことができます．
強いエンコーダが必要な場合は，[timm](https://github.com/huggingface/pytorch-image-models)でも使ってください．
（参考:https://zenn.dev/piment/articles/4ff3b6dfd73103）
```python

from ml_networks import (Encoder, ConvNetConfig,
                         ViTConfig, ResNetConfig,
                         MLPConfig, LinearConfig)

# アーキテクチャを変えたい場合は引数に渡すConfigを変える
## 多層CNN+MLPのエンコーダ
encoder_config = ConvNetConfig(
    channels=[16, 32, 64],
    conv_cfgs=[
        ConvConfig(
            kernel_size=3, # カーネルサイズ
            stride=2, # ストライド
            padding=1, # パディング
            dilation=1, # ダイレーション．変えることはほぼない．Default: 1
            activation="ReLU", # 活性化関数. 大文字小文字を間違えないように．pytorchに実装されているものに加えてml_networksに実装されているものも使える．
            groups=1, # 入力channelを何グループに分けるか．変えることはほぼない．Default: 1
            bias=True, # バイアスを使うかどうか．変えることはほぼない．Default: True
            norm="none", # 正規化を行うかどうか．"none"で正規化なし．"batch"でバッチ正規化．"group"+norm_cfgの設定でGroupNorm, InstanceNorm, LayerNormが使える．
            norm_cfg={}, # 正規化の設定．それぞれの正規化層で変更できる設定はpytorch公式ドキュメントを参照．"num_groups"の設定でGroupNorm, InstanceNorm, LayerNormに切り替え可能.

        ),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
    ]
)

## ResNet+PixelUnShuffleのエンコーダ
encoder_config = ResNetConfig(
    conv_channel=64,
    conv_kernel=3,
    f_kernel=3,
    conv_activation="ReLU",
    output_activation="ReLU",
    n_res_blocks=3,
    scale_factor=2,
    n_scaling=3,
    norm="batch",
    norm_cfg={"affine": True},
    dropout=0.0,
)

full_connection_cfg = MLPConfig(
    hidden_dim=128,
    n_layers=2,
    output_activation="Tanh",
    linaer_cfg=LinearConfig(
        activation="ReLU",
        bias=True,
    )
)
obs_shape = (3, 64, 64)
feature_dim = 64

encoder = Encoder(feature_dim, obs_shape, encoder_config, full_connection_cfg)

obs = torch.randn(32, 3, 64, 64)
z = encoder(obs)
print(z.shape)
>>> torch.Size([32, 64])


```

### Decoder
```python
from ml_networks import Decoder, ConvNetConfig, MLPConfig, LinearConfig


