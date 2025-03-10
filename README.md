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

### 目次
1. [MLP](#MLP)
2. [Encoder](#Encoder)
3. [Decoder](#Decoder)

### MLP

```python
from ml_networks import MLPLayer, MLPConfig, LinearConfig

mlp_config = MLPConfig(
        hidden_dim= 128, # 隠れ層の次元
        n_layers= 2, # 隠れ層の数
        output_activation="Tanh", # 出力層の活性化関数
        linaer_cfg=LinearConfig(
            activation="ReLU", # 活性化関数
            bias=True, # バイアスを使うかどうか Default: True
            norm="none", # 正規化を行うかどうか. Default: "none"
                         # "none"で正規化なし．"layer"でLayerNorm, "rms"でRMSNormが使える．
            norm_cfg={}, # 正規化の設定．それぞれの正規化層で変更できる設定はpytorch公式ドキュメントを参照．
            norm_first=False, # 正規化をnn.Linearの前に行うかどうか．Default: False
            dropout=0.0, # ドロップアウト率．0より大きくするとその割合だけドロップアウトする．Default: 0.0
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

#### Import
```python

from ml_networks import (Encoder, ConvNetConfig,
                         ViTConfig, ResNetConfig,
                         MLPConfig, LinearConfig)

```
#### backboneの設定
画像のDownsampling処理を行うアーキテクチャを変えたい場合は引数に渡すConfigを変える

```python
 
# 多層CNNのエンコーダ
encoder_cfg = ConvNetConfig(
    channels=[16, 32, 64], # 各層のchannel数．最初から順に入力（画像）に近い層のchannel数を指定．
                           # ここの数はconv_cfgsの数と一致している必要がある． 
    conv_cfgs=[
        ConvConfig(
            kernel_size=3, # カーネルサイズ
            stride=2, # ストライド
            padding=1, # パディング
            dilation=1, # ダイレーション．変えることはほぼない．Default: 1
            activation="ReLU", # 活性化関数. 大文字小文字を間違えないように．
                               # pytorchに実装されているものに加えて，
                               # ml_networks.activationsに実装されているものも使える．
            groups=1, # 入力channelを何グループに分けるか．変えることはほぼない．Default: 1
            bias=True, # バイアスを使うかどうか．変えることはほぼない．Default: True
            norm="none", # 正規化を行うかどうか．"none"で正規化なし．"batch"でバッチ正規化．
                         # "group"+norm_cfgの設定でGroupNorm, InstanceNorm, LayerNormが使える．
            norm_cfg={}, # 正規化の設定．それぞれの正規化層で変更できる設定はpytorch公式ドキュメントを参照．
                         # "num_groups"の設定でGroupNorm, InstanceNorm, LayerNormに切り替え可能.
            dropout=0.0, # ドロップアウト率．0より大きくするとその割合だけドロップアウトする．Default: 0.0
            scale_factor=0 # PixelUnShuffle・PixelShuffleのスケールファクタ．
                           # 0より大きいとPixelShuffle, 0より小さいとPixelUnShuffleが設定のスケールで行われる．
                           # Default: 0
        ),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"), # 最低限の設定の場合はこれ
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
    ]
)

# ResNet+PixelUnShuffleのエンコーダ
encoder_cfg = ResNetConfig(
    conv_channel=64, # channel数. 全ての層で同じchannel
    conv_kernel=3, # カーネルサイズ
    f_kernel=3, # 最初 or 最後の層のカーネルサイズ
    conv_activation="ReLU", # 活性化関数
    output_activation="ReLU", # 出力層の活性化関数
    n_res_blocks=3, # ResBlockの数
    scale_factor=2, # PixelUnShuffleのスケールファクタ. 1回のPixelUnShuffleで何分の一にするか．
    n_scaling=3, # PixelUnShuffleの数
    norm="batch", # 正規化の種類. ConvConfigと同じ．
    norm_cfg={"affine": True}, # 正規化の設定. ConvConfigと同じ．
    dropout=0.0, # ドロップアウト率. ConvConfigと同じ．
)

# !! ViTConfigにすればVisionTransformerが使える．
# !! しかし，実装が若干不安なので非推奨．
```
#### 全結合層の設定
特徴次元に変換する全結合層の設定を行う．

```python

# 何層かの全結合層を追加する場合は以下のように設定する
full_connection_cfg = MLPConfig(
    hidden_dim=128,
    n_layers=2,
    output_activation="Tanh",
    linaer_cfg=LinearConfig(
        activation="ReLU",
        bias=True,
    )
)

# 畳み込みの後すぐに特徴次元にする場合は以下のように設定する
full_connection_cfg = LinearConfig(
    activation="ReLU",
    bias=True,
)

# 特徴マップをそのまま出力する場合は以下のように設定する
full_connection_cfg = None

```

#### 使用例
```python

obs_shape = (3, 64, 64)
feature_dim = 64
# 特徴マップをそのまま出力する場合
## feature_dim = <backboneの出力特徴マップ次元>
## 違うのを渡すとエラーで正しいものを教えてくれる

encoder = Encoder(feature_dim, obs_shape, encoder_cfg, full_connection_cfg)

obs = torch.randn(32, 3, 64, 64)
z = encoder(obs)
print(z.shape)
>>> torch.Size([32, 64])


```

### Decoder
#### Import
```python
from ml_networks import Decoder, ConvNetConfig, MLPConfig, LinearConfig, ResNetConfig

```
#### backboneの設定
Upsamplingを行うアーキテクチャを変えたい場合は引数に渡すConfigを変える

```python

# 多層ConvTransposeのデコーダ
# エンコーダにはないoutput_paddingの設定が可能.
# 参考: https://note.com/kiyo_ai_note/n/ne4d78a36de04
decoder_cfg = ConvNetConfig(
    channels=[64, 32, 16], # 各層のchannel数．最初から順に入力（特徴量）に近い層のchannel数を指定．
                           # ここの数はconv_cfgsの数と一致している必要がある． 
    conv_cfgs=[
        ConvConfig(
            output_padding=0, # 出力パディング. ConvTranspose2dのみに利用される．Default: 0
                              # 他はエンコーダの場合と同じ.
            kernel_size=3, # カーネルサイズ
            stride=2, # ストライド
            padding=1, # パディング
        ),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"), # 最低限の設定の場合はこれ
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="Tanh"), # 最後の層は出力層なので活性化関数を変える
    ]

)

# ResNet+PixelShuffleのデコーダ
# エンコーダのPixelUnShuffle -> PixelShuffleとなったバージョン．
# PixelShuffleはPixelUnShuffleの逆なので，縦横がデカくなる．
encoder_config = ResNetConfig(
    conv_channel=64, # channel数. 全ての層で同じchannel
    conv_kernel=3, # カーネルサイズ
    f_kernel=3, # 最初 or 最後の層のカーネルサイズ
    conv_activation="ReLU", # 活性化関数
    output_activation="Tanh", # 出力層の活性化関数
    n_res_blocks=3, # ResBlockの数
    scale_factor=2, # PixelShuffleのスケールファクタ. 1回のPixelUhuffleで何倍にするか
    n_scaling=3, # PixelUnShuffleの数
    norm="batch", # 正規化の種類. ConvConfigと同じ．
    norm_cfg={"affine": True}, # 正規化の設定. ConvConfigと同じ．
    dropout=0.0, # ドロップアウト率. ConvConfigと同じ．
)

```

#### 全結合層の設定
エンコーダと同様．エンコーダにおける説明の「出力」を「入力」に読み変えればそのまま

#### 使用例

```python

obs_shape = (3, 64, 64)
feature_dim = 64
# 特徴マップをそのまま入力する場合
## feature_dim = <backboneの入力特徴マップ次元>
## 違うのを渡すとエラーが出る．

encoder = Encoder(feature_dim, obs_shape, decoder_cfg, full_connection_cfg)

z = torch.randn(32, feature_dim)
predicted_obs = decoder(z)
print(predicted_obs.shape)
>>> torch.Size([32, 3, 64, 64])

```


