# ml-networks


こちらは 村田研共通フレームワーク化計画の一環として，基本的な深層学習モデルのアーキテクチャを提供するリポジトリです．

## Installation
以下のようなコマンドでインストールが可能です．  
（※3つの内どれか一つでOK）

```bash
pip install https://github.com/keio-crl/ml-networks.git
```

```bash
rye add ml-networks --git https://github.com/keio-crl/ml-networks.git
```

```bash
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
4. [Distributions](#Distributions)
5. [Data Save and Load](#データの保存読込)
6. [Loss Functions](#損失関数)
7. [その他便利なものたち](#その他便利なものたち)

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
参考:https://zenn.dev/piment/articles/4ff3b6dfd73103

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

decoder = decoder(feature_dim, obs_shape, decoder_cfg, full_connection_cfg)


z = torch.randn(32, feature_dim)
predicted_obs = decoder(z)
print(predicted_obs.shape)
>>> torch.Size([32, 3, 64, 64])

```

### Distributions
stringで分布を指定．
#### 正規分布

```python
from ml_networks import Distribution

feature_dim = 64

# ガウス分布を使う場合は平均と標準偏差で特徴量次元の2倍の次元が必要
full_connection_cfg = MLPConfig(
    hidden_dim=128,
    n_layers=2,
    output_activation="Identity", # 出力層の活性化関数は分布に変換する場合何もかけないのがいい．
                                  # Identityを指定すると何もかけない．
    linaer_cfg=LinearConfig(
        activation="ReLU",
        bias=True,
    )
)
encoder = Encoder(feature_dim*2, obs_shape, encoder_cfg, full_connection_cfg)


dist = Distribution(
        in_dim = feature_dim, # 分布の次元
                              # 正規分布なら平均（標準偏差）の次元
                              # カテゴリカル分布ならカテゴリ数×各カテゴリの次元
                              # ベルヌーイ分布なら超球の数×超球の次元
        dist = "normal", # 分布の種類. Literal["normal", "categorical", "bernoulli"]
        n_gropus = 1, # 分布のグループ数．ガウス分布の場合は意味ない. Default: 1
                      # カテゴリカル分布の場合はカテゴリ数．ベルヌーイ分布の場合は超球の数
        spherical = False,  # カテゴリカル分布の場合には{0, 1} -> {-1, 1}．Default: False
                            # ベルヌーイ分布の場合に超球にするかどうか．Default: False
)

z = encoder(obs)

# 自動的に分布のパラメータへの変換・再パラメータ化トリックが適用される
dist_z = dist(z)
print(dist_z)
>>> NormalStoch(mean: torch.Size([32, 64]), std: torch.Size([32, 64]), stoch: torch.Size([32, 64])
# mean は平均，std は標準偏差，stoch はサンプリングされた特徴量

# torch.distributions.Distributionに変換
torch_dist_z = dist_z.get_distribution(
                independent=1 # データの次元数はいくつか. 基本的に1にしておけばOK. Default: 1
                )

import torch.distributions as D

normal = D.Normal(0, 1)

# KLDの計算
kld = D.kl_divergence(torch_dist_z, normal).mean()

```

#### カテゴリカル分布

```python
encoder = Encoder(feature_dim, obs_shape, encoder_cfg, full_connection_cfg)

dist = Distribution(
        in_dim = feature_dim,
        dist = "categorical",
        n_gropus = 8, # feature_dimがn_gropusの倍数でないとエラーが出る．
)
z = encoder(obs)

dist_z = dist(z)
print(dist_z)
>>> CategoricalStoch(logits: torch.Size([32, 8, 8]), stoch: torch.Size([32, 8, 8]), probs: torch.Size([32, 8, 8]))

flat_dist = D.OneHotCategorical(probs=torch.ones_like(dist_z.probs)/dist_z.probs.shape[-1])
# KLDの計算
kld = D.kl_divergence(dist_z.get_distribution(), flat_dist).mean()

```

#### 分布データのstack, concatenate
```python

dist_list = []
len(dataloader)
>>> 100
for batch in dataloader:
    obs = batch["obs"]
    obs.shape 
    >>> torch.Size([32, 3, 64, 64])
    z = encoder(obs)
    dist_z = dist(z)
    dist_list.append(dist_z)

# 分布データをstack
from ml_networks import stack_dist
stacked_dist = stack_dist(
    dist_list, 
    dim=0 # どの次元でstackするか．Default: 0
)
print(stacked_dist.shape)
>>> NormalShape(mean: torch.Size([100, 32, 64]), std: torch.Size([100, 32, 64]), stoch: torch.Size([100, 32, 64]))

# 分布データをconcatenate
from ml_networks import cat_dist
concatenated_dist = cat_dist(
    dist_list,
    dim=-1 # どの次元でconcatenateするか．Default: -1
)
print(concatenated_dist.shape)
>>> NormalShape(mean: torch.Size([32, 6400]), std: torch.Size([32, 6400]), stoch: torch.Size([32, 6400]))

```

#### 分布データの保存
```python
from ml_networks import Distribution

dist = Distribution(
        in_dim = feature_dim,
        dist = "normal",
        n_gropus = 1,
)

z = encoder(obs)
dist_z = dist(z)

# 分布データの保存
dist_z.save("reports")
# reportsの下にmean.blosc2, std.blosc2, stoch.blosc2が保存される．
# 他の分布データも同様に保存される．

```

### データの保存読込
blosc2形式でデータを保存・読み込みすることを推奨．
圧縮率が高く，保存も高速．refer to [blosc2](https://zenn.dev/zaburo_ch/articles/a13a0772d2f251)

```python

from ml_networks import save_blosc2, load_blosc2

# numpy形式のデータを作成
data = torch.randn(32, 3, 64, 64).detach().cpu().numpy()

# 保存
save_blosc2(data, "dataset/image.blosc2")

# 読み込み
loaded_data = load_blosc2("dataset/image.blosc2")

```

### 損失関数

#### Focal Loss
分類の学習に良いもの．refer to [Focal Loss](https://qiita.com/agatan/items/53fe8d21f2147b0ac982)
```python
from ml_networks import focal_loss, binary_focal_loss

# 多クラス分類の場合
logits = torch.randn(32, 10)
labels = torch.randint(0, 10, (32,))
loss = focal_loss(
    logits, 
    labels, 
    gamma=2.0, # Focal Lossの重みの調整. 論文を見て．Default: 2.0
    sum_dim=-1 # どの次元でsumするか．他の次元は平均を取る．Default: -1
)

# 二値分類の場合
logits = torch.randn(32)
labels = torch.randint(0, 2, (32,))
loss = binary_focal_loss(
    logits, 
    labels, 
    gamma=2.0, # Focal Lossの重みの調整. 論文を見て．Default: 2.0
    sum_dim=-1 # どの次元でsumするか．他の次元は平均を取る．Default: -1
)

```

#### 画像再構成の損失関数
画像再構成の損失関数．

[charbonnier loss](https://arxiv.org/abs/1701.03077)と[focal frequency loss](https://arxiv.org/abs/2012.12821)が使える．
```python
from ml_networks import FocalFrequencyLoss, charbonnier

# charbonnier loss
# 損失の勾配が安定するらしい
loss = charbonnier(
    predicted_obs, 
    obs, 
    epsilon = 1e-3, # charbonnier lossのパラメータ．Default: 1e-3
    alpha=1, # charbonnier lossのパラメータ．Default: 0.45
    sum_dim=[-1, -2, -3] # どの次元でsumするか．他の次元は平均を取る．Default: [-1, -2, -3] 
)

# focal frequency loss
# 画像自体でなく，画像の周波数成分に焦点を当てた損失関数
# モチベーションとしてはFocal Lossを画像に適用したもの

loss_fn = FocalFrequencyLoss(
    loss_weight=1.0, # Focal Frequency Lossの重み Default: 1.0
    alpha=1.0, # spectrum weightのscaling factor Default: 1.0
    patch_factor=1, # the factor to crop image patches for patch-based focal frequency loss. Default: 1
    ave_spectrum=False, # whether to use minibatch average spectrum. Default: False
    log_matrix=False, # whether to adjust the spectrum weight matrix by logarithm. Default: False
    batch_matrix=False # whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
)

loss = loss_fn(predicted_obs, obs)
```


### その他便利なものたち
#### activations
stringで活性化関数を指定．
pytorchに実装されている活性化関数に加えて，以下の活性化関数が使えます．
- "REReLU" 
    - "Reparametrized ReLU": 逆伝播がGELU等になるReLU. See [here](https://openreview.net/forum?id=lNCnZwcH5Z)
- "SiGLU"
    - "SiLU + GLU": SiLU(Swish)とGLUを組み合わせた活性化関数. See [here](https://arxiv.org/abs/2102.11972v2)
- "CRReLU"
    - "Correction Regularized ReLU": 正則化されたReLU. See [here](https://openreview.net/forum?id=7TZYM6Hm9p)
- "TanhExp"
    - Mishの改善版という位置付け. See [here](https://qiita.com/kuroitu/items/73cd401afd463a78115a)
```python

from ml_networks import Activation

act = Activation("ReLU")
```

#### optimizers
stringで最適化手法を指定．
pytorchに実装されている最適化手法に加えて，  
[pytorch_optimizer](https://pypi.org/project/pytorch_optimizer/)に実装されている最適化手法が使えます．
最新のものが多いので便利．
```python
from ml_networks import get_optimizer
import torch.nn as nn

model = nn.Linear(16, 8)

# **kwargsでoptimizerへの様々な設定に関する引数を渡すことができる．
optimizer = get_optimizer(model.parameters(), "Adam", lr=1e-3, weight_decay=1e-4)

```

#### seed固定
seedを固定する．
```python
from ml_networks import torch_fix_seed, determine_loader

# random, np, torchのseedを固定する．
# さらにGPU関連の再現性も（ある程度）担保．
torch_fix_seed(42)


from torch.utils.data import Dataset

dataset = Dataset(any_data)

# DataLoaderの再現性を担保する．
# 通常のdataloaderの呼び出しでは，データの読み出しに関する再現性は担保されない．
loader = determine_loader(
    dataset, 
    seed=42, # 乱数のseed
    batch_size=32, # バッチサイズ
    shuffle=True, # 毎エポックデータセットの中身を入れ替えるか. Validationの時はFalseにする．
    collate_fn=None, # 特定のミニバッチ作成処理がある場合は指定する．
)
```
