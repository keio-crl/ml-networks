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

