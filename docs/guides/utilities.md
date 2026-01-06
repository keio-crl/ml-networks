# その他の便利な機能

その他の便利な機能を説明します。

## 活性化関数

stringで活性化関数を指定できます。PyTorchに実装されている活性化関数に加えて、以下の活性化関数が使えます：

- **"REReLU"**: Reparametrized ReLU - 逆伝播がGELU等になるReLU
- **"SiGLU"**: SiLU + GLU - SiLU(Swish)とGLUを組み合わせた活性化関数
- **"CRReLU"**: Correction Regularized ReLU - 正則化されたReLU
- **"TanhExp"**: Mishの改善版という位置付け

```python
from ml_networks import Activation

act = Activation("ReLU")
x = torch.randn(32, 64)
y = act(x)
```

## 最適化手法

stringで最適化手法を指定できます。PyTorchに実装されている最適化手法に加えて、[pytorch_optimizer](https://pypi.org/project/pytorch_optimizer/)に実装されている最適化手法が使えます。

```python
from ml_networks import get_optimizer
import torch.nn as nn

model = nn.Linear(16, 8)

# **kwargsでoptimizerへの様々な設定に関する引数を渡すことができる
optimizer = get_optimizer(model.parameters(), "Adam", lr=1e-3, weight_decay=1e-4)
```

## seed固定

再現性を担保するためにseedを固定できます。

```python
from ml_networks import torch_fix_seed, determine_loader

# random, np, torchのseedを固定する
# さらにGPU関連の再現性も（ある程度）担保
torch_fix_seed(42)
```

## DataLoaderの再現性

DataLoaderの再現性を担保するために`determine_loader`を使用します。

```python
from torch.utils.data import Dataset

dataset = Dataset(any_data)

# DataLoaderの再現性を担保する
# 通常のdataloaderの呼び出しでは、データの読み出しに関する再現性は担保されない
loader = determine_loader(
    dataset,
    seed=42,        # 乱数のseed
    batch_size=32,  # バッチサイズ
    shuffle=True,   # 毎エポックデータセットの中身を入れ替えるか
    collate_fn=None,  # 特定のミニバッチ作成処理がある場合は指定する
)
```

## Gumbel Softmax

Gumbel Softmaxを使用できます。

```python
from ml_networks import gumbel_softmax

logits = torch.randn(32, 10)
samples = gumbel_softmax(logits, temperature=1.0, hard=True)
```

## Softmax

カスタムSoftmaxを使用できます。

```python
from ml_networks import softmax

logits = torch.randn(32, 10)
probs = softmax(logits, dim=-1)
```

## MinMaxNormalize

Min-Max正規化を実行できます。

```python
from ml_networks import MinMaxNormalize

normalize = MinMaxNormalize(min=0.0, max=1.0)
data = torch.randn(32, 3, 64, 64)
normalized_data = normalize(data)
```

## SoftmaxTransformation

Softmax変換を実行できます。

```python
from ml_networks import SoftmaxTransformation

transform = SoftmaxTransformation(temperature=1.0)
data = torch.randn(32, 10)
transformed_data = transform(data)
```
