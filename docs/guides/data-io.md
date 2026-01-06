# データの保存と読み込み

データの保存と読み込み機能を説明します。

## blosc2形式での保存・読み込み

`ml-networks`では、blosc2形式でのデータ保存・読み込みを推奨しています。圧縮率が高く、保存も高速です。

### 基本的な使用方法

```python
from ml_networks import save_blosc2, load_blosc2
import torch
import numpy as np

# numpy形式のデータを作成
data = torch.randn(32, 3, 64, 64).detach().cpu().numpy()

# 保存
save_blosc2(data, "dataset/image.blosc2")

# 読み込み
loaded_data = load_blosc2("dataset/image.blosc2")
```

## 分布データの保存

分布データも保存できます：

```python
from ml_networks import Distribution

dist = Distribution(
    in_dim=feature_dim,
    dist="normal",
    n_groups=1,
)

z = encoder(obs)
dist_z = dist(z)

# 分布データの保存
dist_z.save("reports")
# reportsの下にmean.blosc2, std.blosc2, stoch.blosc2が保存される
# 他の分布データも同様に保存される
```

## 注意事項

- blosc2形式は圧縮率が高く、保存も高速です
- 大きなデータセットを扱う場合に特に有効です
- 分布データを保存する場合は、各パラメータ（mean、std、stochなど）が個別のファイルとして保存されます
