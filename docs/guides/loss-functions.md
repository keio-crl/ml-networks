# 損失関数ガイド

損失関数の使用方法を説明します。

## Focal Loss

分類の学習に適した損失関数です。

### 多クラス分類の場合

```python
from ml_networks import focal_loss
import torch

logits = torch.randn(32, 10)
labels = torch.randint(0, 10, (32,))
loss = focal_loss(
    logits,
    labels,
    gamma=2.0,   # Focal Lossの重みの調整
    sum_dim=-1   # どの次元でsumするか
)
```

### 二値分類の場合

```python
from ml_networks import binary_focal_loss

logits = torch.randn(32)
labels = torch.randint(0, 2, (32,))
loss = binary_focal_loss(
    logits,
    labels,
    gamma=2.0,
    sum_dim=-1
)
```

## Charbonnier Loss

画像再構成の損失関数です。損失の勾配が安定します。

```python
from ml_networks import charbonnier

loss = charbonnier(
    predicted_obs,
    obs,
    epsilon=1e-3,  # charbonnier lossのパラメータ
    alpha=1,       # charbonnier lossのパラメータ
    sum_dim=[-1, -2, -3]  # どの次元でsumするか
)
```

## Focal Frequency Loss

画像自体でなく、画像の周波数成分に焦点を当てた損失関数です。Focal Lossを画像に適用したものという位置付けです。

```python
from ml_networks import FocalFrequencyLoss

loss_fn = FocalFrequencyLoss(
    loss_weight=1.0,      # Focal Frequency Lossの重み
    alpha=1.0,            # spectrum weightのscaling factor
    patch_factor=1,       # パッチベースのfocal frequency loss用のクロップファクタ
    ave_spectrum=False,   # ミニバッチ平均スペクトラムを使うかどうか
    log_matrix=False,     # スペクトラム重み行列を対数で調整するかどうか
    batch_matrix=False    # バッチベースの統計でスペクトラム重み行列を計算するかどうか
)

loss = loss_fn(predicted_obs, obs)
```

## KL Divergence

分布間のKLダイバージェンスを計算します。

```python
from ml_networks import kl_divergence
import torch.distributions as D

# 例: 正規分布間のKLダイバージェンス
dist1 = D.Normal(0, 1)
dist2 = D.Normal(1, 2)
kld = kl_divergence(dist1, dist2)
```

## KL Balancing

複数のKLダイバージェンスをバランスするためのユーティリティです。

```python
from ml_networks import kl_balancing

# 複数のKLダイバージェンスをバランス
kld_list = [kld1, kld2, kld3]
balanced_kld = kl_balancing(kld_list, alpha=0.5)
```
