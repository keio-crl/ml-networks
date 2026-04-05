# ml-networks

**JP** | **[EN](README.md)**

[![PyPI](https://img.shields.io/pypi/v/ml-networks)](https://pypi.org/project/ml-networks/)
[![CI](https://github.com/keio-crl/ml-networks/actions/workflows/ci.yml/badge.svg)](https://github.com/keio-crl/ml-networks/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**PyTorch** と **JAX (Flax NNX)** の両方に対応した深層学習モデル構築ライブラリです。
同一の Config 体系で、フレームワークを切り替えてモデルを構築できます。

> **[Documentation](https://keio-crl.github.io/ml-networks/)** | **[JAX Guide](https://keio-crl.github.io/ml-networks/guides/jax/)**

---

## Features

- **Dual Framework**: PyTorch / JAX (Flax NNX) 対応、共通の Config 体系
- **Vision Models**: Encoder, Decoder, ConvNet, ResNet (PixelShuffle/Unshuffle), Vision Transformer
- **Generative Models**: Conditional UNet (1D/2D) for Diffusion Models
- **Distributions**: Normal, Categorical, Bernoulli, BSQ Codebook
- **Loss Functions**: Focal Loss, Charbonnier Loss, Focal Frequency Loss, KL Divergence
- **Advanced**: HyperNetwork, Contrastive Learning, SpatialSoftmax
- **Utilities**: Custom activations, optimizers, blosc2 I/O, seed control

## Installation

```bash
# PyTorch modules only
pip install ml-networks

# With JAX support (Python 3.11+ required)
pip install "ml-networks[jax]"
```

<details>
<summary>uv / rye</summary>

```bash
# uv
uv add ml-networks
uv add "ml-networks[jax]"  # with JAX

# rye
rye add ml-networks
rye add "ml-networks[jax]"  # with JAX
```

</details>

<details>
<summary>開発版 (GitHub から直接インストール)</summary>

```bash
pip install git+https://github.com/keio-crl/ml-networks.git
```

</details>

### Requirements

| | PyTorch backend | JAX backend |
|---|---|---|
| **Python** | >= 3.10 | >= 3.11 |
| **Framework** | PyTorch >= 2.0 | JAX >= 0.4.30, Flax >= 0.10.0 |

## Quick Start

### MLP

```python
from ml_networks.torch import MLPLayer
from ml_networks import MLPConfig, LinearConfig

cfg = MLPConfig(
    hidden_dim=128,
    n_layers=2,
    output_activation="Tanh",
    linear_cfg=LinearConfig(activation="ReLU", bias=True),
)

mlp = MLPLayer(input_dim=16, output_dim=8, cfg=cfg)
y = mlp(torch.randn(32, 16))  # (32, 8)
```

### Encoder

```python
from ml_networks.torch import Encoder
from ml_networks import ConvNetConfig, ConvConfig, LinearConfig

backbone_cfg = ConvNetConfig(
    channels=[16, 32, 64],
    conv_cfgs=[
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=3, stride=2, padding=1, activation="ReLU"),
    ],
)
fc_cfg = LinearConfig(activation="ReLU", bias=True)

encoder = Encoder(feature_dim=64, obs_shape=(3, 64, 64), backbone_cfg=backbone_cfg, fc_cfg=fc_cfg)
z = encoder(torch.randn(32, 3, 64, 64))  # (32, 64)
```

<details>
<summary>Backbone options</summary>

```python
from ml_networks import ResNetConfig, ViTConfig, TransformerConfig

# ResNet + PixelUnshuffle
backbone_cfg = ResNetConfig(
    conv_channel=64,
    conv_kernel=3,
    f_kernel=3,
    conv_activation="ReLU",
    out_activation="ReLU",
    n_res_blocks=3,
    scale_factor=2,
    n_scaling=3,
    norm="batch",
    norm_cfg={"affine": True},
)

# Vision Transformer
backbone_cfg = ViTConfig(
    patch_size=8,
    transformer_cfg=TransformerConfig(
        d_model=64,
        nhead=8,
        dim_ff=256,
        n_layers=3,
    ),
)
```

</details>

<details>
<summary>FC layer options</summary>

```python
from ml_networks import MLPConfig, LinearConfig, SpatialSoftmaxConfig, AdaptiveAveragePoolingConfig

fc_cfg = MLPConfig(
    hidden_dim=128, n_layers=2, output_activation="Tanh", linear_cfg=LinearConfig(activation="ReLU", bias=True)
)
fc_cfg = LinearConfig(activation="ReLU", bias=True)
fc_cfg = SpatialSoftmaxConfig(temperature=1.0)
fc_cfg = AdaptiveAveragePoolingConfig()
fc_cfg = None  # output feature map directly
```

</details>

### Decoder

```python
from ml_networks.torch import Decoder
from ml_networks import ConvNetConfig, ConvConfig, MLPConfig, LinearConfig

backbone_cfg = ConvNetConfig(
    channels=[64, 32, 16],
    conv_cfgs=[
        ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=4, stride=2, padding=1, activation="ReLU"),
        ConvConfig(kernel_size=4, stride=2, padding=1, activation="Tanh"),
    ],
)
fc_cfg = MLPConfig(
    hidden_dim=256, n_layers=2, output_activation="ReLU", linear_cfg=LinearConfig(activation="ReLU", bias=True)
)

decoder = Decoder(feature_dim=64, obs_shape=(3, 64, 64), backbone_cfg=backbone_cfg, fc_cfg=fc_cfg)
img = decoder(torch.randn(32, 64))  # (32, 3, 64, 64)
```

### Conditional UNet

```python
from ml_networks.torch import ConditionalUnet2d, ConditionalUnet1d
from ml_networks import UNetConfig, ConvConfig

cfg = UNetConfig(
    channels=[64, 128, 256],
    conv_cfg=ConvConfig(kernel_size=3, padding=1, stride=1, activation="ReLU"),
    has_attn=True,
    nhead=8,
    cond_pred_scale=True,
)

# 2D (images)
net = ConditionalUnet2d(feature_dim=32, obs_shape=(3, 64, 64), cfg=cfg)
out = net(torch.randn(2, 3, 64, 64), cond=torch.randn(2, 32))  # (2, 3, 64, 64)

# 1D (sequences)
net = ConditionalUnet1d(feature_dim=32, obs_shape=(8, 128), cfg=cfg)
out = net(torch.randn(2, 8, 128), cond=torch.randn(2, 32))  # (2, 8, 128)
```

### Distributions

```python
from ml_networks.torch import Distribution, Encoder, stack_dist, cat_dist

dist = Distribution(in_dim=64, dist="normal")
encoder = Encoder(feature_dim=128, obs_shape=(3, 64, 64), backbone_cfg=backbone_cfg, fc_cfg=fc_cfg)

z = encoder(obs)  # (B, 128) — mean & std concatenated
dist_z = dist(z)  # NormalStoch(mean, std, stoch)

# Convert to torch.distributions for KLD computation
torch_dist = dist_z.get_distribution(independent=1)

# Stack / concatenate distribution objects
stacked = stack_dist(dist_list, dim=0)
catted = cat_dist(dist_list, dim=-1)

# Save to disk (blosc2 format)
dist_z.save("reports/")
```

### Loss Functions

```python
from ml_networks.torch import focal_loss, binary_focal_loss, FocalFrequencyLoss, charbonnier

# Focal loss (classification)
loss = focal_loss(logits, labels, gamma=2.0)

# Charbonnier loss (image reconstruction)
loss = charbonnier(pred, target, epsilon=1e-3)

# Focal frequency loss (frequency-domain reconstruction)
ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
loss = ffl(pred, target)
```

### JAX Backend

同一の Config でフレームワークを切り替え可能です。JAX では初期化時に `rngs` が必要です。

```python
from ml_networks.jax import MLPLayer, Encoder, Decoder
from ml_networks import MLPConfig, LinearConfig, ConvNetConfig, ConvConfig
from flax import nnx
import jax.numpy as jnp

cfg = MLPConfig(
    hidden_dim=128,
    n_layers=2,
    output_activation="Tanh",
    linear_cfg=LinearConfig(activation="ReLU", bias=True),
)

rngs = nnx.Rngs(0)
mlp = MLPLayer(input_dim=16, output_dim=8, cfg=cfg, rngs=rngs)
y = mlp(jnp.ones((32, 16)))  # (32, 8)
```

> JAX modules use NHWC format (channels-last). See the [JAX Guide](https://keio-crl.github.io/ml-networks/guides/jax/) for details.

### Data I/O (blosc2)

```python
from ml_networks import save_blosc2, load_blosc2

save_blosc2(data, "dataset/image.blosc2")
loaded = load_blosc2("dataset/image.blosc2")
```

### Utilities

```python
from ml_networks.torch import Activation, get_optimizer, torch_fix_seed
from ml_networks import determine_loader

# Custom activations: REReLU, SiGLU, CRReLU, TanhExp, L2Norm
act = Activation("REReLU")

# Optimizer (supports pytorch-optimizer library)
optimizer = get_optimizer(model.parameters(), "Adam", lr=1e-3)

# Reproducibility
torch_fix_seed(42)
loader = determine_loader(dataset, seed=42, batch_size=32, shuffle=True)
```

## Package Structure

```
ml_networks/
├── config.py          # Shared config classes (PyTorch/JAX)
├── utils.py           # Shared utilities (blosc2 I/O, conv shape calc)
├── callbacks.py       # PyTorch Lightning callbacks
├── torch/             # PyTorch implementation
│   ├── layers.py      # MLP, Conv, Attention, Transformer
│   ├── vision.py      # Encoder, Decoder, ConvNet, ResNet, ViT
│   ├── unet.py        # ConditionalUnet1d/2d
│   ├── distributions.py
│   ├── loss.py
│   ├── activations.py
│   ├── hypernetworks.py
│   ├── contrastive.py
│   └── torch_utils.py
└── jax/               # JAX (Flax NNX) implementation
    ├── layers.py
    ├── vision.py
    ├── unet.py
    ├── distributions.py
    ├── loss.py
    ├── activations.py
    ├── hypernetworks.py
    ├── contrastive.py
    └── jax_utils.py
```

## Development

```bash
git clone https://github.com/keio-crl/ml-networks.git
cd ml-networks
pip install -e ".[dev]"

# Quality checks
ruff check .          # Lint
ruff format .         # Format
mypy src/             # Type check
pre-commit run --all-files  # All checks
```

### Versioning

Semantic Versioning (`MAJOR.MINOR.PATCH`).

```bash
python scripts/bump_version.py patch   # 0.1.0 -> 0.1.1
python scripts/bump_version.py minor   # 0.1.0 -> 0.2.0
python scripts/bump_version.py major   # 0.1.0 -> 1.0.0
```

Or use the **Version Bump** workflow on GitHub Actions.

### CI/CD

| Workflow | Trigger | Checks |
|----------|---------|--------|
| **CI** | Push / PR to main, develop | ruff, mypy, pytest, build |
| **Release** | Tag push (`vX.Y.Z`) | Build + GitHub Release |
| **Docs** | Push to main | Deploy documentation |

## Authors

- oakwood-fujiken (oakwood.n14.4sp@keio.jp)
- nomutin (nomura0508@icloud.com)
