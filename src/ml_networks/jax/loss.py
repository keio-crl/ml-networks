"""損失関数を扱うモジュール."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ml_networks.jax.distributions import StochState


class FocalFrequencyLoss:
    """Focal frequency loss for image reconstruction and synthesis.

    Uses NHWC (channels-last) format: input (B, H, W, C).

    Reference
    ---------
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Parameters
    ----------
    loss_weight : float
        Weight for focal frequency loss. Default: 1.0
    alpha : float
        Scaling factor alpha of the spectrum weight matrix. Default: 1.0
    patch_factor : int
        Factor to crop image patches. Default: 1
    ave_spectrum : bool
        Whether to use minibatch average spectrum. Default: False
    log_matrix : bool
        Whether to adjust the spectrum weight matrix by logarithm. Default: False
    batch_matrix : bool
        Whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ) -> None:
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x: jax.Array) -> jax.Array:
        """Convert NHWC tensor to frequency domain.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (B, H, W, C).

        Returns
        -------
        jax.Array
            Frequency tensor with real and imaginary stacked on last dim.
        """
        patch_factor = self.patch_factor
        _, h, w, _ = x.shape
        assert h % patch_factor == 0, "Patch factor should be divisible by image height"
        assert w % patch_factor == 0, "Patch factor should be divisible by image width"
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        # NHWC format patches
        patch_list = [
            x[:, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w, :]
            for i in range(patch_factor)
            for j in range(patch_factor)
        ]

        y = jnp.stack(patch_list, axis=1)

        # 2D FFT (ortho normalized)
        # y shape: (B, n_patches, patch_h, patch_w, C)
        freq = jnp.fft.fft2(y, axes=(-3, -2), norm="ortho")
        return jnp.stack([freq.real, freq.imag], axis=-1)

    def loss_formulation(
        self,
        recon_freq: jax.Array,
        real_freq: jax.Array,
        matrix: jax.Array | None = None,
        mean_batch: bool = True,
    ) -> jax.Array:
        """Compute the focal frequency loss."""
        if matrix is not None:
            weight_matrix = jax.lax.stop_gradient(matrix)
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = jnp.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            if self.log_matrix:
                matrix_tmp = jnp.log(matrix_tmp + 1.0)

            if self.batch_matrix:
                matrix_tmp = matrix_tmp / jnp.maximum(matrix_tmp.max(), 1e-10)
            else:
                max_vals = matrix_tmp.max(axis=-1, keepdims=True).max(axis=-2, keepdims=True)
                matrix_tmp = matrix_tmp / jnp.maximum(max_vals, 1e-10)

            matrix_tmp = jnp.where(jnp.isnan(matrix_tmp), 0.0, matrix_tmp)
            matrix_tmp = jnp.clip(matrix_tmp, 0.0, 1.0)
            weight_matrix = jax.lax.stop_gradient(matrix_tmp)

        # Frequency distance (squared Euclidean)
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        loss = weight_matrix * freq_distance
        loss = loss.sum(axis=(-1, -2, -3))
        if mean_batch:
            loss = loss.mean()
        return loss

    def __call__(
        self,
        pred: jax.Array,
        target: jax.Array,
        matrix: jax.Array | None = None,
        mean_batch: bool = True,
    ) -> jax.Array:
        """Forward function to calculate focal frequency loss.

        Parameters
        ----------
        pred : jax.Array
            Predicted tensor of shape (B, H, W, C) in NHWC format.
        target : jax.Array
            Target tensor of shape (B, H, W, C) in NHWC format.
        matrix : jax.Array | None
            Predefined spectrum weight matrix. Default: None.
        mean_batch : bool
            Whether to average over batch dimension.

        Returns
        -------
        jax.Array
            The focal frequency loss.
        """
        if target.shape != pred.shape:
            target = jnp.broadcast_to(target, pred.shape)
        if pred.ndim == 5:
            batch_shape = pred.shape[:2]
            pred = pred.reshape(-1, *pred.shape[2:])
            target = target.reshape(-1, *target.shape[2:])
            flattened = True
        else:
            flattened = False

        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = jnp.mean(pred_freq, axis=0, keepdims=True)
            target_freq = jnp.mean(target_freq, axis=0, keepdims=True)

        loss = self.loss_formulation(pred_freq, target_freq, matrix, mean_batch) * self.loss_weight
        if flattened and not mean_batch:
            loss = loss.reshape(batch_shape)
        return loss


def charbonnier(
    prediction: jax.Array,
    target: jax.Array,
    epsilon: float = 1e-3,
    alpha: float = 1,
    sum_axes: int | list[int] | tuple[int, ...] | None = None,
) -> jax.Array:
    """
    Charbonnier loss function.

    Reference
    ---------
    A General and Adaptive Robust Loss Function
    http://arxiv.org/abs/1701.03077

    Parameters
    ----------
    prediction : jax.Array
        The predicted tensor.
    target : jax.Array
        The target tensor.
    epsilon : float
        A small value to avoid division by zero. Default is 1e-3.
    alpha : float
        The alpha parameter. Default is 1.
    sum_axes : int | list[int] | tuple[int, ...] | None
        The axes to sum the loss. Default is None (sums over [-1, -2, -3]).

    Returns
    -------
    jax.Array
        The Charbonnier loss.
    """
    if sum_axes is None:
        sum_axes = [-1, -2, -3]
    x = prediction - target
    loss = (x**2 + epsilon**2) ** (alpha / 2)
    return jnp.sum(loss, axis=sum_axes)


def kl_divergence(posterior: StochState, prior: StochState) -> jax.Array:
    """KL divergence between two StochState distributions.

    Parameters
    ----------
    posterior : StochState
        The posterior distribution.
    prior : StochState
        The prior distribution.

    Returns
    -------
    jax.Array
        The KL divergence.
    """
    return posterior.get_distribution().kl_divergence(prior.get_distribution())


def kl_balancing(posterior: StochState, prior: StochState, weight: float = 0.8) -> jax.Array:
    """
    KL balancing loss function for StochState.

    Reference
    ---------
    Mastering Atari with Discrete World Models. In NeurIPS 2020.
    https://arxiv.org/abs/2010.02193

    Parameters
    ----------
    posterior : StochState
        The posterior distribution.
    prior : StochState
        The prior distribution.
    weight : float
        The weight of prior gradient for the balancing. Default is 0.8.

    Returns
    -------
    jax.Array
        The KL balancing loss.
    """
    assert 0 <= weight <= 1, "weight should be in the range [0, 1]"
    kld_prior = weight * posterior.detach().get_distribution().kl_divergence(
        prior.get_distribution(),
    )

    kld_posterior = (1 - weight) * posterior.get_distribution().kl_divergence(
        prior.detach().get_distribution(),
    )

    return kld_prior + kld_posterior


def focal_loss(
    prediction: jax.Array,
    target: jax.Array,
    gamma: float = 2.0,
    sum_axis: int = -1,
) -> jax.Array:
    """
    Focal loss function. Mainly for multi-class classification.

    Reference
    ---------
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    prediction : jax.Array
        The predicted tensor. This should be before softmax.
    target : jax.Array
        The target tensor (integer class labels).
    gamma : float
        The gamma parameter. Default is 2.0.
    sum_axis : int
        The axis to sum the loss. Default is -1.

    Returns
    -------
    jax.Array
        The focal loss.
    """
    # Rearrange: unsqueeze(1), transpose sum_axis with 1, squeeze(-1)
    prediction = jnp.expand_dims(prediction, axis=1)
    prediction = jnp.moveaxis(prediction, sum_axis, 1)
    prediction = jnp.squeeze(prediction, axis=-1)

    if gamma:
        log_prob = jax.nn.log_softmax(prediction, axis=1)
        prob = jnp.exp(log_prob)
        # nll_loss equivalent: -log_prob[target]
        n_classes = prediction.shape[1]
        target_one_hot = jax.nn.one_hot(target, n_classes)
        loss = -jnp.sum(((1 - prob) ** gamma) * log_prob * target_one_hot, axis=1)
    else:
        # Cross entropy
        n_classes = prediction.shape[1]
        target_one_hot = jax.nn.one_hot(target, n_classes)
        log_prob = jax.nn.log_softmax(prediction, axis=1)
        loss = -jnp.sum(log_prob * target_one_hot, axis=1)
    return loss.mean(axis=0).sum()


def binary_focal_loss(
    prediction: jax.Array,
    target: jax.Array,
    gamma: float = 2.0,
    sum_axis: int = -1,
) -> jax.Array:
    """
    Binary focal loss function. Mainly for binary classification.

    Reference
    ---------
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    prediction : jax.Array
        The predicted tensor. This should be before sigmoid.
    target : jax.Array
        The target tensor.
    gamma : float
        The gamma parameter. Default is 2.0.
    sum_axis : int
        The axis to sum the loss. Default is -1.

    Returns
    -------
    jax.Array
        The binary focal loss.
    """
    if gamma:
        log_probs = jax.nn.log_sigmoid(prediction)
        neg_log_probs = jax.nn.log_sigmoid(-prediction)
        probs = jax.nn.sigmoid(prediction)
        focal_weight = jnp.where(target == 1, (1 - probs) ** gamma, probs**gamma)
        loss = jnp.where(target == 1, -log_probs, -neg_log_probs)
        loss = focal_weight * loss
    else:
        # Binary cross entropy with logits
        loss = jnp.maximum(prediction, 0) - prediction * target + jnp.log(1 + jnp.exp(-jnp.abs(prediction)))
    return loss.sum(axis=sum_axis)
