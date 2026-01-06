"""損失関数を扱うモジュール."""

from __future__ import annotations

import torch
import torch.distributions as D
import torch.nn.functional as F

from ml_networks.distributions import StochState

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split("+")[0].split("."))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class FocalFrequencyLoss:
    """The torch.nn.Module class that implements focal frequency loss.

    A frequency domain loss function for optimizing generative models.

    Reference
    ---------
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Parameters
    ----------
    loss_weight: float
        weight for focal frequency loss. Default: 1.0
    alpha: float
        the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
    patch_factor: int
        the factor to crop image patches for patch-based focal frequency loss. Default: 1
    ave_spectrum: bool
        whether to use minibatch average spectrum. Default: False
    log_matrix: bool
        whether to adjust the spectrum weight matrix by logarithm. Default: False
    batch_matrix: bool
        whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
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

    def tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0, "Patch factor should be divisible by image height"
        assert w % patch_factor == 0, "Patch factor should be divisible by image width"
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        patch_list: list[torch.Tensor] = [
            x[:, :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
            for i in range(patch_factor)
            for j in range(patch_factor)
        ]

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm="ortho")
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)  # type: ignore[attr-defined]
        return freq

    def loss_formulation(
        self,
        recon_freq: torch.Tensor,
        real_freq: torch.Tensor,
        matrix: torch.Tensor | None = None,
        mean_batch: bool = True,
    ) -> torch.Tensor:
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        min_val = weight_matrix.min().item()
        max_val = weight_matrix.max().item()
        assert min_val >= 0, f"The values of spectrum weight matrix should be >= 0, but got Min: {min_val:.10f}"
        assert max_val <= 1, f"The values of spectrum weight matrix should be <= 1, but got Max: {max_val:.10f}"

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        loss = loss.sum(dim=[-1, -2, -3])
        if mean_batch:
            loss = loss.mean()
        return loss

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: torch.Tensor | None = None,
        mean_batch: bool = True,
    ) -> torch.Tensor:
        """Forward function to calculate focal frequency loss.

        Parameters
        ----------
        pred: torch.Tensor
            of shape (N, C, H, W). Predicted tensor.
        target: torch.Tensor
            of shape (N, C, H, W). Target tensor.
        matrix: torch.Tensor | None
            Default: None (If set to None: calculated online, dynamic).
        mean_batch: bool
            Whether to average over batch dimension.

        Returns
        -------
        torch.Tensor
            The focal frequency loss.
        """
        if target.shape != pred.shape:
            target = target.expand_as(pred)
        if pred.ndim == 5:
            batch_shape = pred.shape[:2]
            pred = pred.flatten(0, 1)
            target = target.flatten(0, 1)
            flattened = True
        else:
            flattened = False

        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        loss = self.loss_formulation(pred_freq, target_freq, matrix, mean_batch) * self.loss_weight
        if flattened and not mean_batch:
            loss = loss.reshape(batch_shape)
        return loss


def charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
    alpha: float = 1,
    sum_dim: int | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """
    Charbonnier loss function.

    Reference
    ---------
    A General and Adaptive Robust Loss Function
    http://arxiv.org/abs/1701.03077

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor.
    target : torch.Tensor
        The target tensor.
    epsilon : float
        A small value to avoid division by zero. Default is 1e-3.
    alpha : float
        The alpha parameter. Default is 1.
    sum_dim : int | list[int] | tuple[int, ...] | None
        The dimension to sum the loss. Default is None (sums over [-1, -2, -3]).

    Returns
    -------
    torch.Tensor
        The Charbonnier loss.

    """
    if sum_dim is None:
        sum_dim = [-1, -2, -3]
    x = prediction - target
    loss = (x**2 + epsilon**2) ** (alpha / 2)
    return torch.sum(loss, dim=sum_dim)


def kl_divergence(posterior: StochState, prior: StochState) -> torch.Tensor:
    """KL divergence between two distributions for StochState in ml-networks.

    Parameters
    ----------
    posterior : StochState
        The posterior distribution.
    prior : StochState
        The prior distribution.

    Returns
    -------
    torch.Tensor
        The KL divergence between the two distributions.

    """
    return D.kl_divergence(posterior.get_distribution(), prior.get_distribution())


def kl_balancing(posterior: StochState, prior: StochState, weight: float = 0.8) -> torch.Tensor:
    """
    KL balancing loss function for StochState in ml-networks.

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
    torch.Tensor
        The KL balancing loss.

    """
    assert 0 <= weight <= 1, "weight should be in the range [0, 1]"
    kld_prior = weight * D.kl_divergence(
        posterior.detach().get_distribution(),
        prior.get_distribution(),
    )

    kld_posterior = (1 - weight) * D.kl_divergence(
        posterior.get_distribution(),
        prior.detach().get_distribution(),
    )

    return kld_prior + kld_posterior


def focal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    sum_dim: int = -1,
) -> torch.Tensor:
    """
    Focal loss function. Mainly for multi-class classification.

    Reference
    ---------
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor. This should be before softmax.
    target : torch.Tensor
        The target tensor.
    gamma : float
        The gamma parameter. Default is 2.0.
    sum_dim : int
        The dimension to sum the loss. Default is -1.

    Returns
    -------
    torch.Tensor
        The focal loss.

    """
    prediction = prediction.unsqueeze(1).transpose(sum_dim, 1).squeeze(-1)
    if gamma:
        log_prob = F.log_softmax(prediction, dim=1)
        prob = torch.exp(log_prob)
        loss = F.nll_loss((1 - prob) ** gamma * log_prob, target, reduction="none")
    else:
        loss = F.cross_entropy(prediction, target, reduction="none")
    return loss.mean(0).sum()


def binary_focal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    sum_dim: int = -1,
) -> torch.Tensor:
    """
    Binary focal loss function. Mainly for binary classification.

    Reference
    ---------
    Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor. This should be before sigmoid.
    target : torch.Tensor
        The target tensor.
    gamma : float
        The gamma parameter. Default is 2.0.
    sum_dim : int
        The dimension to sum the loss. Default is -1.

    Returns
    -------
    torch.Tensor
        The binary focal loss.

    """
    if gamma:
        log_probs = F.logsigmoid(prediction)
        neg_log_probs = F.logsigmoid(-prediction)
        probs = torch.sigmoid(prediction)
        focal_weight = torch.where(target == 1, (1 - probs) ** gamma, probs**gamma)
        loss = torch.where(target == 1, -log_probs, -neg_log_probs)
        loss = focal_weight * loss
    else:
        loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    return loss.sum(sum_dim)
