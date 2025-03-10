import torch
import torch.nn.functional as F
from torch import nn


class Activation(nn.Module):
    def __init__(self, activation: str, **kwargs) -> None:
        super().__init__()
        if "glu" not in activation.lower():
            kwargs.pop("dim", None)
        try:
            self.activation = getattr(nn, activation)(**kwargs)
        except AttributeError:
            if activation == "TanhExp":
                self.activation = TanhExp()
            elif activation == "REReLU":
                self.activation = REReLU(**kwargs)
            elif activation == "SiGLU" or activation == "SwiGLU":
                self.activation = SiGLU(**kwargs)
            elif activation == "CRReLU":
                self.activation = CRReLU(**kwargs)
            else:
                raise NotImplementedError(
                    f"Activation: '{activation}' is not implemented yet.",
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class REReLU(nn.Module):
    def __init__(self, reparametarize_fn: str = "gelu") -> None:
        super().__init__()
        """
        Reparametarized ReLU activation function. This backward pass is differentiable.

        References
        ----------
        https://openreview.net/forum?id=lNCnZwcH5Z

        Parameters:
        ----------
        reparametarize_fn : str
            Reparametarization function. Default is GELU.

        Examples:
        ---------
        >>> rerelu = REReLU()
        >>> x = torch.randn(1, 3)
        >>> output = rerelu(x)
        >>> output.shape
        torch.Size([1, 3])

        """
        reparametarize_fn = reparametarize_fn.lower()
        self.reparametarize_fn = getattr(F, reparametarize_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).detach() + self.reparametarize_fn(x) - self.reparametarize_fn(x).detach()


class CRReLU(nn.Module):
    def __init__(self, lr: float = 0.01) -> None:
        """
        Correction Regularized ReLU activation function. This is a variant of ReLU activation function.

        References
        ----------
        https://openreview.net/forum?id=7TZYM6Hm9p

        Parameters
        ----------
        lr : float
            Learning rate. Default is 0.01.

        Examples
        --------
        >>> crrelu = CRReLU()
        >>> x = torch.randn(1, 3)
        >>> output = crrelu(x)
        >>> output.shape
        torch.Size([1, 3])
        """
        super().__init__()
        self.lr = nn.Parameter(torch.tensor(lr).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) + self.lr * x * torch.exp(-(x**2) / 2)


class SiGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        """
        SiGLU activation function.

        This is equivalent to SwiGLU (Swish variant of Gated Linear Unit) activation function.

        References
        ----------
        https://arxiv.org/abs/2102.11972

        Parameters
        ----------
        dim : int
            Dimension to split the tensor. Default is -1.

        Examples
        --------
        >>> siglu = SiGLU()
        >>> x = torch.randn(1, 4)
        >>> output = siglu(x)
        >>> output.shape
        torch.Size([1, 2])

        >>> siglu = SiGLU(dim=0)
        >>> x = torch.randn(4, 1)
        >>> output = siglu(x)
        >>> output.shape
        torch.Size([2, 1])
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * F.silu(x2)


class TanhExp(nn.Module):
    """
    TanhExp activation function.

    Examples
    --------
    >>> tanhexp = TanhExp()
    >>> x = torch.randn(1, 3)
    >>> output = tanhexp(x)
    >>> output.shape
    torch.Size([1, 3])
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TanhExpBase.apply(x)


class TanhExpBase(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x * x.exp().tanh()

    @staticmethod
    def setup_context(
        ctx,
        inputs: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors

        grad_input = grad_output * (x.exp().tanh() - (x * x.exp() * (x.exp().tanh() ** 2 - 1)))
        return grad_input


if __name__ == "__main__":
    import doctest

    doctest.testmod()
