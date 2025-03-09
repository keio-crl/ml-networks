import torch
import torch.nn as nn
import torch.nn.functional as F


class Activation(nn.Module):
    def __init__(self, activation: str, reparametarize_fn: str = "gelu", **kwargs):
        super().__init__()
        if "glu" not in activation.lower():
            kwargs.pop("dim", None)
        try:
            self.activation = getattr(nn, activation)(**kwargs)
        except AttributeError:
            if activation == "TanhExp":
                self.activation = TanhExp()
            elif activation == "REReLU":
                self.activation = REReLU(reparametarize_fn)
            elif activation == "SiGLU":
                self.activation = SiGLU(**kwargs)
            else:
                raise NotImplementedError(
                    f"Activation: '{activation}' is not implemented yet."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)

class REReLU(nn.Module):
    def __init__(self, reparametarize_fn: str = "gelu") -> None:
        super().__init__()
        reparametarize_fn = reparametarize_fn.lower()
        self.reparametarize_fn = getattr(F, reparametarize_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.relu(x).detach()
            + self.reparametarize_fn(x)
            - self.reparametarize_fn(x).detach()
        )

class SiGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        """
        SiGLU activation function. This is equivalent to SwiGLU (Swish variant of Gated Linear Unit) activation function.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * F.silu(x2)

class TanhExp(nn.Module):
    def __init__(self) -> None:
        """
        TanhExp activation function.
        """
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TanhExpBase.apply(x)


class TanhExpBase(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:

        return x * x.exp().tanh()

    @staticmethod
    def setup_context(
        ctx, inputs: torch.Tensor, output: torch.Tensor
    ):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors

        grad_input = grad_output * (
            x.exp().tanh() - (x * x.exp() * (x.exp().tanh() ** 2 - 1))
        )
        return grad_input

