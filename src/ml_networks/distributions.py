from typing import Literal, Union, Tuple
import os

import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
from ml_networks.utils import save_blosc2, softmax
from dataclasses import dataclass

@dataclass
class NormalShape:
    """
    Shapes of the parameters of a normal distribution and its stochastic sample.

    Attributes
    ----------
    mean : torch.Size
        Shape of the mean of the normal distribution.
    std : torch.Size
        Shape of the standard deviation of the normal distribution.
    stoch : torch.Size
        Shape of the sample from the normal distribution with reparametrization trick.

    """

    mean: torch.Size
    std: torch.Size
    stoch: torch.Size


@dataclass(frozen=True)
class NormalStoch:
    """
    Parameters of a normal distribution and its stochastic sample.

    Attributes
    ----------
    mean : torch.Tensor
        Mean of the normal distribution.
    std : torch.Tensor
        Standard deviation of the normal distribution.
    stoch : torch.Tensor
        sample from the normal distribution with reparametrization trick.

    """
    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor

    def __post_init__(self):
        if self.mean.shape != self.std.shape:
            raise ValueError(f"mean.shape {self.mean.shape} and std.shape {self.std.shape} must be the same.")
        if (self.std < 0).any():
            raise ValueError(f"std must be non-negative.")


    def __getitem__(self, idx):
        return NormalStoch(self.mean[idx], self.std[idx], self.stoch[idx])

    def __len__(self):
        return self.stoch.shape[0]

    @property
    def shape(self):
        """ 
        mean, std, stoch の shape をタプルで返す

        """
        return NormalShape(self.mean.shape, self.std.shape, self.stoch.shape)

    def __getattr__(self, name):
        """
        torch.Tensor に含まれるメソッドを呼び出したら、各メンバに適用する
        例: normal.flatten() → NormalStoch(mean.flatten(), std.flatten(), stoch.flatten())
        """
        if hasattr(torch.Tensor, name):  # torch.Tensor のメソッドか確認
            def method(*args, **kwargs):
                return NormalStoch(
                    getattr(self.mean, name)(*args, **kwargs),
                    getattr(self.std, name)(*args, **kwargs),
                    getattr(self.stoch, name)(*args, **kwargs)
                )
            return method
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def save(self, path: str):
        """
        Save the parameters of the normal distribution to the specified path.

        Parameters
        ----------
        path : str
            Path to save the parameters.

        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_blosc2(f"{path}/mean.blosc2", self.mean.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/std.blosc2", self.std.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/stoch.blosc2", self.stoch.detach().clone().cpu().numpy())

    def get_distribution(self, independent: int = 1):
        return D.Independent(D.Normal(self.mean, self.std), independent)

@dataclass
class CategoricalShape:
    """
    Shapes of the parameters of a categorical distribution and its stochastic sample.

    Attributes
    ----------
    logits : torch.Size
        Shape of the logits of the categorical distribution.
    probs : torch.Size
        Shape of the probabilities of the categorical distribution.
    stoch : torch.Size
        Shape of the sample from the categorical distribution with Straight-Through Estimator.

    """
    logits: torch.Size
    probs: torch.Size
    stoch: torch.Size

@dataclass(frozen=True)
class CategoricalStoch:
    """
    Parameters of a categorical distribution and its stochastic sample.

    Attributes
    ----------
    logits : torch.Tensor
        Logits of the categorical distribution.
    probs : torch.Tensor
        Probabilities of the categorical distribution.
    stoch : torch.Tensor
        sample from the categorical distribution with Straight-Through Estimator.
    
    """
    logits: torch.Tensor
    probs: torch.Tensor
    stoch: torch.Tensor

    def __post_init__(self):
        if self.logits.shape != self.probs.shape:
            raise ValueError(f"logits.shape {self.logits.shape} and probs.shape {self.probs.shape} must be the same.")
        if (self.probs < 0).any() or (self.probs > 1).any():
            raise ValueError(f"probs must be in the range [0, 1].")
        if (self.probs.sum(dim=-1) - 1).abs().max() > 1e-6:
            raise ValueError(f"probs must sum to 1.")

    def __getitem__(self, idx):
        return CategoricalStoch(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self):
        return self.stoch.shape[0]

    @property
    def shape(self):
        """ mean, std, stoch の shape をタプルで返す """
        return CategoricalShape(self.logits.shape, self.probs.shape, self.stoch.shape)

    def __getattr__(self, name):
        """
        torch.Tensor に含まれるメソッドを呼び出したら、各メンバに適用する
        例: normal.flatten() → NormalStoch(mean.flatten(), std.flatten(), stoch.flatten())
        """
        if hasattr(torch.Tensor, name):  # torch.Tensor のメソッドか確認
            def method(*args, **kwargs):
                return CategoricalStoch( 
                    getattr(self.logits, name)(*args, **kwargs),
                    getattr(self.probs, name)(*args, **kwargs),
                    getattr(self.stoch, name)(*args, **kwargs)
                )
            return method
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def save(self, path: str):
        """
        Save the parameters of the categorical distribution to the specified path.

        Parameters
        ----------
        path : str
            Path to save the parameters.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_blosc2(f"{path}/logits.blosc2", self.logits.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/probs.blosc2", self.probs.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/stoch.blosc2", self.stoch.detach().clone().cpu().numpy())

    def get_distribution(self, independent: int = 1):
        return D.Independent(D.OneHotCategoricalStraightThrough(self.probs), independent)

@dataclass
class BernoulliShape:
    """
    Shapes of the parameters of a Bernoulli distribution and its stochastic sample.

    Attributes
    ----------
    logits : torch.Size
        Shape of the logits of the Bernoulli distribution.
    probs : torch.Size
        Shape of the probabilities of the Bernoulli distribution.
    stoch : torch.Size
        Shape of the sample from the Bernoulli distribution with Straight-Through Estimator.

    """
    logits: torch.Size
    probs: torch.Size
    stoch: torch.Size


@dataclass(frozen=True)
class BernoulliStoch:
    """
    Parameters of a Bernoulli distribution and its stochastic sample.

    Attributes
    ----------
    logits : torch.Tensor
        Logits of the Bernoulli distribution.
    probs : torch.Tensor
        Probabilities of the Bernoulli distribution.
    stoch : torch.Tensor
        sample from the Bernoulli distribution with Straight-Through Estimator.
    
    """


    logits: torch.Tensor
    probs: torch.Tensor
    stoch: torch.Tensor

    def __post_init__(self):
        if self.logits.shape != self.probs.shape:
            raise ValueError(f"logits.shape {self.logits.shape} and probs.shape {self.probs.shape} must be the same.")
        if (self.probs < 0).any() or (self.probs > 1).any():
            raise ValueError(f"probs must be in the range [0, 1].")

    def __getitem__(self, idx):
        return CategoricalStoch(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self):
        return self.stoch.shape[0]

    @property
    def shape(self):
        """ mean, std, stoch の shape をタプルで返す """
        return BernoulliShape(self.logits.shape, self.probs.shape, self.stoch.shape)

    def __getattr__(self, name):
        """
        torch.Tensor に含まれるメソッドを呼び出したら、各メンバに適用する
        例: normal.flatten() → NormalStoch(mean.flatten(), std.flatten(), stoch.flatten())
        """
        if hasattr(torch.Tensor, name):  # torch.Tensor のメソッドか確認
            def method(*args, **kwargs):
                return CategoricalStoch( 
                    getattr(self.logits, name)(*args, **kwargs),
                    getattr(self.probs, name)(*args, **kwargs),
                    getattr(self.stoch, name)(*args, **kwargs)
                )
            return method
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_blosc2(f"{path}/logits.blosc2", self.logits.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/probs.blosc2", self.probs.detach().clone().cpu().numpy())
        save_blosc2(f"{path}/stoch.blosc2", self.stoch.detach().clone().cpu().numpy())

    def get_distribution(self, independent: int = 1):
        return D.Independent(BernoulliStraightThrough(self.probs), independent)

StochState = Union[NormalStoch, CategoricalStoch, BernoulliStoch]


def cat_dist(stochs: Tuple[StochState, ...], dim: int = -1):
    """
    Concatenate the parameters of the distributions.

    Parameters
    ----------
    stochs : Tuple[StochState, ...]
        Tuple of the distributions.
    dim : int, optional
        Dimension to concatenate the parameters of the distributions.
        Default is -1.

    Returns
    -------
    StochState
        Concatenated distribution.

    Examples
    --------
    >>> dist1 = NormalStoch(torch.randn(2, 3), torch.rand(2, 3), torch.randn(2, 3))
    >>> dist2 = NormalStoch(torch.randn(2, 3), torch.rand(2, 3), torch.randn(2, 3))
    >>> cat_dist = cat_dist((dist1, dist2))
    >>> cat_dist.shape
    NormalShape(mean=torch.Size([2, 6]), std=torch.Size([2, 6]), stoch=torch.Size([2, 6]))

    """
    if isinstance(stochs[0], NormalStoch):
        return NormalStoch(
            torch.cat([stoch.mean for stoch in stochs], dim=dim),
            torch.cat([stoch.std for stoch in stochs], dim=dim),
            torch.cat([stoch.stoch for stoch in stochs], dim=dim),
        )
    elif isinstance(stochs[0], CategoricalStoch):
        return CategoricalStoch(
            torch.cat([stoch.logits for stoch in stochs], dim=dim),
            torch.cat([stoch.probs for stoch in stochs], dim=dim),
            torch.cat([stoch.stoch for stoch in stochs], dim=dim),
        )
    elif isinstance(stochs[0], BernoulliStoch):
        return BernoulliStoch(
            torch.cat([stoch.logits for stoch in stochs], dim=dim),
            torch.cat([stoch.probs for stoch in stochs], dim=dim),
            torch.cat([stoch.stoch for stoch in stochs], dim=dim),
        )
    else:
        return None


def stack_dist(stochs: Tuple[StochState, ...], dim: int = 0):
    """
    Stack the parameters of the distributions.

    Parameters
    ----------
    stochs : Tuple[StochState, ...]
        Tuple of the distributions.
    dim : int, optional
        Dimension to stack the parameters of the distributions. Default is 0.

    Returns
    -------
    StochState
        Stacked distribution.

    Examples
    --------
    >>> dist1 = NormalStoch(torch.randn(2, 3), torch.rand(2, 3), torch.randn(2, 3))
    >>> dist2 = NormalStoch(torch.randn(2, 3), torch.rand(2, 3), torch.randn(2, 3))
    >>> stack_dist = stack_dist((dist1, dist2))
    >>> stack_dist.shape
    NormalShape(mean=torch.Size([2, 2, 3]), std=torch.Size([2, 2, 3]), stoch=torch.Size([2, 2, 3]))
        
    """

    if isinstance(stochs[0], NormalStoch):
        return NormalStoch(
            torch.stack([stoch.mean for stoch in stochs], dim=dim),
            torch.stack([stoch.std for stoch in stochs], dim=dim),
            torch.stack([stoch.stoch for stoch in stochs], dim=dim),
        )
    elif isinstance(stochs[0], CategoricalStoch):
        return CategoricalStoch(
            torch.stack([stoch.logits for stoch in stochs], dim=dim),
            torch.stack([stoch.probs for stoch in stochs], dim=dim),
            torch.stack([stoch.stoch for stoch in stochs], dim=dim),
        )
    elif isinstance(stochs[0], BernoulliStoch):
        return BernoulliStoch(
            torch.stack([stoch.logits for stoch in stochs], dim=dim),
            torch.stack([stoch.probs for stoch in stochs], dim=dim),
            torch.stack([stoch.stoch for stoch in stochs], dim=dim),
        )
    else:
        return None


class BernoulliStraightThrough(D.Bernoulli):
    has_rsample = True

    def rsample(
            self, 
            sample_shape: torch.Size = torch.Size()
        ) -> torch.Tensor:
        """
        Generate a sample from the Bernoulli distribution with Straight-Through Estimator.

        Parameters
        ----------
        sample_shape : torch.Size, optional
            Shape of the sample. Default is torch.Size().

        Returns
        -------
        torch.Tensor
            Sample from the Bernoulli distribution with Straight-Through Estimator.

        Examples
        --------
        >>> logits = torch.randn(2, 3)
        >>> probs = torch.sigmoid(logits)
        >>> dist = BernoulliStraightThrough(probs=probs)
        >>> sample = dist.rsample()
        >>> sample.shape
        torch.Size([2, 3])

        """
        samples = self.sample(sample_shape)
        probs = self.probs
        return samples + (probs - probs.detach())

class Distribution(nn.Module):
    """
    A distribution function. 

    Parameters
    ----------
    in_dim : int
        Input dimension.
    dist : Literal["normal", "categorical", "bernoulli"]
        Distribution type. 
    n_groups : int, optional
        Number of groups. Default is 1. This is used for the categorical and Bernoulli distributions.
    spherical : bool, optional
        Whether to project samples to the unit sphere. Default is False. 
        This is used for the categorical and Bernoulli distributions.
        If True and dist=="categorical", the samples are projected from {0, 1} to {-1, 1}.
        If True and dist=="bernoulli", the samples are projected from {0, 1} to the unit sphere.

        refer to https://arxiv.org/abs/2406.07548

    Examples
    --------
    >>> dist = Distribution(10, "normal")
    >>> data = torch.randn(2, 20)
    >>> posterior = dist(data)
    >>> posterior.__class__.__name__
    'NormalStoch'
    >>> posterior.shape
    NormalShape(mean=torch.Size([2, 10]), std=torch.Size([2, 10]), stoch=torch.Size([2, 10]))

    >>> dist = Distribution(10, "categorical", n_groups=2)
    >>> data = torch.randn(2, 10)
    >>> posterior = dist(data)
    >>> posterior.__class__.__name__
    'CategoricalStoch'
    >>> posterior.shape
    CategoricalShape(logits=torch.Size([2, 2, 5]), probs=torch.Size([2, 2, 5]), stoch=torch.Size([2, 10]))

    >>> dist = Distribution(10, "bernoulli", n_groups=2)
    >>> data = torch.randn(2, 10)
    >>> posterior = dist(data)
    >>> posterior.__class__.__name__
    'BernoulliStoch'
    >>> posterior.shape
    BernoulliShape(logits=torch.Size([2, 2, 5]), probs=torch.Size([2, 2, 5]), stoch=torch.Size([2, 10]))

    """
    def __init__(
        self,
        in_dim: int,
        dist: Literal["normal", "categorical", "bernoulli"],
        n_groups: int = 1, 
        spherical: bool = False,
    ) -> None:
        super().__init__()


        self.dist = dist
        self.spherical = spherical
        self.n_class = in_dim // n_groups
        self.in_dim = in_dim
        self.n_groups = n_groups

        if dist == "normal":
            self.posterior = self.normal
        elif dist == "categorical":
            self.posterior = self.categorical
        elif dist == "bernoulli":
            self.posterior = self.bernoulli
        else:
            raise NotImplementedError

        if spherical:
            self.codebook = BSQCodebook(self.n_class)

    def normal(self, mu_std: torch.Tensor, deterministic: bool=False, inv_tmp: float = 1.0):
        assert mu_std.shape[-1] == self.in_dim * 2, f"mu_std.shape[-1] {mu_std.shape[-1]} and in_dim {self.in_dim} must be the same."

        mu, std = torch.chunk(mu_std, 2, dim=-1)
        std = F.softplus(std) + 1e-6

        posterior_dist = D.Normal(mu, std)
        posterior_dist = D.Independent(posterior_dist, 1)

        sample = posterior_dist.rsample() if not deterministic else mu

        posterior = NormalStoch(mu, std, sample if not deterministic else mu)

        return posterior

    def categorical(self, logits: torch.Tensor, deterministic: bool=False, inv_tmp: float = 1.0):
        batch_shape = logits.shape[:-1]
        logits_chunk = torch.chunk(logits, self.n_groups, dim=-1)
        logits = torch.stack(logits_chunk, dim=-2)
        logits = logits
        probs = softmax(logits, dim=-1, temperature=1/inv_tmp)
        posterior_dist = D.OneHotCategoricalStraightThrough(probs=probs)
        posterior_dist = D.Independent(posterior_dist, 1)

        sample = posterior_dist.rsample()

        if self.spherical:
            sample = sample * 2 - 1

        posterior = CategoricalStoch(
            logits, probs, sample.reshape([*batch_shape, -1]) if not deterministic else self.deterministic_onehot(probs).reshape([*batch_shape, -1])
            )

        return posterior

    def bernoulli(self, logits: torch.Tensor, deterministic: bool=False, inv_tmp: float = 1.0):
        batch_shape = logits.shape[:-1]
        chunked_logits = torch.chunk(logits, self.n_groups, dim=-1)
        logits = torch.stack(chunked_logits, dim=-2)
        logits = logits * inv_tmp
        probs = torch.sigmoid(logits)

        posterior_dist = BernoulliStraightThrough(probs=probs)  
        posterior_dist = D.Independent(posterior_dist, 1)

        sample = posterior_dist.rsample()

        if self.spherical:
            sample = self.codebook.bits_to_codes(sample)

        if deterministic:
            sample = torch.where(sample > 0.5, torch.ones_like(sample), torch.zeros_like(sample)) + probs - probs.detach()

        posterior = BernoulliStoch(
            logits, probs, sample.reshape([*batch_shape, -1]) 
        )

        return posterior

    def forward(
            self, 
            x: torch.Tensor, 
            deterministic: bool=False, 
            inv_tmp: float = 1.0
        ) -> StochState:
        """
        Compute the posterior distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        deterministic : bool, optional
            Whether to use the deterministic mode. Default is False.
            if True and dist=="normal", the mean is returned.
            if True and dist=="categorical", the one-hot vector computed by argmax is returned.
            if True and dist=="bernoulli", 1 is returned if x > 0.5 or 0 is returned if x <= 0.5.

        inv_tmp : float, optional
            Inverse temperature. Default is 1.0.
            This is used for the categorical and Bernoulli distributions.

        Returns
        -------
        StochState
            Posterior distribution.
        

        """
        return self.posterior(x, deterministic=deterministic, inv_tmp=inv_tmp)

    def deterministic_onehot(self, input:torch.Tensor):
        """
        Compute the one-hot vector by argmax.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            One-hot vector.

        Examples
        --------
        >>> input = torch.arange(6).reshape(2, 3) / 5.0
        >>> dist = Distribution(3, "categorical")
        >>> onehot = dist.deterministic_onehot(input)
        >>> onehot
        tensor([[0., 0., 1.],
                [0., 0., 1.]])
        """
        return F.one_hot(input.argmax(dim=-1), num_classes=self.n_class) + input - input.detach()


class BSQCodebook(nn.Module):
    """
    Binary Spherical Quantization codebook.

    Reference
    ---------
    https://arxiv.org/abs/2406.07548

    Parameters
    ----------
    codebook_dim : int
        Dimension of the codebook.

    Attributes
    ----------
    codebook_dim : int
        Dimension of the codebook.
    codebook_size : int
        Size of the codebook. This is equal to 2 ** codebook_dim.
    codebook : torch.Tensor
        Codebook.

    """
    def __init__(
            self,
            codebook_dim: int,
            ):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.codebook_size = 2 ** codebook_dim
        self.register_buffer("mask", 2 ** torch.arange(codebook_dim-1, -1, -1))
        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits) 
        self.register_buffer("codebook", codebook.float(), persistent=False)

    @staticmethod
    def bits_to_codes(bits):
        """
        Convert bits to codes, which are bits of either 0 or 1.

        Parameters
        ----------
        bits : torch.Tensor
            Bits of either 0 or 1.

        Returns
        -------
        torch.Tensor
            Codes, which are bits depending on codebook_dim(dimmension of the sphery)

        Examples
        --------
        >>> bits = torch.tensor([[0., 1.], [1., 0.]])
        >>> BSQCodebook.bits_to_codes(bits)
        tensor([[-0.7071,  0.7071],
                [ 0.7071, -0.7071]])

        >>> bits = torch.tensor([[0., 0., 1.], [1., 0., 0.]])
        >>> BSQCodebook.bits_to_codes(bits)
        tensor([[-0.5774, -0.5774,  0.5774],
                [ 0.5774, -0.5774, -0.5774]])


        """
        bits = bits * 2 - 1
        return F.normalize(bits, dim = -1)

    def indices_to_codes(
        self,
        indices: torch.Tensor,
        ):
        """
        Convert indices to codes, which are bits of either -1 or 1.

        Parameters
        ----------
        indices : torch.Tensor
            Indices.

        Returns
        -------
        torch.Tensor
            Codes, which are bits depending on codebook_dim(dimmension of the sphery)

        Examples
        --------
        >>> indices = torch.tensor([[31], [19]])
        >>> codebook = BSQCodebook(5)
        >>> codebook.indices_to_codes(indices)
        tensor([[ 0.4472,  0.4472,  0.4472,  0.4472,  0.4472],
                [ 0.4472, -0.4472, -0.4472,  0.4472,  0.4472]])

        """
        indices = indices.squeeze(-1)

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).float()

        codes = self.bits_to_codes(bits)

        return codes

if __name__ == "__main__":
    import doctest
    doctest.testmod()
