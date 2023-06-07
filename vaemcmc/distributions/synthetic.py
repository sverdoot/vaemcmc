from typing import Optional, Tuple

import numpy as np
import torch
from torch.distributions import MultivariateNormal as MNormal, Normal, Distribution, Independent, Uniform

from vaemcmc.registry import Registry
from .base import CustomDistribution


@Registry.register_dist()
class MoIG(CustomDistribution):
    """
    Mixture of Isotropic Gaussians distribution.

    Args:
        locs (torch.Tensor): locations of mean parameters for each Gaussian
        covs (torch.Tensor): covariances for each Gaussian
    """

    def __init__(
        self,
        locs: Optional[torch.Tensor] = None,
        covs: Optional[torch.Tensor] = None,
        dim: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        if locs is None and dim:
            arange = 2 * np.pi * torch.arange(9) / 9.
            locs = 10 * torch.stack([torch.cos(arange), torch.sin(arange)], 1)
            locs = torch.cat([locs, torch.zeros(locs.shape[0], dim - 2)], 1)
            covs = torch.ones(dim)[None, ...].repeat(locs.shape[0], 1)
        self.n_gauss = len(locs)
        self.locs = locs
        self.covs = covs
        self.weights = (
            weights
            if weights is not None
            else torch.ones(self.n_gauss, device=self.device)
        )
        self.weights /= self.weights.sum()
        self.gaussians = [Normal(loc, cov) for loc, cov in zip(locs, covs)]
        self._event_shape = (self.locs.shape[-1],)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            log p(x)
        """
        log_ps = torch.stack(
            [
                torch.log(weight) + gauss.log_prob(x).sum(-1)
                for weight, gauss in zip(self.weights, self.gaussians)
            ],
            dim=0,
        )
        return torch.logsumexp(log_ps, dim=0)

    def sample(self, sample_shape: torch.Size):
        flat_size = np.prod(sample_shape)
        ids = np.random.choice(np.arange(self.locs.shape[0]), flat_size)
        locs = self.locs[ids]
        covs = self.covs[ids]
        noise = (covs**0.5) * torch.randn(flat_size, self.locs.shape[1], device=self.device)
        sample = locs + noise
        return sample.reshape(*sample_shape, -1)

    def plot_2d_contour(self, ax, proj: Tuple[int, int] = (0, 1)):
        rad = self.covs.max().cpu() ** 0.5 * 5
        x = np.linspace(self.locs.cpu().min() - rad, self.locs.cpu().max() + rad, 100)
        y = np.linspace(self.locs.cpu().min() - rad, self.locs.cpu().max() + rad, 100)
        X, Y = np.meshgrid(x, y)
        inp = torch.FloatTensor(np.stack([X, Y], -1), device=self.device)

        proj = list(proj)
        gaussians = [
            Normal(loc[proj], cov[proj]) for loc, cov in zip(self.locs, self.covs)
        ]
        log_ps = torch.stack(
            [
                torch.log(weight) + gauss.log_prob(inp.reshape(-1, 2)).sum(-1)
                for weight, gauss in zip(self.weights, gaussians)
            ],
            dim=0,
        )
        Z = torch.logsumexp(log_ps, dim=0).reshape(inp.shape[:-1]).detach().cpu()
        # levels = np.quantile(Z, np.linspace(0.9, 0.99, 5))

        ax.contour(
            X,
            Y,
            Z.exp(),
            # levels = levels,
            alpha=1.0,
            colors="midnightblue",
            linewidths=1,
        )  # cmap='inferno')


@Registry.register_dist()
class MoG(CustomDistribution):
    """
    Mixture of Gaussians distribution.

    Args:
        locs (torch.Tensor): locations of mean parameters for each Gaussian
        covs (torch.Tensor): covariances for each Gaussian
    """

    def __init__(
        self,
        locs: torch.Tensor,
        covs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.n_gauss = len(locs)
        self.locs = locs
        self.covs = covs
        self.weights = (
            weights
            if weights is not None
            else torch.ones(self.n_gauss, device=self.device)
        )
        self.weights /= self.weights.sum()
        self.gaussians = [MNormal(loc, cov) for loc, cov in zip(locs, covs)]
        self._event_shape = (self.locs.shape[-1],)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_ps = torch.stack(
            [
                torch.log(weight) + gauss.log_prob(x)
                for weight, gauss in zip(self.weights, self.gaussians)
            ],
            dim=0,
        )
        return torch.logsumexp(log_ps, dim=0)

    def sample(self, sample_shape: torch.Size):
        flat_size = np.prod(sample_shape)
        ids = np.random.choice(np.arange(self.locs.shape[0]), flat_size)
        locs = self.locs[ids]
        covs = self.covs[ids]
        noise = torch.einsum(
            "lab,lb->la", (covs**0.5), torch.randn(flat_size, self.locs.shape[1], device=self.device)
        )
        sample = locs + noise
        return sample.reshape(*sample_shape, -1)

    def plot_2d_contour(self, ax, proj: Tuple[int, int] = (0, 1)):
        rad = self.covs.max().cpu() ** 0.5 * 5
        x = np.linspace(self.locs.cpu().min() - rad, self.locs.cpu().max() + rad, 100)
        y = np.linspace(self.locs.cpu().min() - rad, self.locs.cpu().max() + rad, 100)
        X, Y = np.meshgrid(x, y)
        inp = torch.FloatTensor(np.stack([X, Y], -1), device=self.device)

        proj = list(proj)
        gaussians = [
            MNormal(loc[proj], cov[proj, :][:, proj])
            for loc, cov in zip(self.locs, self.covs)
        ]
        log_ps = torch.stack(
            [
                torch.log(weight) + gauss.log_prob(inp.reshape(-1, 2))
                for weight, gauss in zip(self.weights, gaussians)
            ],
            dim=0,
        )
        Z = torch.logsumexp(log_ps, dim=0).reshape(inp.shape[:-1]).detach().cpu()
        # levels = np.quantile(Z, np.linspace(0.9, 0.99, 5))

        ax.contour(
            X,
            Y,
            Z.exp(),
            # levels = levels,
            alpha=1.0,
            colors="midnightblue",
            linewidths=1,
        )  # cmap='inferno')
        
    
@Registry.register_dist()    
class Ring(CustomDistribution):
    """Ring distribution.

    Args:

    Returns:
        _type_: _description_
    """
    def __init__(self, mu: float = 5.0, scale: float = 0.5):
        self.mu = mu
        self.scale = scale
        self.rad_dist = Normal(mu, scale)
        self.angle_dist = Uniform(0, 2 * np.pi)
        self._event_shape = (2,)
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        rad = torch.norm(x, dim=-1)
        angle = torch.atan2(x[..., 1], x[..., 0])
        angle[angle < 0] += 2 * np.pi
        return self.rad_dist.log_prob(rad) + self.angle_dist.log_prob(angle) - torch.log(rad)
    
    def sample(self, sample_shape: torch.Size = ...) -> torch.Tensor:
        rad = self.rad_dist.sample(sample_shape)
        angle = self.angle_dist.sample(sample_shape)
        x = torch.stack([rad * torch.cos(angle), rad * torch.sin(angle)], -1)
        return x
    
    def plot_2d_contour(self, ax, proj: Tuple[int, int] = (0, 1)):
        N = 100
        X = torch.linspace(-7, 7, N)
        Y = torch.linspace(-7, 7, N)

        X, Y = torch.meshgrid(X, Y)
        inp = torch.stack([X, Y], -1)
        Z = self.log_prob(inp).detach().cpu()
        # levels = np.quantile(Z, np.linspace(0.9, 0.99, 5))

        ax.contour(
            X,
            Y,
            Z.exp(),
            # levels = levels,
            alpha=1.0,
            colors="midnightblue",
            linewidths=1,
        )  # cmap='inferno')
    


@Registry.register_dist()
class Funnel(CustomDistribution):
    """
    Funnel distribution.

    “Slice sampling”. R. Neal, Annals of statistics, 705 (2003) https://doi.org/10.1214/aos/1056562461

    Args:
        dim (int) : dimension
        nu - parameter
    """

    def __init__(self, dim: int, a: float = 3):
        self.a = a
        self.normal_first = Normal(0, self.a)
        self._event_shape = (dim,)
            
    def log_prob(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns:
            log p(x)
        """
        normal_last = Normal(
            torch.zeros(x.shape[:-1], device=self.device), torch.exp(x[..., 0] / 2.0)
        )
        return normal_last.log_prob(x[..., 1:].permute(-1, *range(x.ndim - 1))).sum(
            0
        ) + self.normal_first.log_prob(x[..., 0])

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        flat_size = np.prod(sample_shape)
        sample = torch.randn((flat_size, *self.event_shape), device=self.device)
        sample[:, 0] *= self.a
        sample[:, 1:] *= torch.exp(sample[:, [-1]] / 2.0)

        return sample

    def plot_2d_contour(self, ax):
        x = np.linspace(-15, 15, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        inp = torch.FloatTensor(np.stack([X, Y], -1), device=self.device)
        Z = self.log_prob(inp.reshape(-1, 2)).reshape(inp.shape[:-1])

        # levels = np.quantile(Z, np.linspace(0.9, 0.99, 5))
        ax.contour(
            X,
            Y,
            Z.exp(),
            # levels = levels,
            levels=3,
            alpha=1.0,
            colors="midnightblue",
            linewidths=1,
        )


@Registry.register_dist()
class Banana(CustomDistribution):
    """ 
    """
    def __init__(self, dim: int, b: float, sigma: float):
        self.b = b
        self.sigma = sigma
        # self._dist = MNormal(
        #     torch.tensor([0.0, 0.0]),
        #     covariance_matrix=torch.tensor([[1.0, self.rho], [self.rho, 1.0]]),
        # )
        self._event_shape = (dim,)
            
    def log_prob(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Returns:
            log p(x)
        """
        even = np.arange(0, x.shape[-1], 2)
        odd = np.arange(1, x.shape[-1], 2)
        ll = -0.5 * (
            x[..., odd] - self.b * x[..., even] ** 2 + (self.sigma**2) * self.b
        ) ** 2 - ((x[..., even]) ** 2) / (2 * self.sigma**2)
        return ll.sum(-1)

    def plot_2d_contour(self, ax):
        x = np.linspace(-15, 15, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        inp = torch.FloatTensor(np.stack([X, Y], -1), device=self.device)
        Z = self.log_prob(inp.reshape(-1, 2)).reshape(inp.shape[:-1])
        # levels = np.quantile(Z, np.linspace(0.9, 0.99, 5))

        ax.contour(
            X,
            Y,
            Z.exp(),
            # levels = levels,
            levels=5,
            alpha=1.0,
            colors="midnightblue",
            linewidths=1,
        )

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        sample = torch.randn(*sample_shape, *self.event_shape, device=self.device)
        even = np.arange(0, self.event_shape[0], 2)
        odd = np.arange(1, self.event_shape[0], 2)
        sample[..., even] *= self.sigma
        sample[..., odd] += self.b * sample[..., even] ** 2 - (self.sigma**2) * self.b
        return sample


# Registry.register_dist()
# class TwoMoons(Distribution):
#     def __init__(self, dim: int):
#         self._event_shape = (dim,)
