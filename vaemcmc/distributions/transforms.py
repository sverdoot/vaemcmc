from typing import Union, Dict, Optional

import numpy as np
from scipy.stats import ortho_group
from munch import Munch
import torch
from torch.distributions import Normal

from vaemcmc.registry import Registry
from .base import CustomDistribution


@Registry.register_dist('GaussianEmbedding')
class GaussianEmbedding(CustomDistribution):
    def __init__(
        self,
        dist: Union[CustomDistribution, Munch],
        dim: int,
        transform: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
        seed : Optional[int] = None,
    ):
        if isinstance(dist, Dict):
            dist = Registry.dist_registry[dist["name"]](**dist["params"])
        self.dist = dist
        self.dim = dim
        self.split_dim = dist.event_shape[0]
        if transform is None:
            transform = torch.FloatTensor(ortho_group.rvs(dim, random_state=seed))
            # transform = torch.eye(dim)
        self._transform = transform
        self._inv_transform = torch.inverse(transform)
        self.log_det_jac = torch.slogdet(transform).logabsdet
        self.normal = Normal(
            torch.zeros(dim - self.split_dim),
            noise_scale * torch.ones(dim - self.split_dim)
        )
        self._event_shape = (dim,)
            
    def to(self, device: Union[str, torch.device]):
        super().to(device)
        self.dist = self.dist.to(device)
        return self
        
    @property
    def __name__(self) -> str:
        return f'{self.dist.__name__}_embed'

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self._transform.T

    def inv_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self._inv_transform.T

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.normal.sample(x.shape[:-1])
        x = torch.cat([x, noise], -1)
        x = self.transform(x)
        return x

    def inv_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.inv_transform(x)[..., : self.split_dim]

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_inv = self.inv_transform(x)
        x_1, x_2 = x_inv[..., : self.split_dim], x_inv[..., self.split_dim :]
        return (
            self.log_det_jac
            + self.dist.log_prob(x_1)
            + self.normal.log_prob(x_2).sum(-1)
        )

    def sample(self, sample_shape: torch.Size):
        x = self.dist.sample(sample_shape)
        return self.embed(x)

    def plot_2d_contour(self, ax):
        self.dist.plot_2d_contour(ax)
