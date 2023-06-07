from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union

import numpy as np
from munch import Munch
import torch
from torch.distributions import Categorical, Distribution
from pyro.infer import HMC, NUTS

from vaemcmc.registry import Registry


Registry.mcmc_registry.update({"HMC": HMC, "NUTS": NUTS})

class MCMCKernel(ABC):
    @abstractmethod
    def step(self, x: torch.Tensor, meta: Optional[Dict] = None) -> torch.Tensor:
        raise NotImplementedError

    def run(self, x: torch.Tensor, burn_in: int, n_samples: int):
        chain = []
        x_t = x.clone().detach()
        for step_id in range(burn_in + n_samples):
            x_t = self.step(x_t)

            if step_id >= burn_in:
                chain.append(x_t.clone().detach())

        chain = torch.stack(chain, 0)
        return chain


@Registry.register_mcmc()
class Compose(MCMCKernel):
    def __init__(self, *kernels: Union[MCMCKernel, Munch]):
        # if not all([isinstance(kernel, MCMCKernel) for kernel in kernels ]):
        #     self.kernels = []
        #     for kernel in kernels:
        #         if isinstance(kernel, Munch):
        #             kernel = Registry.mcmc_registry[kernel['name']](**kernel['params'])
        self.kernels = kernels
        
    def step(self, x: torch.Tensor, meta: Optional[Dict] = None) -> torch.Tensor:
        for kernel in self.kernels:
            x = kernel.step(x, meta)
            
        return x
        

@Registry.register_mcmc()
class ISIRKernel(MCMCKernel):
    def __init__(
        self, target: Distribution, *, proposal: Distribution, n_particles: int
    ):
        self.target = target
        self.proposal = proposal
        self.n_particles = n_particles

    def step(self, x: torch.Tensor, meta: Optional[Dict] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        new_candidates = self.proposal.sample((batch_size, self.n_particles - 1))
        particles = torch.cat([x[:, None, :], new_candidates], 1)

        log_ps = self.target.log_prob(particles)
        log_qs = self.proposal.log_prob(particles)
        log_weight = log_ps - log_qs
        indices = Categorical(logits=log_weight).sample()

        x_upd = particles[np.arange(x.shape[0]), indices]

        if meta:
            meta[f"{self.__class__.__name__}_log_ps"] = log_ps
            meta[f"{self.__class__.__name__}_log_qs"] = log_qs
            meta[f"{self.__class__.__name__}_indices"] = indices

        return x_upd

    # def run(self, x: torch.Tensor, burn_in: int, n_samples: int):
    #     chain = []
    #     x_t = x.clone().detach()
    #     for step_id in range(burn_in + n_samples):
    #         x_t = self.step(x_t)

    #         chain.append(x_t.clone().detach())

    #     chain = torch.stack(chain, 0)
    #     return chain


@Registry.register_mcmc()
class VAEKernel(MCMCKernel):
    def __init__(self, model):
        self.model = model

    def step(self, x: torch.Tensor, meta: Optional[Dict] = None) -> torch.Tensor:
        x_rec, mean, logvar = self.model(x)
        return x_rec
