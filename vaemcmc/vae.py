from abc import abstractclassmethod, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from munch import Munch
import torch
from torch import nn
import torch.distributions as d
from torch.distributions import Distribution
from torch.optim import Optimizer
from vaemcmc.training.metric_log import MetricLog

from vaemcmc.registry import Registry
from vaemcmc.base_model import BaseModel


class BaseVAE(BaseModel):
    def __init__(
        self,
        encoder: Union[nn.Module, Munch],
        decoder: Union[nn.Module, Munch],
        prior: Optional[Distribution] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(encoder, Dict):
            encoder = Registry.model_registry[encoder["name"]](**encoder["params"])
        if isinstance(decoder, Dict):
            decoder = Registry.model_registry[decoder["name"]](**decoder["params"])
            
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = encoder.n_out // 2
        self.prior = (
            prior
            if prior is not None
            else d.Independent(d.Normal(torch.zeros(self.z_dim), torch.ones(self.z_dim)), 1)
        )
        self.batch_size = batch_size
        
    def to(self, device: Union[str, torch.device]):
        super().to(device)
        if isinstance(self.prior, d.Independent):
            self.prior.base_dist.loc = self.prior.base_dist.loc.to(device)
            self.prior.base_dist.scale = self.prior.base_dist.scale.to(device)
        else:
            self.prior = self.prior.to(device)
        return self
    
    @abstractclassmethod
    def approx_posterior(cls, *params) -> Distribution:
        ...

    @abstractclassmethod
    def cond(cls, *params) -> Distribution:
        ...

    @abstractclassmethod
    def kl_div(cls, *params):
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> Sequence:
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> Sequence:
        ...

    def forward(self, x: torch.Tensor):
        var_params = self.encode(x)
        approx_posterior = self.approx_posterior(*var_params)
        z = approx_posterior.rsample((1,))[0]
        cond_params = self.decode(z)
        cond = self.cond(*cond_params)
        x_rec = cond.rsample((1,))[0]
        return x_rec, var_params, cond_params

    def loss_function(
        self, x: torch.Tensor, y, _, var_params, cond_params,
    ) -> torch.Tensor:
        # var_params, cond_params = params
        kl = self.kl_div(*var_params)
        cond = self.cond(*cond_params)
        ll = cond.log_prob(x)
        neg_elbo = -ll + kl
        return neg_elbo

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        flat_size = np.prod(sample_shape)
        batch_size = self.batch_size or flat_size
        sample = []
        for _ in range(0, flat_size, batch_size):
            z = self.prior.sample((batch_size,))
            cond_params = self.decode(z)
            cond = self.cond(*cond_params)
            x = cond.rsample((1,))[0]
            sample.append(x)

        sample = torch.cat(sample, 0)
        sample = sample.view(*sample_shape, *sample.shape[1:])
        return sample

    def cond_sample(self, z: torch.Tensor) -> torch.Tensor:
        z_flat = z.reshape(-1, z.shape[-1])
        flat_size = z_flat.shape[0]
        batch_size = self.batch_size or flat_size
        sample = []
        for i in range(0, flat_size, batch_size):
            z_batch = z_flat[i * batch_size : (i + 1) * batch_size]
            cond_params = self.decode(z_batch)
            cond = self.cond(*cond_params)
            x = cond.rsample((1,))[0]
            sample.append(x)

        sample = torch.cat(sample, 0)
        sample = sample.view(*z.shape[:-1], *sample.shape[1:])
        return sample
    
    
class IWAE(BaseVAE):
    pass


@Registry.register_model()
class NormalNormalVAE(BaseVAE):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        prior: Optional[Distribution] = None,
        batch_size: Optional[int] = None,
        sigma: Union[float, torch.Tensor] = 1.,
        fix_sigma: bool = False 
    ):
        super().__init__(encoder, decoder, prior, batch_size)
        self.sigma = sigma
        self.fix_sigma = fix_sigma

    def cond(self, mean, logvar):
        return d.Independent(d.Normal(mean, torch.exp(logvar / 2.)), 1)

    def approx_posterior(self, mean, logvar):
        return d.Independent(d.Normal(mean, torch.exp(logvar / 2.)), 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.reshape(-1, x.shape[-1])
        out = self.encoder(x_flat)
        out = out.reshape(*x.shape[:-1], *out.shape[1:])
        mean, logvar = out[..., : out.shape[-1] // 2], out[..., out.shape[-1] // 2 :]
        return mean, logvar

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_flat = z.reshape(-1, z.shape[-1])
        out = self.decoder(z_flat)
        out = out.reshape(*z.shape[:-1], *out.shape[1:])
        mean, logvar = out[..., : out.shape[-1] // 2], out[..., out.shape[-1] // 2 :]
        if self.fix_sigma:
            logvar = 2. * (self.sigma * torch.ones_like(mean)).log()

        return mean, logvar

    def kl_div(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        KLD = -.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(-1)
        return KLD
    

class NormalBernoulliVAE(NormalNormalVAE):
    def cond(self, mean, logvar):
        return d.Independent(d.Bernoulli(mean), 1)
    
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_flat = z.reshape(-1, z.shape[-1])
        out = self.decoder(z_flat)
        out = out.reshape(*z.shape[:-1], *out.shape[1:])
        mean, logvar = out[..., : out.shape[-1] // 2], out[..., out.shape[-1] // 2 :]
        if self.fix_sigma:
            logvar = 2 * (self.sigma * torch.ones_like(mean)).log()

        return mean, logvar
    


class VAETarget(Distribution):
    def __init__(self, model: BaseVAE, target: Optional[Distribution] = None):
        self.model = model
        self.target = target

    def log_prob(self, z: torch.Tensor, meta: Optional[Dict] = None) -> torch.Tensor:
        x = self.model.cond_sample(z)[0]
        # cond_params = self.model.decode(z)
        # cond = self.model.cond(*cond_params)
        # x = cond.rsample((1,))[0]
        # d = cond.log_prob(x)
        # cond_params = self.model.decode(z)
        # x = cond_params[0]
        if self.target:
            log_p = self.target.log_prob(x)
        else:
            pass
        return log_p


@Registry.register_dist()
class VAEProposal(Distribution):
    def __init__(self, vae: BaseVAE, prior: Optional[Distribution] = None):
        self.vae = vae
        self.prior = prior if prior is not None else self.vae.prior

    def sample(self, sample_shape: torch.Size = ...) -> torch.Tensor:
        return self.vae.sample(sample_shape)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        var_params = self.vae.encode(x)
        approx_posterior = self.vae.approx_posterior(*var_params)
        z = approx_posterior.rsample((1,))[0]
        cond_params = self.vae.decode(z)
        cond = self.vae.cond(*cond_params)
        log_qzx = approx_posterior.log_prob(z)
        log_pxz = cond.log_prob(x)
        log_pz = self.prior.log_prob(z)
        log_p = log_pxz + log_pz - log_qzx
        return log_p
    
    
class VAECoupledProposal(Distribution):
    def __init__(self, vae: BaseVAE, prior: Optional[Distribution] = None):
        self.vae = vae
        self.prior = prior if prior is not None else self.vae.prior

    def sample(self, sample_shape: torch.Size, meta: Dict) -> torch.Tensor:
        x = meta['x']
        var_params = self.vae.encode(x)
        approx_posterior = self.vae.approx_posterior(*var_params)
        z = approx_posterior.rsample((sample_shape,))[0]
        return z

    def log_prob(self, z: torch.Tensor, meta: Dict) -> torch.Tensor:
        x = meta['x']
        var_params = self.vae.encode(x)
        approx_posterior = self.vae.approx_posterior(*var_params)
        z = approx_posterior.rsample((1,))[0]
        cond_params = self.vae.decode(z)
        cond = self.vae.cond(*cond_params)
        log_qzx = approx_posterior.log_prob(z)
        log_pxz = cond.log_prob(x)
        log_pz = self.prior.log_prob(z)
        log_p = log_pxz + log_pz - log_qzx
        return log_p
    
    
class VAECoupledTarget(Distribution):
    def __init__(self, vae: BaseVAE, target: Optional[Distribution] = None):
        self.vae = vae

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        var_params = self.vae.encode(x)
        approx_posterior = self.vae.approx_posterior(*var_params)
        z = approx_posterior.rsample((1,))[0]
        cond_params = self.vae.decode(z)
        cond = self.vae.cond(*cond_params)
        log_qzx = approx_posterior.log_prob(z)
        log_pxz = cond.log_prob(x)
        log_pz = self.vae.prior.log_prob(z)
        log_p = log_pxz + log_pz - log_qzx
        return log_p
