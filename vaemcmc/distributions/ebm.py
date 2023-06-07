from typing import Optional, Union, Dict
from munch import Munch

import torch
from torch.distributions import Distribution

from vaemcmc.registry import Registry
from vaemcmc.gan import GAN


@Registry.register_dist()
class ModelPrior(Distribution):
    def __init__(
        self,
        model,
    ):
        self.model = model

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        return self.model.prior.sample(sample_shape)
    
    def log_prob(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.model.prior.log_prob(z)

    # def project(self, z):
    #     return self.model.prior.project(z)


@Registry.register_dist()
class GANEBM(Distribution):
    def __init__(self, gan: Union[GAN, Dict], batch_size: Optional[int] = None, temp: float = 1.0):
        if isinstance(gan, Dict):
            gan = Registry.model_registry[gan["name"]](**gan["params"])
        self.gan = gan
        self.prior = gan.prior
        self.batch_size = batch_size
        self.device = next(self.gan.gen.parameters()).device
        self.temp = temp
        
    def to(self, device: Union[str, torch.device]):
        self.gan.to(device)
        
    @property
    def dim(self):
        return self.gan.gen.z_dim

    def log_prob(self, z: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        init_shape = z.shape
        z = z.reshape(-1, init_shape[-1])
        batch_size = kwargs.get("batch_size", self.batch_size or len(z))
        log_prob = torch.empty((0,), device=self.device)
        for chunk_id, chunk in enumerate(torch.split(z, batch_size)):
            if "x" in kwargs:
                x = kwargs["x"][chunk_id * batch_size : (chunk_id + 1) * batch_size].to(
                    self.device
                )
            else:
                x = self.gan.gen(chunk.to(self.device))
            dgz = self.gan.dis(x).squeeze()
            logp_z = self.prior.log_prob(chunk)
            log_prob = torch.cat([log_prob, (logp_z + dgz) / self.temp])
        return log_prob.reshape(init_shape[:-1])
