from pathlib import Path
from typing import Any, Mapping, Optional, Union, Dict

import torch
from torch import nn, distributions as d
from torch.distributions import Distribution
from torch.optim import Optimizer

from vaemcmc.training.metric_log import MetricLog

from .base_model import BaseModel
from vaemcmc.registry import Registry

# class GANLoss:
#     _register = {}
    
#     def __init__(self, dis: nn.Module, name: str):
#         self.gan = gan
#         self.name = name
#     # @classmethod
#     # def register(cls, name: Optional[str] = None):
#     #     def inner_wrapper(wrapped_class):
#     #         if name is None:
#     #             name_ = wrapped_class.__name__
#     #         else:
#     #             name_ = name
#     #         cls.registry[name_] = wrapped_class
#     #         return wrapped_class

#     #     return inner_wrapper
    
#     # @GANLoss.register()
#     def vanilla(self, fake, real):
#         return self.dis()
    
#     def __call__

@Registry.register_model()
class GAN(BaseModel):
    def __init__(self, 
                 gen: Union[nn.Module, Dict], 
                 dis: Union[nn.Module, Dict], 
                 prior: Optional[Distribution] = None, 
                 batch_size: Optional[int] = None,
                 loss_function: Optional[str] = None,
                 ckpt_file: Optional[Union[str, Path]] = None):
        super().__init__()
        if isinstance(gen, Dict):
            gen = Registry.model_registry[gen["name"]](**gen["params"])
        if isinstance(dis, Dict):
            dis = Registry.model_registry[dis["name"]](**dis["params"])
        self.gen = gen
        self.dis = dis
        self.prior = (
        prior
        if prior is not None
            else d.Independent(d.Normal(torch.zeros(self.gen.z_dim), torch.ones(self.gen.z_dim)), 1)
        )
        self.batch_size = batch_size
        self.ckpt_file = ckpt_file
        if ckpt_file and Path(ckpt_file).exists():
            self.restore_checkpoint(Path(ckpt_file))
        self._loss_function = loss_function
            
    def to(self, device: Union[str, torch.device]):
        super().to(device)
        if isinstance(self.prior, d.Independent):
            self.prior.base_dist.loc = self.prior.base_dist.loc.to(device)
            self.prior.base_dist.scale = self.prior.base_dist.scale.to(device)
        else:
            self.prior = self.prior.to(device)
        return self
        
    # def save_checkpoint(self, directory: Path, global_step: int, optimizer: Optimizer):
    #     return super().save_checkpoint(directory, global_step, optimizer)
        
    def loss_function(self, *args, **kwargs) -> torch.Tensor:
        return super().loss_function(*args, **kwargs)
    
    def train_step(self, batch, optimizer: Optimizer, log_data: MetricLog, global_step: int) -> MetricLog:
        return super().train_step(batch, optimizer, log_data, global_step)
        
    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        sample = self.prior.sample(sample_shape)
        return self.gen(sample)
