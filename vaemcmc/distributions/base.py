from typing import Union
import torch
from torch.distributions import Distribution


class CustomDistribution(Distribution):
    _device: Union[str, torch.device] = 'cpu'
   
    @property
    def __name__(self) -> str:
        return self.__class__.__name__
    
    @property
    def device(self) -> Union[str, torch.device]:
        return self._device
    
    def to(self, device: Union[str, torch.device]):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__setattr__(k, v.to(device))
        self._device = device
        return self
    