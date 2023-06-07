from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from vaemcmc.registry import Registry
from vaemcmc.training.metric_log import MetricLog


class Callback(ABC):
    cnt: int = 0

    @abstractmethod
    def invoke(self, info: Dict[str, Union[float, np.ndarray]], log_data: Optional[MetricLog] = None):
        raise NotImplementedError

    def reset(self):
        self.cnt = 0


# @Registry.register()
# class WandbCallback(Callback):
#     def __init__(
#         self,
#         *,
#         invoke_every: int = 1,
#         init_params: Optional[Dict] = None,
#         keys: Optional[List[str]] = None,
#     ):
#         self.init_params = init_params if init_params else {}
#         import wandb

#         self.wandb = wandb
#         wandb.init(**self.init_params)

#         self.invoke_every = invoke_every
#         self.keys = keys

#         self.img_transform = transforms.Resize(
#             128, interpolation=transforms.InterpolationMode.LANCZOS
#         )

#     def invoke(self, info: Dict[str, Union[float, np.ndarray]]):
#         step = info.get("step", self.cnt)
#         if step % self.invoke_every == 0:
#             wandb = self.wandb
#             if not self.keys:
#                 self.keys = info.keys()
#             log = dict()
#             for key in self.keys:
#                 if key not in info:
#                     continue
#                 if isinstance(info[key], np.ndarray):
#                     log[key] = wandb.Image(
#                         make_grid(
#                             self.img_transform(
#                                 torch.clip(torch.from_numpy(info[key][:25]), 0, 1)
#                             ),
#                             nrow=5,
#                         ),
#                         caption=key,
#                     )
#                 else:
#                     log[key] = info[key]
#             log["step"] = step
#             wandb.log(log)
#         self.cnt += 1
#         return 1

#     def reset(self):
#         super().reset()
#         self.wandb.init(**self.init_params)