from abc import abstractmethod
from typing import Optional
from pathlib import Path
import torch
from torch import nn
from torch.optim import Optimizer

from vaemcmc.training.metric_log import MetricLog


class BaseModel(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device
    
    @abstractmethod
    def loss_function(self, *args, **kwargs) -> torch.Tensor:
        return NotImplementedError
    
    def train_step(self, batch, optimizer: Optimizer, log_data: MetricLog, global_step: int) -> MetricLog:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        optimizer.zero_grad()
        out = self(x)
        loss = self.loss_function(x, y, *out)
        loss = loss.mean(0)
        loss.backward()
        optimizer.step()
        
        log_data.add_metric("loss", loss.mean().item())
        return log_data
    
    def restore_checkpoint(self, ckpt_file: Path, optimizer: Optional[Optimizer] = None) -> int:
        collection = torch.load(ckpt_file, map_location=self.device)
        self.load_state_dict(collection["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(collection["optimizer_state_dict"])
        try:
            global_step = int(ckpt_file.stem[len("ckpt_"):])
        except ValueError:
            global_step = 0
        return global_step
    
    def save_checkpoint(self, directory: Path, global_step: int, optimizer: Optimizer):
        collection = {}
        collection["model_state_dict"] = self.state_dict()
        collection["optimizer_state_dict"] = optimizer.state_dict()
        collection["global_step"] = global_step
        ckpt_path = Path(directory, f"ckpt_{global_step:06d}")
        torch.save(collection, ckpt_path)
        