from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
from pyro.distributions.transforms import AffineCoupling
from pyro.nn import ConditionalDenseNN, DenseNN
from torch.distributions import Independent, Normal

from vaemcmc.registry import Registry
from vaemcmc.base_model import BaseModel


@Registry.register_model()
class RNVP(BaseModel):
    def __init__(
        self,
        dim: int,
        flows: Optional[Iterable] = None,
        num_blocks: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        init_weight_scale: float = 1e-4,
        device: Union[str, int, torch.device] = "cpu",
        scale: float = 1.0,
    ):
        super().__init__()
        self.init_weight_scale = init_weight_scale
        self.x = None
        
        if flows is not None:
            self.flow = nn.ModuleList(flows)
        elif num_blocks and hidden_dim:
            split_dim = max(dim - hidden_dim, dim // 2)
            param_dims = [dim - split_dim, dim - split_dim]
            self.flow = nn.ModuleList(
                [
                    AffineCoupling(
                        split_dim,
                        DenseNN(
                            split_dim,
                            [hidden_dim],
                            param_dims,
                        ),
                    )
                    for _ in range(num_blocks)
                ],
            )
        else:
            raise ValueError
        self.init_params(self.parameters())

        even = [i for i in range(0, dim, 2)]
        odd = [i for i in range(1, dim, 2)]
        reverse_eo = [
            i // 2 if i % 2 == 0 else (i // 2 + len(even)) for i in range(dim)
        ]
        reverse_oe = [(i // 2 + len(odd)) if i % 2 == 0 else i // 2 for i in range(dim)]
        self.register_buffer("eo", torch.tensor(even + odd, dtype=torch.int64))
        self.register_buffer("oe", torch.tensor(odd + even, dtype=torch.int64))
        self.register_buffer(
            "reverse_eo",
            torch.tensor(reverse_eo, dtype=torch.int64),
        )
        self.register_buffer(
            "reverse_oe",
            torch.tensor(reverse_oe, dtype=torch.int64),
        )

        self.prior = Independent(Normal(torch.zeros(dim), scale * torch.ones(dim)), 1)

    def to(self, *args, **kwargs):
        """
        overloads to method to make sure the manually registered buffers are sent to device
        """
        self = super().to(*args, **kwargs)
        self.eo = self.eo.to(*args, **kwargs)
        self.oo = self.oe.to(*args, **kwargs)
        self.reverse_eo = self.reverse_eo.to(*args, **kwargs)
        self.reverse_oe = self.reverse_oe.to(*args, **kwargs)
        self.prior.base_dist.loc = self.prior.base_dist.loc.to(*args, **kwargs)
        self.prior.base_dist.scale = self.prior.base_dist.scale.to(*args, **kwargs)
        return self

    def permute(self, z, i, reverse=False):
        if not reverse:
            if i % 2 == 0:
                z = torch.index_select(z, -1, self.eo)
            else:
                z = torch.index_select(z, -1, self.oe)
        else:
            if i % 2 == 0:
                z = torch.index_select(z, -1, self.reverse_eo)
            else:
                z = torch.index_select(z, -1, self.reverse_oe)
        return z

    def forward(self, x):
        log_jacob = torch.zeros_like(x[..., 0], dtype=torch.float32)
        for i, current_flow in enumerate(self.flow):
            x = self.permute(x, i)
            z = current_flow(x)
            log_jacob += current_flow.log_abs_det_jacobian(x, z)
            z = self.permute(z, i, reverse=True)
            x = z
        return z, log_jacob

    def inverse(self, z):
        log_jacob_inv = torch.zeros_like(z[..., 0], dtype=torch.float32)
        n = len(self.flow) - 1
        for i, current_flow in enumerate(self.flow[::-1]):
            z = self.permute(z, n - i)
            x = current_flow._inverse(z)
            log_jacob_inv -= current_flow.log_abs_det_jacobian(x, z)
            x = self.permute(x, n - i, reverse=True)
            z = x
        return x, log_jacob_inv.reshape(z.shape[:-1])

    def log_prob(self, x):
        if False:  # self.x is not None and torch.equal(self.x, x):
            z, logp = self.z, self.log_jacob
        else:
            z, logp = self.forward(x)
        return self.prior.log_prob(z) + logp

    def sample(self, shape):
        z = self.prior.sample(shape)
        x, log_jacob_inv = self.inverse(z)
        self.log_jacob = -log_jacob_inv
        self.x = x
        self.z = z
        return x

    def init_params(self, params):
        # torch.nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain('relu'))
        for p in params:
            if p.ndim == 2:
                torch.nn.init.sparse_(p, sparsity=0.3, std=self.init_weight_scale)
        # torch.nn.init.normal_(params, 0, self.init_weight_scale)
        
    def loss_function(
        self,
        x,
        y,
        z: Optional[torch.Tensor],
        log_jac: Optional[torch.Tensor],
    ):
        ll = self.prior.log_prob(z) + log_jac
        return -ll
