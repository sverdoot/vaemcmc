# from .base_model import BaseModel

# class FlowML(BaseModel):
#     def __init__(self, flow: nn.Module):
#         super().__init__()
#         self.flow = flow

#     def loss_function(
#         self,
#         _,
#         z: Optional[torch.Tensor] = None,
#         log_jac: Optional[torch.Tensor] = None,
#     ):
#         ll = self.flow.prior.log_prob(z) + log_jac
#         return -ll

#     def forward(self, x: torch.Tensor):
#         return self.flow(x)
