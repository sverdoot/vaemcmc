import torch
import torch.nn as nn

from vaemcmc.registry import Registry


@Registry.register_model()
class EncoderMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hid: int, n_layers: int = 2, double_n_out: bool = True):
        super().__init__()
        n_out = n_out * 2 if double_n_out else n_out
        self.n_in = n_in
        self.n_out = n_out
        layers = [
            nn.Flatten(),
            nn.Linear(n_in, n_hid),
            nn.LeakyReLU(),
        ]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(n_hid, n_hid), nn.LeakyReLU()])
        self.net = nn.Sequential(
            *layers,
            nn.Linear(n_hid, n_out),
            # nn.ReLU(),
            # nn.Linear(n_out, n_out)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


# class FC_encoder(nn.Module):
#     def __init__(self, inp_dim, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(inp_dim, 2 * hidden_dim),
#             nn.ReLU(),
#             nn.Linear(2 * hidden_dim, 2 * hidden_dim)
#         )

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.view(batch_size, -1)
#         return self.net(x)

# class backward_kernel_mnist(nn.Module):
#     def __init__(self, act_func, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(2 * hidden_dim, 200),
#             act_func(),
#             nn.Linear(200, 2 * hidden_dim)
#         )

#     def forward(self, z):
#         out = self.net(z)
#         return out[:, :out.shape[1] // 2], out[:, out.shape[1] // 2:]


# class FC_encoder_mnist(nn.Module):
#     def __init__(self, act_func, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             # nn.Linear(784, 400),
#             # act_func(),
#             nn.Linear(784, 2 * hidden_dim)
#         )

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.view(batch_size, -1)
#         return self.net(x)


# class CONV_encoder(nn.Module):
#     def __init__(self, act_func, hidden_dim, n_channels, shape, upsampling=True):
#         super().__init__()
#         self.n_channels = n_channels
#         self.upsampling = upsampling
#         factor = 2 if upsampling else 1
#         num_maps = 16
#         num_units = ((shape // 8) ** 2) * (8 * num_maps // factor)

#         self.net = nn.Sequential(  # n
#             DoubleConv(n_channels, num_maps, act_func),
#             Down(num_maps, 2 * num_maps, act_func),
#             Down(2 * num_maps, 4 * num_maps, act_func),
#             Down(4 * num_maps, 8 * num_maps // factor, act_func),
#             nn.Flatten(),
#             nn.Linear(num_units, 2 * hidden_dim)
#         )

#     def forward(self, x):
#         return self.net(x)


# class FC_encoder_cifar(nn.Module):
#     def __init__(self, act_func, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(3072, 400),
#             act_func(),
#             nn.Linear(400, 2 * hidden_dim)
#         )

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.view(batch_size, -1)
#         return self.net(x)


# def get_encoder(net_type, act_func, hidden_dim, dataset, shape):
#     if str(dataset).lower().find('mni') > -1:
#         if net_type == 'fc':
#             return FC_encoder_mnist(act_func=act_func,
#                                     hidden_dim=hidden_dim)
#         elif net_type == 'conv':
#             return CONV_encoder(act_func=act_func,
#                                 hidden_dim=hidden_dim, n_channels=1, shape=shape)
#     else:
#         if net_type == 'fc':
#             return FC_encoder_cifar(act_func=act_func,
#                                     hidden_dim=hidden_dim)
#         elif net_type == 'conv':
#             return CONV_encoder(act_func=act_func,
#                                 hidden_dim=hidden_dim, n_channels=3, shape=shape)
