from typing import Tuple
from torch import nn
from torch_mimicry.nets import dcgan, sngan

from vaemcmc.registry import Registry


@Registry.register_model()
class MMCDCGenerator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.gen = dcgan.DCGANGenerator32()
        self.z_dim = self.gen.nz

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.gen.load_state_dict(state_dict["model_state_dict"], strict=strict)

    def forward(self, z):
        return self.gen(z)


@Registry.register_model()
class MMCDCDiscriminator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.dis = dcgan.DCGANDiscriminator32()

    @property
    def penult_layer(self):
        return self.dis.activation

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.dis.load_state_dict(state_dict["model_state_dict"], strict=strict)

    def forward(self, x):
        return self.dis(x)


@Registry.register_model()
class MMCSNGenerator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.gen = sngan.SNGANGenerator32()
        self.z_dim = self.gen.nz

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.gen.load_state_dict(state_dict["model_state_dict"], strict=strict)

    def forward(self, z):
        return self.gen(z)


@Registry.register_model()
class MMCSNDiscriminator(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.dis = sngan.SNGANDiscriminator32()

    @property
    def penult_layer(self):
        return self.dis.activation

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.dis.load_state_dict(state_dict["model_state_dict"], strict=strict)

    def forward(self, x):
        return self.dis(x)