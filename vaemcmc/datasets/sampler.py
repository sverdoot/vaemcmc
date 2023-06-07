from typing import Sequence, Union
import numpy as np
from torch.utils.data import Sampler, Dataset


class InfiniteSampler(Sampler) :
    def __init__(self, dataset: Union[Dataset, Sequence], shuffle: bool = True):
        assert len(dataset) > 0
        self.dataset = dataset
        self.shuffle = shuffle
    
    def __iter__(self):
        order = np.arange((len(self.dataset)))
        idx = 0
        while True:
            yield order[idx]
            idx += 1
            if idx == len(order):
                if self.shuffle:
                    np.random.shuffle(order)
                idx = 0
                
    def __len__(self) -> int:
        return len(self.dataset)
                