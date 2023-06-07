from typing import Any, Sequence

from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, sample: Sequence[Any]):
        self.sample = sample

    def __getitem__(self, idx: int):
        return self.sample[idx], 0

    def __len__(self):
        return len(self.sample)
