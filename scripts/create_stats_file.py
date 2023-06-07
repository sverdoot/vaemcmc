from argparse import ArgumentParser
import importlib
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import __all__

from vaemcmc.registry import Registry
from vaemcmc.metrics.fid import compute_real_dist_stats


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_dir', type=str, default='log')
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.dataset in __all__:
        dataset_class = importlib.import_module(f'torchvision.datasets.{args.dataset}')
        test_dataset = dataset_class(args.data_dir, download=True, train=False)
        train_dataset = dataset_class(args.data_dir, download=True, train=True)
        dataset = ConcatDataset([test_dataset, train_dataset])
    else:
        raise NotImplementedError
    
    compute_real_dist_stats(args.num_samples, args.batch_size, dataset, log_dir=args.log_dir)

    