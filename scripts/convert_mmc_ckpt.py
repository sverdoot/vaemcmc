from argparse import ArgumentParser
from pathlib import Path
import torch
from vaemcmc.registry import Registry
from vaemcmc.config import Config


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('cfg', type=str)
    parser.add_argument('g_path', type=str)
    parser.add_argument('d_path', type=str)
    parser.add_argument('--save_path', type=str)
    
    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.load(args.cfg)
    
    gan = Registry.model_registry[cfg.name](**cfg.params)
    gan.gen.load_state_dict(torch.load(args.g_path, map_location='cpu'))
    gan.dis.load_state_dict(torch.load(args.d_path, map_location='cpu'))
    state_dict = gan.state_dict()
    if args.save_path:
        Path(args.save_path).parent.mkdir(exist_ok=True)
        torch.save({"model_state_dict": state_dict}, Path(args.save_path))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    
