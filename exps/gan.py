from typing import Type
import logging
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from munch import Munch
from matplotlib import pyplot as plt
import seaborn as sns
import pyro
from pyro.infer import MCMC, HMC, NUTS
import torch
from torch.distributions import Independent, Normal
from torch.utils.data import DataLoader

from vaemcmc.utils.general import random_seed
from vaemcmc.config import Config
from vaemcmc.registry import Registry
from vaemcmc.vae import VAEProposal
from vaemcmc.datasets.synthetic import SyntheticDataset
from vaemcmc.datasets.sampler import InfiniteSampler
from vaemcmc.training.trainer import Trainer
# from vaemcmc.viz.synthetic import plot_kde, plot_chain, scatter

from exps import LOG_DIR

sns.set_context('paper')

# logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
    
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    return args


def main(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    if cfg.seed is not None:
        random_seed(cfg.seed)

    target = Registry.dist_registry[cfg.dist.name](**cfg.dist.params)
    target.to(device)
    
    if cfg.use_pyro_gt:
        if cfg.gt_path and Path(cfg.gt_path).exists():
            true_sample = torch.load(Path(cfg.gt_path), map_location=device)
        else:
            proposal = Independent(
                Normal(torch.zeros(target.dim, device=device), torch.ones(target.dim, device=device)), 1
                )
            start = proposal.sample((cfg.gt_chains,))
        
            gt_kernel = Registry.mcmc_registry[cfg.gt_mcmc.name](
                potential_fn=lambda z: -target.log_prob(z["points"]).sum(), **cfg.gt_mcmc.params
                )
            mcmc_true = MCMC(
                kernel=gt_kernel,
                num_samples=cfg.gt_steps,
                warmup_steps=cfg.gt_burn_in,
                initial_params={"points": start}
            )
            mcmc_true.run()
            true_sample = mcmc_true.get_samples(group_by_chain=True)["points"].cpu()
            print(true_sample.shape)
            true_sample = true_sample.reshape(cfg.gt_steps, cfg.gt_chains, -1)
            if cfg.gt_path:
                cfg.gt_path = Path(cfg.gt_path)
                cfg.gt_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(true_sample, cfg.gt_path)
     
    x_start = true_sample[np.random.choice(np.arange(true_sample.shape[0]), cfg.n_chains)]
    
    train_dataset = SyntheticDataset(true_sample)
    train_dataloader = DataLoader(train_dataset, sampler=InfiniteSampler(train_dataset), batch_size=cfg.batch_size)
    #val_dataset = SyntheticDataset(target.sample((cfg.train_data,)))
    #val_dataloader = DataLoader(val_dataset, sampler=InfiniteSampler(train_dataset), batch_size=cfg.batch_size)
    
    models_n = ['vae', 'flow']
    chains = dict()
    samples = dict()
    for model_n in models_n:
        print(model_n)
        model = Registry.model_registry[cfg[model_n].name](**cfg[model_n].params)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        trainer = Trainer(model, optimizer, train_dataloader, num_steps=cfg.n_train_iters, log_dir=cfg.log_dir)
        trainer.train()
        
        samples[model_n] = model.sample((cfg.n_chains * cfg.n_samples,)).detach().cpu()
                
        if model_n == 'vae':
            prior = Independent(Normal(torch.zeros(cfg.lat_dim, device=device), 0.5 * torch.ones(cfg.lat_dim, device=device)), 1)
            proposal = VAEProposal(model, prior)
        else:
            model.prior = Independent(Normal(torch.zeros(cfg.amb_dim, device=device), 1 * torch.ones(cfg.amb_dim, device=device)), 1)
            proposal = model

        kernel = Registry.mcmc_registry[cfg.mcmc.name](target=target, proposal=proposal, **cfg.mcmc.params)
        chain = kernel.run(x_start, cfg.burn_in, cfg.n_samples)
        chains[model_n] = chain.detach().cpu()
    
        
if __name__ == "__main__":
    args = parse_args()
    args = Munch(vars(args))
    cfg = Config.load(args.config)
    cfg.update(args)
    
    log_dir = Path(LOG_DIR, f'{Path(cfg.config).stem}')
    log_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = Path(log_dir, 'figs')
    fig_dir.mkdir(exist_ok=True, parents=True)
    cfg.log_dir = log_dir.as_posix()
    cfg.fig_dir = fig_dir.as_posix()
    cfg.dump(Path(cfg.log_dir, Path(cfg.config).name))

    main(cfg)
    