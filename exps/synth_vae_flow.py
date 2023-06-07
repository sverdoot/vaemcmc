from typing import Type
import logging
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from munch import Munch
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.distributions import Normal, Independent
from torch.utils.data import DataLoader

from vaemcmc.utils.general import random_seed
from vaemcmc.config import Config
from vaemcmc.registry import Registry
from vaemcmc.vae import VAEProposal
from vaemcmc.datasets.synthetic import SyntheticDataset
from vaemcmc.datasets.sampler import InfiniteSampler
from vaemcmc.training.trainer import Trainer
from vaemcmc.viz.synthetic import plot_kde, plot_chain, scatter

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

    PROJ = (0, 1)
   
    target = Registry.dist_registry[cfg.dist.name](**cfg.dist.params)
    target.to(device)
    
    # if cfg.use_mcmc_gt:
    #
    # else:
    # dataset = 
    true_sample = target.sample((cfg.n_chains * cfg.n_samples,))
    x_start = true_sample[np.random.choice(np.arange(true_sample.shape[0]), cfg.n_chains)]
    
    train_dataset = SyntheticDataset(target.sample((cfg.train_data,)))
    train_dataloader = DataLoader(train_dataset, sampler=InfiniteSampler(train_dataset), batch_size=cfg.batch_size)
    val_dataset = SyntheticDataset(target.sample((cfg.train_data,)))
    val_dataloader = DataLoader(val_dataset, sampler=InfiniteSampler(train_dataset), batch_size=cfg.batch_size)
    
    models_n = ['vae'] #, 'flow']
    chains = dict()
    samples = dict()
    for model_n in models_n:
        print(model_n)
        model = Registry.model_registry[cfg[model_n].name](**cfg[model_n].params)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        trainer = Trainer(model, optimizer, train_dataloader, val_dataloader=val_dataloader, num_steps=cfg.n_train_iters, log_dir=cfg.log_dir)
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
    
    target.to('cpu')
    
    if hasattr(target, "inv_embed"):
        proj = lambda sample: target.inv_embed(sample)[..., PROJ]
    else:
        proj = lambda sample: sample[..., PROJ]
        
    true_sample_proj = proj(true_sample)
    samples = {k: proj(v) for k, v in samples.items()}
    chains = {k: proj(v) for k, v in chains.items()}
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for ax in axs:
        scatter(true_sample_proj.view(-1, 2), c='forestgreen', ax=ax, label='ground truth')
        
    for i, ax in enumerate(axs.flatten()):
        model_n = models_n[(i + 1) % 2]
        if i < 2:
            scatter(samples[model_n].view(-1, 2), ax=ax, label=f'{model_n}')
        else:
            scatter(chains[model_n].view(-1, 2), ax=ax, label=f'{model_n} mcmc')

    fig.tight_layout()
    for ax in axs:
        ax.legend()
    fig.savefig(Path(cfg.fig_dir, f'scatter.png').as_posix())
    
    fig, axs = plt.subplots(1, 5, figsize=(12, 2))
    axs = axs.flatten()
    plot_kde(true_sample_proj.view(-1, 2), cfg.xlim, cfg.ylim, ax=axs[0], label='ground truth')
        
    for i, ax in enumerate(axs.flatten()[1:]):
        model_n = models_n[(i + 1) % 2]
        if i < 2:
            plot_kde(samples[model_n].view(-1, 2), cfg.xlim, cfg.ylim, ax=ax, label=f'{model_n}')
        else:
            plot_kde(chains[model_n].view(-1, 2), cfg.xlim, cfg.ylim, ax=ax, label=f'{model_n}')

    fig.tight_layout()
    fig.savefig(Path(cfg.fig_dir, f'kde.png').as_posix())
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs = axs.flatten()

    for i, ax in enumerate(axs.flatten()):
        model_n = models_n[(i + 1) % 2]
        plot_chain(chains[model_n][:, 0].view(-1, 2), ax=ax)
        ax.set_xlim(cfg.xlim)
        ax.set_ylim(cfg.ylim)

    fig.tight_layout()
    fig.savefig(Path(cfg.fig_dir, f'chain.png').as_posix())

        
if __name__ == "__main__":
    args = parse_args()
    args = Munch(vars(args))
    cfg = Config.load(args.config)
    cfg.update(args)
    
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    fig_dir = Path(log_dir, 'figs')
    fig_dir.mkdir(exist_ok=True, parents=True)
    cfg.log_dir = log_dir.as_posix()
    cfg.fig_dir = fig_dir.as_posix()
    cfg.dump(Path(cfg.log_dir, Path(cfg.config).name))

    main(cfg)
    