import io
import re
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Sequence

import torch
from torch import nn
from tqdm import tqdm, trange

from vaemcmc.callback import Callback
from vaemcmc.base_model import BaseModel
from .train_logger import TrainLogger
from .metric_log import MetricLog
from vaemcmc.utils.general import load_json, write_json


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


@dataclass
class Trainer:
    model: BaseModel
    opt: Any
    train_dataloader: Any
    num_steps: int
    val_dataloader: Optional[Any] = None
    save_steps: int = 5000
    print_steps: int = 1000
    vis_steps: int = 500
    log_steps: int = 50
    flush_secs : int = 30
    validate_steps: int = 0
    ckpt_file: Optional[Union[Path, str]] = None
    log_dir: Optional[Union[Path, str]] = None
    logger: Optional[TrainLogger] = None
    scheduler: Optional[Any] = None
    callbacks: Sequence[Callback] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.training_stats = defaultdict(list)
        
        if self.logger is None:
            self.logger = TrainLogger(
            log_dir=self.log_dir,
            num_steps=self.num_steps,
            dataset_size=len(self.train_dataloader),
            flush_secs=self.flush_secs,
            device=self.device,
        )
            
        if self.ckpt_file:
            self.ckpt_dir = Path(self.ckpt_file).parent
        elif self.log_dir:
            self.ckpt_dir = Path(self.log_dir, 'ckpts')
            self.ckpt_dir.mkdir(exist_ok=True, parents=True)
            self.ckpt_file = self._get_latest_checkpoint(self.ckpt_dir)  # can be None
        # self.ckpt_file = Path(self.ckpt_file)
            
    @property
    def device(self):
        return next(self.model.parameters()).device
            
    def _log_params(self, params):
        """
        Takes the argument options to save into a json file.
        """
        if not self.log_dir:
            return
        params_file = Path(self.log_dir, 'params.json')

        # Check for discrepancy with previous training config.
        if Path(params_file).exists():
            check = load_json(params_file)

            if params != check:
                diffs = []
                for k in params:
                    if k in check and params[k] != check[k]:
                        diffs.append('{}: Expected {} but got {}.'.format(
                            k, check[k], params[k]))

                diff_string = '\n'.join(diffs)
                raise ValueError(
                    "Current hyperparameter configuration is different from previously:\n{}"
                    .format(diff_string))

        write_json(params, params_file)

    def _get_latest_checkpoint(self, ckpt_dir: Union[Path, str]):
        """
        Given a checkpoint dir, finds the checkpoint with the latest training step.
        """
        def _get_step_number(k):
            """
            Helper function to get step number from checkpoint files.
            """
            search = re.search(r'(\d+)_steps', k)

            if search:
                return int(search.groups()[0])
            else:
                return -float('inf')

        if not Path(ckpt_dir).exists():
            return None

        files = list(Path(ckpt_dir).glob('*'))
        if len(files) == 0:
            return None

        ckpt_file = max(files, key=lambda x: _get_step_number(x.stem))

        return Path(ckpt_dir, ckpt_file)

    def _restore_models_and_step(self, global_step: int = 0) -> int:
        """
        Restores model and optimizer checkpoints and ensures global step is in sync.
        """

        if self.ckpt_file and Path(self.ckpt_file).exists():
            print("INFO: Restoring checkpoint ...")
            global_step = self.model.restore_checkpoint(ckpt_file=Path(self.ckpt_file), optimizer=self.opt)

        return global_step

    def _save_model_checkpoints(self, global_step: int):
        """
        Saves both discriminator and generator checkpoints.
        """
        self.model.save_checkpoint(directory=self.ckpt_dir, global_step=global_step, optimizer=self.opt)

    # def train_step(self, batch, it: int) -> Tuple[float, Any]:
    #     x, y = batch
    #     out = self.model(x)
    #     loss = self.model.loss_function(x, *out)
    #     loss = loss.mean(0)
    #     self.model.zero_grad()
    #     loss.backward()
    #     self.opt.step()

    #     return loss.item(), out

    # def val_step(self, batch):
    #     x, y = batch
    #     out = self.model(x)
    #     loss = self.model.loss_function(x, *out)

    #     return x, out, loss.item()

    def train(self, start_it: int = 0):
        start_it = self._restore_models_and_step(start_it)
        print(f"INFO: Starting training from global step {start_it}...")
        for callback in self.callbacks:
            callback.cnt = start_it
            
        #tqdm_out = TqdmToLogger(self.logger, level=logging.INFO)
        
        global_step = start_it
        try:
            start_time = time.time()
            
            train_iterator = iter(self.train_dataloader)
            for global_step in trange(start_it, self.num_steps, total=self.num_steps):
                log_data = MetricLog()  # log data for tensorboard
                # tqdm_out_inner = TqdmToLogger(self.logger, level=logging.INFO)
                # for it, batch in tqdm(enumerate(self.train_dataloader), file=tqdm_out_inner, total=len(self.train_dataloader)):
                ep_loss = 0
                batch = next(train_iterator)
                
                log_data = self.model.train_step(
                        batch=batch,
                        optimizer=self.opt,
                        log_data=log_data,
                        global_step=global_step,
                    )
                
                if self.scheduler:
                    self.scheduler.step()

                for callback in self.callbacks:
                    callback.invoke({"step": global_step}, log_data)

                if global_step % self.log_steps == 0:
                    self.logger.write_summaries(
                        log_data=log_data,
                        global_step=global_step,
                    )

                if global_step % self.print_steps == 0:
                    curr_time = time.time()
                    self.logger.print_log(
                        global_step=global_step,
                        log_data=log_data,
                        time_taken=(curr_time - start_time) / self.print_steps,
                    )
                    start_time = curr_time

                if global_step % self.vis_steps == 0:
                    self.logger.vis_images(model=self.model, global_step=global_step)

                if global_step % self.save_steps == 0:
                    print("INFO: Saving checkpoints...")
                    self._save_model_checkpoints(global_step)

            print("INFO: Saving final checkpoints...")
            self._save_model_checkpoints(global_step)
                
        except KeyboardInterrupt:
            print("INFO: Saving checkpoints from keyboard interrupt...")
            self._save_model_checkpoints(global_step)

        finally:
            self.logger.close_writers()

    def validate(self, it: int):
        for _, batch in enumerate(self.train_dataloader):
            self.val_step(batch)

    # def save(self, it: int):
    #     collection = {}
    #     collection["model_state_dict"] = self.model.state_dict()
    #     collection["optimizer_state_dict"] = self.opt.state_dict()
    #     collection["it"] = it
    #     if self.log_dir:
    #         ckpt_path = Path(self.ckpt_dir, f"ckpt_{it:06d}")
    #         torch.save(collection, ckpt_path)

    # def load(self, it: int):
    #     raise NotImplementedError
