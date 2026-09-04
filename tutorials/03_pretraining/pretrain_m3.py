#!/usr/bin/env python3
"""FM4NPP pretraining — paper m3 (width 256), tutorial edition.

    INPUT   a data_root holding BOTH splits (features_pretrain/ and features_test/)
    OUTPUT  runs/<config>/<run_num>/training_checkpoints/ckpt.tar  +  a loss CSV

Vendored from `train/pretrain/nppmamba/train_multi_gpu_mamba1.py` on branch
`downstream-reproducibility`, with six changes. Each is marked `# [TUT-n]` inline so you
can diff this against the original and see exactly what moved.

  [TUT-1] Mamba1GPT -> MambaGPT.
          The original builds a Mamba1 backbone, which (a) needs the compiled `mamba_ssm`
          package and (b) is the wrong architecture: every released checkpoint is Mamba2.
          MambaGPT runs on fm4npp/models/mamba2.py, which is pure PyTorch. One change
          removes a 20-minute CUDA build AND makes the output architecturally comparable
          to pp_nerf_m1_k30.ckpt.

  [TUT-2] Actually initialize the process group.
          The original never calls init_process_group, so `dist.is_initialized()` is
          always False: DDP wrapping is skipped, no DistributedSampler is built, every
          rank reads the same shard, and every rank writes the same checkpoint. SLURM's
          --ntasks-per-node=4 is silently ignored and you get one GPU out of four.

  [TUT-3] mu-parameterization behind --mup / --no-mup, via ../02_mu_parameterization/mup.py.
          The original is single-group AdamW with width-independent init.

  [TUT-4] Save and restore scheduler state.
          The original saves only model + optimizer. On resume the LR scheduler restarts
          from step 0, so a resumed run redoes its warmup and then follows the wrong
          cosine phase for the rest of training. best_loss is not restored either, so the
          first validation after a resume always overwrites ckpt_best.tar.

  [TUT-5] Fix the checkpoint-directory bug.
          The original creates training_checkpoints/ only inside `if not
          os.path.isdir(exp_dir)`. If exp_dir exists but the subdirectory does not — a
          crashed run, or a pre-created directory — the first torch.save fails.

  [TUT-6] --max_steps and --dry_run.
          --max_steps overrides total_steps so a debug-queue job stops cleanly without
          editing the config. --dry_run builds model, data and optimizer then exits,
          so you can validate the whole setup on a login node with no GPU time.

Example (single GPU):

    python pretrain_m3.py --yaml_config configs/tutorial_m3.yaml --config tutorial_m3 \
        --root_dir $FM4NPP_RUNS --run_num debug0 --max_steps 200 --mup
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '02_mu_parameterization'))
_FM4NPP_ROOT = os.environ.get('FM4NPP_ROOT')
if _FM4NPP_ROOT:
    sys.path.insert(0, _FM4NPP_ROOT)

from fm4npp.models.mambagpt import MambaGPT            # [TUT-1] was Mamba1GPT  # noqa: E402
from fm4npp.datasets.dataset_pretrain import get_data_loader                    # noqa: E402
from fm4npp.utils import YParams, pickle_load                                   # noqa: E402

import mup                                             # [TUT-3]                # noqa: E402


def count_parameters(model):
    """Returns (trainable, total).

    These differ: the NeRF embedder's Fourier projection is a FIXED random matrix
    (embedder.embed.projection, 63x256 = 16,128 values, plus a 256-wide scale), so it
    counts toward the checkpoint's size but never receives a gradient. The released m3
    checkpoint contains 4,923,386 values of which 4,907,002 are trainable.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def apply_bin_weights_torch(bin_list, weight_list, target):
    """Bin-based loss weighting (used only when params.loss_reweight is true)."""
    weight_list, bin_list = weight_list.to(target.device), bin_list.to(target.device)
    target = target.unsqueeze(-1)
    mask_in_bin = ((target >= bin_list[:-1].unsqueeze(0)).float()
                   * (target < bin_list[1:].unsqueeze(0)).float())
    return (mask_in_bin * weight_list.unsqueeze(0)).sum(-1)


# --------------------------------------------------------------------------- [TUT-2]
def setup_distributed():
    """Initialize the process group from SLURM/torchrun env vars.

    The sbatch script exports RANK / LOCAL_RANK / WORLD_SIZE per task. With WORLD_SIZE=1
    (a laptop, or a login-node dry run) we stay single-process and never touch NCCL.
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
        if dist.get_rank() == 0:
            print(f'[dist] process group up: world_size={dist.get_world_size()}')
    return world_size


class Trainer:
    def __init__(self, params, args):
        self.params, self.args = params, args
        self.root_dir, self.config, self.run_num = args.root_dir, args.config, args.run_num

        self.world_rank, self.local_rank, self.world_size = 0, 0, 1
        if dist.is_initialized():
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = dist.get_world_size()

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
        else:
            # the original calls set_device unconditionally and dies on CPU-only hosts
            self.device = torch.device('cpu')

        exp_dir = os.path.join(self.root_dir, self.config, self.run_num)
        self.exp_dir = exp_dir
        self.checkpoint_path = os.path.join(exp_dir, 'training_checkpoints/ckpt.tar')

        # [TUT-5] unconditional, not nested inside the exp_dir check
        if self.world_rank == 0:
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(os.path.join(exp_dir, 'training_checkpoints'), exist_ok=True)

        self.iters, self.startEpoch, self.epoch = 0, 0, 0
        self.best_loss = np.inf
        self.logs = {}
        self.logfile = os.path.join(exp_dir, 'train.log')
        self.globalfile = os.path.join(
            exp_dir, f'config_{self.config}_run_{self.run_num}.csv')

        if self.world_rank == 0 and not os.path.exists(self.globalfile):
            with open(self.globalfile, 'w') as f:
                f.write('split,step,loss,lr\n')

        # These are loaded unconditionally even though they are only used when
        # loss_reweight is true. stat_dir must be the repo's stats/ -- if the bin-edges
        # pickle is missing the Voxelizer silently rebins and every number is wrong.
        self.loss_bin = pickle_load(f'{params.stat_dir}/loss_bin_pp.pkl')
        self.loss_weight = pickle_load(f'{params.stat_dir}/loss_weight_pp.pkl')

        self.train_data_loader, self.train_sampler, self.valid_data_loader, _ = \
            get_data_loader(params, dist.is_initialized())

        self.klen = params.klen
        d_state = getattr(params, 'd_state', 16)
        Nu = params.embed_dim

        # [TUT-1] MambaGPT == Mamba2, pure PyTorch, matches the released checkpoints.
        self.model = MambaGPT(
            embed_dim=Nu, num_layers=params.num_layers_backbone, d_state=d_state,
            d_conv=4, expand=2, klen=self.klen, dropout=params.dropout,
            embed_method=params.embed_method, pe_method=params.pe_method)

        # [TUT-3] width-dependent init, or the shipped one
        mup.apply_mup_init(self.model, Nu=Nu, Nx=d_state,
                           enabled=args.mup, verbose=(self.world_rank == 0))
        self.model = self.model.to(self.device)

        if self.world_rank == 0:
            print(f'✅ MambaGPT (Mamba2) initialized  width={Nu} d_state={d_state} '
                  f'layers={params.num_layers_backbone}')
            _tr, _tot = count_parameters(self.model)
            print(f'   Nparams: {_tot:,} total, {_tr:,} trainable '
                  f'({_tot - _tr:,} frozen in the NeRF embedder)')

        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.local_rank],
                output_device=self.local_rank,      # scalar; the original passes a list
                find_unused_parameters=True)

        # [TUT-3] four width-scaled groups, or one flat group
        self.optimizer = mup.build_mup_optimizer(
            self.model, Nu=Nu, Nx=d_state, base_lr=params.min_lr,
            enabled=args.mup, verbose=(self.world_rank == 0))

        total = args.max_steps or params.total_steps          # [TUT-6]
        self.total_steps = total
        self.scheduler = mup.attach_scheduler(
            self.optimizer, total_steps=total, max_lr=params.max_lr,
            min_lr=params.min_lr, warmup_steps=min(params.warmup_steps, total // 5),
            preserve_mup=args.mup, verbose=(self.world_rank == 0))

        self.loss_func = nn.MSELoss(reduction='none')
        self.loss_func_eval = nn.MSELoss(reduction='none')
        self.restore_checkpoint()

    # ------------------------------------------------------------------ logging
    def log_globalfile(self, split, step, loss, lr):
        if self.world_rank == 0:
            with open(self.globalfile, 'a') as f:
                f.write(f'{split},{step},{loss},{lr}\n')

    def log_infile(self, log):
        if self.world_rank == 0:
            with open(self.logfile, 'a') as f:
                f.write(f'{log}\n')

    # ------------------------------------------------------------------ checkpoints
    def restore_checkpoint(self):
        if not os.path.isfile(self.checkpoint_path):
            return
        if self.world_rank == 0:
            print(f'Loading checkpoint {self.checkpoint_path}')
        ck = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(ck['model_state'])
        except Exception:
            (self.model.module if dist.is_initialized() else self.model
             ).load_state_dict(ck['model_state'])
        self.optimizer.load_state_dict(ck['optimizer_state_dict'])
        # [TUT-4] without these the schedule silently restarts and best_loss resets
        if 'scheduler_state_dict' in ck:
            self.scheduler.load_state_dict(ck['scheduler_state_dict'])
        self.best_loss = ck.get('best_loss', np.inf)
        self.iters = ck['iters']
        self.startEpoch = ck['epoch'] + 1
        if self.world_rank == 0:
            print(f'Resuming at epoch {self.startEpoch}, iter {self.iters}, '
                  f'best_loss {self.best_loss}')

    def save_checkpoint(self, path, is_best=False):
        if self.world_rank != 0:
            return
        model_state = (self.model.module if dist.is_initialized() else self.model
                       ).state_dict()
        payload = {
            'iters': self.iters,
            'epoch': self.epoch,
            'model_state': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),   # [TUT-4]
            'best_loss': float(self.best_loss),                    # [TUT-4]
        }
        torch.save(payload, path)
        if is_best:
            torch.save(payload, path.replace('.tar', '_best.tar'))

    def report_loss(self, loss):
        if dist.is_initialized():
            t = loss.clone().detach()
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            return t.item()
        return loss.item()

    # ------------------------------------------------------------------ train
    def _forward_loss(self, grouped, knearest, eval_mode=False):
        b, c = grouped.size(0), grouped.size(-1)
        targets = grouped.reshape(b, -1, 4)[:, :, 1:].to(self.device)
        klabel = knearest.reshape(b, -1, self.klen * 3).to(self.device)
        grouped = grouped.reshape(b, -1, c).to(self.device)

        pred = self.model(grouped)
        kmask = klabel != -100
        tmask = targets[..., 0] != -100

        if eval_mode:
            return self.loss_func_eval(pred[kmask], klabel[kmask]).mean(), tmask

        loss = self.loss_func(pred, klabel)
        if self.params.loss_reweight:
            loss = (loss * kmask).sum(-1).sum(-1) / kmask.sum(-1).sum(-1)
            w = apply_bin_weights_torch(torch.Tensor(self.loss_bin),
                                        torch.Tensor(self.loss_weight), tmask.sum(-1))
            loss = (loss * w).mean()
        else:
            loss = (loss * kmask).sum() / kmask.sum()
        return loss, tmask

    def train_one_epoch(self):
        self.model.train()
        tr_time, tr_start = 0, time.time()

        for grouped, _, knearest in self.train_data_loader:
            self.iters += 1
            self.model.zero_grad()
            loss, _ = self._forward_loss(grouped, knearest)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                            clip_value=self.params.grad_clip_value)
            self.optimizer.step()
            self.scheduler.step()

            if self.world_rank == 0:
                lr_now = self.optimizer.param_groups[-1]['lr']   # the 'other' group
                l = self.report_loss(loss)
                if self.iters % 20 == 0:
                    el = time.time() - tr_start
                    print(f'  iter {self.iters:>5d}/{self.total_steps}  loss={l:.5f}  '
                          f'lr={lr_now:.3e}  {el/max(self.iters,1):.2f}s/it', flush=True)
                self.log_globalfile('train', self.iters, l, lr_now)

            if self.iters % self.params.n_eval_steps == 0:
                tr_time += time.time() - tr_start
                self.val_one_epoch(tr_time)
                tr_start = time.time()

            if self.iters >= self.total_steps:
                break
        return tr_time

    def val_one_epoch(self, tr_time):
        self.model.eval()
        val_start = time.time()
        buff = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.logs['val_loss'] = buff[0].view(-1)

        with torch.no_grad():
            for grouped, _, knearest in self.valid_data_loader:
                loss, _ = self._forward_loss(grouped, knearest, eval_mode=True)
                self.logs['val_loss'] += loss.detach()

        self.logs['val_loss'] /= max(len(self.valid_data_loader), 1)
        if dist.is_initialized():
            dist.all_reduce(self.logs['val_loss'].detach())
            self.logs['val_loss'] /= dist.get_world_size()

        val = float(self.logs['val_loss'])
        is_best = val <= self.best_loss
        if is_best:
            self.best_loss = val
        if self.params.save_checkpoint:
            self.save_checkpoint(self.checkpoint_path, is_best=is_best)

        msg = (f'  [val] step {self.iters}  val_loss={val:.5f}'
               f'{"  <- best" if is_best else ""}  '
               f'({time.time()-val_start:.1f}s val, {tr_time:.1f}s train)')
        if self.world_rank == 0:
            print(msg, flush=True)
            self.log_infile(msg)
            self.log_globalfile('val', self.iters, val,
                                self.optimizer.param_groups[-1]['lr'])
        self.model.train()

    def launch(self):
        if self.world_rank == 0:
            print('=' * 72)
            print(f'FM4NPP pretraining — paper m3 (width {self.params.embed_dim})')
            print(f'  mu-parameterization : {"ON" if self.args.mup else "OFF"}')
            print(f'  total steps         : {self.total_steps}')
            print(f'  world size          : {self.world_size}')
            print(f'  output              : {self.exp_dir}')
            print('=' * 72, flush=True)

        for epoch in range(self.startEpoch, self.params.max_epochs):
            if self.iters >= self.total_steps:
                break
            self.epoch = epoch
            if dist.is_initialized() and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.train_one_epoch()

        if self.iters % self.params.n_eval_steps != 0:
            self.val_one_epoch(0.0)          # always end on a validation + save

        if self.world_rank == 0:
            print('=' * 72)
            print(f'DONE. {self.iters} steps, best val loss {self.best_loss:.5f}')
            print(f'checkpoint: {self.checkpoint_path}')
            print(f'loss log  : {self.globalfile}')
            print('=' * 72)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--yaml_config', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--run_num', default='run0')
    p.add_argument('--root_dir', default=os.environ.get('FM4NPP_RUNS', './runs'))
    p.add_argument('--max_steps', type=int, default=None,
                   help='[TUT-6] override total_steps (for the 10-minute debug queue)')
    p.add_argument('--dry_run', action='store_true',
                   help='[TUT-6] build everything, then exit before training')
    mux = p.add_mutually_exclusive_group()
    mux.add_argument('--mup', dest='mup', action='store_true',
                     help='width-scaled init + per-group LRs (default)')
    mux.add_argument('--no-mup', dest='mup', action='store_false',
                     help='reproduce the shipped behaviour: one group, no width scaling')
    p.set_defaults(mup=True)
    args = p.parse_args()

    setup_distributed()                                          # [TUT-2]
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    trainer = Trainer(params, args)
    if args.dry_run:
        if trainer.world_rank == 0:
            print('\n[dry run] model, data and optimizer built successfully. '
                  'Exiting before the training loop.')
    else:
        trainer.launch()

    if dist.is_initialized():
        dist.destroy_process_group()                             # [TUT-2]


if __name__ == '__main__':
    main()
