# -*- coding: utf-8 -*-
"""Single-/multi-GPU training of BRENet (BRE backbone: BREB + JEPF + HEAL) + TBS.

    python train.py                       # defaults = BRENet + TBS, IRSTD-1k official split (README)
    python train.py --w-tbs 0             # the baseline (BRENet without TBS)
"""

import math
import os

import json
import os.path as ops
import random
import socket
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data_utils import DatasetLoader
from loss import BackgroundReferenceLoss
from metric import ROCMetric, SigmoidMetric, mIoU
from model_base import ISNet


class GradientMask(nn.Module):
    """Generate the inherited structural-branch supervision from the mask."""

    def __init__(self):
        super().__init__()
        vertical = torch.tensor(
            [[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        horizontal = torch.tensor(
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer('weight_v', vertical)
        self.register_buffer('weight_h', horizontal)

    def forward(self, x):
        gv = F.conv2d(x[:, :1], self.weight_v.to(x.device, x.dtype), padding=1)
        gh = F.conv2d(x[:, :1], self.weight_h.to(x.device, x.dtype), padding=1)
        magnitude = torch.sqrt(gv.square() + gh.square() + 1e-6)
        magnitude = magnitude - magnitude.amin(dim=(1, 2, 3), keepdim=True)
        magnitude = magnitude / (magnitude.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
        return magnitude.clamp(0.0, 1.0)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def is_dist_ready():
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_dist_ready():
        dist.barrier()


def reduce_sum_count(value_sum, count, device):
    pair = torch.tensor([float(value_sum), float(count)], dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float((pair[0] / pair[1].clamp_min(1.0)).item())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_locate_dataset(args, verbose=True):
    root = ops.dirname(ops.abspath(__file__))
    candidates = [
        args.data_root,
        ops.join(root, 'datasets', args.dataset),
        ops.join(root, 'datasets', args.dataset.replace('_', '-')),
        ops.join(root, 'datasets', args.dataset.replace('-', '_')),
        ops.join(root, '..', 'datasets', args.dataset),
        ops.join(root, args.dataset),
    ]
    if args.dataset == 'sirst':  # NUAA-SIRST 4:1 split prepared in datasets/sirst_aug
        candidates += [ops.join(root, 'datasets', 'sirst_aug'), ops.join(root, '..', 'datasets', 'sirst_aug')]
    for path in candidates:
        if path and ops.isdir(path):
            args.data_root = ops.abspath(path)
            break
    if verbose:
        print('[Data root]', args.data_root or '(DatasetLoader default)')


def parse_args():
    parser = ArgumentParser(description='BRENet training')
    parser.add_argument('--dataset', type=str, default='IRSTD-1k')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--split', type=str, default='official', choices=['9-1', 'official'],
                        help='IRSTD-1k split: 9-1 = 900/101 (earlier BRENet runs), official = 800/201')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='global batch size; in DDP it must divide evenly by world size')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Adagrad learning rate (recipe of the earlier BRENet runs)')
    parser.add_argument('--lr-schedule', type=str, default='step30', choices=['step30', 'cosine'],
                        help='step30: halve every 30 epochs (earlier runs); cosine: anneal to --min-lr')
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='cosine only: linear warm-up from lr/warmup to lr over these epochs')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--aug', type=str, default='mmlnet', choices=['mmlnet', 'light', 'medium'],
                        help='mmlnet: the inherited photometric/elastic augmentation; light: flips, transposes, scale and crop only; '
                             'medium: light with more frequent rescaling plus brightness, contrast, gamma and noise (no blur or warping)')
    parser.add_argument('--input-scale', type=int, default=1,
                        help='upsample images and masks by this factor before the network (train and test); 1 = off')
    parser.add_argument('--repeats', type=int, default=0,
                        help='repetitions of the training list per epoch (0: dataset default in data_utils.REPEATS)')
    parser.add_argument('--eval-thr', type=float, default=0.5)
    parser.add_argument('--eval-every', type=int, default=1,
                        help='evaluate every N epochs and at the last epoch (1 = every epoch)')
    parser.add_argument('--distance-thresh', type=float, default=3.0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3407)

    # Network-wide deployment is the principal architectural ablation.
    parser.add_argument('--deployment', type=str, default='full', choices=sorted(ISNet.VALID_DEPLOYMENTS),
                        help='full=10 key stage interfaces; dense=former all-block replacement')
    parser.add_argument('--stage-repeat', type=int, default=1,
                        help='number of BREBs per selected stage; use 1/2/3 for repetition-depth ablation')
    parser.add_argument('--evidence-mode', type=str, default='positive',
                        choices=['positive', 'symmetric'],
                        help='positive is BREB; symmetric is the parameter/computation-matched control')
    parser.add_argument('--fusion-mode', type=str, default='partition', choices=sorted(ISNet.VALID_FUSIONS),
                        help='partition = JEPF (BRENet); legacy = MMLNet fusion; uniform = equal-state control')
    parser.add_argument('--fusion-scope', type=str, default='full', choices=sorted(ISNet.VALID_FUSION_SCOPES),
                        help='JEPF deployment: single scale, two scales, or full')
    # New contributions are added one at a time.
    parser.add_argument('--w-tbs', type=float, default=1.0,
                        help='weight of Target-Balanced Supervision (0 = off)')
    parser.add_argument('--tbs-fa', type=int, choices=[0, 1], default=0,
                        help='TBS also counts every false-alarm component once (object-level false-alarm term)')
    parser.add_argument('--tbs-context', type=int, default=4,
                        help='local background ring of each target in TBS (px)')
    # Optional sensitivity switches (defaults reproduce the paper configuration).
    parser.add_argument('--breb-windows', type=str, default='default',
                        choices=['default', 'narrow', 'wide'])
    parser.add_argument('--breb-tau-init', type=float, default=1.0)
    parser.add_argument('--breb-alpha-init', type=float, default=0.1)
    # Representation-objective consistency, subordinate to the architectural block.
    parser.add_argument('--use-reference-loss', type=int, choices=[0, 1], default=1)
    parser.add_argument('--w-aux', type=float, default=0.25)
    parser.add_argument('--w-edge', type=float, default=1.0)
    parser.add_argument('--w-reference', type=float, default=0.15)
    parser.add_argument('--ramp-start', type=int, default=10)
    parser.add_argument('--ramp-end', type=int, default=60)
    parser.add_argument('--target-margin', type=float, default=1.0)
    parser.add_argument('--background-margin', type=float, default=1.0)

    # HEAL is training-only.  It traces persistent false alarms through the three JEPF
    # evidence states and redirects them without changing inference.
    parser.add_argument('--use-heal', type=int, choices=[0, 1], default=1)
    parser.add_argument('--w-heal', type=float, default=0.08)
    parser.add_argument('--heal-scope', type=str, default='full', choices=sorted(ISNet.VALID_FUSION_SCOPES))
    parser.add_argument('--heal-mining', type=str, default='persistent', choices=['persistent', 'final', 'random'])
    parser.add_argument('--heal-threshold', type=float, default=0.35)
    parser.add_argument('--heal-hard-fraction', type=float, default=0.0001)
    parser.add_argument('--heal-min-seeds', type=int, default=2)
    parser.add_argument('--heal-max-seeds', type=int, default=32)
    parser.add_argument('--heal-region-size', type=int, default=3)
    parser.add_argument('--heal-protection-size', type=int, default=5)
    parser.add_argument('--heal-ramp-start', type=int, default=20)
    parser.add_argument('--heal-ramp-end', type=int, default=80)

    parser.add_argument('--pareto-iou-margin', type=float, default=0.015,
                        help='keep checkpoints/pareto/ for epochs within this IoU of the best so far')
    parser.add_argument('--ddp', type=int, choices=[0, 1], default=1)
    parser.add_argument('--ddp-gpus', type=int, default=4)
    parser.add_argument('--master-port', type=int, default=0)
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-1)
    return parser.parse_args()


class Trainer:
    def __init__(self, args, local_rank=0, world_size=1, distributed=False):
        self.args = args
        self.distributed = bool(distributed)
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.rank = dist.get_rank() if self.distributed else 0
        self.is_main = self.rank == 0

        if args.stage_repeat < 1:
            raise ValueError('--stage-repeat must be >= 1')
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda:%d' % self.local_rank)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        seed_everything(args.seed)
        auto_locate_dataset(args, verbose=self.is_main)
        if self.distributed and args.batch_size % self.world_size != 0:
            raise ValueError('--batch-size must divide evenly by world_size')
        self.per_gpu_batch = args.batch_size // self.world_size if self.distributed else args.batch_size
        if self.per_gpu_batch < 1:
            raise ValueError('per-GPU batch size is less than one')

        trainset = DatasetLoader(args, mode='train')
        self.train_sampler = None
        if self.distributed:
            self.train_sampler = DistributedSampler(
                trainset, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, seed=args.seed, drop_last=False,
            )
        self.train_loader = Data.DataLoader(
            trainset, batch_size=self.per_gpu_batch,
            shuffle=self.train_sampler is None, sampler=self.train_sampler,
            num_workers=args.num_workers, pin_memory=self.device.type == 'cuda',
            persistent_workers=bool(args.num_workers > 0),
        )

        self.val_loader = None
        if self.is_main:
            valset = DatasetLoader(args, mode='test')
            self.val_loader = Data.DataLoader(
                valset, batch_size=1, shuffle=False,
                num_workers=args.num_workers, pin_memory=self.device.type == 'cuda',
                persistent_workers=bool(args.num_workers > 0),
            )
            print('[Data] train=%d, test=%d' % (len(trainset), len(valset)))
            print('[DDP] mode=%s | world_size=%d | global_batch=%d | per_gpu_batch=%d' % (
                'on' if self.distributed else 'off', self.world_size,
                args.batch_size, self.per_gpu_batch,
            ))

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.net = ISNet(
            channels=(16, 32, 64, 128),
            deployment=args.deployment,
            stage_repeat=args.stage_repeat,
            evidence_mode=args.evidence_mode,
            fusion_mode=args.fusion_mode,
            fusion_scope=args.fusion_scope,
            breb_windows=args.breb_windows,
            breb_tau_init=args.breb_tau_init,
            breb_alpha_init=args.breb_alpha_init,
        ).to(self.device)
        self.net.apply(self.weight_init)
        if self.is_main:
            model = self.net
            parameters = sum(p.numel() for p in model.parameters())
            print('[Architecture] BREB=%s (%d blocks, %s) | JEPF=%s/%s | HEAL=%s | TBS=%g%s | params=%.3fM' % (
                args.deployment, model.reference_block_count, args.evidence_mode,
                args.fusion_mode, args.fusion_scope, 'on' if args.use_heal else 'off',
                args.w_tbs, '+FA' if args.tbs_fa else '', parameters / 1e6,
            ))

        if self.distributed:
            self.net = DDP(
                self.net, device_ids=[self.local_rank], output_device=self.local_rank,
                broadcast_buffers=True, find_unused_parameters=False,
            )

        self.gradmask = GradientMask().to(self.device)
        self.criterion = BackgroundReferenceLoss(
            w_aux=args.w_aux,
            w_edge=args.w_edge,
            w_reference=args.w_reference,
            ramp_start=args.ramp_start,
            ramp_end=args.ramp_end,
            target_margin=args.target_margin,
            background_margin=args.background_margin,
            use_reference=bool(args.use_reference_loss),
            w_heal=args.w_heal,
            heal_scope=args.heal_scope,
            heal_mining=args.heal_mining,
            heal_threshold=args.heal_threshold,
            heal_hard_fraction=args.heal_hard_fraction,
            heal_min_seeds=args.heal_min_seeds,
            heal_max_seeds=args.heal_max_seeds,
            heal_region_size=args.heal_region_size,
            heal_protection_size=args.heal_protection_size,
            heal_ramp_start=args.heal_ramp_start,
            heal_ramp_end=args.heal_ramp_end,
            use_heal=bool(args.use_heal),
            w_tbs=args.w_tbs,
            tbs_context=args.tbs_context,
            tbs_false_alarm=bool(args.tbs_fa),
        ).to(self.device)
        self.optimizer = torch.optim.Adagrad(
            self.net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        self.best_miou = -1.0
        self.pareto = []
        self.miou = mIoU(1, threshold=args.eval_thr) if self.is_main else None

        self.run_root = self._build_run_root()
        self.ckpt_dir = ops.join(self.run_root, 'checkpoints')
        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            payload = vars(args).copy()
            payload.update({
                'ddp_enabled': self.distributed,
                'world_size': self.world_size,
                'global_batch_size': args.batch_size,
                'per_gpu_batch_size': self.per_gpu_batch,
                'reference_block_count': self.unwrap(self.net).reference_block_count,
                'evidence_stage_count': self.unwrap(self.net).evidence_stage_count,
            })
            with open(ops.join(self.run_root, 'run_args.json'), 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            self.epoch_log = ops.join(self.run_root, 'epoch_log.txt')
            self.csv_path = ops.join(self.run_root, 'metrics_log.csv')
            with open(self.csv_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    'epoch,lr,train_loss,eval_loss,mIoU,IoU,nIoU,AUC,Fa_x1e6,Fa_unm_x1e6,Pd,'
                    'aux,edge_bce,edge_dice,reference,reference_target,reference_background,'
                    'heal,heal_target,heal_false_alarm,heal_hard_pixels,heal_persistent_score,heal_ramp,'
                    'tbs,Pd_single,Pd_multi\n'
                )
            print('[Save to]', self.run_root)
        else:
            self.epoch_log = None
            self.csv_path = None
        barrier()

    def _build_run_root(self):
        project_root = ops.dirname(ops.abspath(__file__))
        if self.is_main:
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            run_name = '%s_BREB-%s_JEPF-%s_HEAL%d_TBS%g%s%s_LR-%s%g_S%d_DDP%d_%s' % (
                'split' + self.args.split if self.args.dataset == 'IRSTD-1k' else 'std',
                self.args.deployment, self.args.fusion_mode, self.args.use_heal,
                self.args.w_tbs, 'FA' if self.args.tbs_fa else '',
                ('_AUG' + self.args.aug if self.args.aug != 'mmlnet' else '')
                + ('_IN%d' % self.args.input_scale if getattr(self.args, 'input_scale', 1) > 1 else ''),
                self.args.lr_schedule, self.args.learning_rate,
                self.args.seed, self.world_size, stamp,
            )
        else:
            run_name = ''
        if self.distributed:
            shared = [run_name]
            dist.broadcast_object_list(shared, src=0)
            run_name = shared[0]
        return ops.join(project_root, 'result', self.args.dataset, run_name)

    @staticmethod
    def weight_init(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def unwrap(net):
        return net.module if isinstance(net, DDP) else net

    def prepare(self, data, labels):
        data = data.to(self.device, non_blocking=True)
        labels = labels[:, :1].to(self.device, non_blocking=True).float()
        if labels.max() > 1.0:
            labels = labels / 255.0
        labels = (labels > 0.5).float()
        if data.size(1) == 1:
            data = data.repeat(1, 3, 1, 1)
        elif data.size(1) > 3:
            data = data[:, :3]
        return data, labels

    def _loss(self, data, labels, epoch, net=None):
        model = self.net if net is None else net
        edge_target = self.gradmask(labels)
        logits, edge_prob, aux_logits, evidence_pyramid = model(data, return_aux=True)
        return self.criterion(
            logits, labels, edge_prob, edge_target,
            aux_logits=aux_logits, evidence_pyramid=evidence_pyramid, epoch=epoch,
        ), logits

    def train_one_epoch(self, epoch):
        self.net.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        warmup = max(0, self.args.warmup_epochs)
        if self.args.lr_schedule == 'cosine' and epoch <= warmup:
            lr = self.args.learning_rate * epoch / warmup
        elif self.args.lr_schedule == 'cosine':
            progress = (epoch - 1 - warmup) / max(1, self.args.epochs - 1 - warmup)
            lr = self.args.min_lr + 0.5 * (self.args.learning_rate - self.args.min_lr) * (
                1.0 + math.cos(math.pi * progress))
        else:
            lr = self.args.learning_rate / (2 ** (epoch // 30))
        for group in self.optimizer.param_groups:
            group['lr'] = lr

        keys = [
            'total', 'final_bce', 'final_dice', 'aux', 'edge_bce', 'edge_dice',
            'reference', 'reference_target', 'reference_background',
            'heal', 'heal_target', 'heal_false_alarm', 'heal_hard_pixels', 'heal_persistent_score', 'heal_ramp',
            'tbs',
        ]
        local_sum = {key: 0.0 for key in keys}
        local_count = 0
        progress = tqdm(self.train_loader, disable=not self.is_main, dynamic_ncols=True)
        for data, labels, _ in progress:
            data, labels = self.prepare(data, labels)
            self.optimizer.zero_grad(set_to_none=True)
            losses, _ = self._loss(data, labels, epoch)
            losses['total'].backward()
            self.optimizer.step()

            local_count += 1
            for key in keys:
                local_sum[key] += float(losses[key].detach().item())
            if self.is_main:
                denom = max(1, local_count)
                progress.set_description(
                    'Epoch:%3d lr:%.6f loss:%.4f seg:%.4f aux:%.4f edge:%.4f ref:%.4f heal:%.4f tbs:%.4f' % (
                        epoch, lr, local_sum['total'] / denom,
                        (local_sum['final_bce'] + local_sum['final_dice']) / denom,
                        local_sum['aux'] / denom,
                        local_sum['edge_bce'] / denom,
                        local_sum['reference'] / denom,
                        local_sum['heal'] / denom,
                        local_sum['tbs'] / denom,
                    )
                )
        means = {key: reduce_sum_count(local_sum[key], local_count, self.device) for key in keys}
        return means['total'], lr, means

    @torch.no_grad()
    def validate(self, epoch):
        if not self.is_main:
            return None
        self.net.eval()
        self.miou.reset()
        metric = SigmoidMetric(
            threshold=self.args.eval_thr, distance_thresh=self.args.distance_thresh
        )
        roc = ROCMetric(bins=100)
        tracked = ['total', 'aux', 'edge_bce', 'edge_dice', 'reference',
                   'reference_target', 'reference_background',
                   'heal', 'heal_target', 'heal_false_alarm', 'heal_hard_pixels', 'heal_persistent_score',
                   'heal_ramp', 'tbs']
        values = {key: [] for key in tracked}
        eval_net = self.unwrap(self.net)
        progress = tqdm(self.val_loader, dynamic_ncols=True)
        for data, labels, _ in progress:
            data, labels = self.prepare(data, labels)
            losses, logits = self._loss(data, labels, epoch, net=eval_net)
            for key in tracked:
                values[key].append(float(losses[key].item()))
            cpu_logits = logits.detach().cpu()
            cpu_labels = labels.detach().cpu()
            self.miou.update(cpu_logits, cpu_labels)
            metric.update(cpu_logits, cpu_labels)
            roc.update(cpu_logits, cpu_labels)
            _, current_miou = self.miou.get()
            progress.set_description(
                'Epoch:%3d eval_loss:%.4f mIoU:%.4f' % (
                    epoch, np.mean(values['total']), current_miou
                )
            )

        _, miou_value = self.miou.get()
        iou, niou, pd, fa_pixel = metric.get()
        _, _, auc = roc.get()
        result = {
            'eval_loss': float(np.mean(values['total'])), 'mIoU': float(miou_value),
            'IoU': float(iou), 'nIoU': float(niou), 'AUC': float(auc),
            'Fa_pixel': float(fa_pixel), 'Fa_x1e6': float(fa_pixel * 1e6), 'Pd': float(pd),
            'Fa_unm_x1e6': float(metric.fa_unmatched() * 1e6),
        }
        by_scene = metric.pd_by_scene()
        result['Pd_single'], result['Pd_multi'] = float(by_scene['single']), float(by_scene['multi'])
        for key in tracked[1:]:
            result[key] = float(np.mean(values[key]))
        return result

    def save_state(self, name):
        if self.is_main:
            torch.save(self.unwrap(self.net).state_dict(), ops.join(self.ckpt_dir, name))

    def keep_pareto(self, epoch, metrics):
        """Keep the weights of every epoch that no other epoch beats on IoU, Pd and unmatched Fa together, so that a
        low-Fa epoch can be scored afterwards with the common evaluator. Epochs far below the best IoU are dropped."""
        point = (metrics['IoU'], metrics['Pd'], -metrics['Fa_unm_x1e6'])
        floor = self.best_miou - self.args.pareto_iou_margin

        def beats(a, b):
            return a != b and all(x >= y for x, y in zip(a, b))

        folder = ops.join(self.ckpt_dir, 'pareto')
        kept = []
        for old_epoch, old in self.pareto:
            if old[0] < floor or beats(point, old):
                path = ops.join(folder, 'ep%03d.pth' % old_epoch)
                if ops.exists(path):
                    os.remove(path)
            else:
                kept.append((old_epoch, old))
        self.pareto = kept
        if point[0] >= floor and not any(beats(old, point) or old == point for _, old in self.pareto):
            os.makedirs(folder, exist_ok=True)
            self.save_state(ops.join('pareto', 'ep%03d.pth' % epoch))
            self.pareto.append((epoch, point))
        with open(ops.join(self.run_root, 'pareto.txt'), 'w', encoding='utf-8') as handle:
            for old_epoch, old in self.pareto:
                handle.write('epoch=%d IoU=%.6f Pd=%.6f Fa_unm_x1e6=%.4f\n' % (old_epoch, old[0], old[1], -old[2]))

    def run(self):
        for epoch in range(1, self.args.epochs + 1):
            train_loss, lr, _ = self.train_one_epoch(epoch)
            barrier()
            if self.is_main and (epoch % max(1, self.args.eval_every) == 0 or epoch == self.args.epochs):
                metrics = self.validate(epoch)
                line = (
                    'Epoch:%03d train=%.6f eval=%.6f mIoU=%.6f IoU=%.6f nIoU=%.6f '
                    'AUC=%.6f Fa(x1e-6)=%.4f Fa_unm(x1e-6)=%.4f Pd=%.6f aux=%.5f edge=%.5f '
                    'ref=%.5f heal=%.5f tbs=%.5f Pd_single=%.4f Pd_multi=%.4f'
                ) % (
                    epoch, train_loss, metrics['eval_loss'], metrics['mIoU'], metrics['IoU'],
                    metrics['nIoU'], metrics['AUC'], metrics['Fa_x1e6'], metrics['Fa_unm_x1e6'],
                    metrics['Pd'], metrics['aux'], metrics['edge_bce'], metrics['reference'],
                    metrics['heal'], metrics['tbs'], metrics['Pd_single'], metrics['Pd_multi'],
                )
                print(line)
                with open(self.epoch_log, 'a', encoding='utf-8') as handle:
                    handle.write(line + '\n')
                columns = [
                    epoch, lr, train_loss, metrics['eval_loss'], metrics['mIoU'], metrics['IoU'],
                    metrics['nIoU'], metrics['AUC'], metrics['Fa_x1e6'], metrics['Fa_unm_x1e6'],
                    metrics['Pd'], metrics['aux'], metrics['edge_bce'], metrics['edge_dice'],
                    metrics['reference'], metrics['reference_target'], metrics['reference_background'],
                    metrics['heal'], metrics['heal_target'], metrics['heal_false_alarm'],
                    metrics['heal_hard_pixels'], metrics['heal_persistent_score'], metrics['heal_ramp'],
                    metrics['tbs'], metrics['Pd_single'], metrics['Pd_multi'],
                ]
                with open(self.csv_path, 'a', encoding='utf-8') as handle:
                    handle.write(','.join(str(value) for value in columns) + '\n')
                self.save_state('latest.pth')
                if metrics['mIoU'] > self.best_miou:
                    self.best_miou = metrics['mIoU']
                    self.save_state('best_miou.pth')
                    with open(ops.join(self.run_root, 'best.txt'), 'w', encoding='utf-8') as handle:
                        handle.write('best_epoch=%d\nbest_mIoU=%.6f\n' % (epoch, self.best_miou))
                    with open(ops.join(self.run_root, 'best_metrics.txt'), 'w', encoding='utf-8') as handle:
                        handle.write(
                            'IoU=%.6f\nnIoU=%.6f\nAUC=%.6f\nFa_pixel=%.12f\nFa_x1e6=%.6f\nPd=%.6f\n'
                            % (metrics['IoU'], metrics['nIoU'], metrics['AUC'],
                               metrics['Fa_pixel'], metrics['Fa_x1e6'], metrics['Pd'])
                        )
                self.keep_pareto(epoch, metrics)
            barrier()
        if self.is_main:
            print('Best mIoU: %.6f' % self.best_miou)


def run_worker(args, local_rank, world_size, distributed):
    if distributed:
        dist.init_process_group(
            backend='nccl', init_method='env://', rank=local_rank, world_size=world_size
        )
    try:
        Trainer(args, local_rank=local_rank, world_size=world_size, distributed=distributed).run()
    finally:
        if is_dist_ready():
            dist.destroy_process_group()


def spawned_worker(local_rank, world_size, args, master_port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    run_worker(args, local_rank=local_rank, world_size=world_size, distributed=True)


def main():
    args = parse_args()
    launched_by_torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if launched_by_torchrun:
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank if args.local_rank >= 0 else 0))
        world_size = int(os.environ['WORLD_SIZE'])
        run_worker(args, local_rank=local_rank, world_size=world_size, distributed=world_size > 1)
        return

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if bool(args.ddp) and gpu_count > 1:
        world_size = min(int(args.ddp_gpus), int(gpu_count))
        port = int(args.master_port) if args.master_port > 0 else find_free_port()
        print('[DDP launcher] spawning %d local CUDA processes on port %d' % (world_size, port))
        mp.spawn(spawned_worker, nprocs=world_size, args=(world_size, args, port), join=True)
    else:
        if bool(args.ddp) and gpu_count < 2:
            print('[DDP launcher] fewer than two CUDA devices; using one process.')
        run_worker(args, local_rank=0, world_size=1, distributed=False)


if __name__ == '__main__':
    main()
