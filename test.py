# -*- coding: utf-8 -*-
"""Evaluate a BRENet run (latest.pth by default) on the test split.

    python test.py --run-dir result/IRSTD-1k/<run>

The model is rebuilt from <run>/run_args.json.  Fa_x1e6 counts all false-positive pixels;
Fa_unm_x1e6 counts pixels of unmatched predicted components (the common evaluator's Fa).
"""

import json
import os.path as ops
from argparse import ArgumentParser

import torch
import torch.utils.data as Data
from tqdm import tqdm

from data_utils import DatasetLoader
from metric import ROCMetric, SigmoidMetric
from model_base import ISNet


def auto_locate_dataset(args):
    root = ops.dirname(ops.abspath(__file__))
    candidates = [
        args.data_root,
        ops.join(root, 'datasets', args.dataset),
        ops.join(root, 'datasets', args.dataset.replace('_', '-')),
        ops.join(root, 'datasets', args.dataset.replace('-', '_')),
        ops.join(root, '..', 'datasets', args.dataset),
    ]
    for path in candidates:
        if path and ops.isdir(path):
            args.data_root = ops.abspath(path)
            break


def parse_args():
    parser = ArgumentParser(description='BRENet evaluation')
    parser.add_argument('--dataset', type=str, default='IRSTD-1k')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--run-dir', type=str, required=True, help='training-result directory')
    # Final-epoch weights by default: no checkpoint selection on the test split.
    parser.add_argument('--checkpoint', type=str, default='latest.pth')
    parser.add_argument('--eval-thr', type=float, default=0.5)
    parser.add_argument('--distance-thresh', type=float, default=3.0)
    parser.add_argument('--split', type=str, default='', choices=['', '9-1', 'official'],
                        help='IRSTD-1k split; read from run_args.json, required for runs that do not record it')
    return parser.parse_args()


def build_model(config, device):
    """Rebuild the network exactly as it was trained."""
    return ISNet(
        channels=(16, 32, 64, 128),
        deployment=config.get('deployment', 'full'),
        stage_repeat=int(config.get('stage_repeat', 1)),
        evidence_mode=config.get('evidence_mode', 'positive'),
        fusion_mode=config.get('fusion_mode', 'legacy'),
        fusion_scope=config.get('fusion_scope', 'full'),
        breb_windows=config.get('breb_windows', 'default'),
        breb_tau_init=float(config.get('breb_tau_init', 1.0)),
        breb_alpha_init=float(config.get('breb_alpha_init', 0.1)),
    ).to(device)


def load_run(run_dir, checkpoint, device):
    config_path = ops.join(run_dir, 'run_args.json')
    config = {}
    if ops.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as handle:
            config = json.load(handle)
    net = build_model(config, device)
    state = torch.load(ops.join(run_dir, 'checkpoints', checkpoint), map_location=device, weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {key.replace('module.', '', 1): value for key, value in state.items()}
    net.load_state_dict(state, strict=True)
    return net.eval(), config


def prepare(data, labels, device):
    data = data.to(device, non_blocking=True)
    labels = labels[:, :1].float()
    if labels.max() > 1.0:
        labels = labels / 255.0
    labels = (labels > 0.5).float()
    if data.size(1) == 1:
        data = data.repeat(1, 3, 1, 1)
    elif data.size(1) > 3:
        data = data[:, :3]
    return data, labels


def evaluate(net, loader, device, args):
    metric = SigmoidMetric(threshold=args.eval_thr, distance_thresh=args.distance_thresh)
    roc = ROCMetric(bins=100)
    with torch.no_grad():
        for data, labels, _ in tqdm(loader, dynamic_ncols=True, leave=False):
            data, labels = prepare(data, labels, device)
            logits = net(data)[0]
            metric.update(logits.cpu(), labels.cpu())
            roc.update(logits.cpu(), labels.cpu())
    iou, niou, pd, fa_pixel = metric.get()
    _, _, auc = roc.get()
    by_scene = metric.pd_by_scene()
    return {'IoU': iou, 'nIoU': niou, 'Pd': pd, 'Fa_x1e6': fa_pixel * 1e6,
            'Fa_unm_x1e6': metric.fa_unmatched() * 1e6, 'AUC': auc,
            'Pd_single': by_scene['single'], 'Pd_multi': by_scene['multi']}


def main():
    args = parse_args()
    auto_locate_dataset(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, config = load_run(args.run_dir, args.checkpoint, device)
    recorded = config.get('split', '')
    if args.dataset == 'IRSTD-1k':
        if recorded and args.split and args.split != recorded:
            raise ValueError('run was trained on split %s; evaluating on %s would mix train and test images'
                             % (recorded, args.split))
        args.split = args.split or recorded
        if not args.split:
            raise ValueError('this run does not record its IRSTD-1k split; pass --split 9-1 or --split official '
                             '(the split it was trained with)')
    parameters = sum(parameter.numel() for parameter in net.parameters())
    print('[Architecture] BREB=%s (%d blocks) | JEPF=%s | HEAL=%s | TBS=%g (training only) | params=%.3fM' % (
        config.get('deployment', 'full'), net.reference_block_count, net.fusion_mode,
        'on' if config.get('use_heal', 0) else 'off', float(config.get('w_tbs', 0.0)), parameters / 1e6))
    if int(config.get('tbs_fa', 0)):
        print('[Architecture] TBS with object-level false-alarm term (training only)')

    dataset = DatasetLoader(args, mode='test')
    loader = Data.DataLoader(dataset, batch_size=1, shuffle=False)
    row = evaluate(net, loader, device, args)
    lines = ['dataset=%s split=%s run=%s checkpoint=%s' % (args.dataset, args.split, args.run_dir, args.checkpoint),
             '%8s %8s %8s %10s %12s %8s %10s %9s' % ('IoU', 'nIoU', 'Pd', 'Fa_x1e6', 'Fa_unm_x1e6', 'AUC',
                                                      'Pd_single', 'Pd_multi'),
             '%8.4f %8.4f %8.4f %10.3f %12.3f %8.4f %10.4f %9.4f' % (
                 row['IoU'], row['nIoU'], row['Pd'], row['Fa_x1e6'], row['Fa_unm_x1e6'], row['AUC'],
                 row['Pd_single'], row['Pd_multi'])]
    result = '\n'.join(lines) + '\n'
    print(result)
    output_path = ops.join(args.run_dir, 'test_results_%s.txt' % args.dataset)
    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write(result)
    with open(ops.join(args.run_dir, 'test_results_%s.json' % args.dataset), 'w', encoding='utf-8') as handle:
        json.dump(row, handle, indent=2)
    print('[Saved]', output_path)


if __name__ == '__main__':
    main()
