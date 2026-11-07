"""Smoke test for BRENet (BREB, JEPF, HEAL, TBS). Run from the code folder (about a minute on CPU):
    python smoke_test.py

Every check prints PASS or raises.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loss import BackgroundReferenceLoss, TargetBalancedSupervision  # noqa: E402
from model_base import ISNet  # noqa: E402


def check(name, condition, detail=''):
    if not condition:
        raise AssertionError('FAIL %s %s' % (name, detail))
    print('PASS', name, detail)


def test_tbs():
    tbs = TargetBalancedSupervision(context=4)
    t = torch.zeros(1, 1, 64, 64)
    t[..., 10:12, 10:12] = 1            # small target (4 px)
    t[..., 40:48, 40:48] = 1            # large target (64 px)
    hit = torch.where(t > 0, torch.tensor(12.0), torch.tensor(-12.0))
    check('TBS: ~0 when every target is segmented', float(tbs(hit, t)) < 1e-3, '%.5f' % float(tbs(hit, t)))
    miss_small, miss_large = hit.clone(), hit.clone()
    miss_small[..., 10:12, 10:12] = -12.0
    miss_large[..., 40:48, 40:48] = -12.0
    a, b = float(tbs(miss_small, t)), float(tbs(miss_large, t))
    check('TBS: missing the small or the large target costs the same', abs(a - b) < 1e-3 and a > 0.45, '%.4f / %.4f' % (a, b))
    dice = lambda z: float(1 - (2 * (torch.sigmoid(z) * t).sum() + 1) / (torch.sigmoid(z).sum() + t.sum() + 1))
    check('TBS vs Dice: Dice barely sees the missed small target', dice(miss_small) < 0.1 < a,
          'Dice %.4f, TBS %.4f' % (dice(miss_small), a))
    fa_tbs = TargetBalancedSupervision(context=4, false_alarm=True)
    check('TBS-FA: identical to TBS when there is no false component',
          abs(float(fa_tbs(hit, t)) - float(tbs(hit, t))) < 1e-6)
    blob = hit.clone()
    blob[..., 55:60, 2:7] = 12.0                      # one false component far from both targets
    blob.requires_grad_(True)
    v = fa_tbs(blob, t)
    v.backward()
    check('TBS-FA: a false component adds one unit and receives gradient',
          float(v) > float(tbs(hit, t)) + 0.5 and float(blob.grad[..., 55:60, 2:7].abs().sum()) > 0,
          '%.4f' % float(v))
    spill = hit.clone()
    spill[..., 38:50, 38:50] = 12.0
    check('TBS: spill into the target\'s own ring is penalised', float(tbs(spill, t)) > 0.1, '%.4f' % float(tbs(spill, t)))


def gradient_mask(x):
    """Same structural target as train.GradientMask (copied to avoid importing train.py)."""
    vertical = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    horizontal = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gv = F.conv2d(x[:, :1], vertical, padding=1)
    gh = F.conv2d(x[:, :1], horizontal, padding=1)
    magnitude = torch.sqrt(gv.square() + gh.square() + 1e-6)
    magnitude = magnitude - magnitude.amin(dim=(1, 2, 3), keepdim=True)
    magnitude = magnitude / (magnitude.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return magnitude.clamp(0.0, 1.0)


def test_model_and_loss(device):
    import model_base
    torch.manual_seed(0)
    base = ISNet(channels=(16, 32, 64, 128))
    n = sum(p.numel() for p in base.parameters())
    if model_base._HAS_WAVELET:
        check('original BRENet unchanged (3,410,728 parameters)', n == 3410728, str(n))
    else:
        print('SKIP parameter-count check: pytorch_wavelets not installed (AvgPool fallback active)')
    image = torch.rand(2, 3, 128, 160, device=device)
    target = torch.zeros(2, 1, 128, 160, device=device)
    target[0, :, 40:44, 50:54] = 1.0
    target[1, :, 90:93, 20:23] = 1.0
    target[1, :, 10:15, 140:146] = 1.0
    configs = [
        ('original BRENet (BREB+JEPF+HEAL)', dict(), dict(use_heal=True)),
        ('BRENet + TBS', dict(), dict(use_heal=True, w_tbs=1.0)),
        ('BRENet + TBS-FA', dict(), dict(use_heal=True, w_tbs=1.0, tbs_false_alarm=True)),
        ('MMLNet fusion control', dict(fusion_mode='legacy'), dict(use_heal=False)),
        ('no BREB (residual control)', dict(deployment='none'), dict(use_heal=True)),
    ]
    for label, net_kw, loss_kw in configs:
        net = ISNet(channels=(16, 32, 64, 128), **net_kw).to(device)
        criterion = BackgroundReferenceLoss(**loss_kw).to(device)
        logits, edge_prob, aux, pyramid = net(image, return_aux=True)
        check('output shape: ' + label, logits.shape == target.shape, str(tuple(logits.shape)))
        shift = torch.zeros_like(logits)
        shift[0, :, 40:44, 50:54] = 20.0
        shift[0, :, 100:104, 100:104] = 20.0
        losses = criterion(logits + shift, target, edge_prob, gradient_mask(target), aux_logits=aux,
                           evidence_pyramid=pyramid, epoch=100)
        losses['total'].backward()
        finite = all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
        missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
        check('forward/backward finite: ' + label, finite)
        check('no unused parameters (DDP-safe): ' + label, not missing, str(missing[:5]))
        if loss_kw.get('use_heal'):
            check('HEAL active: ' + label, float(losses['heal'].detach()) > 0, '%.4f' % float(losses['heal'].detach()))

def test_old_checkpoint(path, device):
    net = ISNet(channels=(16, 32, 64, 128)).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    state = state.get('state_dict', state) if isinstance(state, dict) else state
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    check('original BRENet checkpoint loads strictly', True)


def test_dataset(root, name):
    from data_utils import DatasetLoader

    class Args:
        dataset, data_root = name, root
    for mode in ('train', 'test'):
        ds = DatasetLoader(Args, mode=mode)
        img, mask, item = ds[0]
        check('DatasetLoader %s/%s' % (name, mode), img.shape[0] == 3 and mask.shape[0] == 1,
              '%d samples, image %s, mask %s, %s' % (len(ds), tuple(img.shape), tuple(mask.shape), item))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='', help='optional: dataset folder to test DatasetLoader')
    parser.add_argument('--old-checkpoint', default='', help='optional: checkpoint of the original BRENet')
    parser.add_argument('--dataset', default='IRSTD-1k')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_tbs()
    test_model_and_loss(device)
    if args.old_checkpoint:
        test_old_checkpoint(args.old_checkpoint, device)
    if args.data_root:
        test_dataset(args.data_root, args.dataset)
    print('\nAll smoke tests passed on', device)
