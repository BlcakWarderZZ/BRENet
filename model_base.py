# -*- coding: utf-8 -*-
"""MMLNet-derived infrared small-target detector rebuilt with BREB.

BREB (Background-Referenced Evidence Block) replaces the ordinary residual
processing rule throughout the detection encoder, structural encoder, decoder,
prior encoder, and final fusion.  Every spatial response is judged against an
annular local-background distribution before it is allowed to update the
identity path:

    projection -> annular reference -> dispersion standardization
    -> exceedance selection -> complementary reconstruction -> residual update

The original topology is retained so ``deployment='none'`` supplies a clean
ResidualBlock control.  The proposed ``full`` model inserts one BREB at each
of ten task-critical stage interfaces; ``dense`` keeps the all-block
replacement as an explicit over-deployment control.  JEPF then replaces
decoder additions with a repeated deep/middle/shallow joint-evidence
partition based on cross-source agreement, while ``fusion_mode='legacy'``
preserves the inherited fusion.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward
    _HAS_WAVELET = True
except Exception:
    DWTForward = None
    _HAS_WAVELET = False


def _norm(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _inverse_softplus(value):
    return math.log(math.exp(float(value)) - 1.0)


def _replicate_mean(x, kernel_size):
    if kernel_size == 1:
        return x
    pad = kernel_size // 2
    return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode='replicate'), kernel_size, stride=1)


def annular_mean(x, outer_size, guard_size):
    """Mean over a square reference ring while excluding its guard region."""
    if outer_size <= guard_size or outer_size % 2 == 0 or guard_size % 2 == 0:
        raise ValueError('outer_size and guard_size must be odd with outer_size > guard_size')
    outer_area = float(outer_size * outer_size)
    guard_area = float(guard_size * guard_size)
    outer_sum = _replicate_mean(x, outer_size) * outer_area
    guard_sum = _replicate_mean(x, guard_size) * guard_area
    return (outer_sum - guard_sum) / (outer_area - guard_area)


class ResidualBlock(nn.Module):
    """Control adapter with the same interface as BackgroundReferencedBlock."""
    def __init__(self, in_channels, out_channels, stride=1, outer_size=5, guard_size=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            _norm(out_channels),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), _norm(out_channels))
            if in_channels != out_channels or stride != 1 else nn.Identity()
        )

    def forward(self, x):
        return self.skip(x) + self.body(x)


class BackgroundReferencedBlock(nn.Module):
    """Background-Referenced Evidence Block (BREB).

    The projected feature is decomposed into complementary background and
    innovation routes.  The innovation route is opened only when a positive
    deviation exceeds the dispersion of an annular local reference.  The two routes sum
    exactly to the projected feature before separate channel mixing, preventing
    irreversible information deletion while allowing repeated target/background
    discrimination at every hierarchy level.
    """
    def __init__(self, in_channels, out_channels, stride=1, outer_size=5, guard_size=1,
                 evidence_mode='positive'):
        super().__init__()
        self.outer_size = int(outer_size)
        self.guard_size = int(guard_size)
        if evidence_mode not in {'positive', 'symmetric'}:
            raise ValueError("evidence_mode must be 'positive' or 'symmetric'")
        self.evidence_mode = evidence_mode
        if self.outer_size <= self.guard_size or self.outer_size % 2 == 0 or self.guard_size % 2 == 0:
            raise ValueError('BREB ring sizes must be odd and outer_size > guard_size')

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), _norm(out_channels))
            if in_channels != out_channels or stride != 1 else nn.Identity()
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            _norm(out_channels),
        )
        self.threshold_raw = nn.Parameter(
            torch.full((1, out_channels, 1, 1), _inverse_softplus(1.0))
        )
        self.sharpness_raw = nn.Parameter(torch.tensor(_inverse_softplus(2.0)))
        # The network starts close to its identity adapter and learns safe updates.
        self.residual_scale_logit = nn.Parameter(torch.tensor(-2.1972246))

    def forward(self, x):
        identity = self.skip(x)
        feature = self.project(x)

        reference = annular_mean(feature, self.outer_size, self.guard_size)
        deviation = feature - reference
        dispersion = annular_mean(deviation.abs(), self.outer_size, self.guard_size)
        # Detaching only the denominator prevents the trivial solution of inflating
        # background dispersion while preserving gradients through the reference.
        standardized = deviation / (dispersion.detach() + 1e-4)
        standardized = 6.0 * torch.tanh(standardized / 6.0)

        threshold = F.softplus(self.threshold_raw) + 1e-4
        sharpness = F.softplus(self.sharpness_raw) + 1e-4
        # ``positive`` is the proposed infrared-target primitive. ``symmetric``
        # preserves every tensor operation and parameter while removing the
        # bright-target directional assumption, yielding a matched control.
        evidence_score = standardized if self.evidence_mode == 'positive' else standardized.abs()
        evidence_gate = torch.sigmoid(sharpness * (evidence_score - threshold))

        innovation = evidence_gate * deviation
        background = feature - innovation
        update = self.reconstruct(torch.cat([background, innovation], dim=1))
        scale = torch.sigmoid(self.residual_scale_logit)
        return identity + scale * update


BREB = BackgroundReferencedBlock


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter = max(4, in_channels // 4)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter, 3, padding=1, bias=False),
            _norm(inter),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(inter, out_channels, 1),
        )

    def forward(self, x):
        return self.block(x)


class TensorDilate(nn.Module):
    def __init__(self, ksize=3):
        super().__init__()
        self.ksize = int(ksize)

    def forward(self, x):
        return F.max_pool2d(x, self.ksize, stride=1, padding=self.ksize // 2)


class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super().__init__()
        hidden = max(1, out_planes // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_planes, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(self.pool(x))


class FSP(nn.Module):
    """Inherited selective cross-path projection."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.filter = FilterLayer(channels * 2, channels, reduction)

    def forward(self, guide_path, main_path):
        weight = self.filter(torch.cat([guide_path, main_path], dim=1))
        return main_path + weight * guide_path


class CrossSourceEvidenceEstimator(nn.Module):
    """Estimate evidence from agreement across aligned hierarchy sources."""
    def __init__(self, channels):
        super().__init__()
        self.response = nn.Conv2d(channels, 1, 1)
        self.threshold_raw = nn.Parameter(torch.tensor(_inverse_softplus(0.5)))
        self.sharpness_raw = nn.Parameter(torch.tensor(_inverse_softplus(4.0)))
        self.disagreement_raw = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))

    def forward(self, sources):
        source_probabilities = torch.stack([
            torch.sigmoid(self.response(source)) for source in sources
        ], dim=1)
        mean_support = source_probabilities.mean(dim=1)
        disagreement = (source_probabilities - mean_support.unsqueeze(1)).abs().mean(dim=1)
        penalty = F.softplus(self.disagreement_raw)
        consistent_support = mean_support - penalty * disagreement
        threshold = F.softplus(self.threshold_raw) + 1e-4
        sharpness = F.softplus(self.sharpness_raw) + 1e-4
        return torch.sigmoid(sharpness * (consistent_support - threshold))


class JointEvidencePartitionFusion(nn.Module):
    """Cross-source Joint-Evidence Partition Fusion (JEPF).

    Target-path and structure-path evidence probabilities define four joint
    states: confirmed target, target-only, structure-only, and common
    background.  Only the confirmed state has a direct enhancement route;
    branch-exclusive responses must pass through the joint reconstructor.
    Repeating this operation at every decoder scale prevents an isolated skip
    response from being amplified by unconditional addition.
    """
    def __init__(self, main_in_channels, edge_in_channels, out_channels,
                 partition_mode='evidence'):
        super().__init__()
        if partition_mode not in {'evidence', 'uniform'}:
            raise ValueError("partition_mode must be 'evidence' or 'uniform'")
        self.partition_mode = partition_mode
        self.main_in_channels = tuple(main_in_channels)
        self.edge_in_channels = tuple(edge_in_channels)
        self.out_channels = int(out_channels)

        def projection(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                _norm(self.out_channels),
                nn.SiLU(inplace=True),
            )

        self.main_projections = nn.ModuleList(projection(c) for c in self.main_in_channels)
        self.edge_projections = nn.ModuleList(projection(c) for c in self.edge_in_channels)
        self.main_aggregate = nn.Sequential(
            nn.Conv2d(self.out_channels * len(self.main_in_channels), self.out_channels, 1, bias=False),
            _norm(self.out_channels), nn.SiLU(inplace=True),
        )
        self.edge_aggregate = nn.Sequential(
            nn.Conv2d(self.out_channels * len(self.edge_in_channels), self.out_channels, 1, bias=False),
            _norm(self.out_channels), nn.SiLU(inplace=True),
        )

        self.main_evidence = CrossSourceEvidenceEstimator(self.out_channels)
        self.edge_evidence = CrossSourceEvidenceEstimator(self.out_channels)

        hidden = max(8, self.out_channels // 4)
        self.state_encoders = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.out_channels, hidden, 1, bias=False), nn.SiLU(inplace=True))
            for _ in range(4)
        ])
        self.reconstruct = nn.Sequential(
            nn.Conv2d(hidden * 4, self.out_channels, 1, bias=False),
            _norm(self.out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1,
                      groups=self.out_channels, bias=False),
            _norm(self.out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 1, bias=False),
            _norm(self.out_channels),
        )
        self.main_update = nn.Conv2d(self.out_channels, self.out_channels, 1, bias=False)
        self.edge_update = nn.Conv2d(self.out_channels, self.out_channels, 1, bias=False)
        # Both routes start conservatively so the inherited decoder remains stable.
        self.confirmed_gain_logit = nn.Parameter(torch.tensor(-1.3862944))
        self.update_scale_logit = nn.Parameter(torch.tensor(-2.1972246))

    @staticmethod
    def _resize(feature, size):
        if feature.shape[-2:] == size:
            return feature
        return F.interpolate(feature, size=size, mode='bilinear', align_corners=False)

    def _aggregate(self, sources, projections, expected_channels, aggregate, size):
        if len(sources) != len(projections):
            raise ValueError('unexpected number of JEPF fusion sources')
        aligned = []
        for source, projection, channels in zip(sources, projections, expected_channels):
            if source.size(1) != channels:
                raise ValueError('JEPF source channel mismatch: expected %d, got %d' % (
                    channels, source.size(1)
                ))
            aligned.append(projection(self._resize(source, size)))
        return aggregate(torch.cat(aligned, dim=1)), aligned

    def forward(self, main_sources, edge_sources):
        size = main_sources[0].shape[-2:]
        main, main_aligned = self._aggregate(
            main_sources, self.main_projections, self.main_in_channels,
            self.main_aggregate, size,
        )
        edge, edge_aligned = self._aggregate(
            edge_sources, self.edge_projections, self.edge_in_channels,
            self.edge_aggregate, size,
        )

        p_main = self.main_evidence(main_aligned)
        p_edge = self.edge_evidence(edge_aligned)
        if self.partition_mode == 'uniform':
            # Executes the same estimator and reconstruction graph while
            # removing input-adaptive partitioning: a computation-matched control.
            p_main = p_main * 0.0 + 0.5
            p_edge = p_edge * 0.0 + 0.5
        p_confirmed = p_main * p_edge
        p_main_only = p_main * (1.0 - p_edge)
        p_edge_only = (1.0 - p_main) * p_edge
        p_background = (1.0 - p_main) * (1.0 - p_edge)

        common = 0.5 * (main + edge)
        states = (
            p_confirmed * common,
            p_main_only * main,
            p_edge_only * edge,
            p_background * common,
        )
        state_code = torch.cat([
            encoder(state) for encoder, state in zip(self.state_encoders, states)
        ], dim=1)
        joint_update = self.reconstruct(state_code)

        confirmed_gain = torch.sigmoid(self.confirmed_gain_logit)
        update_scale = torch.sigmoid(self.update_scale_logit)
        confirmed_update = confirmed_gain * p_confirmed * common
        main_out = main + update_scale * self.main_update(joint_update) + confirmed_update
        edge_out = edge + update_scale * self.edge_update(joint_update) + confirmed_update
        evidence = {
            'main': p_main,
            'edge': p_edge,
            'confirmed': p_confirmed,
            'main_only': p_main_only,
            'edge_only': p_edge_only,
            'background': p_background,
        }
        return main_out, edge_out, evidence


class Unit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.body(x)


class Down_wt(nn.Module):
    """Inherited Haar downsampling with an AvgPool fallback for portability."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.use_wavelet = _HAS_WAVELET
        if self.use_wavelet:
            self.wt = DWTForward(J=1, mode='zero', wave='haar')
            proj_in = in_ch * 4
        else:
            self.wt = None
            self.pool = nn.AvgPool2d(2, 2)
            proj_in = in_ch
        self.proj = nn.Sequential(nn.Conv2d(proj_in, out_ch, 1, bias=False), _norm(out_ch), nn.SiLU(inplace=True))

    def forward(self, x):
        if self.use_wavelet:
            low, high = self.wt(x)
            high = high[0]
            x = torch.cat([low, high[:, :, 0], high[:, :, 1], high[:, :, 2]], dim=1)
        else:
            x = self.pool(x)
        return self.proj(x)


class SobelConv(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel', torch.cat([kx, ky], dim=0))

    def forward(self, x):
        kernel = self.kernel.to(device=x.device, dtype=x.dtype).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=x.size(1))


class CCFE(nn.Module):
    """Inherited candidate-local feature extraction."""
    def __init__(self, k=3, down=2):
        super().__init__()
        self.k = int(k)
        self.down = int(down)

    def forward(self, prob, feat):
        b, _, h, w = prob.shape
        r = self.down
        patch_mean = F.unfold(prob, kernel_size=r, stride=r).mean(dim=1)
        _, idx = torch.topk(patch_mean, k=min(self.k, patch_mean.size(1)), dim=1)
        grid_w = w // r
        iy, ix = (idx // grid_w) * r, (idx % grid_w) * r
        regions = []
        for j in range(idx.size(1)):
            samples = []
            for bi in range(b):
                y0 = int(iy[bi, j].clamp(max=h - r).item())
                x0 = int(ix[bi, j].clamp(max=w - r).item())
                patch = feat[bi:bi + 1, :, y0:y0 + r, x0:x0 + r]
                samples.append(F.interpolate(patch, size=(h // 2, w // 2), mode='bilinear', align_corners=False))
            regions.append(torch.cat(samples, dim=0))
        while len(regions) < self.k:
            regions.append(regions[-1])
        return torch.cat(regions[:self.k], dim=1)


def white_tophat(x, kernel_size):
    """Differentiable grayscale white top-hat: image minus opening."""
    pad = kernel_size // 2
    eroded = -F.max_pool2d(
        -F.pad(x, (pad, pad, pad, pad), mode='replicate'), kernel_size, stride=1
    )
    opened = F.max_pool2d(
        F.pad(eroded, (pad, pad, pad, pad), mode='replicate'), kernel_size, stride=1
    )
    return x - opened


class ISNet(nn.Module):
    """MMLNet topology rebuilt with BREB processing and JEPF reconstruction."""
    VALID_DEPLOYMENTS = {
        'none', 'shallow', 'middle', 'deep', 'encoder', 'decoder', 'full', 'dense'
    }
    VALID_FUSIONS = {'legacy', 'partition', 'uniform'}
    VALID_FUSION_SCOPES = {
        'deep', 'middle', 'shallow', 'deep_middle', 'middle_shallow', 'full'
    }
    def __init__(self, layer_blocks=None, channels=(16, 32, 64, 128),
                 deployment='full', stage_repeat=1, evidence_mode='positive',
                 fusion_mode='partition', fusion_scope='full'):
        super().__init__()
        if deployment not in self.VALID_DEPLOYMENTS:
            raise ValueError('deployment must be one of %s' % sorted(self.VALID_DEPLOYMENTS))
        self.deployment = deployment
        self.stage_repeat = max(1, int(stage_repeat))
        if evidence_mode not in {'positive', 'symmetric'}:
            raise ValueError("evidence_mode must be 'positive' or 'symmetric'")
        if fusion_mode not in self.VALID_FUSIONS:
            raise ValueError('fusion_mode must be one of %s' % sorted(self.VALID_FUSIONS))
        if fusion_scope not in self.VALID_FUSION_SCOPES:
            raise ValueError('fusion_scope must be one of %s' % sorted(self.VALID_FUSION_SCOPES))
        self.evidence_mode = evidence_mode
        self.fusion_mode = fusion_mode
        self.fusion_scope = fusion_scope
        self.reference_block_count = 0
        self.evidence_stage_count = 0 if fusion_mode == 'legacy' else 3
        c1, c2, c3, c4 = channels

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.down_half = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.dilate = TensorDilate(3)
        self.down1, self.down2 = Down_wt(c1, c1), Down_wt(c2, c2)
        self.down3, self.down4 = Down_wt(c3, c3), Down_wt(c2, c3)

        # Detection encoder.
        self.stage1 = self._make_layer(2, 7, c1, level='shallow', scope='encoder', core=True)
        self.stage1_1 = self._make_layer(1, c1, c1, level='shallow', scope='encoder')
        self.stage1_2 = self._make_layer(1, c1 * 2, c1 * 2, level='shallow', scope='encoder')
        self.stage2_1 = self._make_layer(1, c1, c1, level='shallow', scope='encoder')
        self.stage2_2 = self._make_layer(
            1, c1 * 2, c1 * 2, level='middle', scope='encoder', core=True
        )
        self.stage2_3 = self._make_layer(1, c2, c2, level='middle', scope='encoder')
        self.stage3_1 = self._make_layer(1, c2, c2, level='middle', scope='encoder')
        self.stage3_2 = self._make_layer(
            1, c2 * 2, c2 * 2, level='middle', scope='encoder', core=True
        )
        self.stage2_4 = self._make_layer(1, c2 * 2, c2 * 2, level='middle', scope='encoder')
        self.stage3_3 = self._make_layer(1, c3, c3, level='deep', scope='encoder')
        self.stage4_1 = self._make_layer(1, c3, c3, level='deep', scope='encoder')
        self.stage4_2 = self._make_layer(
            1, c3 * 2, c3 * 2, level='deep', scope='encoder', core=True
        )
        self.stage3_4 = self._make_layer(1, c3 * 2, c3 * 2, level='deep', scope='encoder')

        # Structural encoder uses the identical task-derived processing rule.
        self.edge_stage1 = self._make_layer(
            2, 3, c1, level='shallow', scope='encoder', core=True
        )
        self.edge_stage1_1 = self._make_layer(1, c1, c1, level='shallow', scope='encoder')
        self.edge_stage2_1 = self._make_layer(1, c1, c1, level='shallow', scope='encoder')
        self.edge_stage2_2 = self._make_layer(
            1, c1 * 2, c1 * 2, level='middle', scope='encoder', core=True
        )
        self.edge_stage2_3 = self._make_layer(1, c2, c2, level='middle', scope='encoder')
        self.edge_stage3_1 = self._make_layer(1, c2, c2, level='middle', scope='encoder')
        self.edge_stage3_2 = self._make_layer(
            1, c2 * 2, c2 * 2, level='middle', scope='encoder', core=True
        )
        self.edge_stage2_4 = self._make_layer(1, c2 * 2, c2 * 2, level='middle', scope='encoder')
        self.edge_stage3_3 = self._make_layer(1, c3, c3, level='deep', scope='encoder')
        self.edge_stage4_1 = self._make_layer(1, c3, c3, level='deep', scope='encoder')
        self.edge_stage4_2 = self._make_layer(
            1, c3 * 2, c3 * 2, level='deep', scope='encoder', core=True
        )
        self.edge_stage3_4 = self._make_layer(1, c3 * 2, c3 * 2, level='deep', scope='encoder')

        if self.fusion_mode == 'legacy':
            self.fsp_main4, self.fsp_edge4 = FSP(c4, 16), FSP(c4, 16)
            self.fsp_main3, self.fsp_edge3 = FSP(c3, 8), FSP(c3, 8)
            self.reduce2_edge = nn.Conv2d(c4, c3, 1, bias=False)
            self.reduce1_edge = nn.Conv2d(c3, c2, 1, bias=False)
            self.reduce2 = nn.Conv2d(c4, c3, 1, bias=False)
            self.reduce1 = nn.Conv2d(c3, c2, 1, bias=False)
        else:
            # The same cross-source partition operation replaces every cross-path
            # FSP and unconditional decoder addition at the selected scales.
            scope_levels = {
                'deep': {'deep'},
                'middle': {'middle'},
                'shallow': {'shallow'},
                'deep_middle': {'deep', 'middle'},
                'middle_shallow': {'middle', 'shallow'},
                'full': {'deep', 'middle', 'shallow'},
            }[self.fusion_scope]
            def partition_mode(level):
                if self.fusion_mode == 'uniform' or level not in scope_levels:
                    return 'uniform'
                return 'evidence'

            self.fusion_deep = JointEvidencePartitionFusion(
                (c4,), (c4,), c4, partition_mode=partition_mode('deep'),
            )
            self.fusion_middle = JointEvidencePartitionFusion(
                (c4, c4, c3, c3), (c4, c4), c3,
                partition_mode=partition_mode('middle'),
            )
            self.fusion_shallow = JointEvidencePartitionFusion(
                (c3, c3, c2, c4), (c3, c3), c2,
                partition_mode=partition_mode('shallow'),
            )
        self.uplayer2 = self._make_layer(
            2, c4, c4, level='deep', scope='decoder', core=True
        )
        self.uplayer2_edge = self._make_layer(2, c4, c4, level='deep', scope='decoder')
        self.uplayer1 = self._make_layer(
            2, c3, c3, level='middle', scope='decoder', core=True
        )
        self.uplayer1_edge = self._make_layer(2, c3, c3, level='middle', scope='decoder')

        self.fuse_edge = nn.Conv2d(c2, 1, 1, bias=False)

        self.ccfe = CCFE(k=3, down=2)
        self.fuse_local = nn.Conv2d(c2 * 3, c2 * 4, 1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.sobel = SobelConv()

        self.prior_fuse1 = self._make_layer(2, 9, 3, level='shallow', scope='encoder')
        self.prior_fuse2 = self._make_layer(2, 3, 3, level='shallow', scope='encoder')

        # The same block also defines the final multi-source fusion interface.
        self.tail_fusion = self._make_layer(1, c2 * 3 + 3, c2, level='shallow', scope='decoder')
        self.unit = Unit(c2)
        self.end = _FCNHead(c2, 1)
        self.aux_deep = nn.Conv2d(c4, 1, 1)
        self.aux_mid = nn.Conv2d(c3, 1, 1)

    def _use_reference(self, level, scope, core):
        d = self.deployment
        if d == 'dense':
            return True
        if not core:
            return False
        return (
            d == 'full' or d == level or
            (d == 'encoder' and scope == 'encoder') or
            (d == 'decoder' and scope == 'decoder')
        )

    @staticmethod
    def _ring_for_level(level):
        return {'shallow': (7, 3), 'middle': (5, 1), 'deep': (3, 1)}[level]

    def _make_layer(self, block_num, in_channels, out_channels, level, scope, core=False):
        use_reference = self._use_reference(level, scope, core)
        outer, guard = self._ring_for_level(level)
        base_count = int(block_num)
        common = {'outer_size': outer, 'guard_size': guard}

        if not use_reference:
            layers = [ResidualBlock(in_channels, out_channels, **common)]
            layers.extend(
                ResidualBlock(out_channels, out_channels, **common)
                for _ in range(base_count - 1)
            )
            return nn.Sequential(*layers)

        breb_args = dict(common, evidence_mode=self.evidence_mode)
        # ``dense`` reproduces the former all-block replacement.  The proposed
        # modes expose one deep module at each selected seam and retain ordinary
        # residual adapters for any remaining depth in that stage.
        breb_count = base_count + self.stage_repeat - 1 if self.deployment == 'dense' else self.stage_repeat
        layers = [BackgroundReferencedBlock(in_channels, out_channels, **breb_args)]
        layers.extend(
            BackgroundReferencedBlock(out_channels, out_channels, **breb_args)
            for _ in range(breb_count - 1)
        )
        residual_count = 0 if self.deployment == 'dense' else max(0, base_count - breb_count)
        layers.extend(
            ResidualBlock(out_channels, out_channels, **common)
            for _ in range(residual_count)
        )
        self.reference_block_count += breb_count
        return nn.Sequential(*layers)

    @staticmethod
    def _pad_to_pyramid_stride(x, stride=8):
        """Pad only the bottom/right edges so wavelet and pooling grids agree."""
        height, width = x.shape[-2:]
        pad_height = (-height) % stride
        pad_width = (-width) % stride
        if pad_height == 0 and pad_width == 0:
            return x, (height, width)
        mode = 'reflect' if height > pad_height and width > pad_width else 'replicate'
        return F.pad(x, (0, pad_width, 0, pad_height), mode=mode), (height, width)

    @staticmethod
    def _crop_evidence_pyramid(evidence_pyramid, original_size, padded_size):
        """Remove padded border cells from every scale used by HEAL."""
        original_height, original_width = original_size
        padded_height, padded_width = padded_size
        cropped = {}
        for level, states in evidence_pyramid.items():
            cropped_states = {}
            for name, value in states.items():
                feature_height, feature_width = value.shape[-2:]
                crop_height = min(
                    feature_height,
                    (original_height * feature_height + padded_height - 1) // padded_height,
                )
                crop_width = min(
                    feature_width,
                    (original_width * feature_width + padded_width - 1) // padded_width,
                )
                cropped_states[name] = value[..., :crop_height, :crop_width]
            cropped[level] = cropped_states
        return cropped

    def forward(self, x, return_aux=False):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3]
        x, original_size = self._pad_to_pyramid_stride(x)
        h, w = x.shape[-2:]

        x_dilate = self.dilate(x)
        prior = torch.cat([white_tophat(x, 3), white_tophat(x, 5), white_tophat(x, 7)], dim=1)
        prior = self.prior_fuse2(self.prior_fuse1(prior))
        sobel = self.sobel(x_dilate)
        c = sobel.size(1) // 2
        x_grad = torch.sqrt(sobel[:, :c].pow(2) + sobel[:, c:].pow(2) + 1e-6).mean(dim=1, keepdim=True)

        stage1 = self.stage1(torch.cat([x, x_grad, x_dilate], dim=1))
        first = self.stage1_1(stage1)
        second = self.stage2_1(self.pool(stage1))
        third = self.stage2_2(torch.cat([self.down1(first), second], dim=1))
        stage1_2 = self.stage1_2(torch.cat([self.up(second), first], dim=1))

        first = self.stage2_3(third)
        second = self.stage3_1(self.pool(third))
        third = self.stage3_2(torch.cat([self.down2(first), second], dim=1))
        stage2_4 = self.stage2_4(torch.cat([self.up(second), first], dim=1))

        first = self.stage3_3(third)
        second = self.stage4_1(self.pool(third))
        stage4_2 = self.stage4_2(torch.cat([self.down3(first), second], dim=1))
        stage3_4 = self.stage3_4(torch.cat([self.up(second), first], dim=1))

        e_stage1 = self.edge_stage1(x)
        e_first = self.edge_stage1_1(e_stage1)
        e_second = self.edge_stage2_1(self.pool(e_stage1))
        e_third = self.edge_stage2_2(torch.cat([self.down_half(e_first), e_second], dim=1))

        e_first = self.edge_stage2_3(e_third)
        e_second = self.edge_stage3_1(self.pool(e_third))
        e_third = self.edge_stage3_2(torch.cat([self.down_half(e_first), e_second], dim=1))
        edge_stage2_4 = self.edge_stage2_4(torch.cat([self.up(e_second), e_first], dim=1))

        e_first = self.edge_stage3_3(e_third)
        e_second = self.edge_stage4_1(self.pool(e_third))
        edge_stage4_2 = self.edge_stage4_2(torch.cat([self.down_half(e_first), e_second], dim=1))
        edge_stage3_4 = self.edge_stage3_4(torch.cat([self.up(e_second), e_first], dim=1))

        evidence_pyramid = {}
        if self.fusion_mode == 'legacy':
            rec_main = self.uplayer2(self.fsp_main4(edge_stage4_2, stage4_2))
            rec_edge = self.uplayer2_edge(self.fsp_edge4(stage4_2, edge_stage4_2))
            aux_deep = F.interpolate(
                self.aux_deep(rec_main), size=(h, w), mode='bilinear', align_corners=False
            )

            rec_edge = self.reduce2_edge(self.up(rec_edge) + edge_stage3_4)
            rec_main = (self.reduce2(self.up(rec_main) + stage3_4)
                        + self.down3(stage2_4) + self.down4(self.down2(stage1_2)))
            rec_main = self.uplayer1(self.fsp_main3(rec_edge, rec_main))
            rec_edge = self.uplayer1_edge(self.fsp_edge3(rec_main, rec_edge))
            aux_mid = F.interpolate(
                self.aux_mid(rec_main), size=(h, w), mode='bilinear', align_corners=False
            )

            rec_edge = self.reduce1_edge(self.up(rec_edge) + edge_stage2_4)
            rec_main = (self.reduce1(self.up(rec_main) + stage2_4)
                        + self.down2(stage1_2) + self.reduce1(self.reduce2(self.up(stage3_4))))
        else:
            rec_main, rec_edge, evidence_pyramid['deep'] = self.fusion_deep(
                [stage4_2], [edge_stage4_2]
            )
            rec_main = self.uplayer2(rec_main)
            rec_edge = self.uplayer2_edge(rec_edge)
            aux_deep = F.interpolate(
                self.aux_deep(rec_main), size=(h, w), mode='bilinear', align_corners=False
            )

            stage1_down = self.down2(stage1_2)
            rec_main, rec_edge, evidence_pyramid['middle'] = self.fusion_middle(
                [self.up(rec_main), stage3_4, self.down3(stage2_4), self.down4(stage1_down)],
                [self.up(rec_edge), edge_stage3_4],
            )
            rec_main = self.uplayer1(rec_main)
            rec_edge = self.uplayer1_edge(rec_edge)
            aux_mid = F.interpolate(
                self.aux_mid(rec_main), size=(h, w), mode='bilinear', align_corners=False
            )

            rec_main, rec_edge, evidence_pyramid['shallow'] = self.fusion_shallow(
                [self.up(rec_main), stage2_4, stage1_down, self.up(stage3_4)],
                [self.up(rec_edge), edge_stage2_4],
            )
        edge_full = F.interpolate(rec_edge, size=(h, w), mode='bilinear', align_corners=False)
        edge_prob = torch.sigmoid(self.fuse_edge(edge_full))
        rec_main = F.interpolate(rec_main, size=(h, w), mode='bilinear', align_corners=False)

        local = self.ccfe(edge_prob, rec_main)
        local = self.pixel_shuffle(self.fuse_local(local))
        fused = self.tail_fusion(torch.cat([rec_main, local, edge_full, prior], dim=1))
        logits = self.end(self.unit(fused))

        original_h, original_w = original_size
        logits = logits[..., :original_h, :original_w]
        edge_prob = edge_prob[..., :original_h, :original_w]
        aux_mid = aux_mid[..., :original_h, :original_w]
        aux_deep = aux_deep[..., :original_h, :original_w]

        if return_aux:
            evidence_pyramid = self._crop_evidence_pyramid(
                evidence_pyramid, original_size, (h, w)
            )
            return logits, edge_prob, [aux_mid, aux_deep], evidence_pyramid
        return logits, edge_prob


BREBNet = ISNet


if __name__ == '__main__':
    net = ISNet(channels=(16, 32, 64, 128), deployment='full', stage_repeat=1)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y, edge, aux, evidence = net(x, return_aux=True)
    print('logits:', y.shape, 'edge:', edge.shape, 'aux:', [z.shape for z in aux])
    print('HEAL evidence:', sorted(evidence))
    print('BREB blocks:', net.reference_block_count)
