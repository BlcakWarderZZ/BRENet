# -*- coding: utf-8 -*-
"""BRENet losses: segmentation, deep supervision, edge, BREB reference term, HEAL and TBS."""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_dice_loss(prob, target, eps=1e-6):
    prob = prob.flatten(1)
    target = target.float().flatten(1)
    intersection = (prob * target).sum(dim=1)
    score = (2.0 * intersection + eps) / (prob.sum(dim=1) + target.sum(dim=1) + eps)
    return 1.0 - score.mean()


def _masked_mean(value, mask, eps=1.0):
    dims = tuple(range(1, value.ndim))
    numerator = (value * mask).sum(dim=dims)
    denominator = mask.sum(dim=dims).clamp_min(eps)
    return (numerator / denominator).mean()


def _replicate_mean(x, kernel_size):
    if kernel_size == 1:
        return x
    pad = kernel_size // 2
    return F.avg_pool2d(
        F.pad(x, (pad, pad, pad, pad), mode='replicate'), kernel_size, stride=1
    )


def _annular_mean(x, outer_size, guard_size):
    outer_area = float(outer_size * outer_size)
    guard_area = float(guard_size * guard_size)
    outer = _replicate_mean(x, outer_size) * outer_area
    guard = _replicate_mean(x, guard_size) * guard_area
    return (outer - guard) / (outer_area - guard_area)


class SoftLoULoss(nn.Module):
    """Kept for compatibility with earlier experiment scripts."""

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1.0
        intersection = pred * target
        loss = (intersection.sum() + smooth) / (
            pred.sum() + target.sum() - intersection.sum() + smooth
        )
        return 1.0 - loss


class SoftLoULoss1(nn.Module):
    """Batch-wise IoU variant retained for compatibility."""

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1.0
        intersection = pred * target
        loss = (intersection.sum((1, 2, 3)) + smooth) / (
            pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
            - intersection.sum((1, 2, 3)) + smooth
        )
        return (1.0 - loss).mean()


class HierarchicalEvidenceAttribution(nn.Module):
    """Hierarchical Evidence Attribution Learning (HEAL).

    A false alarm is considered persistent only when the final prediction and
    at least two JEPF scales jointly support the same background location.  The
    detached mining step locates those rare regions; differentiable attribution
    then redirects every selected JEPF state from ``confirmed`` to ``background``.
    Ground-truth targets are explicitly protected and receive the complementary
    confirmed-evidence constraint, avoiding a global response-suppression tradeoff.
    """

    VALID_SCOPES = {
        'deep': {'deep'},
        'middle': {'middle'},
        'shallow': {'shallow'},
        'deep_middle': {'deep', 'middle'},
        'middle_shallow': {'middle', 'shallow'},
        'full': {'deep', 'middle', 'shallow'},
    }
    VALID_MINING = {'persistent', 'final', 'random'}

    def __init__(
        self,
        scope='full',
        mining='persistent',
        threshold=0.35,
        hard_fraction=0.0001,
        min_seeds=2,
        max_seeds=32,
        region_size=3,
        protection_size=5,
        target_weight=1.0,
        false_alarm_weight=1.0,
    ):
        super().__init__()
        if scope not in self.VALID_SCOPES:
            raise ValueError('HEAL scope must be one of %s' % sorted(self.VALID_SCOPES))
        if mining not in self.VALID_MINING:
            raise ValueError('HEAL mining must be one of %s' % sorted(self.VALID_MINING))
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError('HEAL threshold must lie in (0, 1)')
        if hard_fraction <= 0.0 or hard_fraction > 1.0:
            raise ValueError('HEAL hard_fraction must lie in (0, 1]')
        if min_seeds < 1 or max_seeds < min_seeds:
            raise ValueError('HEAL requires 1 <= min_seeds <= max_seeds')
        if region_size < 1 or region_size % 2 == 0:
            raise ValueError('HEAL region_size must be a positive odd integer')
        if protection_size < 1 or protection_size % 2 == 0:
            raise ValueError('HEAL protection_size must be a positive odd integer')
        self.levels = self.VALID_SCOPES[scope]
        self.mining = mining
        self.threshold = float(threshold)
        self.hard_fraction = float(hard_fraction)
        self.min_seeds = int(min_seeds)
        self.max_seeds = int(max_seeds)
        self.region_size = int(region_size)
        self.protection_size = int(protection_size)
        self.target_weight = float(target_weight)
        self.false_alarm_weight = float(false_alarm_weight)

    @staticmethod
    def _resize_mask(mask, size):
        if mask.shape[-2:] == size:
            return mask
        if mask.shape[-2] >= size[0] and mask.shape[-1] >= size[1]:
            return F.adaptive_max_pool2d(mask, size)
        return F.interpolate(mask, size=size, mode='nearest')

    def _mine(self, logits, target, evidence_pyramid):
        full_size = logits.shape[-2:]
        confirmed = [
            F.interpolate(
                state['confirmed'], size=full_size,
                mode='bilinear', align_corners=False,
            )
            for state in evidence_pyramid.values()
            if 'confirmed' in state
        ]
        final_probability = torch.sigmoid(logits).detach()
        if self.mining == 'persistent' and len(confirmed) >= 2:
            stacked = torch.stack([value.detach() for value in confirmed], dim=1)
            second_support = torch.topk(stacked, k=2, dim=1).values[:, 1]
            score = torch.sqrt((final_probability * second_support).clamp_min(1e-8))
        elif self.mining == 'random':
            score = torch.rand_like(final_probability)
        else:
            score = final_probability

        protected = target.float()
        if self.protection_size > 1:
            protected = F.max_pool2d(
                protected, self.protection_size, stride=1,
                padding=self.protection_size // 2,
            )
        background = (protected < 0.5).to(dtype=logits.dtype)
        candidate_score = score * background
        seeds = torch.zeros_like(candidate_score)

        for index in range(candidate_score.size(0)):
            background_count = int(background[index].sum().item())
            if background_count == 0:
                continue
            requested = int(round(background_count * self.hard_fraction))
            count = min(background_count, self.max_seeds, max(self.min_seeds, requested))
            flat_score = candidate_score[index].flatten()
            values, locations = torch.topk(flat_score, k=count, sorted=False)
            if self.mining == 'random':
                valid = torch.ones_like(values)
            else:
                valid = (values >= self.threshold).to(dtype=values.dtype)
            seeds[index].view(-1).scatter_(0, locations, valid)

        if self.region_size > 1:
            hard_region = F.max_pool2d(
                seeds, self.region_size, stride=1,
                padding=self.region_size // 2,
            )
            hard_weight = F.max_pool2d(
                seeds * candidate_score, self.region_size, stride=1,
                padding=self.region_size // 2,
            )
        else:
            hard_region = seeds
            hard_weight = seeds * candidate_score
        hard_region = hard_region * background
        hard_weight = hard_weight * hard_region
        if self.mining == 'random':
            hard_weight = hard_region
        return hard_region.detach(), hard_weight.detach(), candidate_score.detach()

    def forward(self, logits, target, evidence_pyramid):
        zero = logits.new_zeros(())
        active = {
            name: state for name, state in evidence_pyramid.items()
            if name in self.levels
        }
        if not active:
            return {
                'total': zero, 'target': zero, 'false_alarm': zero,
                'hard_pixels': zero, 'persistent_score': zero,
            }

        hard_region, hard_weight, persistent_score = self._mine(
            logits, target, evidence_pyramid
        )
        target_terms, false_terms = [], []
        for state in active.values():
            confirmed = state['confirmed'].clamp(1e-6, 1.0 - 1e-6)
            background = state['background'].clamp(1e-6, 1.0 - 1e-6)
            size = confirmed.shape[-2:]
            target_mask = self._resize_mask(target.float(), size)
            hard_mask = self._resize_mask(hard_region, size)
            hard_scale = self._resize_mask(hard_weight, size) * hard_mask

            target_terms.append(_masked_mean(-torch.log(confirmed), target_mask))
            reject_confirmed = _masked_mean(-torch.log1p(-confirmed), hard_scale)
            accept_background = _masked_mean(-torch.log(background), hard_scale)
            false_terms.append(reject_confirmed + 0.5 * accept_background)

        target_loss = torch.stack(target_terms).mean()
        false_alarm_loss = torch.stack(false_terms).mean()
        total = (
            self.target_weight * target_loss
            + self.false_alarm_weight * false_alarm_loss
        )
        hard_pixels = hard_region.flatten(1).sum(dim=1).mean().detach()
        persistent = _masked_mean(persistent_score, hard_region).detach()
        return {
            'total': total,
            'target': target_loss,
            'false_alarm': false_alarm_loss,
            'hard_pixels': hard_pixels,
            'persistent_score': persistent,
        }


class TargetBalancedSupervision(nn.Module):
    """TBS: every ground-truth target contributes equally to the loss.

    Measured on IRSTD-1k: detectors miss targets far more often in images that contain several
    targets (Pd 0.80-0.86 vs 0.94-0.95 for BRENet variants; 0.84-0.91 vs 0.90-0.95 for seven
    published detectors), and the baseline misses 11% of such targets even on its TRAINING images.
    Pixel-level BCE/Dice weight a target by its share of the batch's pixels, so once the dominant
    target is fitted a secondary target receives almost no gradient, while Pd counts every target.

    For each connected target T_k, its window W_k is T_k plus its local background ring
    (dilation by ``context`` px, other targets excluded).  The loss is the mean over all targets
    of 1 - softIoU_k, with softIoU_k = sum(p T_k) / sum_{W_k}(p + T_k - p T_k): a missed target
    costs 1 whatever its size or the number of targets in its image, and spill into the target's
    own background ring is penalised in the same term.
    """
    def __init__(self, context=4, eps=1e-6, false_alarm=False):
        super().__init__()
        if context < 0:
            raise ValueError('TBS context must be >= 0')
        self.context = int(context)
        self.eps = float(eps)
        # Optional object-level counterpart for false alarms (off by default): every predicted
        # component (p >= 0.5) that lies outside all target windows is one more unit of the loss,
        # its mean binary cross-entropy towards background, so each false-alarm component counts
        # once regardless of its area -- the way Fa counts unmatched components.
        self.false_alarm = bool(false_alarm)

    def _false_alarm_terms(self, logits, binary_np, index, kernel):
        guard = cv2.dilate(binary_np, np.ones((kernel, kernel), np.uint8)) if self.context else binary_np
        pred = (torch.sigmoid(logits[index, 0]).detach() >= 0.5).cpu().numpy().astype(np.uint8)
        pred[guard > 0] = 0
        count, labels = cv2.connectedComponents(pred, connectivity=8)
        if count <= 1:
            return None
        # Per-component mean by scatter over the label map: memory O(H*W), independent of the
        # number of components (early in training the thresholded output has thousands of them).
        labels = torch.from_numpy(labels.astype(np.int64)).to(logits.device).view(-1)
        bce = F.softplus(logits[index, 0]).reshape(-1)                           # -log(1 - p)
        sums = bce.new_zeros(count).scatter_add(0, labels, bce)
        sizes = torch.bincount(labels, minlength=count).to(bce.dtype)
        return sums[1:] / sizes[1:].clamp_min(1.0)

    def forward(self, logits, target):
        probability = torch.sigmoid(logits)
        binary = (target > 0.5)
        masks = binary.detach().cpu().numpy().astype(np.uint8)
        kernel = 2 * self.context + 1
        terms = []
        for index in range(masks.shape[0]):
            if self.false_alarm:
                fa = self._false_alarm_terms(logits, masks[index, 0], index, kernel)
                if fa is not None:
                    terms.append(fa)
            count, labels = cv2.connectedComponents(masks[index, 0], connectivity=8)
            if count <= 1:
                continue
            labels = torch.from_numpy(labels.astype(np.int64)).to(logits.device)
            ids = torch.arange(1, count, device=logits.device).view(-1, 1, 1)
            objects = (labels.unsqueeze(0) == ids).unsqueeze(1).to(probability.dtype)   # (K,1,H,W)
            windows = F.max_pool2d(objects, kernel, stride=1, padding=self.context) if self.context else objects
            others = (binary[index:index + 1].to(probability.dtype) - objects).clamp(0.0, 1.0)
            windows = windows * (1.0 - others)
            p = probability[index:index + 1]
            inter = (p * objects).sum(dim=(1, 2, 3))
            union = ((p + objects - p * objects) * windows).sum(dim=(1, 2, 3))
            terms.append(1.0 - inter / (union + self.eps))
        if not terms:
            return logits.sum() * 0.0
        return torch.cat(terms).mean()


class BackgroundReferenceLoss(nn.Module):
    """Supervise segmentation, BREB representation, and JEPF attribution.

    The reference term does not introduce a second unrelated objective.  It
    applies the same rule as BREB to the prediction space: target pixels should
    exceed their annular background distribution, whereas background pixels
    should not produce a large positive standardized deviation.  A ramp avoids
    forcing this relation before the segmentation prediction is meaningful.
    """

    def __init__(
        self,
        w_final_bce=1.0,
        w_final_dice=1.0,
        w_aux=0.25,
        w_edge=1.0,
        w_edge_bce=5.0,
        w_edge_dice=1.0,
        w_reference=0.15,
        ramp_start=10,
        ramp_end=60,
        target_margin=1.0,
        background_margin=1.0,
        outer_size=7,
        guard_size=3,
        use_reference=True,
        w_heal=0.08,
        heal_scope='full',
        heal_mining='persistent',
        heal_threshold=0.35,
        heal_hard_fraction=0.0001,
        heal_min_seeds=2,
        heal_max_seeds=32,
        heal_region_size=3,
        heal_protection_size=5,
        heal_ramp_start=20,
        heal_ramp_end=80,
        use_heal=True,
        w_tbs=0.0,
        tbs_context=4,
        tbs_false_alarm=False,
    ):
        super().__init__()
        if outer_size <= guard_size or outer_size % 2 == 0 or guard_size % 2 == 0:
            raise ValueError('outer_size and guard_size must be odd with outer_size > guard_size')
        self.w_final_bce = float(w_final_bce)
        self.w_final_dice = float(w_final_dice)
        self.w_aux = float(w_aux)
        self.w_edge = float(w_edge)
        self.w_edge_bce = float(w_edge_bce)
        self.w_edge_dice = float(w_edge_dice)
        self.w_reference = float(w_reference)
        self.ramp_start = int(ramp_start)
        self.ramp_end = int(ramp_end)
        self.target_margin = float(target_margin)
        self.background_margin = float(background_margin)
        self.outer_size = int(outer_size)
        self.guard_size = int(guard_size)
        self.use_reference = bool(use_reference)
        self.w_heal = float(w_heal)
        self.use_heal = bool(use_heal)
        self.heal_ramp_start = int(heal_ramp_start)
        self.heal_ramp_end = int(heal_ramp_end)
        # Target-Balanced Supervision (0 = off).
        self.w_tbs = float(w_tbs)
        self.tbs = TargetBalancedSupervision(context=tbs_context, false_alarm=tbs_false_alarm)
        self.heal = HierarchicalEvidenceAttribution(
            scope=heal_scope,
            mining=heal_mining,
            threshold=heal_threshold,
            hard_fraction=heal_hard_fraction,
            min_seeds=heal_min_seeds,
            max_seeds=heal_max_seeds,
            region_size=heal_region_size,
            protection_size=heal_protection_size,
        )

    def _segmentation_loss(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target.float())
        dice = _soft_dice_loss(torch.sigmoid(logits), target)
        return self.w_final_bce * bce + self.w_final_dice * dice, bce, dice

    def _reference_loss(self, logits, target):
        probability = torch.sigmoid(logits)
        reference = _annular_mean(probability, self.outer_size, self.guard_size)
        deviation = probability - reference
        dispersion = _annular_mean(deviation.abs(), self.outer_size, self.guard_size)
        # Stop the denominator from shrinking as a shortcut.  BREB makes the
        # same stabilization choice inside the architectural primitive.
        standardized = (deviation / (dispersion.detach() + 0.05)).clamp(-8.0, 8.0)

        foreground = target.float()
        background = 1.0 - foreground
        target_term = _masked_mean(
            F.relu(self.target_margin - standardized), foreground
        )
        background_term = _masked_mean(
            F.relu(standardized - self.background_margin), background
        )
        return target_term + background_term, target_term, background_term

    def _ramp(self, epoch):
        if self.ramp_end <= self.ramp_start:
            return 1.0
        return max(0.0, min(1.0, (float(epoch) - self.ramp_start) /
                            (self.ramp_end - self.ramp_start)))

    def _heal_ramp(self, epoch):
        if self.heal_ramp_end <= self.heal_ramp_start:
            return 1.0
        return max(0.0, min(1.0, (float(epoch) - self.heal_ramp_start) /
                            (self.heal_ramp_end - self.heal_ramp_start)))

    def forward(
        self, logits, target, edge_prob, edge_target, aux_logits=None,
        evidence_pyramid=None, epoch=0,
    ):
        total_final, final_bce, final_dice = self._segmentation_loss(logits, target)

        aux_logits = [] if aux_logits is None else list(aux_logits)
        if aux_logits:
            aux_terms = []
            for aux in aux_logits:
                resized_target = F.interpolate(target.float(), size=aux.shape[-2:], mode='nearest')
                aux_terms.append(self._segmentation_loss(aux, resized_target)[0])
            aux_loss = torch.stack(aux_terms).mean()
        else:
            aux_loss = logits.new_zeros(())

        edge_target = edge_target.float()
        edge_bce = F.binary_cross_entropy(edge_prob.clamp(1e-6, 1.0 - 1e-6), edge_target)
        edge_dice = _soft_dice_loss(edge_prob, edge_target)
        edge_loss = self.w_edge_bce * edge_bce + self.w_edge_dice * edge_dice

        reference_loss = logits.new_zeros(())
        reference_target = logits.new_zeros(())
        reference_background = logits.new_zeros(())
        schedule = self._ramp(epoch)
        ramp = schedule if self.use_reference else 0.0
        if self.use_reference and self.w_reference > 0.0:
            reference_loss, reference_target, reference_background = self._reference_loss(logits, target)
            if aux_logits:
                aux_reference = []
                for aux in aux_logits:
                    resized_target = F.interpolate(target.float(), size=aux.shape[-2:], mode='nearest')
                    aux_reference.append(self._reference_loss(aux, resized_target)[0])
                reference_loss = reference_loss + 0.5 * torch.stack(aux_reference).mean()

        heal = {
            'total': logits.new_zeros(()),
            'target': logits.new_zeros(()),
            'false_alarm': logits.new_zeros(()),
            'hard_pixels': logits.new_zeros(()),
            'persistent_score': logits.new_zeros(()),
        }
        heal_ramp = self._heal_ramp(epoch) if self.use_heal else 0.0
        if self.use_heal and self.w_heal > 0.0 and evidence_pyramid:
            heal = self.heal(logits, target, evidence_pyramid)

        tbs = self.tbs(logits, target) if self.w_tbs > 0.0 else logits.new_zeros(())

        total = (
            total_final
            + self.w_tbs * tbs
            + self.w_aux * aux_loss
            + self.w_edge * edge_loss
            + self.w_reference * ramp * reference_loss
            + self.w_heal * heal_ramp * heal['total']
        )
        return {
            'total': total,
            'final_bce': final_bce,
            'final_dice': final_dice,
            'aux': aux_loss,
            'edge_bce': edge_bce,
            'edge_dice': edge_dice,
            'reference': reference_loss,
            'reference_target': reference_target,
            'reference_background': reference_background,
            'ramp': logits.new_tensor(ramp),
            'heal': heal['total'],
            'heal_target': heal['target'],
            'heal_false_alarm': heal['false_alarm'],
            'heal_hard_pixels': heal['hard_pixels'],
            'heal_persistent_score': heal['persistent_score'],
            'heal_ramp': logits.new_tensor(heal_ramp),
            'tbs': tbs,
        }


BREBLoss = BackgroundReferenceLoss
HEALLoss = HierarchicalEvidenceAttribution


if __name__ == '__main__':
    criterion = BackgroundReferenceLoss()
    prediction = torch.randn(2, 1, 64, 64, requires_grad=True)
    mask = (torch.rand(2, 1, 64, 64) > 0.98).float()
    edge = torch.rand_like(mask).sigmoid()
    evidence = {
        'deep': {
            'confirmed': torch.sigmoid(prediction[:, :, ::4, ::4]),
            'background': 1.0 - torch.sigmoid(prediction[:, :, ::4, ::4]),
        },
        'middle': {
            'confirmed': torch.sigmoid(prediction[:, :, ::2, ::2]),
            'background': 1.0 - torch.sigmoid(prediction[:, :, ::2, ::2]),
        },
        'shallow': {
            'confirmed': torch.sigmoid(prediction),
            'background': 1.0 - torch.sigmoid(prediction),
        },
    }
    losses = criterion(
        prediction, mask, edge, mask,
        [prediction[:, :, ::2, ::2]], evidence_pyramid=evidence, epoch=30,
    )
    losses['total'].backward()
    print({key: float(value.detach()) for key, value in losses.items()})
