"""Training/evaluation metrics with the interface used by train.py and test.py.

  mIoU(nclass, threshold)          .update(logits, labels)  .get() -> (pixel_acc, IoU)
  SigmoidMetric(threshold, distance_thresh)
                                    .update(logits, labels)  .get() -> (IoU, nIoU, Pd, Fa_pixel)
  ROCMetric(bins)                   .update(logits, labels)  .get() -> (tpr, fpr, AUC)

logits: [B,1,H,W] pre-sigmoid scores; labels: [B,1,H,W] binary masks.
Definitions follow the manuscript: dataset IoU, image-averaged nIoU, 8-connected
objects matched by centroid distance < distance_thresh (one-to-one, raster order),
Fa_pixel = all false-positive pixels / all pixels; SigmoidMetric.fa_unmatched() gives the
common evaluator's definition (pixels of unmatched predicted components / all pixels).  These numbers are for training
logs only; paper tables come from the common evaluator applied to latest.pth.
"""
import cv2
import numpy as np
import torch


def _prob(logits):
    return torch.sigmoid(logits.detach().float()).cpu().numpy()[:, 0]


def _mask(labels):
    return (labels.detach().float().cpu().numpy()[:, 0] > 0.5)


class mIoU:
    def __init__(self, nclass=1, threshold=0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self.inter = self.union = self.correct = self.total = 0

    def update(self, logits, labels):
        pred = _prob(logits) >= self.threshold
        target = _mask(labels)
        self.inter += int(np.logical_and(pred, target).sum())
        self.union += int(np.logical_or(pred, target).sum())
        self.correct += int((pred == target).sum())
        self.total += int(target.size)

    def get(self):
        pixel_acc = self.correct / max(1, self.total)
        iou = self.inter / self.union if self.union else 1.0
        return pixel_acc, iou


def _components(mask):
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return centroids[1:count], stats[1:count, cv2.CC_STAT_AREA]


class SigmoidMetric:
    def __init__(self, threshold=0.5, distance_thresh=3.0):
        self.threshold = float(threshold)
        self.distance = float(distance_thresh)
        self.reset()

    def reset(self):
        self.inter = self.union = 0
        self.image_iou = []
        self.matched = self.objects = 0
        self.false_pixels = self.pixels = 0
        self.unmatched_pixels = 0
        self.matched_by = {'single': 0, 'multi': 0}
        self.objects_by = {'single': 0, 'multi': 0}

    def update(self, logits, labels):
        preds = _prob(logits) >= self.threshold
        targets = _mask(labels)
        for pred, target in zip(preds, targets):
            inter = int(np.logical_and(pred, target).sum())
            union = int(np.logical_or(pred, target).sum())
            self.inter += inter
            self.union += union
            self.image_iou.append(inter / union if union else 1.0)
            gt_centres, _ = _components(target)
            pred_centres, pred_areas = _components(pred)
            available = np.ones(len(pred_centres), dtype=bool)
            scene = 'single' if len(gt_centres) == 1 else 'multi'
            for centre in gt_centres:
                candidates = np.flatnonzero(available)
                if len(candidates):
                    dist = np.linalg.norm(pred_centres[candidates] - centre, axis=1)
                    close = candidates[dist < self.distance]
                    if len(close):
                        available[close[0]] = False
                        self.matched += 1
                        if len(gt_centres):
                            self.matched_by[scene] += 1
            self.objects += len(gt_centres)
            if len(gt_centres):
                self.objects_by[scene] += len(gt_centres)
            self.unmatched_pixels += int(pred_areas[available].sum()) if len(pred_areas) else 0
            self.false_pixels += int(np.logical_and(pred, ~target).sum())
            self.pixels += int(target.size)

    def get(self):
        iou = self.inter / self.union if self.union else 1.0
        niou = float(np.mean(self.image_iou)) if self.image_iou else 1.0
        pd = self.matched / self.objects if self.objects else 1.0
        fa = self.false_pixels / max(1, self.pixels)
        return iou, niou, pd, fa

    def pd_by_scene(self):
        """Pd on images with one target and on images with several targets."""
        return {key: self.matched_by[key] / self.objects_by[key] if self.objects_by[key] else float('nan')
                for key in ('single', 'multi')}

    def fa_unmatched(self):
        return self.unmatched_pixels / max(1, self.pixels)


class ROCMetric:
    def __init__(self, nclass=1, bins=100):
        if isinstance(nclass, int) and bins is None:
            bins = nclass
        self.bins = int(bins)
        self.thresholds = np.linspace(0.0, 1.0, self.bins + 1)
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.bins + 1)
        self.fp = np.zeros(self.bins + 1)
        self.pos = self.neg = 0

    def update(self, logits, labels):
        prob = _prob(logits).ravel()
        target = _mask(labels).ravel()
        positives, negatives = prob[target], prob[~target]
        self.pos += positives.size
        self.neg += negatives.size
        pos_sorted, neg_sorted = np.sort(positives), np.sort(negatives)
        self.tp += pos_sorted.size - np.searchsorted(pos_sorted, self.thresholds, side='left')
        self.fp += neg_sorted.size - np.searchsorted(neg_sorted, self.thresholds, side='left')

    def get(self):
        tpr = self.tp / max(1, self.pos)
        fpr = self.fp / max(1, self.neg)
        order = np.argsort(fpr)
        auc = float(np.trapz(tpr[order], fpr[order])) if hasattr(np, 'trapz') else \
            float(np.trapezoid(tpr[order], fpr[order]))
        return tpr, fpr, auc
