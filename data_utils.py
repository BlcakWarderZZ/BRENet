"""Dataset loader for BRENet (IRSTD-1k, NUAA-SIRST, NUDT-SIRST, sirst, sirst_aug, MDvsFA_cGAN).

Reconstructed from the official MMLNet data_utils.py (github.com/qianngli/MMLNet,
commit 6809f21), which BRENet was built on. Differences from that file:
  * the dataset root comes from --data_root (train.py/test.py also search ./datasets,
    ../datasets); the old hard-coded '/data/liqiang/<dataset>' is only a last fallback;
  * IRSTD-1k split lists come from the package (splits/, chosen by --split); ids may be
    written with or without the extension, and every image/mask path is checked once;
  * single-channel 'L' masks are supported (the original cvtColor(RGB2BGR) on masks
    fails for 2-D arrays);
  * augmentations that were renamed/removed in albumentations >= 2.0 fall back to
    the closest available transform (see _build_augmentation).

Expected layout:
  IRSTD-1k/{images,masks}/<id>.png  + trainval.txt, test.txt
  sirst_aug/{train,test}/{images,masks}/*.png
  MDvsFA_cGAN/{train,test}/{images,masks}/*.png
  NUAA-SIRST, NUDT-SIRST (BasicIRSTD layout): {images,masks}/<id>.png + img_idx/{train,test}_<name>.txt
"""
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')  # no version check over the network
try:
    import albumentations as A
except ImportError:  # augmentation pipeline is optional
    A = None


def _maybe(name, *args, **kwargs):
    """Return albumentations.<name>(...) if available in this version, else None."""
    if A is None or not hasattr(A, name):
        return None
    try:
        return getattr(A, name)(*args, **kwargs)
    except Exception:
        return None


def _one_of(items, p):
    items = [item for item in items if item is not None]
    return A.OneOf(items, p=p) if items else None


IMAGE_EXTS = ('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')


def read_id_list(list_file):
    """Image ids from a split file; accepts 'XDU1' and 'XDU1.png' alike."""
    ids = []
    with open(list_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            item = osp.basename(line.strip())
            if not item:
                continue
            stem, ext = osp.splitext(item)
            ids.append(stem if ext.lower() in IMAGE_EXTS else item)
    return ids


def resolve_file(directory, image_id):
    """Path of <directory>/<image_id>.<ext> for the first extension that exists."""
    for ext in IMAGE_EXTS + tuple(e.upper() for e in IMAGE_EXTS):
        path = osp.join(directory, image_id + ext)
        if osp.isfile(path):
            return path
    raise FileNotFoundError('no image file for id %r in %s' % (image_id, directory))


# Official ISNet split (800 train / 201 test), as stated in the manuscript.
# IRSTD-1k split lists shipped with the code (the dataset folder's own .txt files are not used):
#   official : ISNet/BasicIRSTD split, 800 train / 201 test (as stated in the ISNet and MMLNet papers)
#   9-1      : 900 train / 101 test, the random 9:1 split of the earlier BRENet runs
# Both are disjoint.  State in the paper which one is used; numbers on the two test sets
# are not comparable.  Released baseline weights were trained on the official 800 images,
# which include 84 of the 101 '9-1' test images.
SPLIT_DIRS = {
    'official': osp.join(osp.dirname(osp.abspath(__file__)), 'splits', 'IRSTD-1k'),
    '9-1': osp.join(osp.dirname(osp.abspath(__file__)), 'splits', 'IRSTD-1k_9-1'),
}
OFFICIAL_SPLIT_DIR = SPLIT_DIRS['official']
_checked_splits = set()


def irstd1k_split_lists(split):
    """(train ids, test ids) of a shipped IRSTD-1k split; refuses overlapping lists."""
    if split not in SPLIT_DIRS:
        raise ValueError('IRSTD-1k split must be one of %s, got %r' % (sorted(SPLIT_DIRS), split))
    folder = SPLIT_DIRS[split]
    train = read_id_list(osp.join(folder, 'trainval.txt'))
    test = read_id_list(osp.join(folder, 'test.txt'))
    overlap = sorted(set(train) & set(test))
    if overlap:
        raise ValueError('%d ids are in both lists of split %s (e.g. %s): test leakage'
                         % (len(overlap), split, ', '.join(overlap[:5])))
    if split not in _checked_splits:
        _checked_splits.add(split)
        print('[Data] IRSTD-1k split %s: %d train / %d test (lists from %s)' % (
            split, len(set(train)), len(set(test)), folder))
    return train, test


def _build_augmentation(p=1.0):
    """Same transforms and probabilities as MMLNet's Img_AUG."""
    if A is None:
        return None
    flip = _maybe('Flip')
    if flip is None:  # removed in albumentations 2.x
        flip = _one_of([_maybe('HorizontalFlip', p=1.0), _maybe('VerticalFlip', p=1.0)], p=0.5)
    shift_scale_rotate = _maybe('ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.2,
                                rotate_limit=45, p=0.2)
    if shift_scale_rotate is None:
        shift_scale_rotate = _maybe('Affine', translate_percent=(-0.0625, 0.0625),
                                    scale=(0.8, 1.2), rotate=(-45, 45), p=0.2)
    pipeline = [
        _maybe('RandomRotate90'),
        flip,
        _maybe('Transpose'),
        _maybe('ToGray'),
        _one_of([_maybe('GaussNoise')], p=0.2),
        shift_scale_rotate,
        _one_of([_maybe('OpticalDistortion', p=0.3), _maybe('GridDistortion', p=0.1),
                 _maybe('PiecewiseAffine', p=0.3)], p=0.2),
        _one_of([_maybe('CLAHE', clip_limit=2), _maybe('Sharpen'), _maybe('Emboss'),
                 _maybe('RandomBrightnessContrast')], p=0.5),
        _maybe('HueSaturationValue', p=0.5),
    ]
    return A.Compose([t for t in pipeline if t is not None], p=p)


def _photometric(img):
    """Brightness/contrast, gamma and Gaussian noise on the grey levels; target shapes are left untouched."""
    a = np.asarray(img, dtype=np.float32)
    if random.random() < 0.5:
        a = a * random.uniform(0.8, 1.2) + random.uniform(-20.0, 20.0)
    if random.random() < 0.5:
        a = 255.0 * (np.clip(a, 0, 255) / 255.0) ** random.uniform(0.8, 1.25)
    if random.random() < 0.3:
        a = a + np.random.normal(0.0, random.uniform(1.0, 5.0), a.shape[:2])[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


class DatasetLoader(Data.Dataset):
    SIZES = {'IRSTD-1k': 480, 'sirst_aug': 256, 'sirst': 256, 'MDvsFA_cGAN': 128, 'NUAA-SIRST': 256, 'NUDT-SIRST': 256}
    # training lists are repeated so that one epoch has about 4,000 samples on every dataset (800 x 5 for IRSTD-1k,
    # 213 x 19 for NUAA-SIRST, 663 x 6 for NUDT-SIRST): the same number of iterations and learning-rate steps everywhere
    REPEATS = {'IRSTD-1k': 5, 'NUAA-SIRST': 19, 'NUDT-SIRST': 6}

    def __init__(self, args, mode='train'):
        if mode not in {'train', 'test'}:
            raise ValueError('Unknown mode %s' % mode)
        self.dataset = args.dataset
        self.mode = mode
        base_dir = getattr(args, 'data_root', '') or osp.join('/data/liqiang', self.dataset)
        if not osp.isdir(base_dir):
            raise FileNotFoundError('dataset root not found: %s (set --data_root)' % base_dir)

        if self.dataset == 'IRSTD-1k':
            self.imgs_dir = osp.join(base_dir, 'images')
            self.label_dir = osp.join(base_dir, 'masks')
            train_ids, test_ids = irstd1k_split_lists(getattr(args, 'split', '9-1'))
            ids = list(train_ids if mode == 'train' else test_ids)
        elif self.dataset in ('NUAA-SIRST', 'NUDT-SIRST'):
            # standard BasicIRSTD splits (NUAA-SIRST 213/214, NUDT-SIRST 663/664)
            self.imgs_dir = osp.join(base_dir, 'images')
            self.label_dir = osp.join(base_dir, 'masks')
            lists = {m: read_id_list(osp.join(base_dir, 'img_idx', '%s_%s.txt' % (m, self.dataset)))
                     for m in ('train', 'test')}
            overlap = sorted(set(lists['train']) & set(lists['test']))
            if overlap:
                raise ValueError('%d ids are in both %s lists (e.g. %s): test leakage'
                                 % (len(overlap), self.dataset, ', '.join(overlap[:5])))
            ids = list(lists[mode])
        elif self.dataset == 'sirst' and not osp.isdir(osp.join(base_dir, 'train', 'images')):
            # flat layout: images/ + masks/ with id lists (trainval.txt or train.txt, test.txt)
            self.imgs_dir = osp.join(base_dir, 'images')
            self.label_dir = osp.join(base_dir, 'masks')
            names = ('trainval.txt', 'train.txt') if mode == 'train' else ('test.txt',)
            found = [osp.join(base_dir, n) for n in names if osp.isfile(osp.join(base_dir, n))]
            found += [osp.join(base_dir, 'idx_427', n) for n in names if osp.isfile(osp.join(base_dir, 'idx_427', n))]
            if not found:
                raise FileNotFoundError('sirst: neither %s/train/images nor an id list %s was found' % (base_dir, names))
            ids = read_id_list(found[0])
        elif self.dataset in ('sirst_aug', 'MDvsFA_cGAN', 'sirst'):
            split = 'train' if mode == 'train' else 'test'
            self.imgs_dir = osp.join(base_dir, split, 'images')
            self.label_dir = osp.join(base_dir, split, 'masks')
            ids = sorted(osp.splitext(f)[0] for f in os.listdir(self.imgs_dir)
                         if osp.splitext(f)[1].lower() in IMAGE_EXTS and not f.startswith('.'))
        else:
            raise ValueError('Unsupported dataset %s' % self.dataset)

        # resolve every file once, so a bad split file fails here with a full list
        # instead of inside a DataLoader worker after the DDP processes have started
        self.files, missing = {}, []
        for image_id in ids:
            try:
                self.files[image_id] = (resolve_file(self.imgs_dir, image_id),
                                        resolve_file(self.label_dir, image_id))
            except FileNotFoundError:
                missing.append(image_id)
        if missing:
            raise FileNotFoundError('%d of %d ids in the %s split have no image or mask in %s / %s, e.g. %s'
                                    % (len(missing), len(ids), mode, self.imgs_dir, self.label_dir,
                                       ', '.join(missing[:5])))
        repeats = int(getattr(args, 'repeats', 0) or 0) or self.REPEATS.get(self.dataset, 1)
        if mode == 'train' and repeats > 1:
            ids = ids * repeats  # MMLNet repeats the IRSTD-1k training list five times
            random.shuffle(ids)
        self.names = ids

        self.scale = max(1, int(getattr(args, 'input_scale', 1) or 1))
        self.size = self.SIZES[self.dataset] * self.scale
        self.aug = getattr(args, 'aug', 'mmlnet')
        self.augmentation = _build_augmentation(p=1.0) if mode == 'train' and self.aug == 'mmlnet' else None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        img_path, mask_path = self.files[name]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if mask.size != img.size:
            # NUAA-SIRST Misc_111: image 325x220, mask 592x400; the mask follows the image grid
            mask = mask.resize(img.size, Image.NEAREST)
        if self.scale > 1:  # --input-scale: finer grid for small targets, same model
            w, h = img.size
            img = img.resize((w * self.scale, h * self.scale), Image.BILINEAR)
            mask = mask.resize((w * self.scale, h * self.scale), Image.NEAREST)
        if self.mode == 'train':
            img, mask = self.train_transform(img, mask)
        return self.transform(img), transforms.ToTensor()(mask), osp.basename(img_path)

    def train_transform(self, img, mask):
        size = self.size
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < (0.5 if self.aug == 'medium' else 0.2):  # random scale of the long edge
            long_size = random.randint(int(size * 0.75), int(size * 1.5))
            w, h = img.size
            if h > w:
                oh, ow = long_size, int(1.0 * w * long_size / h + 0.5)
                short_size = ow
            else:
                ow, oh = long_size, int(1.0 * h * long_size / w + 0.5)
                short_size = oh
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            if short_size < size:
                padh = size - oh if oh < size else 0
                padw = size - ow if ow < size else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        if w > size or h > size:  # random crop (the original only checked the width)
            x1 = random.randint(0, max(0, w - size))
            y1 = random.randint(0, max(0, h - size))
            img = img.crop((x1, y1, x1 + size, y1 + size))
            mask = mask.crop((x1, y1, x1 + size, y1 + size))
        if img.size != (size, size):  # keep batch shapes equal after scaling
            img = img.resize((size, size), Image.BILINEAR)
            mask = mask.resize((size, size), Image.NEAREST)
        if self.aug in ('light', 'medium'):  # shape-preserving: the dihedral group only, no blur or warping
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.5:
                img = img.transpose(Image.TRANSPOSE)
                mask = mask.transpose(Image.TRANSPOSE)
            if self.aug == 'medium':
                img = _photometric(img)
            return img, mask
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        if self.augmentation is not None:
            image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            label = np.asarray(mask)
            out = self.augmentation(image=image, mask=label)
            img = Image.fromarray(cv2.cvtColor(out['image'], cv2.COLOR_BGR2RGB))
            mask = Image.fromarray(np.asarray(out['mask'], dtype=np.uint8))
        return img, mask
