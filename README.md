# BRENet

Code for **BRENet: A Background-Referenced Evidence Network for Mitigating Multi-Target Suppression in Infrared Small Target Detection**.

BRENet applies one principle at the feature, fusion, and supervision levels: a response is target evidence only relative to its own local background.

- **BREB** (Background-Referenced Evidence Block) compares feature responses with the surrounding background at each selected scale.
- **EPA** (Evidence Partition and Attribution) separates the evidence shared by the detection and structural branches from background clutter during fusion. It consists of JEPF and HEAL.
- **TBS** (Target-Balanced Supervision) computes the loss of each target within its own local window.

## Results

| Dataset | Split | IoU (%) | Pd (%) | Fa (10^-6) |
|---|---|---|---|---|
| IRSTD-1K | 800 / 201 | 66.25 | 95.62 | 15.54 |
| NUAA-SIRST | 341 / 86 | 79.34 | 100.00 | 3.90 |
| NUDT-SIRST | 663 / 664 | 88.04 | 98.52 | 14.78 |

## Installation

Install PyTorch and torchvision for your CUDA version from [pytorch.org](https://pytorch.org), and then run:

```bash
pip install -r requirements.txt
python smoke_test.py
```

## Data

Download the datasets from their original authors and place them under `datasets/`:

```
datasets/IRSTD-1k/{images,masks}/<id>.png
datasets/sirst_aug/{train,test}/{images,masks}/*.png
datasets/NUDT-SIRST/{images,masks}/<id>.png
datasets/NUDT-SIRST/img_idx/{train,test}_NUDT-SIRST.txt
```

The split lists used in the paper are in `splits/`.

## Training

The default settings are those of the paper: Adagrad with a learning rate of 0.01 halved every 30 epochs, 200 epochs, seed 3407, and four GPUs.

```bash
python train.py                                  # IRSTD-1K
python train.py --dataset sirst                  # NUAA-SIRST, 4:1 split
python train.py --dataset NUDT-SIRST --aug light # NUDT-SIRST
python train.py --w-tbs 0                        # without TBS
```

## Evaluation

```bash
python test.py --run-dir result/IRSTD-1k/<run> --checkpoint latest.pth
```

The model is rebuilt from `<run>/run_args.json`.

## Notes

- The class of the network in `model_base.py` is named `ISNet` for compatibility with the code base.
- `pytorch_wavelets` is required for the Haar downsampling. Without it, an average pooling fallback changes the model.
- The trained weights and the per-image predictions will be added to this repository.

## Acknowledgment

This code builds on the implementations of [MMLNet](https://github.com/qianngli/MMLNet) and [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD).
