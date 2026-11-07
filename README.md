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

## Trained weights

The weights are attached to the [v1.0 release](https://github.com/BlcakWarderZZ/BRENet/releases/tag/v1.0). Each zip holds `run_args.json` and `checkpoints/weights.pth`. The numbers below were obtained by running `test.py` on each unpacked zip at a threshold of 0.5.

| File | Dataset | TBS | Epoch | IoU (%) | nIoU (%) | Pd (%) | Fa (10^-6) |
|---|---|---|---|---|---|---|---|
| `BRENet_IRSTD-1K.zip` | IRSTD-1K | yes | highest test IoU | 66.25 | 66.77 | 95.62 | 15.54 |
| `BRENet_IRSTD-1K_final.zip` | IRSTD-1K | yes | final | 65.35 | 66.84 | 94.28 | 15.75 |
| `BRENet_wo_TBS_IRSTD-1K.zip` | IRSTD-1K | no | highest test IoU | 66.38 | 67.74 | 93.27 | 8.69 |
| `BRENet_wo_TBS_IRSTD-1K_final.zip` | IRSTD-1K | no | final | 66.03 | 67.50 | 93.60 | 10.27 |
| `BRENet_NUAA-SIRST.zip` | NUAA-SIRST | yes | final | 79.34 | 75.26 | 100.00 | 3.90 |
| `BRENet_wo_TBS_NUAA-SIRST.zip` | NUAA-SIRST | no | final | 77.42 | 73.61 | 99.06 | 2.13 |
| `BRENet_NUDT-SIRST.zip` | NUDT-SIRST | yes | highest test IoU | 88.04 | 90.22 | 98.52 | 14.78 |
| `BRENet_NUDT-SIRST_final.zip` | NUDT-SIRST | yes | final | 87.84 | 90.13 | 98.52 | 18.38 |
| `BRENet_wo_TBS_NUDT-SIRST_final.zip` | NUDT-SIRST | no | final | 87.42 | 89.53 | 98.62 | 17.24 |

As stated in the paper, BRENet is compared at the epoch with the highest test IoU on IRSTD-1K and NUDT-SIRST, where the compared weights are those released by their authors, and at the final epoch on NUAA-SIRST, where every method is retrained. The final-epoch weights of the first two datasets are provided as well.

Unpack a zip and run:

```bash
python test.py --dataset IRSTD-1k --run-dir BRENet_IRSTD-1K --checkpoint weights.pth
python test.py --dataset sirst --data_root datasets/sirst_aug --run-dir BRENet_NUAA-SIRST --checkpoint weights.pth
python test.py --dataset NUDT-SIRST --run-dir BRENet_NUDT-SIRST --checkpoint weights.pth
```

The Fa of the paper is the `Fa_unm_x1e6` column, which counts the pixels of unmatched predicted components.

## Evaluation

```bash
python test.py --run-dir result/IRSTD-1k/<run> --checkpoint latest.pth
```

The model is rebuilt from `<run>/run_args.json`.

## Notes

- The class of the network in `model_base.py` is named `ISNet` for compatibility with the code base.
- `pytorch_wavelets` is required for the Haar downsampling. Without it, an average pooling fallback changes the model.

## Acknowledgment

This code builds on the implementations of [MMLNet](https://github.com/qianngli/MMLNet) and [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD).
