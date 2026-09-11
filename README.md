# HERA-Net

Repository for **HERA-Net: Hierarchical evidence reasoning and attribution for infrared small target detection**.

## Release status

This initial source release contains the model and loss implementations:

- `model_base.py`: network architecture, including BREB and JEPF.
- `loss.py`: background reference and hierarchical evidence attribution losses.

The architecture builds on MMLNet. The model entry point retains the class name `ISNet` used by the local implementation.

Training and evaluation utilities, pretrained models, and dataset access instructions will be added in a later release.

## Dependencies

The two source files require PyTorch. Install a PyTorch build appropriate for your operating system and CUDA version using the [official installation guide](https://pytorch.org/get-started/locally/).
