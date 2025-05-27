# README GENERATED VIA AI! 
# TODO: FIX IT BETTER

Equivariant Neural Networks for Zero-Field Splitting (ZFS) Prediction

Overview

equivariant_nn_zfs is a PyTorch-based deep learning framework implementing equivariant neural networks designed to predict tensor properties related to zero-field splitting (ZFS) in materials or molecules. Leveraging group theory and equivariant representations (via e3nn), this repository focuses on accurate, symmetry-aware modeling of tensorial data.

Features

Equivariant architecture utilizing spherical harmonics and irreducible representations (Irreps).
Custom radial and angular embeddings tailored for three-body interactions.
Support for weighted MSE loss to emphasize specific tensor components.
Modular design with configurable radial basis functions and cutoff functions.
Separate logging for training and validation with detailed component-wise losses.
Utilities for training, validation, and testing with restart capability via checkpointing.
Easy switching between coarse and fine training dynamics.
Installation

git clone https://github.com/enrdrigo/equivariant_nn_zfs.git
cd equivariant_nn_zfs
pip install -r requirements.txt
Dependencies include:

Python 3.8+
PyTorch
e3nn
mace (for radial and product modules)
numpy, scipy (for numerical utilities)
Usage

Model Definition
from model import TensorRegressor
import torch
from e3nn.o3 import Irreps

model = TensorRegressor(
    nbessel=6,
    zlist=[1, 6, 8],  # Example atomic numbers
    radial_cutoff=5.0,
    pol_cut_num=5,
    nchannels=16,
    irreps_sh=Irreps("0e+1o+2e"),
    irreps_out=Irreps("2e"),
    weights=[1.0]*9
)
Training
Use the provided nntrain function for training:

nntrain(
    model=model,
    loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    nepochs=100,
    start_dyn=start_dynamics_config,
    fine_dyn=fine_dynamics_config,
    device=torch.device('cuda')
)
Logging

Training logs are saved in training.log
Validation logs are saved in validation.log
Console outputs show epoch progress, losses, and learning rates
Checkpoints & Resume Training

Checkpoints are saved as checkpoint.pth and checkpoint_final.pth.
The repository supports saving/loading of model states and optimizer/scheduler states for seamless training restarts.
Code Structure

model.py: Contains the TensorRegressor class implementing the equivariant neural network.
train.py: Training and evaluation loop implementations with logging and checkpointing.
utils.py: Helper functions and utilities.
data/: Data loaders and preprocessing (not included here, user should provide).
Contributing

Contributions and issues are welcome! Please open a pull request or issue on GitHub.

