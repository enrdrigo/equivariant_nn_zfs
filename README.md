# README GENERATED VIA AI! 
# TODO: FIX IT BETTER

Equivariant Matrix Regressor for Spherical Tensor Prediction

This repository contains code to train an equivariant neural network that predicts 3×3 symmetric matrices (converted to spherical tensor components) from atomic structures using E(3)-equivariant neural network techniques.

The model is particularly suited for learning tensorial properties (e.g., quadrupole tensors, inertia tensors) in a physically consistent and symmetry-aware manner.

📦 Features

Custom torch.utils.data.Dataset to handle atomic structures and transform tensors to spherical harmonics.
E(3)-equivariant neural network architecture with radial/angular embeddings.
Modular training framework with weighted loss and parity plotting.
Compatible with atomic simulation environments (ASE, MACE, e3nn).
Built-in utilities to:
Convert Cartesian tensors to spherical harmonics.
Visualize learning progress and prediction parity.

📁 Structure

EquivariantMatrixDataset: Loads atomic structures and creates edge/node attributes and targets in spherical harmonics.
SymmetricMatrixRegressor: Neural network built from radial/angular embeddings and E(3)-equivariant layers.
Training loop: Implements weighted MSE loss with per-component tracking and visualization.
Evaluation: Includes parity plotting between predicted and true tensor components.

🧪 Requirements

dataset_pol_L2.extxyz: The input structure dataset (in ASE format).

🚀 Usage

Prepare your dataset:
The dataset must be an .extxyz file with an attribute target_L2 in each structure's info dictionary, storing the 3×3 tensor to be predicted.
Run training:
python main.py
This script will:

Load and preprocess the data.
Train the model on 80% of the dataset.
Evaluate on the remaining 20%.
Custom Model Settings (optional):
Adjust parameters such as:

pol_cut_num=6
nbessel=8
rcut=5.0
nchannels=4
irreps_sh=Irreps('0e + 1o + 2e')
to tune model complexity and radial/angular resolution.
