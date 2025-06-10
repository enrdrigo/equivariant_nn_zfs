# Equivariant Neural Networks for Zero-Field Splitting (ZFS) Prediction

## Overview

equivariant_nn_zfs is a PyTorch-based deep learning framework implementing equivariant neural networks designed to predict tensor properties related to zero-field splitting (ZFS) in materials or molecules.
Leveraging equivariant representations (via e3nn), this repository focuses on accurate, symmetry-aware modeling of tensorial data.

## 🚀 Features

Equivariant architecture utilizing spherical harmonics and irreducible representations (Irreps).
Custom radial and angular embeddings tailored for three-body interactions.
Modular design with configurable radial basis functions and cutoff functions.
Separate logging for training and validation with detailed component-wise losses.

- Tensor regression model using E(3)-equivariant operations
- Flexible training with:
  - Cyclic training over data segments
  - Adaptive learning rate scheduling
  - Model checkpointing & logging
- Support for `.extxyz` molecular datasets (via ASE)
- Automatic batching, segmenting, and device selection
- Minimal dataset interface (`MinimalDataset`)
- Fine-grained logging (train/validation/test losses separately)

## 📦 Installation
```
git clone https://github.com/enrdrigo/equivariant_nn_zfs.git
cd equivariant_nn_zfs
```
Dependencies include:

```
Python 3.8+
torch
e3nn
mace  # for radial and product modules
```

---

## 📁 Repository Structure
```
equivariant_nn_zfs/
│
├── train/                  # Training loop and validation logic
│   └── train.py
├── model/                  # Equivariant tensor regression model
│   └── model.py
├── dataset/                # Dataset and descriptor processing
│   └── dataset.py
├── tools/                # tools for the evaluation of the model
│   └── embedding.py
│   └── prod.py
│   └── contract.py
├── d_test.extxyz             # Example test dataset
├── d_train.extxyz            # Example training dataset
└── README.md
```
---
Usage

```
python main.py \
  --data_path train.extxyz \
  --data_test_path test.extxyz \
  --epochs 1000 \
  --rcut 6.0 \
  --nchannels 128 \
  --num_segments 5 \
  --batch_size 32 \
  --use_cuda \
  --mlp "[64, 64, 64]"
```
Optional Arguments

- ``--restart``: Resume from checkpoint
- ``--patience``: Patience for LR scheduler
- ``--lr``, ``--lr_factor``, ``--min_lr``: Learning rate controls
- ``--num_segments``, ``--len_segment``: For cyclic training over subsets
- ``--max_l_hidden``: Maximum l for spherical harmonics in hidden layers

## 📊 Outputs
- ``checkpoint_model.pth``: Latest model weights
- ``checkpoint.pth``: Full training state (model + optimizer + scheduler)
- ``training.log``, ``validation.log``, ``testing.log``: Detailed logs


Code Structure

- model.py: Contains the TensorRegressor class implementing the equivariant neural network.
- train.py: Training and evaluation loop implementations with logging and checkpointing.

Contributions and issues are welcome! Please open a pull request or issue on GitHub.

