import mace
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
import numpy as np
from ase.io import read
from e3nn.o3 import Irreps
from equivariant_nn_zfs.train.train import nntrain
from equivariant_nn_zfs.model.model import SymmetricMatrixRegressor
from equivariant_nn_zfs.dataset.dataset import EquivariantMatrixDataset

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)


def collate_fn(batch):
    """
    Custom collate function to handle variable-length descriptors in the batch.
    """
    vectors, lengths, nodeattr, edgeindex, targets = zip(*batch)

    # We can't stack the descriptors directly because they have different sizes
    # Instead, we keep them in a list
    targets = torch.stack(targets)

    return list(vectors), list(lengths), list(nodeattr), list(edgeindex), targets


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an equivariant neural network for ZFS prediction.")

    parser.add_argument('--data_path', type=str, default='train.extxyz', help='Path to input EXTXYZ file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--nchannels', type=int, default=8, help='Number of hidden channels in model')
    parser.add_argument('--use_cuda', action='store_true', help='Force use of CUDA if available')

    args = parser.parse_args()

    db = read(args.data_path, ':')

    batch_size = args.batch_size

    epochs = args.epochs

    nchannels = args.nchannels

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    lr = {'SGD': 1e-4,
          'adam': 1e-2
          }

    START_FINE = -1

    fine_dyn = {"optimizer": lambda params: optim.SGD(params,
                                                      lr=lr['SGD'],
                                                      momentum=0.2,
                                                      weight_decay=5e-7,
                                                      dampening=1e-4
                                                      ),
                "scheduler": lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                    mode='min',
                                                                                    threshold=1e-5,
                                                                                    factor=0.7,
                                                                                    patience=1
                                                                                    ),
                "START_FINE": START_FINE
                }

    start_dyn = {"optimizer": lambda params: optim.AdamW(params,
                                                         lr=lr['adam'],
                                                         weight_decay=5e-7
                                                         ),
                 "scheduler": lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                     mode='min',
                                                                                     threshold=1e-4,
                                                                                     factor=0.95,
                                                                                     patience=0
                                                                                     ),
                 "START_FINE": START_FINE
                 }

    dataset = EquivariantMatrixDataset(db,
                                       pol_cut_num=6,
                                       nbessel=8,
                                       rcut=5.0,
                                       irreps_sh=Irreps('0e + 1o + 2e'),
                                       device=device
                                       )

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True
                        )

    total_size = len(dataset)
    test_ratio = 0.1
    validation_ratio = 0.1

    # Calculate split sizes
    test_size = int(test_ratio * total_size)

    validation_size = int(validation_ratio * total_size)

    train_size = total_size - test_size - validation_size  # ensures all data is used

    print([train_size, test_size, validation_size])

    # Randomly split
    train_data, test_data, validation_data = random_split(dataset,
                                                          [train_size, test_size, validation_size]
                                                          )

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn
                              )

    test_loader = DataLoader(test_data,
                             batch_size=1,
                             collate_fn=collate_fn
                             )

    validation_loader = DataLoader(validation_data,
                                   batch_size=1,
                                   collate_fn=collate_fn
                                   )

    model = SymmetricMatrixRegressor(nbessel=dataset.nbessel,
                                     zlist=dataset.z_table,
                                     nchannels=nchannels,
                                     weights=[1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1
                                              ],
                                     device=device,
                                     irreps_sh=dataset.irreps_sh
                                     )

    logging.info(f"{device}")

    model = model.to(device)

    nntrain(model=model,
            loader=train_loader,
            val_loader=validation_loader,
            test_loader=test_loader,
            NEPOCHS=epochs,
            start_dyn=start_dyn,
            fine_dyn=fine_dyn
            )

    print("\nEvaluating on test set:")

    model.eval()
    errors = []
    # Extracting true and predicted inertia tensor components
    with torch.no_grad():
        for X, X_v, node_attr, edge_index,  Y_true in test_loader:
            Y_pred = model(X, X_v, node_attr, edge_index)
            mse = model.loss_fn(Y_pred, Y_true).item()
            errors.append(mse)

    mean_mse = np.mean(errors)
    print(f"Test Mean Squared Error (MSE): {mean_mse:.6f}")
