import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from ase.io import read
from mace import data, modules, tools
from e3nn.o3 import Irreps
from equivariant_nn_zfs.train.train import nntrain
from equivariant_nn_zfs.model.model import SymmetricMatrixRegressor
from equivariant_nn_zfs.dataset.dataset import EquivariantMatrixDataset

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
    db = read('dataset_pol_L2.extxyz', ':500')

    batch_size = 1

    NEPOCHS = 20

    nchannels = 2

    lr = 1e-1

    dataset = EquivariantMatrixDataset(db,
                                       pol_cut_num=6,
                                       nbessel=8,
                                       rcut=5.0,
                                       irreps_sh=Irreps('0e + 1o + 2e + 3o')
                                       )


    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True
                        )

    total_size = len(dataset)
    train_ratio = 0.8
    validation_ratio = 0.05

    # Calculate split sizes
    train_size = int(train_ratio * total_size)

    validation_size = int(validation_ratio * total_size)

    test_size = total_size - train_size - validation_size# ensures all data is used

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
                             batch_size=batch_size,
                             collate_fn=collate_fn
                             )

    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   collate_fn=collate_fn
                                   )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SymmetricMatrixRegressor(nbessel=dataset.nbessel,
                                     zlist=dataset.z_table,
                                     nchannels=nchannels,
                                     lr=lr,
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
    nntrain(model=model,
            loader=train_loader,
            val_loader=validation_loader,
            NEPOCHS=NEPOCHS
            )
    model.eval()
    errors = []
    # Extracting true and predicted inertia tensor components
    true = []
    pred = []
    print("\nEvaluating on test set:")
    with torch.no_grad():
        for X, X_v, node_attr, edge_index,  Y_true in test_loader:
            Y_pred = model(X, X_v, node_attr, edge_index)
            mse = model.loss_fn(Y_pred, Y_true).item()
            errors.append(mse)
            true.append(Y_true.view(-1).numpy())  # Convert to numpy array
            pred.append(Y_pred.view(-1).numpy())  # Convert to numpy array

    mean_mse = np.mean(errors)
    print(f"Test Mean Squared Error (MSE): {mean_mse:.6f}")

    true = np.array(true)  # Remove unnecessary dimensions
    pred = np.array(pred)  # Remove unnecessary dimensions

    # Labels for each component of the inertia tensor
    labels = [r'$Y^0_0$', r'$Y^1_{-1}$', r'$Y^1_0$', r'$Y^1_1$',
              r'$Y^2_{-2}$', r'$Y^2_{-1}$', r'$Y^2_0$', r'$Y^2_1$', r'$Y^2_1$']