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
import ast

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--nchannels', type=int, default=8, help='Number of hidden channels in model')
    parser.add_argument('--use_cuda', action='store_true', help='Force use of CUDA if available')
    parser.add_argument('--rcut', type=float, help='cutoff for the ML')
    parser.add_argument('--patience', type=int, default=1, help='patience interval for scheduler')
    parser.add_argument('--restart', type=str2bool, default=False, help='restart the training from last iteration')
    parser.add_argument('--lr', type=float, default=1e-2, help='starting learning rate')
    parser.add_argument('--lr_ratio', type=float, default=100, help='ratio of the final lr')
    parser.add_argument('--mlp', type=str, default=None, help='architecture of the MLP, default [64, 64, 64]')
    # TODO: MODIFY RESTART IN ORDER TO INCLUDE ALSO LR, OPTIMIZER AND SCHEDULER

    args = parser.parse_args()

    if args.mlp is None:
        mlp = None
    else:
        mlp = ast.literal_eval(args.mlp)

    data_path_list = args.data_path.split(':')

    db = read(data_path_list[0], ':500')

    if len(data_path_list) > 1:
        for d in data_path_list[1:]:
            db = db + read(d, ':500')

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    lr = {'SGD': 1e-4,
          'adam': args.lr
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
                                                                                     threshold=args.lr/args.lr_ratio,
                                                                                     factor=args.lr_ratio **
                                                                                            (- (args.patience + 2) /
                                                                                             args.epochs),
                                                                                     patience=args.patience
                                                                                     ),
                 "START_FINE": START_FINE
                 }

    dataset = EquivariantMatrixDataset(db,
                                       pol_cut_num=6,
                                       nbessel=8,
                                       rcut=args.rcut,
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
                              batch_size=args.batch_size,
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

    if args.restart:
        print('restart', args.restart)
        model = torch.load('checkpoint_final.pth', weights_only=False)
    else:
        model = SymmetricMatrixRegressor(nbessel=dataset.nbessel,
                                         zlist=dataset.z_table,
                                         nchannels=args.nchannels,
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
                                         irreps_sh=dataset.irreps_sh,
                                         mlp=mlp
                                         )

    logging.info(f"{device}")

    model = model.to(device)

    nntrain(model=model,
            loader=train_loader,
            val_loader=validation_loader,
            test_loader=test_loader,
            NEPOCHS=args.epochs,
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
