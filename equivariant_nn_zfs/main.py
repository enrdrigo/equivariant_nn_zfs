import mace
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
import numpy as np
from ase.io import read
from e3nn.o3 import Irreps
from equivariant_nn_zfs.train.train import train
from equivariant_nn_zfs.model.model import TensorRegressor
from equivariant_nn_zfs.dataset.dataset import MinimalDataset
import random
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


def collate_fn(batch_):
    """
    Custom collate function to handle variable-length descriptors in the batch.
    """
    batches, targets_ = zip(*batch_)

    # We can't stack the descriptors directly because they have different sizes
    # Instead, we keep them in a list
    targets_ = torch.stack(targets_)

    return {'batches': list(batches), 'targets': targets_}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an equivariant neural network for ZFS prediction.")

    parser.add_argument('--data_path', type=str, default='train.extxyz', help='Path to input train EXTXYZ file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--nchannels', type=int, default=128, help='Number of hidden channels in model')
    parser.add_argument('--use_cuda', action='store_true', help='Force use of CUDA if available')
    parser.add_argument('--rcut', type=float, help='cutoff for the ML')
    parser.add_argument('--patience', type=int, default=60, help='patience interval for scheduler')
    parser.add_argument('--restart', type=str2bool, default=False, help='restart the training from last iteration')
    parser.add_argument('--lr', type=float, default=1e-3, help='starting learning rate')
    parser.add_argument('--lr_factor', type=float, default=0.75, help='ratio of the final lr')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='ratio of the final lr')
    parser.add_argument('--mlp', type=str, default=None, help='architecture of the MLP, default [64, 64, 64]')
    parser.add_argument('--data_test_path', type=str, default='test.extxyz', help='Path to input test EXTXYZ file')
    # TODO: MODIFY RESTART IN ORDER TO INCLUDE ALSO LR, OPTIMIZER AND SCHEDULER

    args = parser.parse_args()

    if args.mlp is None:
        mlp = None
    else:
        mlp = ast.literal_eval(args.mlp)

    data_path_list = args.data_path.split(':')

    db = read(data_path_list[0], ':')

    if len(data_path_list) > 1:
        for d in data_path_list[1:]:
            db = db + read(d, ':')

    db_test = read(args.data_test_path, ':10')

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
                                                                                    min_lr=1e-5,
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
                                                                                     min_lr=args.min_lr,
                                                                                     factor=args.lr_factor,
                                                                                     patience=args.patience
                                                                                     ),
                 "START_FINE": START_FINE
                 }

    dataset = MinimalDataset(db,
                             device=device,
                             irreps_out=Irreps('2e'),
                             radial_cutoff=args.rcut
                             )

    dataset_test = MinimalDataset(db_test,
                                  device=device,
                                  irreps_out=Irreps('2e'),
                                  radial_cutoff=args.rcut
                                  )

    total_size = len(dataset)
    validation_ratio = 0.05

    # Calculate split sizes

    validation_size = int(validation_ratio * total_size)

    train_size = total_size  - validation_size  # ensures all data is used

    print([train_size,  validation_size])

    generator = torch.Generator().manual_seed(1234)

    # Randomly split
    train_data, validation_data = random_split(dataset,
                                               [train_size, validation_size],
                                               generator=generator
                                               )

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn
                              )

    test_loader = DataLoader(dataset_test,
                             batch_size=1,
                             collate_fn=collate_fn
                             )

    validation_loader = DataLoader(validation_data,
                                   batch_size=1,
                                   collate_fn=collate_fn
                                   )

    if args.restart:
        print('restart', args.restart)
        model = torch.load('checkpoint_model_final.pth', weights_only=False)
    else:
        model = TensorRegressor(radial_cutoff=args.rcut,
                                pol_cut_num=5,
                                n_bessel=8,
                                zlist=dataset.z_table,
                                n_channels=args.nchannels,
                                weights=[1,
                                         1,
                                         1,
                                         1,
                                         1
                                         ],
                                device=device,
                                irreps_sh=Irreps('0e + 1o +2e'),
                                mlp=mlp,
                                irreps_out=dataset.irreps_out
                                )

    logging.info(f"{device}")

    model = model.to(device)

    train(model=model,
          loader=train_loader,
          val_loader=validation_loader,
          test_loader=test_loader,
          nepochs=args.epochs,
          start_dyn=start_dyn,
          fine_dyn=fine_dyn
          )

    print("\nEvaluating on test set:")

    model.eval()
    errors = []
    # Extracting true and predicted inertia tensor components
    with torch.no_grad():
        for batches in test_loader:
            batch_data = batches['batches']

            y_true = batches['targets']

            y_pred = model(batch_data)

            mse = model.loss_fn(y_pred, y_true).item()

            errors.append(mse)

    mean_mse = np.mean(errors)
    print(f"Test Mean Squared Error (MSE): {mean_mse:.6f}")
