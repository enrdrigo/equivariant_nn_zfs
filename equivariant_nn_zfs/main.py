import mace
import argparse
import torch
import torch.optim as optim
from torch.utils.data import random_split
import logging
from ase.io import read
from e3nn.o3 import Irreps
from equivariant_nn_zfs.train.train import train
from equivariant_nn_zfs.model.model import TensorRegressor
from equivariant_nn_zfs.dataset.dataset import MinimalDataset
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an equivariant neural network for ZFS prediction.")

    parser.add_argument('--data_path', type=str, default='train.extxyz', help='Path to input train EXTXYZ file')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training, default is epochs * #train_set / 100000 + 1')
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
    parser.add_argument('--pol_cut_num', type=int, default=5, help='number of cutoff polynomials in the descriptor')
    parser.add_argument('--n_bessel', type=int, default=8, help='number of bessel polynomials in the descriptor')
    parser.add_argument('--max_l_hidden', type=int, default=2, help='max l in hidden irreps')
    parser.add_argument('--num_segments', type=int, default=None, help='number of segments of train set')
    parser.add_argument('--seed', type=int, default=123456789, help='seed')

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

    if args.num_segments is None:
        args.num_segments = min(len(db) // 500, args.patience) + 1

    if args.batch_size is None:
        args.batch_size = int(args.epochs // args.num_segments * len(db) / 200000 + 1)

    db_test = read(args.data_test_path, ':')

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

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

    generator = torch.Generator().manual_seed(args.seed)

    # Randomly split
    train_data, validation_data = random_split(dataset,
                                               [train_size, validation_size],
                                               generator=generator
                                               )

    optimizer_lambda = lambda parameters : optim.AdamW(parameters,
                                                       lr=args.lr,
                                                       weight_decay=5e-7
                                                       )

    scheduler_lambda = lambda optimizer_ : optim.lr_scheduler.ReduceLROnPlateau(optimizer_,
                                                                                mode='min',
                                                                                min_lr=args.min_lr,
                                                                                factor=args.lr_factor,
                                                                                patience=args.patience
                                                                                )

    if args.restart:
        model = torch.load('checkpoint_model.pth', weights_only=False)

        checkpoint = torch.load('checkpoint.pth', map_location=device)

        optimizer = optimizer_lambda(model.parameters())

        scheduler = scheduler_lambda(optimizer)

        optimizer.load_state_dict(checkpoint['optimizer_state'])

        scheduler.load_state_dict(checkpoint['scheduler_state'])

        start_epoch = checkpoint['epoch'] + 1
    else:
        model = TensorRegressor(radial_cutoff=args.rcut,
                                pol_cut_num=args.pol_cut_num,
                                n_bessel=args.n_bessel,
                                zlist=dataset.z_table,
                                n_channels=args.nchannels,
                                weights=[1,
                                         1,
                                         1,
                                         1,
                                         1
                                         ],
                                device=device,
                                irreps_sh=Irreps([(1, (l, int(-2*(l%2-0.5)))) for l in range(args.max_l_hidden+1)]),
                                mlp=mlp,
                                irreps_out=dataset.irreps_out
                                )

        start_epoch = 0

        optimizer = optimizer_lambda(model.parameters())

        scheduler = scheduler_lambda(optimizer)

    logging.info("\n🔧 Training Configuration:")
    for key, value in vars(args).items():
        logging.info(f"{key:24s}: {value}")

    logging.info(f"{'test set size':24s}: {len(dataset_test)}")

    logging.info(f"{'train set size':24s}: {len(train_data)}")

    logging.info(f"{'validation set size':24s}: {len(validation_data)}")

    logging.info(f"{'species':24s}: {str(dataset.z_table)}")

    logging.info(f"{'device':24s}: {device}")

    model = model.to(device)

    train(model=model,
          train_data=train_data,
          val_data=validation_data,
          test_data=dataset_test,
          n_epochs=args.epochs,
          optimizer=optimizer,
          scheduler=scheduler,
          start_epoch=start_epoch,
          batch_size=args.batch_size,
          num_segments=args.num_segments,
          seed=args.seed
          )
