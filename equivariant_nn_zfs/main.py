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
# from equivariant_nn_zfs.tools.utils_readout import augment_data
import ast
import sys

# Logger that logs only to stdout
console_logger = logging.getLogger('console_logger')
console_logger.setLevel(logging.INFO)
console_logger.propagate = False  # prevent message propagation
console_handler = logging.StreamHandler(sys.stdout)

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
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training, default is epochs * #train_set / 100000 + 1')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--nchannels', type=int, default=128, help='Number of hidden channels in model')
    parser.add_argument('--use_cuda', action='store_true', help='Force use of CUDA if available')
    parser.add_argument('--rcut', type=float, help='cutoff for the ML')
    parser.add_argument('--patience', type=int, default=60, help='patience interval for scheduler')
    parser.add_argument('--restart', type=str2bool, default=False, help='restart the training from last iteration')
    # parser.add_argument('--augment_data', type=str2bool, default=False, help='augment training data')
    parser.add_argument('--lr', type=float, default=1e-3, help='starting learning rate')
    parser.add_argument('--lr_factor', type=float, default=0.75, help='ratio of the final lr')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='ratio of the final lr')
    parser.add_argument('--mlp', type=str, default=None, help='architecture of the MLP, default [64, 64, 64]')
    parser.add_argument('--data_test_path', type=str, default='test.extxyz', help='Path to input test EXTXYZ file')
    parser.add_argument('--pol_cut_num', type=int, default=5, help='number of cutoff polynomials in the descriptor')
    parser.add_argument('--n_bessel', type=int, default=8, help='number of bessel polynomials in the descriptor')
    parser.add_argument('--number_of_centers', type=int, default=1, help='number of centers for the selection of active atoms')
    parser.add_argument('--length_train_segments', type=int, default=None, help='length of training segments')
    parser.add_argument('--max_l_hidden', type=int, default=2, help='max l in hidden irreps')
    parser.add_argument('--length_test', type=str, default='', help='length of test set')
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

    # if args.augment_data:
    #     db = augment_data(db)

    if args.length_train_segments is not None:
        len_segment = args.length_train_segments
        num_segments = len(db)//len_segment
    else:
        num_segments = 1
        len_segment = len(db)

    db_test = read(args.data_test_path, ':'+args.length_test)

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    dataset = MinimalDataset(db,
                             device=device,
                             rule='ij=ji',
                             radial_cutoff=args.rcut
                             )

    dataset_test = MinimalDataset(db_test,
                                  device=device,
                                  rule='ij=ji',
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

        num_centers = checkpoint['num_centers']
    else:
        model = TensorRegressor(radial_cutoff=args.rcut,
                                pol_cut_num=args.pol_cut_num,
                                n_bessel=args.n_bessel,
                                zlist=dataset.z_table,
                                n_channels=args.nchannels,
                                device=device,
                                irreps_sh=Irreps([(1, (l, int(-2*(l%2-0.5)))) for l in range(args.max_l_hidden+1)]),
                                mlp=mlp,
                                irreps_out=Irreps(str(dataset.irreps_out)),
                                number_of_centers=args.number_of_centers,
                                )

        start_epoch = 0

        optimizer = optimizer_lambda(model.parameters())

        scheduler = scheduler_lambda(optimizer)

    console_logger.info("\n🔧 Training Configuration:")
    for key, value in vars(args).items():
        console_logger.info(f"{key:24s}: {value}")

    console_logger.info(f"{'test set size':24s}: {len(dataset_test)}")

    console_logger.info(f"{'train set size':24s}: {len(train_data)}")

    console_logger.info(f"{'validation set size':24s}: {len(validation_data)}")

    console_logger.info(f"{'species':24s}: {str(dataset.z_table)}")

    console_logger.info(f"{'device':24s}: {device}")

    console_logger.info(f"{'# of parameters updates':24s}: {int(len(train_data)/args.batch_size/num_segments*args.epochs)}")

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
          num_segments=num_segments,
          seed=args.seed
          )
