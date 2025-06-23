from equivariant_nn_zfs.dataset.dataset import EvaluationDataset, collate_fn_eval
from equivariant_nn_zfs.model.model import TensorRegressor
from torch.utils.data import DataLoader
import torch
from ase.io import read,write
from tqdm import tqdm
from e3nn.o3 import Irreps

from equivariant_nn_zfs.tools.convert_matrix import cartesian_to_spherical_irreps


def evaluating(data: list,
               model: TensorRegressor,
               radial_cutoff=None,
               irreps_out = None,
               do_irreps = False,
               batch_size=10,
               num_workers=4,
               path_to_evaluate=None,
               ):

    if path_to_evaluate is None:
        path_to_evaluate = 'evaluate_dataset.extxyz'

    if irreps_out is None:
        try:
            irreps_out = model.irreps_out
        except: do_irreps = False
    else: do_irreps = True

    assert isinstance(irreps_out, Irreps), "irreps_out must be an instance of Irreps"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model.device = device

    dataset = EvaluationDataset(data,
                                model=model,
                                radial_cutoff=radial_cutoff
                                )
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=collate_fn_eval
                        )

    y_pred_tot=[]
    y_pred_local=[]

    with torch.no_grad():
        for batches in tqdm(loader, desc="Evaluating"):
            batches_data = batches['batches'].to(model.device)

            y_pred= model(batches_data)  # Forward pass

            y_pred_tot.append(y_pred['target'][:, :].cpu())
            y_pred_local.append(y_pred['local target'][:, :].cpu())

    y_pred_tot = torch.cat(y_pred_tot, dim=0)
    y_pred_local = torch.cat(y_pred_local, dim=0)

    start_idx=0
    for idx, data_ in enumerate(data):
        end_idx=start_idx+len(data_)
        data[idx].info['eval_target']=y_pred_tot[idx].numpy()
        if do_irreps:
            data[idx].info['true_target'] = cartesian_to_spherical_irreps(data[idx].info['target_L2'].reshape(3, 3),
                                                                          irreps=irreps_out).numpy()
        data[idx].arrays['eval_local_target'] = y_pred_local[start_idx:end_idx].numpy()
        data[idx].arrays['eval_local_norm_target']=y_pred_local[start_idx:end_idx].norm(dim=1).numpy()
        start_idx=end_idx

    write(path_to_evaluate,data)

    return