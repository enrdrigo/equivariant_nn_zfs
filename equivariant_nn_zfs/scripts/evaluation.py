from equivariant_nn_zfs.dataset.dataset import EvaluationDataset, collate_fn_eval
from equivariant_nn_zfs.model.model import TensorRegressor
from torch.utils.data import DataLoader
import torch
from ase.io import read,write


def evaluating(data: list,
               model: TensorRegressor,
               radial_cutoff=None,
               batch_size=10,
               num_workers=4,
               path_to_evaluate='',
               ):
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
        for batches in loader:
            batches_data = batches['batches'].to(model.device)

            y_pred= model(batches_data)  # Forward pass

            y_pred_tot.append(y_pred['target'][:, :].cpu())
            y_pred_local.append(y_pred['local target'][:, :].cpu())

    y_pred_tot = torch.cat(y_pred_tot, dim=0).T
    y_pred_local = torch.cat(y_pred_local, dim=0).T

    start_idx=0
    for idx, data_ in enumerate(data):
        end_idx=start_idx+len(data_)
        data[idx].info['target']=y_pred_tot[idx]
        data[idx].arrays['local_target']=y_pred_local[start_idx:end_idx]
        start_idx=end_idx

    write(path_to_evaluate+'evaluate_dataset.extxyz',data)

    yield data

