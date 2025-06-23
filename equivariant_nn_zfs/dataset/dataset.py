import torch
from torch.utils.data import Dataset
from mace import data, tools
from equivariant_nn_zfs.tools.convert_matrix import cartesian_to_spherical_irreps
from mace.tools.torch_geometric import Batch
from mace.tools.torch_geometric.data import Data
from equivariant_nn_zfs.model.model import TensorRegressor
from mace.tools import AtomicNumberTable


class MinimalDataset(Dataset):

    def __init__(self,
                 structures,
                 irreps_out,
                 radial_cutoff,
                 device,
                 z_table=None
                 ):
        self.radial_cutoff = radial_cutoff
        self.structures = structures
        self.irreps_out = irreps_out
        self.targets = torch.stack([cartesian_to_spherical_irreps(torch.tensor(s.info['target_L2'].reshape(3, 3)),
                                                                  irreps=irreps_out) for s in structures], dim=0)
        self.device = device

        if z_table is None:
            z_table = set()
            for s in structures:
                s_z_table = s.get_atomic_numbers()
                z_table.update(s_z_table)
            self.z_table = tools.AtomicNumberTable(list(z_table))
        else:
            self.z_table = tools.AtomicNumberTable(z_table)

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        struct = self.structures[idx]

        config = data.Configuration(
            atomic_numbers=struct.numbers,
            positions=struct.positions,
            cell=struct.cell,
            pbc=struct.pbc,
            properties={'positions': 'positions'},
            property_weights={'positions': 1}
        )

        # we handle configurations using the Data class
        atomic = data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.radial_cutoff)

        # Convert atomic → torch_geometric.data.Data
        batch = Data(
            edge_index=atomic.edge_index,
            pos=atomic.positions,
            node_attrs=atomic.node_attrs,
            shifts=atomic.shifts,
            batch=None  # gets set by Batch.from_data_list
        )

        target = self.targets[idx]

        return batch, target

class EvaluationDataset(Dataset):

    def __init__(self,
                 structures,
                 model: TensorRegressor
                 ):
        self.radial_cutoff = model.radial_cutoff
        self.structures = structures
        self.device = model.device
        self.z_table = model.z_table
        assert isinstance(self.z_table, AtomicNumberTable), "z_table must be an instance of AtomicNumberTable"

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        struct = self.structures[idx]

        config = data.Configuration(
            atomic_numbers=struct.numbers,
            positions=struct.positions,
            cell=struct.cell,
            pbc=struct.pbc,
            properties={'positions': 'positions'},
            property_weights={'positions': 1}
        )

        # we handle configurations using the Data class
        atomic = data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.radial_cutoff)

        # Convert atomic → torch_geometric.data.Data

        return Data(
            edge_index=atomic.edge_index,
            pos=atomic.positions,
            node_attrs=atomic.node_attrs,
            shifts=atomic.shifts
        )

def collate_fn(batch_):
    """
    Custom collate function to handle variable-length descriptors in the batch.
    """
    batches, targets_ = zip(*batch_)

    # We can't stack the descriptors directly because they have different sizes
    # Instead, we keep them in a list
    targets_ = torch.stack(targets_)

    return {'batches': Batch.from_data_list(list(batches)), 'targets': targets_}

def collate_fn_eval(batch_):
    """
    Custom collate function to handle variable-length descriptors in the batch.
    """

    # We can't stack the descriptors directly because they have different sizes
    # Instead, we keep them in a list

    return {'batches': Batch.from_data_list(list(batch_))}
