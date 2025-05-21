import torch
from torch.utils.data import Dataset
from mace import data, modules, tools
from e3nn.o3 import SphericalHarmonics
from mace.modules.radial import BesselBasis
from mace.modules.radial import PolynomialCutoff
from e3nn.io import CartesianTensor
from equivariant_nn_zfs.tools.convert_matrix import cartesian_to_spherical_irreps
import warnings


class EquivariantMatrixDataset(Dataset):

    def __init__(self,
                 structures,
                 pol_cut_num,
                 nbessel,
                 rcut,
                 irreps_sh,
                 device
                 ):
        self.structures = structures
        self.pol_cut_num = pol_cut_num
        self.nbessel = nbessel
        self.rcut = rcut
        self.irreps_sh = irreps_sh
        cartesian = CartesianTensor('ij=ij')
        #  warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._check")
        self.targets = torch.stack([cartesian_to_spherical_irreps(torch.tensor(s.info['target_L2'].reshape(3, 3))) for s in structures], dim=0)
        self.device = device

        z_table = set()
        for s in structures:
            s_z_table = s.get_atomic_numbers()
            z_table.update(s_z_table)
        self.z_table = tools.AtomicNumberTable(list(z_table))

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        struct = self.structures[idx]

        config = data.Configuration(
            atomic_numbers=struct.numbers,
            positions=struct.positions,
            properties={'positions': 'positions'},
            property_weights={'positions': 1}
        )

        # we handle configurations using the AtomicData class
        batch = data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.rcut)

        vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
            positions=batch["positions"],
            edge_index=batch["edge_index"],
            shifts=batch["shifts"],
        )

        node_attr = batch.node_attrs

        edge_index = batch.edge_index

        node_attr_len = vectors.shape[0]

        cutoff = PolynomialCutoff(r_max=self.rcut, p=self.pol_cut_num)

        bf = BesselBasis(r_max=self.rcut, num_basis=self.nbessel)

        spherical_harmonics = SphericalHarmonics(irreps_in='1o', irreps_out=self.irreps_sh, normalize=True)

        vector_descriptor = spherical_harmonics(vectors)

        length_descriptor = cutoff(lengths) * bf(lengths)

        target = self.targets[idx]

        return length_descriptor, vector_descriptor, node_attr, edge_index, target
