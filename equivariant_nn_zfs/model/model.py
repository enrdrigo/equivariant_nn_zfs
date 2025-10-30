import torch
from mace.tools.torch_geometric import Batch
from torch import nn
from e3nn import o3
from e3nn.o3 import Irreps
from mace.tools import AtomicNumberTable
from equivariant_nn_zfs.tools.embedding import NodeFeaturesStart, RadialAngularEmbedding
from equivariant_nn_zfs.tools.prod import ReadoutL2, Product3body
from mace import modules
from e3nn.o3 import SphericalHarmonics
from mace.modules.radial import BesselBasis
from mace.modules.radial import PolynomialCutoff
from mace.tools.scatter import scatter_sum, scatter_mean
import logging
from torch import Tensor
from typing import List, Tuple
import sys
from equivariant_nn_zfs.tools.utils_readout import get_centers

# Logger that logs only to stdout
console_logger = logging.getLogger('console_logger_model')
console_logger.setLevel(logging.INFO)
console_logger.propagate = False  # prevent message propagation
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
console_logger.addHandler(console_handler)


# --- Tensor Regressor ---
class TensorRegressor(nn.Module):
    def __init__(self,
                 n_bessel: int,
                 zlist: AtomicNumberTable,
                 radial_cutoff: float,
                 radial_cutoff_zfs: float,
                 pol_cut_num: int,
                 n_channels: int,
                 irreps_sh: Irreps,
                 irreps_out: Irreps,
                 number_of_centers: int,
                 device=None,
                 mlp=None
                 ):
        super().__init__()
        self.device = device

        self.zlist = zlist

        self.number_of_centers = number_of_centers

        self.irreps_out = irreps_out

        self.radial_cutoff = radial_cutoff

        self.radial_cutoff_zfs = radial_cutoff_zfs

        self.args = {'n_bessel': n_bessel,
                     'zlist': zlist,
                     'radial_cutoff': radial_cutoff,
                     'radial_cutoff_zfs': radial_cutoff_zfs,
                     'pol_cut_num': pol_cut_num,
                     'n_channels': n_channels,
                     'irreps_sh': irreps_sh,
                     'irreps_out': irreps_out,
                     'number_of_centers': number_of_centers,
                     'device': device,
                     'mlp': mlp
                     }


        self.cutoff = PolynomialCutoff(r_max=radial_cutoff, p=pol_cut_num)

        self.bf = BesselBasis(r_max=radial_cutoff, num_basis=n_bessel)

        self.spherical_harmonics = SphericalHarmonics(irreps_in='1o', irreps_out=irreps_sh, normalize=True)

        if mlp is None:
            mlp = [64, 64, 64]

        node_attr_len = len(self.zlist)

        node_attr_irreps = o3.Irreps([(node_attr_len, (0, 1))])

        node_feat_irreps_start = o3.Irreps(f"{n_channels}x0e")

        hidden_irreps = (irreps_sh * n_channels).sort()[0].simplify()

        self.node_features = NodeFeaturesStart(node_attr_irreps=node_attr_irreps,
                                               node_feat_irreps=node_feat_irreps_start
                                               )

        self.radialemb = nn.ModuleList()
        self.radialemb.append(RadialAngularEmbedding(nbessel=n_bessel,
                                                     node_feat_irreps=node_feat_irreps_start,
                                                     irreps_sh=irreps_sh,
                                                     hidden_irreps=hidden_irreps,
                                                     node_attr_irreps=node_attr_irreps,
                                                     mlp=mlp
                                                     )
                              )
        self.radialemb.append(RadialAngularEmbedding(nbessel=n_bessel,
                                                     node_feat_irreps=hidden_irreps,
                                                     irreps_sh=irreps_sh,
                                                     hidden_irreps=hidden_irreps,
                                                     node_attr_irreps=node_attr_irreps,
                                                     mlp=mlp
                                                     )
                              )

        self.prod = nn.ModuleList()
        self.prod.append(Product3body(irreps_sh=irreps_sh,
                                      hidden_irreps=hidden_irreps,
                                      node_attr_irreps=node_attr_irreps,
                                      ncor=[0, 1, 2]
                                      )
                         )
        self.prod.append(Product3body(irreps_sh=irreps_sh,
                                      hidden_irreps=hidden_irreps,
                                      node_attr_irreps=node_attr_irreps,
                                      ncor=[0, 1, 2]
                                      )
                         )

        self.readout = nn.ModuleList()
        self.readout.append(ReadoutL2(hidden_irreps=hidden_irreps,
                                      out_irreps=irreps_out
                                      )
                            )
        self.readout.append(ReadoutL2(hidden_irreps=hidden_irreps,
                                      out_irreps=irreps_out
                                      )
                            )

        self.loss_fn = self.weighted_mse_loss

        self.to(self.device)

        if irreps_out == Irreps('0e+1o+2e'):
            console_logger.info(r"                          " +
                                "$Y^0_0$    " +
                                "$Y^1_{-1}$ " +
                                "$Y^1_0$    " +
                                "$Y^1_1$    " +
                                "$Y^2_{-2}$ " +
                                "$Y^2_{-1}$ " +
                                "$Y^2_0$    " +
                                "$Y^2_1$    " +
                                "$Y^2_1$"
                                )
        elif irreps_out == Irreps('2e'):
            console_logger.info(r"                          " +
                                "$Y^2_{-2}$ " +
                                "$Y^2_{-1}$ " +
                                "$Y^2_0$    " +
                                "$Y^2_1$    " +
                                "$Y^2_1$"
                                )


    def weighted_mse_loss(self,
                          pred,
                          target,
                          lambda_=5e-7
                          ):

        device = pred['pred'].device  # Get the device of prediction

        target = target.to(device)

        pred_flat = pred['pred'].view(pred['pred'].size(0), -1)
        base_flat = pred['base'].view(pred['base'].size(0), -1)
        target_flat = target.view(target.size(0), -1)
        reg=0
        for params in self.parameters(recurse=True):
            reg+=(params.flatten()**2).sum()
        loss = ((pred_flat - target_flat) ** 2).mean(axis=0) + (base_flat**2).mean(axis=0) + reg * lambda_
        return loss.mean()

    def mse_components(self,
                       pred,
                       target
                       ):

        device = pred['pred'].device  # Get the device of prediction

        target = target.to(device)

        pred_flat = pred['pred'].view(pred['pred'].size(0), -1)
        base_flat = pred['base'].view(pred['base'].size(0), -1)

        target_flat = target.view(target.size(0), -1)

        loss = ((pred_flat - target_flat) ** 2) + base_flat**2
        return loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_graph_edge_attributes(self, data: Batch):

        vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
            positions=data.pos,
            edge_index=data.edge_index,
            shifts=data.shifts,
        )

        vectors = vectors.to(self.device)

        lengths = lengths.to(self.device)

        node_attributes = data.node_attrs

        node_attributes = node_attributes.to(self.device)

        edge_index = data.edge_index

        edge_index = edge_index.to(self.device)

        vector_descriptor = self.spherical_harmonics(vectors)

        self.cutoff.r_max = self.cutoff.r_max.to(self.device)

        self.bf.r_max = self.bf.r_max.to(self.device)

        self.cutoff.p = self.cutoff.p.to(self.cutoff.r_max.device).to(torch.int)

        length_descriptor = self.cutoff(lengths) * self.bf(lengths)

        return length_descriptor, vector_descriptor, node_attributes, edge_index

    def forward(self,
                batch_data_: Tuple[Batch, Batch],
                ):

        batch_data, batch_data_zfs = batch_data_

        num_graphs = batch_data.ptr.numel() - 1

        number_nodes = batch_data.num_nodes // num_graphs

        length_b, edge_attr_b, node_attr_b, edge_index_b = self.get_graph_edge_attributes(batch_data)

        length_zfs_b, edge_attr_zfs_b, node_attr_zfs_b, edge_index_zfs_b = self.get_graph_edge_attributes(batch_data_zfs)

        node_attr_b = node_attr_b.detach().requires_grad_()

        node_features = self.node_features(node_attr_b)

        total_readout = 0

        for i in range(2):

            message, sc = self.radialemb[i](length_b,
                                            node_features,
                                            node_attr_b,
                                            edge_attr_b,
                                            edge_index_b
                                            )

            node_features = self.prod[i](message,
                                         node_attr_b,
                                         sc)

            readout = self.readout[i](node_features)

            total_readout += readout


        attention_neighbour, neighbour_center = get_centers(atomic=total_readout,
                                                            k=self.number_of_centers,
                                                            batch=batch_data,
                                                            edge_index=edge_index_zfs_b,
                                                            thr=-1
                                                            )

        not_neighbour_center = torch.ones(total_readout.size(0),
                                          dtype=torch.bool,
                                          device=total_readout.device)

        active_atoms = torch.zeros(total_readout.size(0),
                                   dtype=torch.bool,
                                   device=total_readout.device)

        active_atoms[neighbour_center] = 1

        not_neighbour_center[neighbour_center] = False

        self.attention_atoms = attention_neighbour

        self.active_atoms = neighbour_center

        self.dormient_atoms = torch.argwhere(not_neighbour_center).squeeze(1)

        baseline = scatter_sum(total_readout[not_neighbour_center], batch_data.batch[not_neighbour_center],
                                dim=0,
                                dim_size=num_graphs)


        final_readout = scatter_sum(total_readout[attention_neighbour], batch_data.batch[attention_neighbour],
                                     dim=0,
                                     dim_size=num_graphs)

        return {'target': {'pred':final_readout,
                           'base': baseline},
                'local target': total_readout,
                'active atoms': active_atoms
                }
