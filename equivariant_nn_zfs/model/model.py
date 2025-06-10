import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Irreps
from equivariant_nn_zfs.tools.embedding import NodeFeaturesStart, RadialAngularEmbedding
from equivariant_nn_zfs.tools.prod import ReadoutL2, Product3body
from mace import modules
from e3nn.o3 import SphericalHarmonics
from mace.modules.radial import BesselBasis
from mace.modules.radial import PolynomialCutoff
import logging
import sys

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
                 zlist,
                 radial_cutoff: float,
                 pol_cut_num: int,
                 n_channels: int,
                 irreps_sh: Irreps,
                 irreps_out: Irreps,
                 weights: list,
                 device=None,
                 mlp=None
                 ):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')

        radial_cutoff = torch.tensor(radial_cutoff, device=self.device)

        self.cutoff = PolynomialCutoff(r_max=radial_cutoff, p=pol_cut_num)

        self.bf = BesselBasis(r_max=radial_cutoff, num_basis=n_bessel)

        self.spherical_harmonics = SphericalHarmonics(irreps_in='1o', irreps_out=irreps_sh, normalize=True)

        if mlp is None:
            mlp = [64, 64, 64]

        node_attr_len = len(zlist)

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

        self.loss_weights = torch.tensor(weights)

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
                          target
                          ):
        # Extract upper triangle components (batch_size, 9)

        device = pred.device  # Get the device of prediction

        target = target.to(device)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        weights = self.loss_weights.to(pred.device).unsqueeze(0)
        loss = weights * ((pred_flat - target_flat) ** 2).mean(axis=0)
        return loss.mean()

    def mse_components(self,
                       pred,
                       target
                       ):
        # Extract upper triangle components (batch_size, 9)

        device = pred.device  # Get the device of prediction

        target = target.to(device)

        pred_flat = pred.view(pred.size(0), -1)

        target_flat = target.view(target.size(0), -1)

        loss = ((pred_flat - target_flat) ** 2)
        return loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_graph_edge_attributes(self, data):

        vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
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

        length_descriptor = self.cutoff(lengths) * self.bf(lengths)

        return length_descriptor, vector_descriptor, node_attributes, edge_index

    def forward(self,
                batch_data
                ):
        outputs = []

        for idx, data in enumerate(batch_data):

            length_b, edge_attr_b, node_attr_b, edge_index_b = self.get_graph_edge_attributes(data)

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

                total_readout += readout.sum(dim=0)

            outputs.append(total_readout)

        # Stack to form final output tensor
        return torch.stack(outputs, dim=0)
