from mace import data, modules, tools
import torch
from torch import nn
from e3nn.o3 import Linear
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct, FullyConnectedTensorProduct
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from e3nn.o3 import Irreps
from mace.modules.irreps_tools import reshape_irreps
from mace.tools.scatter import scatter_sum
from e3nn.util.jit import compile_mode

# TODO: REVIEW THIS PART EXTENSIVELY, FIND THE BUG


@compile_mode("script")
class NodeFeaturesStart(nn.Module):
    def __init__(self,
                 node_attr_irreps,
                 node_feat_irreps
                 ):
        super().__init__()

        self.linear = Linear(irreps_in=node_attr_irreps,
                             irreps_out=node_feat_irreps
                             )

    def forward(self,
                node_attributes):
        node_attributes = node_attributes.to(self.linear.weight.device)
        return self.linear(node_attributes)


@compile_mode("script")
class RadialAngularEmbedding(nn.Module):
    def __init__(self,
                 nbessel,
                 node_feat_irreps,
                 node_attr_irreps,
                 irreps_sh,
                 hidden_irreps
                 ):
        super().__init__()

        irreps_mid, instructions = tp_out_irreps_with_instructions(
            node_feat_irreps,
            irreps_sh,
            target_irreps=hidden_irreps,
        )

        self.conv_tp = TensorProduct(irreps_in1=node_feat_irreps,
                                     irreps_in2=irreps_sh,
                                     irreps_out=irreps_mid,
                                     instructions=instructions,
                                     shared_weights=False,
                                     internal_weights=False
                                     )

        irreps_in1 = Irreps(f'{nbessel}x0e')

        self.fcn = FullyConnectedNet(
            [irreps_in1.num_irreps, 64, 64, 64, self.conv_tp.weight_numel],
            act=torch.nn.functional.silu
        )

        self.linear = Linear(
            irreps_mid,
            hidden_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        self.reshape = reshape_irreps(hidden_irreps)

        self.fctp_attributes = FullyConnectedTensorProduct(irreps_in1=node_attr_irreps,
                                                           irreps_in2=node_feat_irreps,
                                                           irreps_out=hidden_irreps
                                                           )

        print(self.fcn, self.linear, self.fctp_attributes)

    def forward(self,
                length,
                node_features,
                node_attributes,
                edge_attributes,
                edge_index
                ):
        device = self.fcn[0].weight.device  # or self.linear.weight.device

        length = length.to(device)

        node_features = node_features.to(device)

        edge_attributes = edge_attributes.to(device)

        edge_index = edge_index.to(device)

        sender = edge_index[0]

        receiver = edge_index[1]

        num_nodes = node_features.shape[0]

        tp_weights = self.fcn(length)

        mij = self.conv_tp(node_features[sender], edge_attributes, tp_weights)  # [n_edges, target_irreps]

        message = scatter_sum(
            src=mij, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, target_irreps]

        message = self.linear(message)

        sc = self.fctp_attributes(node_attributes, node_features)  # [n_nodes, nchannel*irreps] hidden_irreps

        return self.reshape(message), sc
