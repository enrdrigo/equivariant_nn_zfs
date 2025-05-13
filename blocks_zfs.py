import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Linear, FullTensorProduct, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from e3nn.o3 import Irreps
from mace.modules.irreps_tools import reshape_irreps
from mace.tools.scatter import scatter_sum
from e3nn.util.jit import compile_mode
from typing import Union, Tuple, List
from collections import defaultdict

@compile_mode("script")
class NodeFeaturesStart(nn.Module):
    def __init__(self,
                 zlist,
                 nchannels
                 ):
        super().__init__()
        self.zlist = zlist

        self.nchannels = nchannels

        self.node_attr_len = len(zlist)

        self.node_attr_irreps = o3.Irreps([(self.node_attr_len, (0, 1))])

        self.hidden_irreps = o3.Irreps(f"{nchannels}x0e + {nchannels}x1o + {nchannels}x2e")

        self.node_feat_irreps = o3.Irreps([(self.hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])

        self.linear = Linear(irreps_in=self.node_attr_irreps,
                             irreps_out=self.node_feat_irreps,
                             internal_weights=True,
                             shared_weights=True,
                             )

    def forward(self,
                node_attributes):
        node_attributes = node_attributes.to(self.linear.weight.device)
        return self.linear(node_attributes)

@compile_mode("script")
class RadialAngularEmbedding(nn.Module):
    def __init__(self,
                 nbessel,
                 nchannels,
                 zlist,
                 irreps_node_feat
                 ):
        super().__init__()
        self.zlist = zlist

        self.node_attr_len = len(zlist)

        irreps_sh = Irreps('0e + 1o +2e')

        target_irreps = (irreps_sh * nchannels).sort()[0].simplify()

        irreps_mid, instructions = tp_out_irreps_with_instructions(
            irreps_node_feat,
            irreps_sh,
            target_irreps=target_irreps,
        )

        self.conv_tp = TensorProduct(irreps_in1=irreps_node_feat,
                                     irreps_in2=irreps_sh,
                                     irreps_out=irreps_mid,
                                     instructions=instructions,
                                     shared_weights=False,
                                     internal_weights=False
                                     )


        irreps_in1=Irreps(f'{nbessel}x0e')

        self.fcn = FullyConnectedNet(
            [irreps_in1.num_irreps, 6, 6, 6, self.conv_tp.weight_numel],
            act=torch.nn.functional.silu
        )

        self.irreps_out = target_irreps

        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )


        self.reshape = reshape_irreps(self.irreps_out)

    def forward(self,
                lenght,
                node_features,
                edge_attributes,
                edge_index
                ):
        device = self.fcn[0].weight.device  # or self.linear.weight.device

        lenght = lenght.to(device)

        node_features = node_features.to(device)

        edge_attributes = edge_attributes.to(device)

        edge_index = edge_index.to(device)

        sender = edge_index[0]

        receiver = edge_index[1]

        num_nodes = node_features.shape[0]

        tp_weights = self.fcn(lenght)

        mij = self.conv_tp(node_features[sender], edge_attributes, tp_weights)# [n_edges, irreps]

        message = scatter_sum(
            src=mij, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]

        message = self.linear(message)

        #self.reshape(message)

        return message

@compile_mode("script")
class UpdateNodeAttributes_readoutl2(nn.Module):
    def __init__(self,
                 nchannels
                 ):
        super().__init__()
        target_irreps = o3.Irreps(f"{nchannels}x0e + {nchannels}x1o + {nchannels}x2e")

        out_irreps = o3.Irreps("0e + 1o + 2e")

        self.linear = Linear(irreps_in=target_irreps,
                             irreps_out=target_irreps
                             )

        self.linear_readout = Linear(irreps_in=target_irreps,
                                     irreps_out=out_irreps
                                     )

    def forward(self,
                message
                ):
         message = message.to(self.linear.weight.device)

         node_features = self.linear(message)

         readout = self.linear_readout(node_features)


         return (readout, node_features)

@compile_mode("script")
class ConvolveTensor3body(nn.Module):
    def __init__(self,
                 irreps_in1
                 ):
        super().__init__()

        self.linear1 = Linear(irreps_in=irreps_in1, irreps_out=irreps_in1)

        self.linear2 = Linear(irreps_in=irreps_in1, irreps_out=irreps_in1)

        self.linear3 = Linear(irreps_in=irreps_in1, irreps_out=irreps_in1)

        self.fctp1 = FullTensorProduct(irreps_in1=irreps_in1, irreps_in2=irreps_in1)

        self.fctp2 = FullTensorProduct(irreps_in1=self.fctp1.irreps_out,
                                       irreps_in2=irreps_in1,
                                       filter_ir_out=[ir for mul, ir in irreps_in1],

                                       )

        self.linear4 = Linear(irreps_in=self.fctp2.irreps_out, irreps_out=irreps_in1)



    def forward(self,
                node_feature_i
                ):

        self.nonlinearity = torch.nn.functional.silu

        node1 = self.nonlinearity(self.linear1(node_feature_i))

        node2 = self.nonlinearity(self.linear2(node_feature_i))

        iter2 = self.fctp1(node1, node2)

        node3 = self.nonlinearity(self.linear3(node_feature_i))

        iter3 = self.fctp2(iter2, node3)


        return self.linear4(iter3)




