from mace import data, modules, tools
import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Linear, FullyConnectedTensorProduct
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from equivariant_nn_zfs.tools.contract import ContractProduct3j


@compile_mode("script")
class Product3body(nn.Module):
    def __init__(self,
                 irreps_sh,
                 hidden_irreps,
                 node_attr_irreps,
                 ncor
                 ):
        super().__init__()

        self.contraction = ContractProduct3j(irreps_in1=irreps_sh,
                                             irreps_in2=irreps_sh,
                                             irreps_target=irreps_sh
                                             )
        ncor_irreps_sh = (len(ncor)*irreps_sh).sort()[0].simplify()

        self.linear_body = Linear(irreps_in=ncor_irreps_sh,
                                  irreps_out=irreps_sh
                                  )

        self.linear = Linear(irreps_in=hidden_irreps,
                             irreps_out=hidden_irreps
                             )

        self.fctp_attributes = FullyConnectedTensorProduct(irreps_in1=node_attr_irreps,
                                                           irreps_in2=hidden_irreps,
                                                           irreps_out=hidden_irreps
                                                           )

        print(self.linear_body, self.linear, self.fctp_attributes)

    def forward(self,
                node_feature_i,
                node_attr,
                sc
                ):

        node2 = self.contraction(node_feature_i, node_feature_i)

        node3 = self.contraction(node2, node_feature_i)  # nodes, channels, irreps

        node_out = torch.cat([node_feature_i, node2, node3],
                             dim=-1
                             )  # nodes, channels, irreps*nbody

        node_out = self.linear_body(node_out)  # nodes, channels, irreps

        out = []
        for s in torch.unbind(node_out, -1):
            out.append(s)

        node_out = self.linear(torch.cat(out, dim=-1))

        node_out = self.fctp_attributes(node_attr, node_out)

        return node_out + sc


@compile_mode("script")
class ReadoutL2(nn.Module):
    def __init__(self,
                 hidden_irreps
                 ):
        super().__init__()

        out_irreps = o3.Irreps("0e + 1o + 2e")

        self.linear_readout = Linear(irreps_in=hidden_irreps,
                                     irreps_out=out_irreps
                                     )

        print(self.linear_readout)

    def forward(self,
                node_features
                ):

        readout = self.linear_readout(node_features)

        return readout
