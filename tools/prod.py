from mace import data, modules, tools
import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Linear
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from e3nn.o3 import wigner_3j
from equivariant_nn_zfs.tools.contract import ContractProduct3j


@compile_mode("script")
class UpdateNodeAttributesReadoutL2(nn.Module):
    def __init__(self,
                 hidden_irreps,
                 node_attr_irreps
                 ):
        super().__init__()

        out_irreps = o3.Irreps("0e + 1o + 2e")


        self.linear = Linear(irreps_in=hidden_irreps,
                             irreps_out=hidden_irreps
                             )

        self.linear_intermediate = nn.ModuleList()

        self.linear_intermediate.append(Linear(irreps_in=hidden_irreps,
                                               irreps_out=hidden_irreps
                                               )
                                        )

        self.linear_readout = Linear(irreps_in=hidden_irreps,
                                     irreps_out=out_irreps
                                     )

    def forward(self,
                message,
                node_features_
                ):

        nodes, channels, irreps_dim = message.shape

        message = message.to(self.linear.weight.device)


        node_features_intermid = self.linear_intermediate[0](node_features_)

        message = message.transpose(1,2).reshape(nodes, channels*irreps_dim)

        node_features = self.linear(message) + node_features_intermid

        readout = self.linear_readout(node_features)

        return readout, node_features


@compile_mode("script")
class NewNodeFeaturesFrom3Body(nn.Module):
    def __init__(self,
                 irreps_sh,
                 node_attr_irreps,
                 hidden_irreps,
                 ncor=[0, 1, 2]
                 ):
        super().__init__()

        ncor_irreps = Irreps(f"{len(ncor)}x0e")

        ncor_irreps_final = Irreps(f"{1}x0e")

        hidden_irreps_body = (hidden_irreps * len(ncor)).sort()[0].simplify()

        node_attr_irreps_body = (node_attr_irreps * len(ncor)).sort()[0].simplify()

        self.linear_nodes_chemical = Linear(irreps_in=node_attr_irreps_body,
                                            irreps_out=hidden_irreps_body
                                            )

        self.contraction = ContractProduct3j(irreps_in1=irreps_sh,
                                             irreps_in2=irreps_sh,
                                             irreps_target=irreps_sh
                                            )

        self.linear_nodes_body = Linear(irreps_in=ncor_irreps,
                                        irreps_out=ncor_irreps_final
                                        )

    def forward(self,
                node_nbody,
                node_attr
                ):
        # PREPARE THE SHAPES AND THE FEATURES IN THE RIGHT SHAPE

        nodes, channels, irreps_dim = node_nbody[..., 0].shape  # WE WANT THESE SHAPES

        nodes, lenz = node_attr.shape

        node3_ = torch.stack(tensors=(node_nbody[..., 0], node_nbody[..., 1], node_nbody[..., 2]),
                             dim=-1
                             ).transpose(1, 2)  # nodes, irreps, channels, nbody

        node_attr_body = torch.stack(tensors=(node_attr,
                                              node_attr,
                                              node_attr
                                              ),
                                     dim=0)  # body, nodes, lenz


        node3_ = node3_.reshape(nodes, irreps_dim, channels * 3).transpose(1, 2)  # nodes, channels*nbody, irreps

        node_attr_body = node_attr_body.transpose(1, 2).reshape(nodes, lenz * 3)

        weights_z = self.linear_nodes_chemical(node_attr_body)  # nodes, body*channel*irreps

        weights_z = weights_z.reshape(nodes, irreps_dim, 3 * channels).transpose(1, 2)  # nodes, body*channels, irreps

        m_eta = self.contraction(weights_z, node3_)  # nodes, channels*nbody, irreps

        m_eta = m_eta.transpose(1, 2).reshape(nodes * channels * irreps_dim, 3)

        output = self.linear_nodes_body(m_eta).reshape(nodes,
                                                       irreps_dim,
                                                       channels) # nodes, irreps, channels

        return output.reshape(nodes, irreps_dim*channels)

@compile_mode("script")
class ConvolveTensor3body(nn.Module):
    def __init__(self,
                 irreps_sh,
                 nchannels,
                 ncor=[0, 1, 2]
                 ):
        super().__init__()

        nchannels_irreps = Irreps(f"{nchannels}x0e")

        self.linear_channels = nn.ModuleList()
        for _ in ncor:
            self.linear_channels.append(Linear(irreps_in=nchannels_irreps,
                                               irreps_out=nchannels_irreps
                                               )
                                        )

        self.contraction = ContractProduct3j(irreps_in1=irreps_sh,
                                             irreps_in2=irreps_sh,
                                             irreps_target=irreps_sh
                                             )

        self.nonlinearity = torch.nn.functional.silu

    def mix_channels(self,
                     x,
                     ncor
                     ):
        nodes, channels, irreps_dim = x.shape


        x_reshaped = x.transpose(2, 1)

        x_reshaped = x_reshaped.reshape((nodes * irreps_dim, channels))

        # Apply the linear operation on the reshaped input
        x_transformed = self.linear_channels[ncor](x_reshaped)

        x_transformed = x_transformed.reshape((nodes, irreps_dim, channels))

        # Reshape back to [nodes, channels, irreps] (note: here channels_out = channels_in)
        x_transformed = x_transformed.transpose(2, 1)

        return x_transformed

    def forward(self,
                node_feature_i
                ):

        node1 = self.nonlinearity(self.mix_channels(node_feature_i, 0))  # [nodes, channels, irreps]

        node1_ = self.nonlinearity(self.mix_channels(node_feature_i, 1))

        node2 = self.contraction(node1, node1_)

        node2_ = self.nonlinearity(self.mix_channels(node_feature_i, 2))

        node3 = self.contraction(node2, node2_)  # nodes, channels, irreps

        node_out = torch.stack(tensors=(node1, node2, node3),
                             dim=-1
                             )  # nodes, irreps, channels, nbody

        return node_out
