from mace import data, modules, tools
import torch
from torch import nn
from e3nn import o3
from e3nn.o3 import Linear
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from e3nn.o3 import Irreps
from mace.modules.irreps_tools import reshape_irreps
from mace.tools.scatter import scatter_sum
from e3nn.util.jit import compile_mode
from e3nn.o3 import wigner_3j


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
                 nchannels,
                 node_feat_irreps,
                 irreps_sh
                 ):
        super().__init__()

        hidden_irreps = (irreps_sh * nchannels).sort()[0].simplify()

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
            [irreps_in1.num_irreps, 6, 6, 6, self.conv_tp.weight_numel],
            act=torch.nn.functional.silu
        )

        self.linear = Linear(
            irreps_mid,
            hidden_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        self.reshape = reshape_irreps(hidden_irreps)

    def forward(self,
                length,
                node_features,
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

        message = self.reshape(message)  # [n_nodes, nchannels, hidden_irreps]

        return message


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

        # TODO FIX THIS PART: IT IS ACTUALLY WRONG

        self.linear_intermediate = nn.ModuleList()

        self.linear_intermediate.append(Linear(irreps_in=node_attr_irreps,
                                               irreps_out=hidden_irreps
                                               )
                                        )

        self.linear_intermediate.append(Linear(irreps_in=hidden_irreps,
                                               irreps_out=hidden_irreps
                                               )
                                        )

        self.linear_readout = Linear(irreps_in=hidden_irreps,
                                     irreps_out=out_irreps
                                     )

    def forward(self,
                message,
                node_attributes
                ):

        message = message.to(self.linear.weight.device)

        node_features_intermid = self.linear_intermediate[1](self.linear_intermediate[0](node_attributes))

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

        nodes, channels, irreps_dim = node_nbody[0].shape  # WE WANT THESE SHAPES

        nodes, lenz = node_attr.shape

        node3_ = torch.stack(tensors=(node_nbody[0], node_nbody[1], node_nbody[2]),
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
                                                       channels).transpose(1,
                                                                           2)  # nodes, channels, irreps

        return output

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

        print(x.shape)

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

        return node1, node2, node3


@compile_mode("script")
class ContractProduct3j(nn.Module):
    def __init__(self,
                 irreps_in1,
                 irreps_in2,
                 irreps_target
                 ):
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_target = irreps_target

    def extractl(self, x: torch.Tensor, irreps: Irreps, l_target: int) -> torch.Tensor:
        """
        Extract components from a tensor with given irreps corresponding to angular momentum l_target.

        Args:
            x: tensor of shape (..., irreps.dim)
            irreps: e3nn.o3.Irreps
            l_target: integer angular momentum l

        Returns:
            Tensor of shape (..., selected_dim), containing only components with l = l_target
        """
        out = []
        offset = 0
        for mul, ir in irreps:
            dim = mul * ir.dim
            if ir.l == l_target:
                out.append(x[..., offset:offset + dim])
            offset += dim
        if not out:
            raise ValueError(f"No irreps with l = {l_target} found in {irreps}")
        return torch.cat(out, dim=-1)  # nodes, nchannels, 2l + 1

    def forward(self,
                tensor_1,
                tensor_2
                ):
        # tensors are of shape [node, nchannels, irreps]

        device = tensor_1.device

        nodes, channels, _ = tensor_2.shape

        out = torch.zeros((nodes, channels, self.irreps_target.dim), device=device)

        dims = []

        for mul3_, (mul3, ir3) in enumerate(self.irreps_target):
            dims.append(ir3.dim)

        offset = [0]

        for idx, dim in enumerate(dims[:-1]):
            offset.append(offset[idx] + dim)

        for mul1_, (mul1, ir1) in enumerate(self.irreps_in1):
            for mul2_, (mul2, ir2) in enumerate(self.irreps_in2):
                for ir3 in ir1 * ir2:
                    if ir3 not in self.irreps_target:
                        continue
                    w123 = wigner_3j(ir1.l, ir2.l, ir3.l)  # with shape [2l_1+1, 2l_2+1, 2l_3+1]
                    tir1 = self.extractl(tensor_1, self.irreps_in1, ir1.l)
                    tir2 = self.extractl(tensor_2, self.irreps_in2, ir2.l)

                    out[..., offset[ir3.l]:int(2*ir3.l + 1 + offset[ir3.l])] += torch.einsum('ncl, ncm, lms -> ncs',
                                                                                             tir1,
                                                                                             tir2,
                                                                                             w123
                                                                                             )

        return out  # tensor is of shape [node, nchannels, irreps]


# irr1 = Irreps('0e')
# irr2 = Irreps('0e + 1o + 2e + 3o')
# irrt = Irreps('0e + 1o + 2e')
#
# conv = ContractProduct3j(irr1, irr2, irrt)
# irra1=irr1.randn(5, 5, -1)
# irra2=irr2.randn(5, 5, -1)
# print(conv(irra1, irra2).shape)
# print(conv(irra1, irra2)[:, :, :])
