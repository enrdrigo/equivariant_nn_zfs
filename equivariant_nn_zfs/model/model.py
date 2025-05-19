import torch
from torch import nn
from e3nn import o3
from equivariant_nn_zfs.tools.embedding import NodeFeaturesStart, RadialAngularEmbedding
from equivariant_nn_zfs.tools.prod import ReadoutL2, Product3body


# --- Symmetric Matrix Regressor ---
class SymmetricMatrixRegressor(nn.Module):
    def __init__(self,
                 nbessel,
                 zlist,
                 nchannels,
                 irreps_sh,
                 weights,
                 device = None
                 ):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.to(self.device)

        node_attr_len = len(zlist)

        node_attr_irreps = o3.Irreps([(node_attr_len, (0, 1))])

        node_feat_irreps_start = o3.Irreps(f"{nchannels}x0e")

        hidden_irreps = (irreps_sh * nchannels).sort()[0].simplify()

        self.node_features = NodeFeaturesStart(node_attr_irreps=node_attr_irreps,
                                               node_feat_irreps=node_feat_irreps_start
                                               )

        self.radialemb = nn.ModuleList()
        self.radialemb.append(RadialAngularEmbedding(nbessel=nbessel,
                                                     node_feat_irreps=node_feat_irreps_start,
                                                     irreps_sh=irreps_sh,
                                                     hidden_irreps=hidden_irreps,
                                                     node_attr_irreps=node_attr_irreps
                                                     )
                              )
        self.radialemb.append(RadialAngularEmbedding(nbessel=nbessel,
                                                     node_feat_irreps=hidden_irreps,
                                                     irreps_sh=irreps_sh,
                                                     hidden_irreps=hidden_irreps,
                                                     node_attr_irreps=node_attr_irreps
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
        self.readout.append(ReadoutL2(hidden_irreps=hidden_irreps
                                      )
                            )
        self.readout.append(ReadoutL2(hidden_irreps=hidden_irreps
                                      )
                            )

        self.loss_weights = torch.tensor(weights)

        self.loss_fn = self.weighted_mse_loss

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

    def forward(self,
                x,
                x_v,
                node_attr,
                edge_index
                ):
        outputs = []

        for idx, (lenght_b, edge_attr_b, node_attr_b, edge_index_b) in enumerate(zip(x, x_v, node_attr, edge_index)):
            node_attr_b = node_attr_b.detach().requires_grad_()

            node_features = self.node_features(node_attr_b)

            total_readout = 0

            for i in range(2):

                message, sc = self.radialemb[i](lenght_b,
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
