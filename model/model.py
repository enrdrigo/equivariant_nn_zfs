from mace import data, modules, tools
import torch
from torch import nn
import torch.optim as optim
from e3nn import o3
from equivariant_nn_zfs.tools.embedding import NodeFeaturesStart, RadialAngularEmbedding
from equivariant_nn_zfs.tools.prod import UpdateNodeAttributesReadoutL2, ConvolveTensor3body, NewNodeFeaturesFrom3Body


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
                                                     nchannels=nchannels,
                                                     node_feat_irreps=node_feat_irreps_start,
                                                     irreps_sh=irreps_sh,
                                                     hidden_irreps=hidden_irreps
                                                     )
                              )
        self.radialemb.append(RadialAngularEmbedding(nbessel=nbessel,
                                                     nchannels=nchannels,
                                                     node_feat_irreps=hidden_irreps,
                                                     irreps_sh=irreps_sh,
                                                     hidden_irreps=hidden_irreps
                                                     )
                              )

        self.prod = nn.ModuleList()
        self.prod.append(ConvolveTensor3body(irreps_sh=irreps_sh,
                                             nchannels=nchannels
                                             )
                         )
        self.prod.append(ConvolveTensor3body(irreps_sh=irreps_sh,
                                             nchannels=nchannels
                                             )
                         )

        self.nbodyfeatures = nn.ModuleList()
        self.nbodyfeatures.append(NewNodeFeaturesFrom3Body(irreps_sh=irreps_sh,
                                                           node_attr_irreps=node_attr_irreps,
                                                           hidden_irreps=hidden_irreps,
                                                           )
                                  )
        self.nbodyfeatures.append(NewNodeFeaturesFrom3Body(irreps_sh=irreps_sh,
                                                           node_attr_irreps=node_attr_irreps,
                                                           hidden_irreps=hidden_irreps,
                                                           )
                                  )

        self.update_readout = nn.ModuleList()
        self.update_readout.append(UpdateNodeAttributesReadoutL2(hidden_irreps=hidden_irreps,
                                                                 node_attr_irreps=node_attr_irreps
                                                                 )
                                   )
        self.update_readout.append(UpdateNodeAttributesReadoutL2(hidden_irreps=hidden_irreps,
                                                                 node_attr_irreps=node_attr_irreps
                                                                )
                                   )

        self.optimizer = optim.AdamW(self.parameters(), lr=5e-3, weight_decay=5e-7)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', threshold=1e-3,
                                                         factor=0.9, patience=2)

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

    def weighted_mse_loss_xcomponent(self,
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

            node_features_initial = self.node_features(node_attr_b)

            message1 = self.radialemb[0](lenght_b,
                                         node_features_initial,
                                         edge_attr_b,
                                         edge_index_b
                                         )

            prod1_ = self.prod[0](message1)

            prod1 = self.nbodyfeatures[0](prod1_, node_attr_b)

            readout1, node_features1 = self.update_readout[0](message1, prod1)

            message2 = self.radialemb[1](lenght_b,
                                         node_features1,
                                         edge_attr_b,
                                         edge_index_b
                                         )

            prod2_ = self.prod[1](message2)

            prod2 = self.nbodyfeatures[0](prod2_, node_attr_b)

            readout2, _ = self.update_readout[1](message2, prod2)

            total_readout = readout1.sum(dim=0) + readout2.sum(dim=0)

            outputs.append(total_readout)

        # Stack to form final output tensor
        return torch.stack(outputs, dim=0)
