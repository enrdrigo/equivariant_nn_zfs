import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from mace import data, modules, tools
from e3nn import o3
from matplotlib import pyplot as plt
from e3nn.o3 import SphericalHarmonics
from e3nn.o3 import Irreps
from mace.modules.radial import BesselBasis
from mace.modules.radial import PolynomialCutoff
from convert_matrix import cartesian_to_spherical_irreps
from blocks_zfs import NodeFeaturesStart, RadialAngularEmbedding, UpdateNodeAttributes_readoutl2

# 1. Utility: Compute inertia tensor
def compute_inertia_tensor(atoms):
    com = atoms.get_center_of_mass()
    positions = atoms.get_positions() - com
    masses = atoms.get_masses()
    I = np.zeros((3, 3))
    for r, m in zip(positions, masses):
        I += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    I = I - np.eye(3) * I.trace() / 3

    I = torch.tensor(I)

    I = cartesian_to_spherical_irreps(I)

    return I

# --- Dummy dataset (replace with your own structures and target matrices) ---
class EquivariantMatrixDataset(Dataset):
    def __init__(self,
                 structures,
                 pol_cut_num,
                 nbessel,
                 Rcut
                 ):
        self.structures = structures
        self.targets = np.array([compute_inertia_tensor(s) for s in structures])
        self.pol_cut_num=pol_cut_num
        self.nbessel=nbessel
        self.Rcut=Rcut

        z_table = set()
        for struct in structures:
            s_z_table = struct.get_atomic_numbers()
            z_table.update(s_z_table)
        self.z_table = tools.AtomicNumberTable(list(z_table))

    def __len__(self):
        return len(self.structures)


    def __getitem__(self, idx):
        self.struct = self.structures[idx]


        config = data.Configuration(
            atomic_numbers=self.struct.numbers,
            positions=self.struct.positions,
            properties={'positions': 'positions'},
            property_weights={'positions': 1}
        )

        # we handle configurations using the AtomicData class
        batch = data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.Rcut)

        vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
            positions=batch["positions"],
            edge_index=batch["edge_index"],
            shifts=batch["shifts"],
        )

        self.node_attr = batch.node_attrs

        self.edge_index = batch.edge_index

        self.node_attr_len = vectors.shape[0]

        cutoff = PolynomialCutoff(r_max=self.Rcut, p=self.pol_cut_num)
        bf = BesselBasis(r_max=self.Rcut, num_basis=self.nbessel)

        irreps_sh = Irreps('0e + 1o +2e')

        spherical_harmonics = SphericalHarmonics(irreps_in='1o', irreps_out=irreps_sh, normalize=True)

        vector_descriptor = spherical_harmonics(vectors)

        length_descriptor = cutoff(lengths) * bf(lengths)

        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)

        return length_descriptor, vector_descriptor, self.node_attr, self.edge_index, target_tensor


def collate_fn(batch):
    """
    Custom collate function to handle variable-length descriptors in the batch.
    """
    vectors, lengths, node_attr, edge_index, targets = zip(*batch)

    # We can't stack the descriptors directly because they have different sizes
    # Instead, we keep them in a list
    targets = torch.stack(targets)

    return list(vectors), list(lengths), list(node_attr), list(edge_index), targets

# --- Symmetric Matrix Regressor ---
class SymmetricMatrixRegressor(nn.Module):
    def __init__(self,
                 nbessel,
                 zlist,
                 nchannels,
                 weights=[1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1,
                          1
                          ],
                 device=None
                 ):
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.to(self.device)
        self.node_features = NodeFeaturesStart(zlist=zlist,
                                               nchannels=nchannels
                                               )



        self.radialemb = RadialAngularEmbedding(nbessel=nbessel,
                                                zlist=zlist,
                                                nchannels=nchannels,
                                                irreps_node_feat=Irreps(f'{nchannels}x0e')
                                                )

        self.update_readout = UpdateNodeAttributes_readoutl2(nchannels=nchannels)

        #TODO: FIX THIS HARD IMPLEMENTATION OF THE IRREPS_NODE_FEAT

        self.radialemb2 = RadialAngularEmbedding(nbessel=nbessel,
                                                zlist=zlist,
                                                nchannels=nchannels,
                                                 irreps_node_feat=(Irreps('0e + 1o + 2e') * nchannels).sort()[0].simplify()
                                                )

        self.update_readout2 = UpdateNodeAttributes_readoutl2(nchannels=nchannels)

        self.optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=5e-7,)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.94) #ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

        self.loss_weights = torch.tensor(weights)
        self.loss_fn = self.weighted_mse_loss


    def weighted_mse_loss(self, pred, target):
        # Extract upper triangle components (batch_size, 9)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        weights = self.loss_weights.to(pred.device).unsqueeze(0)
        loss = weights * (pred_flat - target_flat) ** 2
        return loss.mean()



    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



    def forward(self, x, x_v, node_attr, edge_index):
        outputs = []

        for idx, (lenght_b, edge_attr_b, node_attr_b, edge_index_b) in enumerate(zip(x, x_v, node_attr, edge_index)):
            node_attr_b = node_attr_b.detach().requires_grad_()


            node_features_initial = self.node_features(node_attr_b)

            message1 = self.radialemb(lenght_b,
                                     node_features_initial,
                                     edge_attr_b,
                                     edge_index_b
                                     )

            readout1, updated_node_features = self.update_readout(message1)


            message2 = self.radialemb2(lenght_b,
                                     updated_node_features,
                                     edge_attr_b,
                                     edge_index_b
                                     )

            readout2, _ = self.update_readout2(message2)

            total_readout = readout1.sum(dim=0) + readout2.sum(dim=0)

            outputs.append(total_readout)

        # Stack to form final output tensor
        return torch.stack(outputs, dim=0)


    # --- Training loop ---
    def NNtrain(self,
              loader,
                device=None
                ):
        device = device if device is not None else self.device
        print(self.count_parameters())
        for epoch in range(100):
            total_loss = 0
            for X, X_v, node_attr, edge_index, Y_true in loader:
                Y_true = Y_true.to(device)
                X = [x.to(device) for x in X]
                X_v = [xv.to(device) for xv in X_v]
                node_attr = [na.to(device) for na in node_attr]
                edge_index = [ei.to(device) for ei in edge_index]
                self.optimizer.zero_grad()  # Zeroing gradients
                Y_pred = model(X, X_v, node_attr, edge_index)  # Forward pass

                loss = model.loss_fn(Y_pred, Y_true)  # Loss calculation
                loss.backward()  # Backpropagation
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(
                #             f"{name}: grad mean = {param.grad.mean():.6e}, std = {param.grad.std():.6e}, max = {param.grad.max():.6e}")
                #     else:
                #         print(f"{name}: grad is None")
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e4)
                self.optimizer.step()  # Update weights
                total_loss += loss.item()



            # --- Validation Phase ---
            self.scheduler.step()
            for param_group in self.optimizer.param_groups:
                print(f"LR: {param_group['lr']}")

            print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")


def plot_parity(true_values, predicted_values, labels):
    """
    Generate parity plot (predicted vs. true) for each component of the inertia tensor.

    Args:
    - true_values: Array of true values (target values).
    - predicted_values: Array of predicted values from the model.
    - labels: List of labels for the tensor components.
    """
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.ravel()  # Flatten the 2D array of axes

    for i in range(9):
        ax = axes[i]

        ax.plot(true_values[:, i], predicted_values[:, i], '.', color='b')
        lx, hx = ax.get_xlim()
        ax.set_ylim(lx, hx)

        ax.set_aspect('equal', 'box')

        ax.set_xlabel(f'True {labels[i]}')
        ax.set_ylabel(f'Predicted {labels[i]}')
        ax.set_title(f'Parity plot for {labels[i]}')


    plt.show()

if __name__ == "__main__":
    db = read('../../../Tutorials/data/solvent_configs.xyz', ':')

    dataset = EquivariantMatrixDataset(db,
                                       pol_cut_num=6,
                                       nbessel=15,
                                       Rcut=5.0
                                       )


    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=True
                        )

    total_size = len(dataset)
    train_ratio = 0.8
    test_ratio = 0.2

    # Calculate split sizes
    train_size = int(train_ratio * total_size)

    test_size = total_size - train_size # ensures all data is used

    # Randomly split
    train_data, test_data = random_split(dataset,
                                         [train_size, test_size]
                                         )


    train_loader = DataLoader(train_data,
                              batch_size=10,
                              shuffle=True,
                              collate_fn=collate_fn
                              )

    test_loader = DataLoader(test_data,
                             batch_size=1,
                             collate_fn=collate_fn
                             )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SymmetricMatrixRegressor(nbessel = dataset.nbessel,
                                     zlist = dataset.z_table,
                                     nchannels=4,
                                     weights=[0.001,
                                              0.001,
                                              0.001,
                                              0.001,
                                              1,
                                              1,
                                              1,
                                              1,
                                              1
                                              ],
                                     device = device
                                     )
    model.NNtrain(loader=train_loader
                  )
    model.eval()
    errors = []
    # Extracting true and predicted inertia tensor components
    true_inertia = []
    pred_inertia = []
    print("\nEvaluating on test set:")
    with torch.no_grad():
        for X, X_v, node_attr, edge_index,  Y_true in test_loader:
            Y_pred = model(X, X_v, node_attr, edge_index)
            mse = model.loss_fn(Y_pred, Y_true).item()
            errors.append(mse)
            true_inertia.append(Y_true.view(-1).numpy())  # Convert to numpy array
            pred_inertia.append(Y_pred.view(-1).numpy())  # Convert to numpy array

    mean_mse = np.mean(errors)
    print(f"Test Mean Squared Error (MSE): {mean_mse:.6f}")

    true_inertia = np.array(true_inertia)  # Remove unnecessary dimensions
    pred_inertia = np.array(pred_inertia)  # Remove unnecessary dimensions


    # Labels for each component of the inertia tensor
    labels = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']

    # Generate the parity plot
    plot_parity(true_inertia, pred_inertia, labels)
