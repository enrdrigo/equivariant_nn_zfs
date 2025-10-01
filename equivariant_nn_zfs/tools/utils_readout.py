import torch
from mace.tools.torch_geometric import Batch

def get_disjoint_centers_batch_with_attention(top_k_batch_indices: torch.tensor,
                                              attention: torch.tensor,
                                              edge_index: torch.tensor,
                                              k: int,
                                              device: torch.device,
                                              ):

    centers_batch = [top_k_batch_indices[0]]

    # Gather neighbors of the first center
    centers_tensor = torch.tensor(centers_batch, device=device)

    # Find all edges whose source is in first center
    source_nodes = edge_index[0]

    center_mask = torch.isin(source_nodes, centers_tensor)

    neighs = edge_index[1][center_mask]

    unique_neighs = torch.unique(torch.cat([neighs, centers_tensor], dim=0))

    for _ in range(k-1):

        mask = ~torch.isin(top_k_batch_indices, unique_neighs)

        if not mask.any():
            attention_frac = attention[unique_neighs] / attention[unique_neighs].sum(dim=0)

            return unique_neighs[attention_frac>0.01]

        centers_batch = [top_k_batch_indices[mask][0]]

        # Gather neighbors of the center
        centers_tensor = torch.tensor(centers_batch, device=device)

        center_mask = torch.isin(source_nodes, centers_tensor)

        neighs = edge_index[1][center_mask]

        unique_neighs = torch.unique(torch.cat([unique_neighs,  neighs, centers_tensor,], dim=0))

    attention_frac=attention[unique_neighs]/attention[unique_neighs].sum(dim=0)

    return unique_neighs[attention_frac>0.001], unique_neighs

def fast_get_centers_batch(top_k_batch_indices,
                           edge_index,
                           k,
                           device,
                           ):

    num_nodes = edge_index.max().item() + 1

    centers_batch = [top_k_batch_indices[0]]
    mask_centers = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask_centers[centers_batch[0]] = True

    for next_center in top_k_batch_indices[1:]:
        if len(centers_batch) == k:
            break
        centers_batch.append(next_center)
        mask_centers[next_center] = True
        if len(centers_batch) == k:
            break

    # Gather neighbors of all centers
    centers_tensor = torch.tensor(centers_batch, device=device)

    # Find all edges whose source is in centers
    source_nodes = edge_index[0]
    center_mask = torch.isin(source_nodes, centers_tensor)

    neighs = edge_index[1][center_mask]

    unique_neighs = torch.unique(torch.cat([neighs, centers_tensor], dim=0))

    return unique_neighs

def get_attention(atomic: torch.Tensor,):

    # cart = CartesianTensor('ij=ji')
    #
    # atomic_cart=cart.to_cartesian(atomic)
    #
    # atomic_cart = atomic_cart - (torch.einsum('jii->j', atomic_cart)*(torch.ones(atomic_cart.shape)*torch.eye(3)/3).T).T
    #
    # energies=torch.linalg.eigvalsh(atomic_cart)
    #
    # attention=(energies**2).sum(axis=1)
    #
    #THE FROBENIUS NORM IS ROTATIONAL INVARIANT: <H_ZFS, H,ZFS>=SUM_I E_I^2 = VAR (E). I AM EXCLUDING THE L=0 TERM, THE TRACE

    attention = atomic[:, 1:].norm(p='fro', dim=1)**2

    return attention

def get_centers(atomic: torch.Tensor,
                edge_index,
                k,
                batch: Batch):

    # given the top_k indexes I divide them in the respective graphs and look for the centers. I will have K centers per graph.
    num_graphs = batch.ptr.numel() - 1

    attention=get_attention(atomic)

    top_k = torch.topk(attention, dim=0, k=len(batch.batch))

    neighbours_centers = []
    attention_neighbours = []

    for b in range(num_graphs):
        top_k_batch_ = top_k.indices[batch.batch[top_k.indices] == b]

        # neighbours_centers_b=fast_get_centers_batch(top_k_batch_,
        #                                             k=k,
        #                                             edge_index=edge_index,
        #                                             device=atomic.device)

        attention_neighbours_b, neighbours_centers_b = get_disjoint_centers_batch_with_attention(top_k_batch_,
                                                                                                 attention=attention,
                                                                                                 k=k,
                                                                                                 edge_index=edge_index,
                                                                                                 device=atomic.device)


        neighbours_centers.append(neighbours_centers_b)
        attention_neighbours.append(attention_neighbours_b)

    return torch.cat(attention_neighbours, dim=0), torch.cat(neighbours_centers, dim=0)

# def augment_data(data: List[Atoms]):
#     data_aug=[]
#     for i in data:
#         for m in spglib.get_symmetry((i.cell.array, [[0, 0, 0]], [1]))['rotations']:
#             # if np.linalg.det(m) < 0:
#             #     continue
#
#             j = i.copy()
#             c = i.cell.array.copy()
#             r = c @ m @ np.linalg.inv(c)
#             j.set_positions(wrap_positions(i.positions @ r.T, i.cell))
#             j.info['target_L2'] = (r @ i.info['target_L2'].reshape(3, 3) @ r.T).flatten()
#             if np.linalg.norm(np.linalg.eigvalsh(i.info['target_L2'].reshape(3, 3)) - np.linalg.eigvalsh(
#                     r @ i.info['target_L2'].reshape(3, 3) @ r.T)) > 1e-2:
#                 print(np.linalg.eigvalsh(i.info['target_L2'].reshape(3, 3)),
#                       np.linalg.eigvalsh(r @ i.info['target_L2'].reshape(3, 3) @ r.T))
#                 raise ValueError('MALISSIMO')
#             data_aug.append(j)
#
#     return data_aug