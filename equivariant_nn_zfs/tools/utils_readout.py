import torch
from mace.tools.torch_geometric import Batch

def get_centers_batch(top_k_batch,
                      edge_index,
                      k,
                      device,
                      ):

    edge_set = set()
    for i, j in zip(edge_index[0], edge_index[1]):
        edge_set.add((int(i), int(j)))
        edge_set.add((int(j), int(i)))

    center = top_k_batch[0]
    centers_batch = [center]

    if k==1:
        neighs = set()
        for center_ in centers_batch:
            snd_mask = edge_index[0] == center_

            neighs.update(edge_index[1][snd_mask].tolist())
        return torch.tensor(list(neighs), device=device)

    for next_center in top_k_batch[1:]:
        if any((next_center, center_) in edge_set for center_ in centers_batch) :
            continue
        else:
            centers_batch.append(next_center)

        if len(centers_batch) == k:
            neighs = set()
            for center_ in centers_batch:
                snd_mask = edge_index[0] == center_

                neighs.update(edge_index[1][snd_mask].tolist())
            return torch.tensor(list(neighs), device=device)

    return ValueError('NOPE!')

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

def get_centers(atomic: torch.Tensor,
                edge_index,
                k,
                batch: Batch):

    # given the top_k indexes I divide them in the respective graphs and look for the centers. I will have K centers per graph.
    num_graphs = batch.ptr.numel() - 1

    top_k = torch.topk(atomic.norm(dim=1), dim=0, k=len(batch.batch))

    neighbours_centers = []

    for b in range(num_graphs):
        top_k_batch_ = top_k.indices[batch.batch[top_k.indices] == b]

        neighbours_centers_b=fast_get_centers_batch(top_k_batch_,
                                                    k=k,
                                                    edge_index=edge_index,
                                                    device=atomic.device)
        neighbours_centers.append(neighbours_centers_b)

    return torch.cat(neighbours_centers, dim=0)
