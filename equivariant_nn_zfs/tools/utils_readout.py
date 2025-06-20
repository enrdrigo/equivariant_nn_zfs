import torch
from mace.tools.torch_geometric import Batch

def get_centers_batch(top_k_batch,
                      edge_index,
                      k,
                      device,
                      ):
    edge_set = set((int(i), int(j)) for i, j in zip(edge_index[0], edge_index[1]))

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

def get_centers(atomic: torch.Tensor,
                edge_index,
                k,
                batch: Batch):

    # given the top_k indexes I divide them in the respective graphs and look for the centers. I will have K centers per graph.
    num_graphs = batch.ptr.numel() - 1

    top_k = torch.topk(atomic.norm(dim=1), dim=0, k=len(batch.batch))

    neighbours_centers = []

    list_batches = []

    norms = []

    for b in range(num_graphs):
        top_k_batch_ = top_k.indices[batch.batch[top_k.indices] == b]

        neighbours_centers_b=get_centers_batch(top_k_batch_,
                                    k=k,
                                    edge_index=edge_index,
                                    device=atomic.device)
        neighbours_centers.append(neighbours_centers_b)

        list_batches.append(batch.batch[neighbours_centers_b])

        norms.append(len(neighbours_centers_b))

    return torch.cat(neighbours_centers, dim=0), torch.cat(list_batches, dim=0), torch.tensor(norms, device=atomic.device)

