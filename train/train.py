from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def nntrain(model,
            loader,
            device=None
            ):
    device = device if device is not None else model.device
    counts = defaultdict(int)
    for name, param in model.named_parameters():
        top_level = name.split('.')[0]  # e.g., "radialemb", "prod", etc.
        if param.requires_grad:
            counts[top_level] += param.numel()
    for block, count in counts.items():
        print(f"{block:<25}: {count:,} params")

    print(model.count_parameters())
    error = []
    for epoch in range(4):
        total_loss = 0
        for X, X_v, node_attr, edge_index, Y_true in loader:

            model.optimizer.zero_grad()  # Zeroing gradients
            Y_true = Y_true.to(device)
            X = [x.to(device) for x in X]
            X_v = [xv.to(device) for xv in X_v]
            node_attr = [na.to(device) for na in node_attr]
            edge_index = [ei.to(device) for ei in edge_index]

            Y_pred = model(X, X_v, node_attr, edge_index)  # Forward pass

            loss = model.loss_fn(Y_pred, Y_true)  # Loss calculation
            loss.backward()# Backpropagation
            loss_xcomponent = model.weighted_mse_loss_xcomponent(Y_pred, Y_true)
            error.append(loss_xcomponent.mean(axis=0).tolist())

            model.optimizer.step()  # Update weights
            total_loss += loss.item()
        for param_group in model.optimizer.param_groups:
            print(f"LR: {param_group['lr']}")

            print(f"Epoch {epoch + 1}: Loss = {total_loss / len(loader):.4f}")
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.ravel()  # Flatten the 2D array of axes

    labels = [r'$Y^0_0$', r'$Y^1_{-1}$', r'$Y^1_0$', r'$Y^1_1$',
              r'$Y^2_{-2}$', r'$Y^2_{-1}$', r'$Y^2_0$', r'$Y^2_1$', r'$Y^2_1$']

    for i in range(9):
        ax = axes[i]

        ax.plot(np.array(error)[:, i], '.', color='b')
        ax.set_ylabel(labels[i])
        ax.set_yscale('log')

    plt.tight_layout()
