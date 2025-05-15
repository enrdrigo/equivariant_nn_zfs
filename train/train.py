import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, X_v, node_attr, edge_index, Y_true in loader:

            Y_true = Y_true.to(device)
            X = [x.to(device) for x in X]
            X_v = [xv.to(device) for xv in X_v]
            node_attr = [na.to(device) for na in node_attr]
            edge_index = [ei.to(device) for ei in edge_index]

            Y_pred = model(X, X_v, node_attr, edge_index)  # Forward pass

            loss = model.loss_fn(Y_pred, Y_true)  # Loss calculation
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Validation Loss = {avg_loss:.4f}")
    return avg_loss


def nntrain(model,
            loader,
            val_loader,
            NEPOCHS,
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
    for epoch in range(NEPOCHS):
        error_batches = []
        total_loss = 0
        for X, X_v, node_attr, edge_index, Y_true in loader:

            model.optimizer.zero_grad()  # Zeroing gradients

            # PREPARE THE DATA ON THE CORRECT DEVICE

            Y_true = Y_true.to(device)
            X = [x.to(device) for x in X]
            X_v = [xv.to(device) for xv in X_v]
            node_attr = [na.to(device) for na in node_attr]
            edge_index = [ei.to(device) for ei in edge_index]

            Y_pred = model(X, X_v, node_attr, edge_index)  # Forward pass

            loss = model.loss_fn(Y_pred, Y_true)  # Loss calculation
            loss.backward()  # Backpropagation

            loss_xcomponent = model.mse_loss_xcomponent(Y_pred, Y_true)
            error_batches.append(loss_xcomponent.mean(axis=0).tolist())

            model.optimizer.step()  # Update weights

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}")

        for param_group in model.optimizer.param_groups:
            print(f"LR: {param_group['lr']}")

        print(f"Training Loss = {total_loss / len(loader):.4f} ", end='')

        val_loss = validate(model, val_loader, device)

        error_batches = np.array(error_batches)
        print(r'$Y^0_0$ $Y^1_{-1}$ $Y^1_0$ $Y^1_1$ $Y^2_{-2}$ $Y^2_{-1}$ $Y^2_0$ $Y^2_1$ $Y^2_1$')
        print("MEAN Loss values:", ", ".join(f"{x:.3e}" for x in error_batches.mean(axis=0)))
        print("STD Loss values:", ", ".join(f"{x:.3e}" for x in error_batches.std(axis=0)))

        error.append(np.array(error_batches))

        model.scheduler.step(val_loss)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.ravel()  # Flatten the 2D array of axes

    labels = [r'$Y^0_0$', r'$Y^1_{-1}$', r'$Y^1_0$', r'$Y^1_1$',
              r'$Y^2_{-2}$', r'$Y^2_{-1}$', r'$Y^2_0$', r'$Y^2_1$', r'$Y^2_2$']

    for i in range(9):
        ax = axes[i]

        ax.plot(np.array(error)[..., i], '.', color='b')
        ax.set_ylabel(labels[i])
#        ax.set_yscale('log')

    plt.tight_layout()
    plt.show()
