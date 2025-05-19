import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)



def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, X_v, node_attr, edge_index, Y_true in loader:

            Y_true = Y_true.to(device)
            X = [xx.to(device) for xx in X]
            X_v = [xv.to(device) for xv in X_v]
            node_attr = [na.to(device) for na in node_attr]
            edge_index = [ei.to(device) for ei in edge_index]

            Y_pred = model(X, X_v, node_attr, edge_index)  # Forward pass

            loss = model.loss_fn(Y_pred, Y_true)  # Loss calculation
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    logging.info(f"VAL        Loss =                    {avg_loss:.4f}")
    return avg_loss


def test(model, loader, device, epoch, nepoch):

    model.eval()
    rmse_rel = []

    y_true_list = []
    y_pred_list = []

    error_batches = []

    with torch.no_grad():
        for X, X_v, node_attr, edge_index, Y_true in loader:

            Y_true = Y_true.to(device)

            X = [xx.to(device) for xx in X]

            X_v = [xv.to(device) for xv in X_v]

            node_attr = [na.to(device) for na in node_attr]

            edge_index = [ei.to(device) for ei in edge_index]

            Y_pred = model(X, X_v, node_attr, edge_index)  # Forward pass

            loss = model.mse_components(Y_pred, Y_true).mean(axis=0).tolist()  # Loss calculation

            ch_zeros = []

            for idx, y_pred_ in enumerate(Y_pred[0]):
                ch = 0
                if abs(y_pred_) > 1e-6:
                    ch = np.sqrt(loss[idx])/y_pred_
                ch_zeros.append(ch)

            rmse_rel.append(np.array(ch_zeros))
            y_true_list.append(Y_true[0])
            y_pred_list.append(Y_pred[0])

            error_batches.append(loss)

    error_batches = np.array(error_batches)
    logging.info("TEST  MEAN Loss values:   " + ", ".join(f"{x:.3e}" for x in error_batches.mean(axis=0)))
    logging.info("TEST  STD  Loss values:   " + ", ".join(f"{x:.3e}" for x in error_batches.std(axis=0)))

    rmse_rel = np.array(rmse_rel)
    logging.info("TEST relative RMSE values:" + ", ".join(f"{abs(x):.3e}" for x in rmse_rel.mean(axis=0)))
    if epoch == nepoch:
        with open('failed_test.pkl', 'wb') as g:
            torch.save([y_true_list, y_pred_list], g)
    return


def nntrain(model,
            loader,
            val_loader,
            test_loader,
            NEPOCHS,
            start_dyn,
            fine_dyn,
            device=None
            ):
    device = device if device is not None else model.device
    counts = defaultdict(int)
    for name, param in model.named_parameters():
        top_level = name.split('.')[0]  # e.g., "radialemb", "prod", etc.
        if param.requires_grad:
            counts[top_level] += param.numel()

    for block, count in counts.items():
        logging.info(f"{block:<25}: {count:,} params")

    logging.warning(f"{model.count_parameters()}")

    optimizer = start_dyn['optimizer'](model.parameters())
    scheduler = start_dyn['scheduler'](optimizer)

    error = []
    logging.info(r"                          " +
                 "$Y^0_0$    " +
                 "$Y^1_{-1}$ " +
                 "$Y^1_0$    " +
                 "$Y^1_1$    " +
                 "$Y^2_{-2}$ " +
                 "$Y^2_{-1}$ " +
                 "$Y^2_0$    " +
                 "$Y^2_1$    " +
                 "$Y^2_1$")

    for epoch in range(NEPOCHS):
        error_batches = []
        total_loss = 0

        if epoch == start_dyn['START_FINE']:
            optimizer = fine_dyn['optimizer'](model.parameters())
            scheduler = fine_dyn['scheduler'](optimizer)

        for X, X_v, node_attr, edge_index, Y_true in loader:

            optimizer.zero_grad()  # Zeroing gradients

            # PREPARE THE DATA ON THE CORRECT DEVICE

            Y_true = Y_true.to(device)
            X = [xx.to(device) for xx in X]
            X_v = [xv.to(device) for xv in X_v]
            node_attr = [na.to(device) for na in node_attr]
            edge_index = [ei.to(device) for ei in edge_index]

            Y_pred = model(X, X_v, node_attr, edge_index)  # Forward pass

            loss = model.loss_fn(Y_pred, Y_true)  # Loss calculation
            loss.backward()  # Backpropagation

            mse_components = model.mse_components(Y_pred, Y_true)
            error_batches.append(mse_components.mean(axis=0).tolist())

            optimizer.step()  # Update weights

            total_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}")

        for param_group in optimizer.param_groups:
            logging.info(f"LR: {param_group['lr']}")

        error_batches = np.array(error_batches)

        logging.info("TRAIN MEAN Loss values:   " + ", ".join(f"{x:.3e}" for x in error_batches.mean(axis=0)))
        logging.info("TRAIN STD  Loss values:   " + ", ".join(f"{x:.3e}" for x in error_batches.std(axis=0)))

        logging.info(f"TRAIN      Loss =                    {total_loss / len(loader):.4f} ")

        val_loss = validate(model, val_loader, device)

        error.append(np.array(error_batches))

        test(model, test_loader, device, epoch=epoch, nepoch=NEPOCHS-1)

        scheduler.step(val_loss)

        torch.save(model, "../checkpoint.pth")

    np.save('../training', np.array(error))
