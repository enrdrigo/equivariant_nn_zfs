import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
import logging
import sys

# Logger that logs only to stdout
console_logger = logging.getLogger('console_logger')
console_logger.setLevel(logging.INFO)
console_logger.propagate = False  # prevent message propagation
console_handler = logging.StreamHandler(sys.stdout)

console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
console_logger.addHandler(console_handler)

# Logger that logs only to file
file_logger = logging.getLogger('file_logger')
file_logger.setLevel(logging.INFO)
file_logger.propagate = False

file_handler = logging.FileHandler('training.log', mode='w')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('[%(levelname)s] %(message)s')
file_handler.setFormatter(file_formatter)
file_logger.addHandler(file_handler)

# Logger that logs only to file
file_validation_logger = logging.getLogger('file_validation_logger')
file_validation_logger.setLevel(logging.INFO)
file_validation_logger.propagate = False

file_validation_handler = logging.FileHandler('validation.log', mode='w')
file_validation_handler.setLevel(logging.INFO)
file_validation_formatter = logging.Formatter('[%(levelname)s] %(message)s')
file_validation_handler.setFormatter(file_validation_formatter)
file_validation_logger.addHandler(file_validation_handler)

# Logger that logs only to file
file_test_logger = logging.getLogger('file_test_logger')
file_test_logger.setLevel(logging.INFO)
file_test_logger.propagate = False

file_test_handler = logging.FileHandler('testing.log', mode='w')
file_test_handler.setLevel(logging.INFO)
file_test_formatter = logging.Formatter('[%(levelname)s] %(message)s')
file_test_handler.setFormatter(file_test_formatter)
file_test_logger.addHandler(file_test_handler)

def collate_fn(batch_):
    """
    Custom collate function to handle variable-length descriptors in the batch.
    """
    batches, targets_ = zip(*batch_)

    # We can't stack the descriptors directly because they have different sizes
    # Instead, we keep them in a list
    targets_ = torch.stack(targets_)

    return {'batches': list(batches), 'targets': targets_}

def validate(epoch, model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batches in loader:
            # PREPARE THE DATA ON THE CORRECT DEVICE
            batch_data = [data.to(device) for data in batches['batches']]

            y_true = batches['targets'].to(device)

            y_pred = model(batch_data)  # Forward pass

            loss = model.loss_fn(y_pred, y_true)  # Loss calculation

            total_loss += loss.item()

            mse_components = model.mse_components(y_pred, y_true)

            file_validation_logger.info(f"{epoch:.3e}  "+" ".join(f"{mse:.3e}" for mse in mse_components.mean(axis=0).tolist()))

    avg_loss = total_loss / len(loader)

    console_logger.info(f"VAL        Loss =                    {avg_loss:.4f}")
    return avg_loss


def test(epoch, model, loader, device):

    model.eval()

    error_batches = []

    total_loss = 0

    with torch.no_grad():
        for batches in loader:
            # PREPARE THE DATA ON THE CORRECT DEVICE
            batch_data = [data.to(device) for data in batches['batches']]

            y_true = batches['targets'].to(device)

            y_pred = model(batch_data)  # Forward pass

            loss = model.loss_fn(y_pred, y_true)

            mse_components = model.mse_components(y_pred, y_true).mean(axis=0)  # Loss calculation

            error_batches.append(mse_components.tolist())

            total_loss += loss.item()

    error_batches = torch.tensor(error_batches)
    avg_loss = total_loss / len(loader)

    console_logger.info(f"TEST       Loss =                    {avg_loss:.4f}")
    console_logger.info("TEST  MEAN Loss values:   " + ", ".join(f"{err:.3e}" for err in error_batches.mean(dim=0)))
    console_logger.info("TEST  STD  Loss values:   " + ", ".join(f"{err:.3e}" for err in error_batches.std(dim=0)))
    file_test_logger.info(f"{epoch:.3e}  "+f"{avg_loss:.4f}")
    return


def train(model,
          train_data,
          val_data,
          test_data,
          n_epochs,
          optimizer,
          scheduler,
          start_epoch,
          batch_size,
          num_segments,
          device=None
          ):
    device = device if device is not None else model.device

    test_loader = DataLoader(test_data,
                             batch_size=1,
                             collate_fn=collate_fn
                             )

    val_loader = DataLoader(val_data,
                                   batch_size=1,
                                   collate_fn=collate_fn
                                   )

    counts = defaultdict(int)
    for name, param in model.named_parameters():
        top_level = name.split('.')[0]  # e.g., "radialemb", "prod", etc.
        if param.requires_grad:
            counts[top_level] += param.numel()

    for block, count in counts.items():
        console_logger.info(f"{block:24s}: {count:,} params")

    console_logger.warning(f"{model.count_parameters()}")

    error = []

    total_data = len(train_data)

    segment_size = total_data // num_segments

    console_logger.warning(f"{segment_size} segments' size")

    for epoch in range(start_epoch, n_epochs):
        error_batches = []

        total_loss = 0

        # Cycle through segments: 0,1,2,0,1,2,...
        segment_idx = epoch % num_segments
        start_segment = segment_idx * segment_size
        end_segment = start_segment + segment_size
        if segment_idx == num_segments - 1: end_segment = total_data

        sub_train = Subset(train_data, range(start_segment, end_segment))

        loader = DataLoader(sub_train,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn
                            )

        console_logger.info(f"Epoch {epoch + 1}: using segment {segment_idx + 1} of {num_segments} (data {start_segment}-{end_segment - 1})")

        for batches in loader:

            optimizer.zero_grad()  # Zeroing gradients

            # PREPARE THE DATA ON THE CORRECT DEVICE
            batch_data = [data.to(device) for data in batches['batches']]

            y_true = batches['targets'].to(device)

            y_pred = model(batch_data)  # Forward pass

            loss = model.loss_fn(y_pred, y_true)  # Loss calculation

            loss.backward()  # Backpropagation

            mse_components = model.mse_components(y_pred, y_true)

            error_batches.append(mse_components.mean(axis=0).tolist())

            file_logger.info(f"{epoch:.3e}  "+" ".join(f"{mse:.3e}" for mse in mse_components.mean(axis=0).tolist()))

            optimizer.step()  # Update weights

            total_loss += loss.item()

        for param_group in optimizer.param_groups:
            console_logger.info(f"LR: {param_group['lr']}")

        error_batches = torch.tensor(error_batches)

        console_logger.info("TRAIN MEAN Loss values:   " + ", ".join(f"{err:.3e}" for err in error_batches.mean(dim=0)))

        console_logger.info("TRAIN STD  Loss values:   " + ", ".join(f"{err:.3e}" for err in error_batches.std(dim=0)))

        console_logger.info(f"TRAIN      Loss =                    {total_loss / len(loader):.4f} ")

        val_loss = validate(epoch, model, val_loader, device)

        error.append(torch.tensor(error_batches))

        test(epoch, model, test_loader, device)

        scheduler.step(val_loss)

        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': epoch,
        }, 'checkpoint.pth')

        torch.save(model, 'checkpoint_model.pth')

    torch.save(model, 'checkpoint_model_final.pth')

    torch.save(torch.stack(error, dim=-1), 'training.pth')
