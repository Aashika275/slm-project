import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    step,
    loss,
    path
):
    """
    Save model and training state.
    """

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint saved: {path}"
    )


def load_checkpoint(
    model,
    optimizer,
    path,
    device
):
    """
    Load model and training state.
    """

    if not os.path.exists(path):

        print(
            "No checkpoint found."
        )

        return 0, 0, None

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    loss = checkpoint["loss"]

    print(
        f"Checkpoint loaded: {path}"
    )

    print(
        f"Epoch: {epoch}"
    )

    print(
        f"Step: {step}"
    )

    print(
        f"Loss: {loss}"
    )

    return epoch, step, loss
