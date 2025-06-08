import torch
from torch import Tensor
import torch.nn as nn
from model import NCA_3D
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from enum import Enum
from loss import lossFn
from util import (
    save_tensor,
    save_model_state,
    minimise_voxel,
    new_seed,
    new_numpy_seed,
    load_image,
    getLosses,
    initialiseGPU,
    visualise,
)

"""
target_voxel : rgba, x, y, z
seed: rgba, x, y, z
output: batch, rgba, x, y, z
"""


class Debug(Enum):
    OFF = 0
    VERBOSE = 1
    CONCISE = 2


if torch.cuda.is_available():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)
    # torch.set_default_device("cpu")


def forward_pass(model: nn.Module, state, updates, record=False):  # TODO
    """
    Run a forward pass consisting of `updates` number of updates
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """
    if record:
        frames_array = Tensor(
            updates,
            CHANNELS,
            target_voxel.shape[1],
            target_voxel.shape[2],
            target_voxel.shape[3],
        )
        for i in range(updates):
            state = model(state)
            frames_array[i] = state
        return frames_array

    else:
        for i in range(updates):
            state = model(state)

    return state


def update_pass(model, batch, target_voxel, optimiser):
    """
    Back calculate gradient and update model paramaters
    """
    device = next(model.parameters()).device
    batch_losses = torch.zeros(BATCH_SIZE, device=device)
    for batch_idx in range(BATCH_SIZE):
        optimiser.zero_grad()
        updates = random.randrange(UPDATES_RANGE[0], UPDATES_RANGE[1])

        output = forward_pass(
            model=model, state=batch[batch_idx].unsqueeze(0), updates=updates
        )

        ## Apply voxel-wise MSE loss between RGBA channels in the grid and the target_voxel pattern
        output = output.squeeze(0)[0:4, :, :, :]

        loss = LOSS_FN(output, target_voxel)
        batch_losses[batch_idx] = loss.item()
        loss.backward()
        optimiser.step()
    return batch_losses.cpu().numpy()


def train(
    model: nn.Module,
    target_voxel: torch.Tensor,
    optimiser,
    record=False,
    DEBUG_MODE=Debug.OFF,
    training_losses=[],
):
    device = next(model.parameters()).device

    target_voxel = target_voxel.to(device)

    try:
        for epoch in range(EPOCHS):
            model.train()
            if record:
                outputs = torch.zeros_like(batch)

            batch = new_seed(target_voxel=target_voxel, batch_size=BATCH_SIZE)
            batch = batch.to(device)

            losses = update_pass(model, batch, target_voxel, optimiser)
            training_losses.append(np.mean(losses))

            # Print loss statistics
            if DEBUG_MODE == Debug.VERBOSE:
                print(f"epoch {epoch+1}/{EPOCHS} loss = {losses}")
            elif DEBUG_MODE == Debug.CONCISE:
                print(
                    f"""
                Epoch {epoch + 1}/{EPOCHS}
                    Mean loss = {np.mean(losses):.4e}
                    Std loss  = {np.std(losses):.4e}
                    Min loss  = {np.min(losses):.4e}
                    Max loss  = {np.max(losses):.4e}
                """.strip().replace(
                        " " * 16, "    "
                    )
                )

            recordRate = 5  # Loss graph will be updated every x epochs, and model will be saved every x epochs.
            if epoch % recordRate == 0 and epoch != 0:
                print(f"saving model, epoch: {epoch}")
                torch.save(MODEL.state_dict(), f"{VOXEL_PATH_NAME}.pth")
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)

                ax.cla()
                ax.set_yscale("log")
                ax.set_xlim(0, len(training_losses))
                ax.set_ylim(min(training_losses), max(training_losses))
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Loss")
                ax.plot(training_losses, ".", alpha=0.2)
                plt.savefig("loss.png")
                if LOSS_LOGGING:
                    with open("losses.csv", "a") as f:
                        losses_str = (
                            ",".join(
                                f"{loss:.6f}"
                                for loss in training_losses[-recordRate - 1 : -1]
                            )
                            + ","
                        )
                        f.write(losses_str)

    except KeyboardInterrupt:
        pass

    if record:
        return (model, training_losses, outputs)
    else:
        return model, training_losses


if __name__ == "__main__":

    DEBUG_MODE = Debug.CONCISE  # OFF, VERBOSE, CONCISE
    LOSS_LOGGING = True

    torch.manual_seed(0)
    TRAINING = True
    GRID_SIZE = 32
    CHANNELS = 16
    VOXEL_PATH_NAME = "potted_flower"

    MODEL = NCA_3D()
    EPOCHS = 1
    BATCH_SIZE = 2
    UPDATES_RANGE = [64, 96]

    LR = 1e-4  # Suggestion: 1e-3 for hours of training, 1e-4 for tens of hours.
    initialiseGPU(MODEL)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)

    # LOSS_FN = torch.nn.MSELoss(reduction="mean")
    LOSS_FN = lossFn

    target_voxel = load_image(f"./voxel_models/{VOXEL_PATH_NAME}.vox")
    target_voxel = minimise_voxel(target_voxel).cpu()
    # anim = visualise(target_voxel, save=False, show=True)

    if os.path.exists(f"{VOXEL_PATH_NAME}.pth"):
        MODEL.load_state_dict(
            torch.load(f"{VOXEL_PATH_NAME}.pth", map_location=torch.device("cpu"))
        )

    if TRAINING:
        losses = getLosses()
        MODEL, losses = train(
            MODEL,
            target_voxel,
            optimizer,
            DEBUG_MODE=DEBUG_MODE,
            training_losses=losses,
        )
        torch.save(MODEL.state_dict(), f"{VOXEL_PATH_NAME}.pth")

        # Plot losses, and save to loss.png
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.cla()
        ax.set_yscale("log")
        ax.set_xlim(0, EPOCHS)
        ax.set_ylim(min(losses), losses[0])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.plot(losses, ".", alpha=0.2)
        plt.savefig("loss.png")

    # ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()

    # ## Plot final state of evaluation OR evaluation animation
    img = new_seed(target_voxel=target_voxel, batch_size=1)
    model_generated_voxel = forward_pass(MODEL, img, 96, record=True)
    save_tensor(model_generated_voxel, VOXEL_PATH_NAME)
    numpy_seed_state = new_numpy_seed(target_voxel=target_voxel, batch_size=1)
    save_model_state(MODEL, numpy_seed_state, VOXEL_PATH_NAME)
    anim = visualise(model_generated_voxel, VOXEL_PATH_NAME, save=True, show=True)
