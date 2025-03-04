import torch
from torch import Tensor
import torch.nn as nn
from model import NCA_3D
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation
from midvoxio.voxio import vox_to_arr

"""
target_voxel : rgba, x, y, z
seed: rgba, x, y, z
output: batch, rgba, x, y, z
"""

if torch.cuda.is_available():
    torch.set_default_device("cuda")


def visualise(imgTensor, filenameBase="minecraft", save=True, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ## If imgTensor does not have batch dimension, add batch dimension of size 1
    if imgTensor.ndim < 5:
        imgTensor = imgTensor.unsqueeze(0)

    ## Permute the tensor to (batch, x, y, z, channel) from (batch, channel, x, y, z)
    imgTensor = imgTensor.permute(0, 2, 3, 4, 1)

    ## Convert tensor to numpy array (as otherwise matplotlib cannot transpose it (when moveaxis is used))
    imgTensor = imgTensor.detach().numpy()

    ## Voxels look like they have their x and y swapped when plotted with matplotlib, so swap them for visualisation
    imgTensor = np.moveaxis(imgTensor, (1, 2), (1, 2))

    ## Calculate minimum edge length to ensure all voxels are cuboid
    ax.set_box_aspect(imgTensor.shape[1:4] / np.max(imgTensor.shape[1:4]))

    def update(imgIdx):
        ## Clear axis to avoid overlaying plots
        ax.cla()

        ## Fix all axis range to show growth better
        ax.set_xlim([0, imgTensor.shape[1]])
        ax.set_ylim([0, imgTensor.shape[2]])
        ax.set_zlim([0, imgTensor.shape[3]])

        ## TODO: might need to reset this Set aspect ratio to be equal

        ## Only plot voxels with alpha channel > 0.1, and clip the RGBA channels to be between 0 and 1
        ax.voxels(
            filled=(imgTensor[imgIdx, :, :, :, 3] > 0.1),
            facecolors=np.clip(imgTensor[imgIdx, :, :, :, :4], 0, 1),
        )
        ax.set_title(f"Frame {imgIdx}")

    if save:
        ## Create an animation with the number of frames equal to the time dimension
        ani = animation.FuncAnimation(fig, update, frames=len(imgTensor), repeat=False)
        writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(filenameBase + ".gif", writer=writer)

    if show:
        update(imgIdx=0)
        plt.show()
        plt.close()

    return


def new_seed(target_voxel, batch_size=1):
    """
    Seed is a cube map that sets a singular pixel activated in form
    """
    SHAPE = [target_voxel.shape[i] for i in range(len(target_voxel.shape))]
    seed = torch.zeros(batch_size, CHANNELS, SHAPE[1], SHAPE[2], SHAPE[3])

    ## Batch, channels, x, y, z
    seed[:, 3, SHAPE[1] // 2, SHAPE[2] // 2, 0] = (
        1  #  Alpha channel = 3 (as 4th value in RGBA channel)
    )
    return seed


def load_image(imagePath: str):
    voxel = vox_to_arr(imagePath)
    voxel_tensor = torch.tensor(voxel).float()
    return voxel_tensor.permute(3, 0, 1, 2)


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

    print(f"batch loss = {batch_losses.cpu().numpy()}")


def train(model: nn.Module, target_voxel: torch.Tensor, optimiser, record=False):
    device = next(model.parameters()).device

    target_voxel = target_voxel.to(device)

    try:
        training_losses = []
        for epoch in range(EPOCHS):
            model.train()
            if record:
                outputs = torch.zeros_like(batch)

            batch = new_seed(target_voxel=target_voxel, batch_size=BATCH_SIZE)
            batch = batch.to(device)

            update_pass(model, batch, target_voxel, optimiser)

    except KeyboardInterrupt:
        pass

    if record:
        return (model, training_losses, outputs)
    else:
        return model, training_losses


def initialiseGPU(model):
    ## Check if GPU available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")

    ## Configure device as GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    return model


if __name__ == "__main__":
    TRAINING = True
    GRID_SIZE = 32
    CHANNELS = 16

    MODEL = NCA_3D()
    EPOCHS = 20
    BATCH_SIZE = 32
    UPDATES_RANGE = [48, 64]

    LR = 1e-3
    initialiseGPU(MODEL)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
    LOSS_FN = torch.nn.MSELoss(reduction="mean")

    target_voxel = load_image("./voxel_models/donut.vox")
    # anim = visualise(target_voxel, save=False, show=True)

    if TRAINING:
        if os.path.exists("Minecraft.pth"):
            MODEL.load_state_dict(
                torch.load("Minecraft.pth", map_location=torch.device("cpu"))
            )
        MODEL, losses = train(MODEL, target_voxel, optimizer)
        torch.save(MODEL.state_dict(), "Minecraft.pth")

    ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()

    ## Plot final state of evaluation OR evaluation animation
    img = new_seed(target_voxel=target_voxel, batch_size=1)
    model_generated_voxel = forward_pass(MODEL, img, 64, record=True)
    anim = visualise(model_generated_voxel, save=True, show=True)
