from preprocessing import preprocessing
from torch import Tensor
import torch
import torch.nn as nn
from model import NCA_3D
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation
from midvoxio.voxio import vox_to_arr


frames = preprocessing()
print(f'{frames[0].shape = }')


def forward_pass(model: nn.Module, state, updates, record=False):  # TODO
    """
    Run a forward pass consisting of `updates` number of updates
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """
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
        output = output[0:1, :, :, :]
        target_voxel = target_voxel[:, 0:1]

        loss = LOSS_FN(output, target_voxel)
        batch_losses[batch_idx] = loss.item()
        loss.backward()
        optimiser.step()

    print(f"batch loss = {batch_losses.cpu().numpy()}")

def train(model: nn.Module, target_voxels: torch.Tensor, optimiser, record=False):
    device = next(model.parameters()).device
    print(device)
    target_voxel = target_voxels[-1]
    target_voxel = target_voxel.to(device)

    try:
        training_losses = []
        for epoch in range(EPOCHS):
            model.train()
            if record:
                outputs = torch.zeros_like(batch)

            batch = frames[0]
            batch = batch.to(device)
            print(batch.shape)
            assert batch.shape == target_voxel.shape
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
    torch.manual_seed(0)
    TRAINING = True
    GRID_SIZE = 32
    CHANNELS = 16
    VOXEL_PATH_NAME = "donut"

    MODEL = NCA_3D()
    EPOCHS = 50
    BATCH_SIZE = 32
    UPDATES_RANGE = [64, 96]

    LR = 1e-3
    initialiseGPU(MODEL)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
    LOSS_FN = torch.nn.MSELoss(reduction="mean")

    if TRAINING:
        MODEL, losses = train(MODEL, frames, optimizer)
        torch.save(MODEL.state_dict(), f"{VOXEL_PATH_NAME}.pth")
