import torch
from torch import Tensor
import torch.nn as nn
from model_xyzc import NCA3DModel
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from enum import Enum
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

if torch.cuda.is_available():
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    # torch.set_default_device("cpu")

def generate_gif_tensor(model: nn.Module, seed):
    """
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """

    return model.generate_gif(seed, target_voxel.shape, CHANNELS)


if __name__ == "__main__":
    LOSS_LOGGING = True

    torch.manual_seed(0)
    TRAINING = False
    GRID_SIZE = 32
    CHANNELS = 16
    VOXEL_PATH_NAME = "donut"
    SAVED_PATH_NAME = "donut"
    UPDATES_RANGE = [64, 96]

    MODEL = NCA3DModel(hidden_channels=12, update_steps=UPDATES_RANGE, useGPU=False)

    ## Configure device as GPU
    device = torch.device("cpu")
    MODEL = MODEL.to(device)
    BATCH_SIZE = 1
    
    LR = 1e-4  # Suggestion: 1e-3 for hours of training, 1e-4 for tens of hours.
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)

    target_voxel = load_image(f"./voxel_models/{VOXEL_PATH_NAME}.vox")
    # target_voxel = minimise_voxel(target_voxel).cpu()
    target_voxel = minimise_voxel(target_voxel).permute(1,2,3,0) 

    if os.path.exists(f"saved_models/{SAVED_PATH_NAME}.pth"):
        MODEL.load_state_dict(
            torch.load(f"saved_models/{SAVED_PATH_NAME}.pth", map_location=torch.device("cpu"))
        )

    # ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()

    # ## Plot final state of evaluation OR evaluation animation
    gif_seed = new_seed(target_voxel=target_voxel, batch_size=1)
    gif_tensor = generate_gif_tensor(MODEL, gif_seed).permute(0,4,1,2,3)
    save_tensor(gif_tensor, VOXEL_PATH_NAME)
    numpy_seed_state = new_numpy_seed(target_voxel=target_voxel, batch_size=1)
    # save_model_state(MODEL, numpy_seed_state, VOXEL_PATH_NAME)
    anim = visualise(gif_tensor, VOXEL_PATH_NAME, save=True, show=True)

