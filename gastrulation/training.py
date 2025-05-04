import random
import time     
import logging
import sys

from torch import Tensor
import torch.nn as nn
import torch

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from preprocessing import preprocess
from model import NCA_3D
from loss import lossFn 

original_print = print
log_file_name = "log.log"

def print(*args, **kwargs):
    kwargs.pop('file', None)
    with open(log_file_name, "a") as log_file:
        original_print(*args, file=log_file)


print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")


def visualise(imgTensor, filenameBase="mouse_embryo", save=True, show=False):

    if isinstance(imgTensor, list):
        imgTensor = torch.cat([frame[0:1] for frame in imgTensor], dim=0)
    
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
        ax.cla()
        ax.set_xlim([0, imgTensor.shape[1]])
        ax.set_ylim([0, imgTensor.shape[2]])
        ax.set_zlim([0, imgTensor.shape[3]])

        alpha = imgTensor[imgIdx, :, :, :, 0]  # Alpha = channel 0
        filled = alpha > 0.05  # Visibility threshold

        # Fixed black color with full opacity
        color = np.zeros((*alpha.shape, 4))
        color[..., 3] = 1.0  # Alpha channel = 1 (opaque)

        ax.voxels(filled=filled, facecolors=color)
        ax.set_title(f"Frame {imgIdx}")

    if save:
        ani = animation.FuncAnimation(fig, update, frames=len(imgTensor), repeat=False)
        writer = animation.PillowWriter(fps=1, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(filenameBase + ".gif", writer=writer)

    if show:
        update(0)
        plt.show()
        plt.close()

    return


frames = preprocess()

frames = [frame.to(torch.float32) for frame in frames]

if torch.cuda.is_available():
    frames = [frame.to(torch.float64) for frame in frames]


print(f'{frames[0].shape = }')
print(f'{len(frames) = }')


def forward_pass(model: nn.Module, state, updates=1, record=False):  # TODO
    """
    Run a forward pass consisting of `updates` number of updates
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """

    if record:
        # Num batches is disregarded
        _, channels, x_dim, y_dim, z_dim = state.shape

        frames_array = Tensor(
            updates,
            channels, # small thing here but would it be best practice to use the global variable channels or the unpacked variable?
            x_dim,
            y_dim, 
            z_dim,
        )
        for i in range(updates):
            state = model(state)
            # print("processed state shape is", state.shape)
            frames_array[i] = state[0]
        return frames_array

    else:
        for _ in range(updates):
            state = model(state)

    return state

def update_pass(model, batch, updates, target_voxel, optimiser):
    """
    Back calculate gradient and update model paramaters
    """
    device = next(model.parameters()).device
    batch_losses = torch.zeros(BATCH_SIZE, device=device)

    for batch_idx in range(BATCH_SIZE):
        optimiser.zero_grad()

        input_state = batch[batch_idx:batch_idx+1].detach().clone().requires_grad_(True)

        output = forward_pass(
            model = model, 
            state = input_state, 
            updates = updates[batch_idx]
        )

        output_alpha = output[:, 0:1, :, :, :]
        target = target_voxel[0:1, 0:1, :, :, :] 

        loss = LOSS_FN(output_alpha, target)

        batch_losses[batch_idx] = loss.item()
        loss.backward()
        optimiser.step()

    return batch_losses.cpu().numpy().mean()

def get_batch(target_index):
    """
    Get a batch for the input of training. Returns input images and 
    the number of updates needed to reach the target frame.
    """
    batch_images = []
    updates = []
    quarter = BATCH_SIZE//4
    
    if target_index == 0:
        # Just return the first frame with no updates needed
        return frames[0], [0] * BATCH_SIZE
    
    for i in range(BATCH_SIZE):
        if i < quarter:
            batch_images.append(frames[0][i])
            updates.append(target_index)
        elif i < quarter * 3:
            rand_idx = random.randint(0, target_index - 1)
            batch_images.append(frames[rand_idx][i])
            updates.append(target_index-rand_idx)
        else:
            batch_images.append(frames[target_index - 1][i])
            updates.append(1)

    # Ensure the tensors are properly set up for gradient computation
    output = torch.stack(batch_images)
    return output, updates



def train(model: nn.Module, target_voxels: torch.Tensor, optimiser, record=False):
    device = next(model.parameters()).device
    print(device)
    # target voxel is modulo length of data where half of batch is 1 -> this image 
    # and the other half is -1 of this image -> this image 
    # for example say we are up to epoch 471 

    # target_voxel = target_voxels[-1]
    # target_voxel = target_voxel.to(device)
    start = time.time()
    training_losses = []
    try:
        for epoch in range(EPOCHS):
            # eg epoch = 571, frames_length = 265; 571 % 265 = 41
            target_index  = epoch % FRAMES_LENGTH
            target_voxel = target_voxels[target_index]
            target_voxel = target_voxel.to(device)

            model.train()
            if record:
                outputs = torch.zeros_like(batch)

            # first quater frames are the input image, then the remaining 3/4 are random indexes between index 0 and current target index
            batch, updates = get_batch(target_index)
            batch = batch.to(device)
            # batch = frames[0]
            # print("BATCH SHAPE IS: ", batch.shape)
            assert batch.shape == target_voxel.shape

            loss = update_pass(model, batch, updates, target_voxel, optimiser)
            training_losses.append(loss)
            
            # EVERY 265 WE SAVE WEIGHTS AND WRITE A VISUALISE GIF!
            # print(f"Epoch {epoch} - Loss: {loss:.10f}")
            if epoch % FRAMES_LENGTH == 0:
                checkpoint_path = f"UPDATED_model_checkpoint_epoch_{epoch}.pth"
                # torch.save(model.state_dict(), checkpoint_path)
                # print(f"Model weights saved at {checkpoint_path}")
                if len(training_losses) != 0:
                    print(sum(training_losses[-FRAMES_LENGTH:])/len(training_losses[-FRAMES_LENGTH:]))
                # VISUALISE AT CHECKPOINT 
                # with torch.no_grad():
                #     model.eval()  # Set model to evaluation mode
                #     #if i can get this to anim = visualise frame[0] that would be good 
                #     # but having problems with gpu and cpu mixup
                #     model.train()  # Set model back to training mode
                #     # make sure to move the model back to CUDA after evaluation
            # Timing for epoch
            # end = time.time()
            # print(f"Time taken to train for epoch {epoch} is {end - start} seconds")
            # start = time.time()

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
    FRAMES_LENGTH = 14 # standard = 264
    TRAINING = True
    GRID_SIZE = 32
    CHANNELS = 16
    OUTPUT_NAME = "EMBRYO"
    MODEL_WEIGHTS = "./UPDATED_model_checkpoint_epoch_265.pth"

    MODEL = NCA_3D()
    EPOCHS = 10000# 10 * FRAMES_LENGTH 1 epoch should iterate over the entire dataset.
    BATCH_SIZE = 32

    LR = 1e-4
    initialiseGPU(MODEL)
    # try:
    #     MODEL.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    #     print(f"Loaded model weights from {MODEL_WEIGHTS}")
    # except:
    #     pass
        
    print("Device: :", next(MODEL.parameters()).device)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
    LOSS_FN = torch.nn.MSELoss(reduction="mean")
    # LOSS_FN = lossFn

    if TRAINING:
        MODEL, losses = train(MODEL, frames, optimizer)

        torch.save(MODEL.state_dict(), f"{OUTPUT_NAME}.pth")
        plt.plot(losses)
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_loss.png")
        # plt.show()

    # Switch state to evaluation to disable dropout etc.
    print("EVAL MODE")
    MODEL.eval()

    # Plot final state of evaluation OR evaluation animation
    MODEL.eval()
    img = frames[0][0].unsqueeze(0)
    print("FORWARD PASSING")
    
    # Generate sequence showing development over time (200 steps)
    model_generated_voxel = forward_pass(MODEL, img, updates=14, record=True)
    print("VISUALISING")
    anim = visualise(model_generated_voxel, save=True, show=False)