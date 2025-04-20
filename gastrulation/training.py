from preprocessing import preprocess
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
import timeit


def visualise(imgTensor, filenameBase="mouse_embryo", save=True, show=False):
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
        writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(filenameBase + ".gif", writer=writer)

    if show:
        update(0)
        plt.show()
        plt.close()

    return


frames = preprocess()
print(f'{frames[0].shape = }')
print(f'{len(frames) = }')


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
            # THIS MIGHT NEED TO CHANGE IM NOT SURE IF THIS IS RIGHT! (32,32,32 it was target_voxel.shape[1], 2 and 3 before...)
            32,
            32,
            32,
        )
        for i in range(updates):
            state = model(state)
            print("processed state shape is", state.shape)
            frames_array[i] = state[0]
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
        output = output[:, 0:1, :, :, :]
        target_voxel = target_voxel[0:1, 0:1, :, :, :]

        # print(f"Batch index: {batch_idx}")
        # print(f"Output shape: {output.shape}")
        # print(f"Target voxel shape: {target_voxel[batch_idx:batch_idx+1, 0:1, :, :, :].shape}")

        loss = LOSS_FN(output, target_voxel)
        batch_losses[batch_idx] = loss.item()
        loss.backward()
        optimiser.step()

    # print loss as a percentage 
    print(f"BATCH LOSS = {batch_losses.cpu().numpy().mean() * 100:.4f}%")
    return batch_losses.cpu().numpy().mean()

def get_batch(target_index):
    """
    Get a batch for the input of training. Current implementation is 
    1/4 is the seed image (start image ) to the target
    1/2 is random images between start and target idx - 1 -> target
    1/4 is target-1 -> target

    This is more experiemental to see how the model will learn.
    Right now i also think that this will only work for a batch size of 32 since 
    that is what matches the framesoutput from preprocessing. This should change here in the 
    indexing we dont need to change preprocessing probably.
    """
    batch_images = []
    quarter = BATCH_SIZE//4
    if target_index == 0:
        return frames[0]
    else:
        for i in range(BATCH_SIZE):
            if i < quarter:
                img = frames[0][i]
            elif i < quarter * 3:
                rand_idx = random.randint(0, target_index - 1)
                img = frames[rand_idx][i]
            else:
                img = frames[target_index -1][i]
            batch_images.append(img)
        output = torch.stack(batch_images)
        print("BATCH IMAGES AFTER USING GET_BATCH FUNCTION IS", output.shape)
        return output


def train(model: nn.Module, target_voxels: torch.Tensor, optimiser, record=False):
    device = next(model.parameters()).device
    print(device)
    # target voxel is modulo length of data where half of batch is 1 -> this image 
    # and the other half is -1 of this image -> this image 
    # for example say we are up to epoch 471 

    # target_voxel = target_voxels[-1]
    # target_voxel = target_voxel.to(device)
    start = timeit.timeit()
    try:
        training_losses = []
        for epoch in range(EPOCHS):
            # eg epoch = 571, frames_length = 265; 571 % 265 = 41
            target_index  = epoch % FRAMES_LENGTH
            target_voxel = target_voxels[target_index]
            target_voxel = target_voxel.to(device)

            model.train()
            if record:
                outputs = torch.zeros_like(batch)

            # first quater frames are the input image, then the remaining 3/4 are random indexes between index 0 and current target index
            batch = get_batch(target_index)
            # batch = frames[0]
            batch = batch.to(device)
            print("BATCH SHAPE IS: ", batch.shape)
            assert batch.shape == target_voxel.shape
            loss = update_pass(model, batch, target_voxel, optimiser)
            training_losses.append(loss)
    except KeyboardInterrupt:
        pass
    end = timeit.timeit()
    print(f"Time taken to train for epoch {epoch} is", end - start, "seconds")
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
    FRAMES_LENGTH = 265
    TRAINING = True
    GRID_SIZE = 32
    CHANNELS = 16
    OUTPUT_NAME = "EMBRYO"

    MODEL = NCA_3D()
    EPOCHS = 20 * FRAMES_LENGTH # 1 epoch should iterate over the entire dataset.
    BATCH_SIZE = 32
    UPDATES_RANGE = [64, 96]

    LR = 1e-4
    initialiseGPU(MODEL)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)
    LOSS_FN = torch.nn.MSELoss(reduction="mean")

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

           # ## Switch state to evaluation to disable dropout e.g.
    print("EVAL MODE")
    MODEL.eval()

    # ## Plot final state of evaluation OR evaluation animation
    img = frames[0]
    print("FORWARD PASSING")
    model_generated_voxel = forward_pass(MODEL, img, 96, record=True)
    print("VISUALISING")
    anim = visualise(model_generated_voxel, save=True, show=False)