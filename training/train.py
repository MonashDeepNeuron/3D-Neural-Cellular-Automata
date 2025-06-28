import torch
from torch import Tensor
import torch.nn as nn
from model import NCA3DModel
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from enum import Enum
from loss import Loss
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

# memory issues
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()  

"""
target_voxel : rgba, x, y, z
seed: batch, rgba, x, y, z
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


def generate_gif_tensor(model: nn.Module, seed):
    """
    If `record` is true, then records the state in a tensor to animate and saves the video
    Returns the final state
    """
    return model.generate_gif(seed, target_voxel.shape, CHANNELS)

def update_pass_sample_pool(device, model, batch, target_voxel, optimiser):
    """
    Back calculate gradient and update model paramaters
    """
    outputs = torch.zeros(batch.shape, device=device)
    batch_losses = torch.zeros(BATCH_SIZE, device=device)
    for batch_idx in range(BATCH_SIZE):
        optimiser.zero_grad()

        output = model(batch[batch_idx].unsqueeze(0))
        outputs[batch_idx] = output

        ## Apply voxel-wise MSE loss between RGBA channels in the grid and the target_voxel pattern
        output = output.squeeze(0)[0:4, :, :, :]

        loss = LOSS_FN(output, target_voxel)
        batch_losses[batch_idx] = loss.item()
        loss.backward()
        optimiser.step()
    return batch_losses.cpu().numpy(), outputs # might error due to device


def update_pass(device, model, batch, target_voxel, optimiser):
    """
    Back calculate gradient and update model paramaters
    """
    batch_losses = torch.zeros(BATCH_SIZE, device=device)
    for batch_idx in range(BATCH_SIZE):
        optimiser.zero_grad()

        output = model(batch[batch_idx].unsqueeze(0))

        ## Apply voxel-wise MSE loss between RGBA channels in the grid and the target_voxel pattern
        output = output.squeeze(0)[0:4, :, :, :]

        loss = LOSS_FN(output, target_voxel)
        batch_losses[batch_idx] = loss.item()
        loss.backward()
        optimiser.step()
    return batch_losses.cpu().numpy()

def sample_pool_train(
    device, 
    model: nn.Module,
    target_voxel: torch.Tensor,
    optimiser,
    scheduler,
    record=False,
    DEBUG_MODE=Debug.OFF,
    num_samples=8,
    training_losses=[],
):
    
    target_voxel = target_voxel.to(device)

    start_seed = new_seed(target_voxel=target_voxel, batch_size=1)

    batch = start_seed.repeat(SAMPLE_POOL_SIZE, 1, 1, 1, 1).to(device) # batch_size = pool size

    #print(batch.shape)

    try:
        for epoch in range(EPOCHS):
            model.train()
            # if record:
            #     outputs = torch.zeros_like(batch) # would this error?

            indices = torch.randperm(batch.size(0))[:num_samples]
            sampled_batch = batch[indices]

            #print(sampled_batch.shape)

            # replace one item in sample with starting seed
            reset_index = torch.randint(0, sampled_batch.size(0), (1,)).item()
            sampled_batch[reset_index] = start_seed.squeeze(0)

            losses, outputs = update_pass_sample_pool(device, model, sampled_batch, target_voxel, optimiser)
            training_losses.append(np.mean(losses))
            batch[indices] = outputs.detach() # detach comp graph as no longer needed

            scheduler.step() # for learning rate decay

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
                torch.save(MODEL.state_dict(), f"saved_models/{VOXEL_PATH_NAME}.pth")
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

def train(
    device, 
    model: nn.Module,
    target_voxel: torch.Tensor,
    optimiser,
    record=False,
    DEBUG_MODE=Debug.OFF,
    training_losses=[],
):
    #device = next(model.parameters()).device

    target_voxel = target_voxel.to(device)

    try:
        for epoch in range(EPOCHS):
            model.train()
            if record:
                outputs = torch.zeros_like(batch) # would this error?

            batch = new_seed(target_voxel=target_voxel, batch_size=BATCH_SIZE).to(device)

            losses = update_pass(device, model, batch, target_voxel, optimiser)
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
                torch.save(MODEL.state_dict(), f"saved_models/{VOXEL_PATH_NAME}.pth")
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
    VOXEL_PATH_NAME = "donut"

    MODEL = NCA3DModel(hidden_channels=12)
    MODEL, device = initialiseGPU(MODEL) # initialiseGPU returns the Model that is moved onto GPU
    EPOCHS = 100
    SAMPLE_POOL_SIZE = 64 #BATCH_SIZE = 1 # 64 for sample pooling
    BATCH_SIZE = 8
    UPDATES_RANGE = [48, 64]
    LR = 1e-3  # Suggestion: 1e-3 for hours of training, 1e-4 for tens of hours.
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LR)

    # StepLR: decay LR by gamma every step_size epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.0002)

    LOSS_FN = Loss(loss_fn=1)

    target_voxel = load_image(f"./voxel_models/{VOXEL_PATH_NAME}.vox")
    # target_voxel = minimise_voxel(target_voxel).cpu()
    target_voxel = minimise_voxel(target_voxel) 

    if os.path.exists(f"saved_models/{VOXEL_PATH_NAME}.pth"):
        MODEL.load_state_dict(
            torch.load(f"saved_models/{VOXEL_PATH_NAME}.pth", map_location=torch.device("cpu"))
        )

    if TRAINING:
        # losses = getLosses()
        # MODEL, losses = train(
        #     device,
        #     MODEL,
        #     target_voxel,
        #     optimizer,
        #     DEBUG_MODE=DEBUG_MODE,
        #     # training_losses=None,
        # )
        #torch.save(MODEL.state_dict(), f"saved_models/{VOXEL_PATH_NAME}.pth") #saving
        MODEL, losses = sample_pool_train(
            device,
            MODEL,
            target_voxel,
            optimizer,
            scheduler=scheduler,
            DEBUG_MODE=DEBUG_MODE
        )

    # ## Switch state to evaluation to disable dropout e.g.
    MODEL.eval()

    # ## Plot final state of evaluation OR evaluation animation
    gif_seed = new_seed(target_voxel=target_voxel, batch_size=1)
    gif_tensor = generate_gif_tensor(MODEL, gif_seed)
    save_tensor(gif_tensor, VOXEL_PATH_NAME)
    numpy_seed_state = new_numpy_seed(target_voxel=target_voxel, batch_size=1)
    save_model_state(MODEL, numpy_seed_state, VOXEL_PATH_NAME)
    anim = visualise(gif_tensor, VOXEL_PATH_NAME, save=True, show=True)
