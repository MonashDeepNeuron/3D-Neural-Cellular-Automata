import numpy as np
import os
import torch
from midvoxio.voxio import vox_to_arr
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════════════════════════
#                           INPUT OUTPUT
# ════════════════════════════════════════════════════════════════════


def load_image(imagePath: str):
    voxel = vox_to_arr(imagePath)
    voxel_tensor = torch.tensor(voxel).float()
    if torch.cuda.is_available():
        voxel_tensor = voxel_tensor.to(torch.float64)
    else:
        voxel_tensor = voxel_tensor.to(torch.float32)

    return voxel_tensor.permute(3, 0, 1, 2)


def save_tensor(simulation_tensor, filenameBase="test", directory="tensors"):
    """
    Saves the tensor containing the simiulation as an npy file
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    ## If imgTensor does not have step dimension, add step dimension of size 1
    if simulation_tensor.ndim < 5:
        simulation_tensor = simulation_tensor.unsqueeze(0)

    ## Permute the tensor to (step, x, y, z, channel) from (step, channel, x, y, z)
    simulation_tensor = simulation_tensor.permute(0, 2, 3, 4, 1)

    ## Convert tensor to numpy array (as otherwise matplotlib cannot transpose it (when moveaxis is used))
    simulation_tensor = simulation_tensor.detach().numpy()

    # ## Voxels look like they have their y and z swapped when plotted with matplotlib, so swap them for visualisation
    simulation_tensor = np.moveaxis(simulation_tensor, (2, 3), (3, 2))

    ## Set alpha values < 0.1 to 0
    alphas = simulation_tensor[:, :, :, :, 3]
    simulation_tensor[:, :, :, :, 3] = np.where(alphas > 0.1, alphas, 0)

    ## Clip RGBA channels between [0,1]
    simulation_tensor[:, :, :, :, :4] = np.clip(simulation_tensor[:, :, :, :, :4], 0, 1)

    ## Remove hidden channels
    simulation_tensor = simulation_tensor[..., :4]

    ## Serialise and save tensor
    np.save(f"{directory}/{filenameBase}", simulation_tensor)


def save_model_state(model, seed, filenameBase="test", directory="state", has_learnable_network = True):
    """
    Saves the model state as a npy file. This state is a dictionary with structure:
    {
        "seed" : np.array,
        "layers.0.weight" : np.array,
        "layers.0.bias" : np.array,
        "layers.2.weight" : np.array
    }
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    state_dict = {
        k: v.detach().cpu().numpy().astype(np.float32)
        for k, v in model.state_dict().items()
    }

    state_dict["seed"] = seed
     
    if has_learnable_network:
        state_dict = {k: v for k, v in state_dict.items() if ".update_net." not in k}

    np.save(f"{directory}/{filenameBase}", state_dict)


# ════════════════════════════════════════════════════════════════════
#                           TRAINING UTIL
# ════════════════════════════════════════════════════════════════════


def getLosses(fileName="losses.csv"):
    losses = []

    if not os.path.exists(fileName):
        print("No existing losses detected")
        return []
    try:
        with open(fileName, mode="r") as file:
            while True:
                buffer = file.read(8)  # Load float into buffer
                if not buffer:  # exit if EOF
                    break

                file.read(1)  # skip the comma
                losses.append(float(buffer))
    except ValueError:
        print("Losses file corrupted")
        return []
    return losses


def initialiseGPU(model):
    ## Check if GPU available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")

    ## Configure device as GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    return model, device


# ════════════════════════════════════════════════════════════════════
#                           TENSOR OPERATIONS
# ════════════════════════════════════════════════════════════════════


def minimise_voxel(imgTensor):
    imgTensor
    non_zeros = (torch.nonzero(imgTensor[3, :, :, :])).permute(1, 0)

    min_bounds = [torch.min(non_zeros[i]) for i in range(non_zeros.shape[0])]
    max_bounds = [torch.max(non_zeros[i]) for i in range(non_zeros.shape[0])]

    imgTensor = imgTensor[
        :,
        min_bounds[0] : max_bounds[0] + 1,
        min_bounds[1] : max_bounds[1] + 1,
        min_bounds[2] : max_bounds[2] + 1,
    ]
    return imgTensor


def new_seed(target_voxel, channels=16, batch_size=1):
    """
    Seed is a cube map that sets a singular pixel activated in form
    """
    SHAPE = [target_voxel.shape[i] for i in range(len(target_voxel.shape))]
    seed = torch.zeros(batch_size, channels, SHAPE[0], SHAPE[1], SHAPE[2]) # need to be changed depending on xyzc or cxyz

    x, y, z = compute_seed_position(target_voxel, SHAPE)
    seed[:, 3, x, y, z] = 1
    return seed

def compute_seed_position(target_voxel, shape):
    # note that y and z are swapped around in target_voxel, thus height = z
    c, x, y, z = shape
    x_ideal_index = x // 2
    y_ideal_index = y // 2
    z_ideal_index = 0
    
    # alive_cells_positions = np.where(target_voxel[3, ..., 0] > 0) ## positions of alive cells on bottom row 
    alive_cells_positions = torch.where(target_voxel[3, ..., 0] > 0) ## positions of alive cells on bottom row 

    ## TODO: use argmin to figure out which index in alive_cell_positions
    # index = (np.abs(alive_cells_positions[0] - x_ideal_index) + np.abs(alive_cells_positions[1] - y_ideal_index)).argmin()
    index = (torch.abs(alive_cells_positions[0] - x_ideal_index) + torch.abs(alive_cells_positions[1] - y_ideal_index)).argmin()

    ## find the index in which we minimise the x and y dimensions whilst keep z = 0 (if none exist when z = 0 this 
    ## should be cropped away by minimal cropping)

    return (
        alive_cells_positions[0][index],
        alive_cells_positions[1][index],
        0
    )


def new_numpy_seed(target_voxel, channels=16, batch_size=1):
    """
    Seed is a cube map that sets a singular pixel activated in form.
    Returns a NumPy array of shape (batch_size, channels, X, Y, Z).
    """
    SHAPE = [target_voxel.shape[i] for i in range(len(target_voxel.shape))]
    seed = np.zeros((batch_size, channels, SHAPE[0], SHAPE[1], SHAPE[2]), dtype=np.float32) # shape need to be changed depending on xyzc or cxyz
    x, y, z = compute_seed_position(target_voxel, SHAPE)
    seed[:, 3, x, y, z] = 1
    return seed


# ════════════════════════════════════════════════════════════════════
#                           VISUALISATION
# ════════════════════════════════════════════════════════════════════


def visualise(imgTensor, filenameBase, save=True, show=False, loss_suffix = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ## If imgTensor does not have batch dimension, add batch dimension of size 1
    if imgTensor.ndim < 5:
        imgTensor = imgTensor.unsqueeze(0)

    ## Permute the tensor to (batch, x, y, z, channel) from (batch, channel, x, y, z)
    imgTensor = imgTensor.permute(0, 2, 3, 4, 1)

    ## Convert tensor to numpy array (as otherwise matplotlib cannot transpose it (when moveaxis is used))
    imgTensor = imgTensor.detach().cpu().numpy()

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
        if loss_suffix:
            ani.save("gif/"+filenameBase + "_" + str(loss_suffix) + ".gif", writer=writer)
        else:
            ani.save("gif/"+filenameBase + ".gif", writer=writer)

    if show:
        update(imgIdx=0)
        plt.show()
        plt.close()

    return
