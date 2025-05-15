import numpy as np
import os


def save_tensor(simulation_tensor, filenameBase="tensor_test", directory="tensors"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    ## If imgTensor does not have step dimension, add step dimension of size 1
    if simulation_tensor.ndim < 5:
        simulation_tensor = simulation_tensor.unsqueeze(0)

    ## Permute the tensor to (step, x, y, z, channel) from (step, channel, x, y, z)
    simulation_tensor = simulation_tensor.permute(0, 2, 3, 4, 1)

    ## Convert tensor to numpy array (as otherwise matplotlib cannot transpose it (when moveaxis is used))
    simulation_tensor = simulation_tensor.detach().numpy()

    # ## Voxels look like they have their x and y swapped when plotted with matplotlib, so swap them for visualisation
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
