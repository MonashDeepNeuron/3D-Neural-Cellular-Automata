import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch import Tensor
import os
import torch

RESO_FUCKING_LUTION = 15


# Load the data
# 323-59 = 264 (THERE ARE 264 UNIQUE IMAGES) 265 LOL

def visualise(iterations, filenameBase="gastrulation_animation", save=True, show=False):
    import matplotlib.animation as animation
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    # batch, channels, x, y, z
    # Convert list of 3D iterations to a 4D iterations (time, x, y, z)
    if isinstance(iterations, list):
        tensors_array = np.array(iterations)
    else:
        tensors_array = iterations
        
    # Calculate threshold for visualization (e.g., 0.05 of max value)
    threshold = 0.05
    
    def update(frame_idx):
        ax.cla()
        
        # Get the data for this frame
        frame_data = tensors_array[frame_idx]
        
        # Find non-zero positions (cells) that exceed threshold
        x, y, z = np.where(frame_data > threshold)
        values = frame_data[x, y, z]
        
        # Plot points with color mapping based on density
        scatter = ax.scatter(x, y, z, c=values, cmap='viridis', 
                            alpha=0.8, s=2)
        
        # Set consistent view
        ax.set_xlim(0, RESO_FUCKING_LUTION)
        ax.set_ylim(0, RESO_FUCKING_LUTION)
        ax.set_zlim(0, RESO_FUCKING_LUTION)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Frame {frame_idx}")
        
        # Add colorbar
        # if frame_idx == 0:
        #     fig.colorbar(scatter, ax=ax, label='Density')
            
    if save:
        ani = animation.FuncAnimation(fig, update, frames=len(tensors_array), 
                                     interval=200, blit=False)
        writer = animation.PillowWriter(fps=5)
        ani.save(f"{filenameBase}.gif", writer=writer)
        print(f"Animation saved as {filenameBase}.gif")
    
    if show:
        update(0)
        plt.show()
        
    plt.close(fig)
    return

'''
USING IQR WE FOUND 800~ outliers probably bullshit (not actually outliers)
'''

def preprocessing():

    csv_path = "./gastrulation/Database.csv"
    df = pd.read_csv(csv_path)
    print("DATAFRAME IS", df)
    df.columns = df.columns.str.strip()
    print("Columns in df:", df.columns.tolist())

    # Only keep relevant columns
    data = df[['x', 'y', 'z', 't', 'label', 'id', 'mother_id']].copy()
    print("called after copy")

    Q1 = data[['x', 'y', 'z']].quantile(0.25)
    Q3 = data[['x', 'y', 'z']].quantile(0.75)
    IQR = Q3 - Q1

    print("called IQR")

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    data = data[~((data['x'] < lower_bound['x']) | (data['x'] > upper_bound['x']))]
    data = data[~((data['y'] < lower_bound['y']) | (data['y'] > upper_bound['y']))]
    data = data[~((data[
        'z'] < lower_bound['z']) | (data['z'] > upper_bound['z']))]

    for axis in ['x', 'y', 'z']:
        print("axis", axis)
        data[axis] -= data[axis].mean()

    # Global normalization
    global_max = max([data['x'].max(), data['y'].max(), data['z'].max()])
    global_min = min([data['x'].min(), data['y'].min(), data['z'].min()])


    for axis in ['x', 'y', 'z']:
        print(axis)
        data[axis] = RESO_FUCKING_LUTION* (data[axis] - global_min) / (global_max - global_min)

    # Per axis normalisation
    # for axis in ['x', 'y', 'z']:
    #     min_val = data[axis].min()
    #     max_val = data[axis].max()
    #     data[axis] = RESO_FUCKING_LUTION * (data[axis] - min_val) / (max_val - min_val)

    # sort data by x values
    data = data.sort_values(by=['x', 'y', 'z'])
    # print the max of the sorted
    # print(data["x"])
    # print("POST NORM", data[['x', 'y', 'z']].describe())

    # Get the unique sorted 't' values
    unique_t_values = sorted(data['t'].unique())

    iterations = []
    # create a list mapping the t value (a unique time point) to the corresponding dataframe
    # at that time point

    for t_value in unique_t_values:
        iteration = data[data['t'] == t_value]
        iterations.append(iteration)

    # we have our list of iterations (which are dataframes) and we want to cast them into 

    voxels = [] # each item in here is gonna be of type 
    # batch, channels, x, y, z

    for index, dataframe in enumerate(iterations):
        three_dee = np.zeros([RESO_FUCKING_LUTION+1, RESO_FUCKING_LUTION+1, RESO_FUCKING_LUTION+1], dtype=np.int32)
        # going through a row in the dataframe and add the corresponding integer value (eg round it)
        # to that corresponding position in the three dee tensor 
        for i, row in dataframe.iterrows():
            x = int(row['x'])
            y = int(row['y'])
            z = int(row['z'])
            three_dee[x, y, z] += 1  # Increment for density counting
            if x < 0 or x >= RESO_FUCKING_LUTION or y < 0 or y >= RESO_FUCKING_LUTION or z < 0 or z >= RESO_FUCKING_LUTION:
                print(f"Out of bounds: x={x}, y={y}, z={z} for index {index} in iteration {index}")
                continue
        # print(f"Iteration {index}: {len(dataframe)} points, max value in 3D tensor: {np.max(three_dee)}")
        # Normalize the tensor
        # three_dee = three_dee.astype(np.float32) / np.max(three_dee)
        # print(three_dee)
        print("done iteration", index)
        voxels.append(three_dee)


    max_val = max([np.max(voxel) for voxel in voxels])
    voxels = [voxel/max_val for voxel in voxels]
    # print(max_val)

    if max_val > 0:  # Avoid division by zero
        iterations = [tensor.astype(np.float32) / max_val for tensor in voxels]

    BATCH_SIZE = 32
    CHANNELS = 16

    # we are trying to store a list of each input tensor to the model, where each index of this storage is a frame from the mouse embryo development.
    # this should be a tensor which can be inputted/outputted by the model.
    # we want to fit the alpha mask into this input_image variable. 
    #  

    images = []
    
    for tensor in voxels:
        # Create the input_image tensor
        input_image = torch.zeros(BATCH_SIZE, CHANNELS, RESO_FUCKING_LUTION+1, RESO_FUCKING_LUTION+1, RESO_FUCKING_LUTION+1)
        tensor = torch.tensor(tensor, dtype=torch.float32)  # Convert to PyTorch tensor if it's not already
        input_image[:, 0, :, :, :] = tensor  # Assign the 3D tensor to the first channel (alpha mask)
        images.append(input_image)


    # Minimise Voxel
    min_bounds = (RESO_FUCKING_LUTION, RESO_FUCKING_LUTION, RESO_FUCKING_LUTION)
    max_bounds = (0, 0, 0)

    for iteration in images:
        non_zeros = torch.nonzero(iteration[0, 0, :, :, :], as_tuple=True)
        
        # Update min
        min_bounds = (
            min(min_bounds[0], torch.min(non_zeros[0]).item()),
            min(min_bounds[1], torch.min(non_zeros[1]).item()),
            min(min_bounds[2], torch.min(non_zeros[2]).item())
        )
        
        # Update max
        max_bounds = (
            max(max_bounds[0], torch.max(non_zeros[0]).item()),
            max(max_bounds[1], torch.max(non_zeros[1]).item()),
            max(max_bounds[2], torch.max(non_zeros[2]).item())
        )
    
    # Apply min bounds
    images = [iteration[:, :, min_bounds[0]:max_bounds[0]+1, min_bounds[1]:max_bounds[1]+1, min_bounds[2]:max_bounds[2]+1] for iteration in images]

    return images



def preprocess():
    tensor_path = "./gastrulation/iterations"
    if os.path.exists(tensor_path):
        print("File exists")
        try:
            print("Attempting to fetch data from file")
            return torch.load(tensor_path)
        except Exception as e:
            print("Failed to load tensor:", e)

    print("No data array, preprocess it all")
    data = preprocessing()
    # visualise(data, save=False, show=True)
    if os.path.exists(tensor_path):
        os.remove(tensor_path)  
    print("WRITING SAVE")
    torch.save(data, tensor_path)

    return data

if __name__ == "__main__":
    preprocess()