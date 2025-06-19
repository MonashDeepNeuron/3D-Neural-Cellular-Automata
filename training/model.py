import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init

FLOAT_TYPE = torch.float64 if torch.cuda.is_available() else torch.float32

"""
    Author: Nyan Kyaw
    Last Modified: 15/06/2025

    This script represents the model described in "Growing 3D Artefacts and Functional Machines with Neural Cellular Automata"
    by Sudhakaran et al.

    Note that the expected input is an occupancy matrix [x,y,z] that has colour (RGB) and alpha values, where the alpha value represents
    the occupancy of that cell. 

    Hence, the expected shape of the input is [4, x, y, z]. We can add additional channels if we want to encode additional hidden 
    information. 
    
    If we perform batching (or in our case, sampling), the expected shape of the input becomes [b, 4, x, y, z]. This is what we
    assume we are given.
"""


class LearnablePerceptionNetwork(torch.nn.Module):
    """
    The paper suggests to utilise a 3D Convolutional Learnable Perception Network. We no longer utilise hardcoded sobel
    filters to extract features. The model will learn how to extract useful features using this approach.

    The paper utilises a 3D Convolutional layer with the following hyperparameters:
        - kernel_size = 3
        - stride = 1
        - output_channels = 3 * input_channels (3 as we are getting features in X,Y,Z direction)
        - padding = 1

    No ReLU.
    """

    def __init__(
        self,
        num_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        #weight_mean=0.001,
        normal_std=0.02,
        zero_bias=True
    ):
        super(
            LearnablePerceptionNetwork, self
        ).__init__()  # torch.nn.Module super function

        self.conv = torch.nn.Conv3d(
            in_channels=num_channels,
            out_channels=3
            * num_channels,  # we want features in the X,Y and Z directions (hence we have 3 times the number of input channels)
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_channels,  # when groups = in_channels, each input channel (RGBA) is convolved with its own filters (so we get a convolution per channel, which is what we want when picking up features per channel)
            bias=bias,
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.normal_(m.weight, std=normal_std)
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        torch.nn.init.zeros_(m.bias)
                    else:
                        torch.nn.init.normal_(m.bias, std=normal_std)

        with torch.no_grad():
            self.apply(init_weights)

        # # Apply weight initialisation if results are poor
        # init.normal_(self.conv.weight, mean=weight_mean, std=weight_std)

    def forward(self, x):
        return self.conv(x)


class NCA3DUpdateNetwork(torch.nn.Module):
    """
    The paper utilises a 3-Layer Update Network. This network applies two 3D convolution intermediate layers, and
    finishes with an update step, taking in the feature map from the learnable perception layer and applying a 3D convolution to this feature map.
    The output of this network should be a delta which can be added to the input occupancy grid to give the occupancy grid for the next
    timestamp.

    Note that the paper utilises a kernel_size of 1, which essentially turns the 3D convolutional layer into a 3D linear layer.

    We apply ReLU to the intermediate layers. Note that hidden_layer_dims should be about 4 * input_channels, to align with original
    NCA paper utilising X, Y and Identity (in our case, we have Z too).
    """

    def __init__(
        self,
        num_channels=4,
        hidden_layer_dims=[16, 16],
        # weight_mean=0.001,
        # weight_std=0.0005
        normal_std=0.02,
        zero_bias=True
    ):
        super(NCA3DUpdateNetwork, self).__init__()
        layers = []

        self.conv1 = torch.nn.Conv3d(
            in_channels=num_channels * 3,
            out_channels=hidden_layer_dims[0],
            kernel_size=1,
        )
        #init.normal_(self.conv1.weight, mean=weight_mean, std=weight_std)

        layers.append(self.conv1)

        self.relu1 = torch.nn.ReLU()
        layers.append(self.relu1)

        self.conv2 = torch.nn.Conv3d(
            in_channels=hidden_layer_dims[0],
            out_channels=hidden_layer_dims[1],
            kernel_size=1,
        )
        #init.normal_(self.conv2.weight, mean=weight_mean, std=weight_std)

        layers.append(self.conv2)

        self.relu2 = torch.nn.ReLU()
        layers.append(self.relu2)

        self.conv3 = torch.nn.Conv3d(
            in_channels=hidden_layer_dims[1],
            out_channels=num_channels,  # want to convert back to original shape to add delta back to input occupancy matrix
            kernel_size=1,
            bias=False,
        )
        #init.normal_(self.conv3.weight, mean=weight_mean, std=weight_std)

        layers.append(self.conv3)

        self.update_net = torch.nn.Sequential(*layers)

        def init_weights(m):
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.normal_(m.weight, std=normal_std)
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        torch.nn.init.zeros_(m.bias)
                    else:
                        torch.nn.init.normal_(m.bias, std=normal_std)

        with torch.no_grad():
            self.apply(init_weights)

    def forward(self, x):
        return self.update_net(x)


class NCA3DModel(torch.nn.Module):
    """
    This class represents the end-to-end NCA model that takes in an input occupancy matrix (4,X,Y,Z) and outputs a
    delta that represents the update for the next time stamp.

    Note that we apply alive masking and stochastic updating.

    Add additional hidden channels if we want to have hidden information.

    Update steps refers to how many updates we make in a single "forward" pass.
    """

    def __init__(
        self,
        alpha_living_threshold=0.1,
        update_rate=0.5,
        hidden_channels=0,
        update_network_hidden_layer_dims=[16, 16],
        update_steps=[48, 64],
    ):
        super(NCA3DModel, self).__init__()
        self.num_hidden_channels = hidden_channels
        self.update_network_channel_dims = update_network_hidden_layer_dims
        self.alpha_living_threshold = alpha_living_threshold
        self.update_rate = update_rate
        self.num_channels = (
            4 + self.num_hidden_channels
        )  # 4 for RGBA, but could make this an input param as well
        self.update_steps = update_steps

        self.perception_layer = LearnablePerceptionNetwork(
            num_channels=self.num_channels
        )

        self.update_network = NCA3DUpdateNetwork(
            num_channels=self.num_channels,  # 1*48 perception vector
            hidden_layer_dims=self.update_network_channel_dims,
        )

        # Use GPU if available
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            torch.set_default_dtype(FLOAT_TYPE)
            self.device = "cuda"
        else:
            self.device = "cpu"

    def alive(self, x):
        """
        This function assists us in checking if a cell/voxel is alive by checking if any of its neighbours are alive.
        This is part of the cellular automata rule that we need to tune (using alpha_living_threshold), as mentioned in the paper:

            Each cell state has an “alive channel” with an alpha
            value; a cell is “alive” when it or one of its neighbors has an
            alpha value greater than 0.1 and “dead” otherwise

        This will return a feature_map where each cell's value is the maximum value of its 3x3x3 neighbourhood (using maxpool)

        If this maximum value is greater than the alpha_living_threshold, then this cell will be set to alive.

        We utilise kernel_size=3, stride=1, padding=1 to keep the output shape the same as the input shape.
        """
        return F.max_pool3d(
            x[:, 3:4, :, :, :],  # [b,4,x,y,z] -> Alpha channel is the 4th channel
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def update(self, x):
        """
        Perform perception and update steps, including alive masking and stochastic updating.
        """
        alive_cells = (
            self.alive(x) > self.alpha_living_threshold
        )  # store which cells are already alive

        percept = self.perception_layer(x)
        delta = self.update_network(percept)

        # return x + delta
        stochastic_mask = (
            torch.rand_like(x[:, :1, :, :, :]) < self.update_rate
        )  # :1 because we want binary mask with shape [b,1,X,Y,Z]
        delta_masked = delta * stochastic_mask.float().to(
            self.device
        )  # delta is on device, stochastic mask needs to be sent to device
        out = x + delta_masked

        alive_cells_after_update = (
            self.alive(out) > self.alpha_living_threshold
        )  # store which cells are still alive after update
        alive_mask = (alive_cells & alive_cells_after_update).float().to(self.device)

        out = out * alive_mask  # all cells that are dead will be set to 0

        return out

    def forward(self, x):
        self.random_number_steps = np.random.randint(
            self.update_steps[0], self.update_steps[1]
        )
        for step in range(self.random_number_steps):  # TODO: Seeding
            x = self.update(x)
        return x

    def generate_gif(self, x, shape, channels):
        random_number_steps = np.random.randint(
            self.update_steps[0], self.update_steps[1]
        )
        frames_array = torch.Tensor(
            random_number_steps,
            channels,
            shape[1],
            shape[2],
            shape[3],
        )

        frames_array[0] = x
        for step in range(1, random_number_steps):
            frames_array[step] = self.update(frames_array[step-1].unsqueeze(0))
        return frames_array
    