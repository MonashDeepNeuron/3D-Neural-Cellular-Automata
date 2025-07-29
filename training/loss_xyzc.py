import torch as torch
import torch.nn.functional as F

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

class Loss():
    def __init__(self, loss_fn=0):
        self.loss_fn = loss_fn

    def __call__(self, output, target):
        if self.loss_fn == 0:
            return self.lossFn(output, target)
        elif self.loss_fn == 1:
            return self.updated_loss_fn(output, target)
        else:
            raise ValueError(f"{self.loss_fn} is not implemented. Please check your inputted loss function.")

    def lossFn(self, output, target):
        ## 1. normalise alpha channel values between 0 and 1 for both target and output
        targ_min = torch.min(target[3:4, :, :, :])
        targ_max = torch.max(target[3:4, :, :, :])
        target[3:4, :, :, :] = (target[3:4, :, :, :] - targ_min) / (targ_max - targ_min)
        binary_target = target.type(torch.int64)
        binary_output = (
            torch.clamp(output[3:4, :, :, :], min=0, max=1)
            .round(decimals=0)
            .type(torch.int64)
        )

        # 2. Calculate IOU loss (int64) clamp output image values between 0 and 1
        intersect = torch.sum(binary_output & binary_target).float()
        union = torch.sum(binary_output | binary_target).float()
        iou_loss = (union - intersect) / (union + 1e-8)

        ## 3. calculate the MSE loss for RGB channels of pixels that have an alpha channel == 1
        mse_loss = torch.nn.MSELoss(reduction="mean")(
            output[0:3, :, :, :], target[0:3, :, :, :]
        )

        ## Calculate the overall loss that is the sum of the IOU loss and MSE loss
        return iou_loss * 0.05 + mse_loss * 0.95  # (float32 with grad function) /2.0

    def iou_loss_fn(self, output, target): # assuming BS = 1 (per target/output pair iou calc, not batch output -> target iou calc)
        # clamp values between 0, 1 (we are only interested in alpha, which is either alive = 1 or dead = 0)
        target_alpha = torch.clamp(target[3], min=0, max=1) # with how we load in voxels, the alpha values should already be in between 0 and 1\
        output_alpha  = torch.clamp(output[3], min=0, max=1) # NYAN NOTE: MAKE SURE CHANNELS IS FIRST DIMENSION DUMBASS

        # Convert to boolean masks
        target_mask = target_alpha > 0.1 # living threshold, only consider living cells for intersection and union
        output_mask = output_alpha > 0.1

        intersect = torch.sum(output_mask & target_mask).float()
        union = torch.sum(output_mask | target_mask).float()
        iou = (union - intersect) / (union + 1e-8)
        return iou

    def updated_loss_fn(self, output, target):
        output = output.permute(3,0,1,2).to(dtype=torch.float32).cpu() # NYAN NOTE: PERMUTE THE TENSOR. IT SHOULD BE [C,X,Y,Z] after this
        target = target.permute(3,0,1,2).to(dtype=torch.float32).cpu()

        # Get the IoU Loss
        iou_loss = self.iou_loss_fn(output, target)

        output_cells = torch.clamp(output[3, ...], 0.0, 1.0) # could honestly do this before IoU, and pass tensors into IoU
        target_cells = torch.clamp(target[3, ...], 0.0, 1.0)

        # Only do MEAN squared error over alive cells, if you do not mask, you mean over all cells and this can reduce the loss value
        # A lower loss value when there is semantically high error can stunt the model's ability to learn
        alive_output = output_cells > 0.1
        alive_target = target_cells > 0.1
        any_activity_mask = (alive_output | alive_target) # this will be used for loss where we are interested in any activity
        target_activity_mask = (alive_output & alive_target)
        colour_mask = target_activity_mask.unsqueeze(0).expand(3, -1, -1, -1) # only want to compare colour where there should be alive cells

        # penalize model for having incorrect alpha predictions 
        alive_loss = torch.nn.functional.mse_loss(output_cells[any_activity_mask], target_cells[any_activity_mask]) 
        
        # penalize model not growing (no alive cells where there should be) 
        false_negatives = (alive_target & (output_cells <= 0.1)).float().sum() 
        fn_count = alive_output.sum()
        fn_loss = false_negatives / fn_count # this gives us proportion of false negatives over all POSSIBLE false negative locations
        #fn_loss = false_negatives / target_cells.numel() -> numel() sums over entire space, what if we want to just get the spaces where there are living cells in the target (for false negs)
        
        # penalize model growing too much (alive cells where there shouldn't be)
        false_positives = ((output_cells >= 0.1) & (target_cells < 0.1)).float().sum() 
        fp_count = (target_cells <= 0.1).sum() # this gives us proportion of false positives over all POSSIBLE false positive locations
        fp_loss = false_positives / fp_count
        #  #fp_loss = false_positives / target_cells.numel()

        # we want the model to overfit, and have high precision. The model must grow to the shape of the target voxel, and not overgrow/undergrow
        true_positives = target_activity_mask.float().sum()
        precision = true_positives / (true_positives + false_positives)
        precision_loss = 1 - precision # want this to be low when precision is high

        # penalize model for poor colour predictions at cells where it should be alive
        colour_loss = torch.nn.functional.mse_loss(output[:3, ...][colour_mask], target[:3,...][colour_mask])

        # Debugging
        print(f'##### Alive Cells Output: {len(torch.nonzero(alive_output))} #####')
        print(f'true_positives: {true_positives}')
        print(f'fn loss:  {fn_loss.item()}')
        print(f'false_positives: {false_positives}')
        print(f'fp loss:  {fp_loss.item()}')
        print(f'precision: {precision}')
        print(f'IoU Loss: {iou_loss.item()}')
        print(f'Alive Loss:  {alive_loss.item()}')
        print(f'Colour Loss: {colour_loss.item()}')

        loss = (1.5 * colour_loss +  alive_loss + 2 * iou_loss + fn_loss + 2 * fp_loss + 2 * precision_loss) # false negatives are important to fit to the target structure, false positives are important for preventing overgrowth
        
        #NEED TO PENALIZE OVERGROWTH IN SPARSER PROBLEMS

        return loss

