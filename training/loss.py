import torch as torch
import torch.nn.functional as F


def lossFn(output, target):
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

def iou_loss_fn(output, target): # assuming BS = 1 (per target/output pair iou calc, not batch output -> target iou calc)
    # clamp values between 0, 1 (we are only interested in alpha, which is either alive = 1 or dead = 0)
    target_alpha = torch.clamp(target[3], min=0, max=1) # with how we load in voxels, the alpha values should already be in between 0 and 1
    output_alpha  = torch.clamp(output[3], min=0, max=1) 

    # Convert to boolean masks
    target_mask = target_alpha > 0.1 # living threshold
    output_mask = output_alpha > 0.1 

    intersect = torch.sum(output_mask & target_mask).float()
    union = torch.sum(output_mask | target_mask).float()
    iou = (union - intersect) / (union + 1e-8)
    return iou

def updated_loss_fn(output, target):
    output = output.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)

    iou_loss = iou_loss_fn(output, target)

    alive_cells = torch.clamp(output[3], 0.0, 1.0)
    alive_target_cells = torch.clamp(target[3], 0.0, 1.0)

    alive_loss = torch.nn.functional.mse_loss(alive_cells, alive_target_cells)
    colour_loss = torch.nn.functional.mse_loss(output[:3], target[:3])

    loss = (0.5 * colour_loss + 0.5 * alive_loss + iou_loss) 
    return loss