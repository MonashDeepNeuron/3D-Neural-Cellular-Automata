import torch as torch
import torch.nn.functional as F

def lossFn(output, target):    
    ## 1. normalise alpha channel values between 0 and 1 for both target and output
    targ_min = torch.min(target[3:4, :, :, :])
    targ_max = torch.max(target[3:4, :, :, :])
    target[3:4, :, :, :] = (target[3:4, :, :, :] - targ_min) / (targ_max - targ_min)
    binary_target = target.type(torch.int64)
    binary_output = torch.clamp(output[3:4, :, :, :], min=0, max=1).round(decimals=0).type(torch.int64)

    # 2. Calculate IOU loss (int64) clamp output image values between 0 and 1 
    intersect = torch.sum(binary_output & binary_target).float()
    union = torch.sum(binary_output | binary_target).float()
    iou_loss = (union - intersect) / (union + 1e-8)

    ## 3. calculate the MSE loss for RGB channels of pixels that have an alpha channel == 1
    mse_loss = torch.nn.MSELoss(reduction="mean")(output[0:3,:,:,:], target[0:3,:,:,:])

    ## Calculate the overall loss that is the sum of the IOU loss and MSE loss 
    return (iou_loss*0.05 + mse_loss*0.95) #(float32 with grad function) /2.0
