import torch

def lossFn(output, target):
    # This will ensure we maintain the gradients as we modify the normalised values
    target_norm = target.clone()
    
    ## 1. normalise alpha channel values between 0 and 1 for target
    targ_min = torch.min(target_norm)
    targ_max = torch.max(target_norm)
    target_norm = (target_norm - targ_min) / (targ_max - targ_min + 1e-8)
    
    # Clamp output values between 0 and 1
    output_clamped = torch.clamp(output, min=0, max=1)
    
    # 2. Calculate soft IOU loss to maintain gradients
    # Convert to binary while preserving gradients using differentiable operations
    output_soft = torch.sigmoid(20 * (output_clamped - 0.5))  # Sharpen sigmoid around 0.5
    
    # Compute intersection and union in a differentiable way
    intersection = torch.sum(output_soft * target_norm)
    union = torch.sum(output_soft) + torch.sum(target_norm) - intersection
    
    iou_loss = 1.0 - (intersection / (union + 1e-8))
    return iou_loss