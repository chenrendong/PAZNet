"""

DiceBCELoss created by CRD and Chatgpt in 2024.12.11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-4):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        # Flatten tensors to compute intersection and union
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        # Compute intersection and union
        intersection = (pred * target).sum(dim=1)
        pred_sum = pred.sum(dim=1)
        target_sum = target.sum(dim=1)

        # Dice coefficient
        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        if torch.any(dice < 0) or torch.any(dice > 1):
            print(f"Dice coefficients: {dice}")
        # Dice loss
        return 1 - dice.mean()

# Combined Loss: Dice Loss + BCE
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-4, weight_dice=0.5, weight_bce=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCELoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        # Dice Loss
        dice_loss = self.dice_loss(pred, target)
        # BCE Loss
        bce_loss = self.bce_loss(pred, target)
        # Combined Loss
        return self.weight_dice * dice_loss + self.weight_bce * bce_loss

class Dice(nn.Module):
    def __init__(self, smooth=1e-4):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)
        # Flatten tensors to compute intersection and union
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        # Compute intersection and union
        intersection = (pred * target).sum(dim=1)
        pred_sum = pred.sum(dim=1)
        target_sum = target.sum(dim=1)

        # Dice coefficient
        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Dice loss
        return dice.mean()
