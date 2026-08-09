import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=8, dice_weight=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):

        ce_loss = self.ce(logits, target)

        probabilities = F.softmax(logits, dim=1)

        target_one_hot = F.one_hot(
            target,
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = (
            probabilities * target_one_hot
        ).sum(dims)

        denominator = (
            probabilities.sum(dims)
            + target_one_hot.sum(dims)
        )

        dice = (
            2 * intersection + 1e-6
        ) / (
            denominator + 1e-6
        )

        dice_loss = 1 - dice.mean()

        return (
            self.dice_weight * dice_loss
            + (1 - self.dice_weight) * ce_loss
        )


def create_dice_ce_loss(
    num_classes=8,
    dice_weight=0.5,
):
    return DiceCrossEntropyLoss(
        num_classes=num_classes,
        dice_weight=dice_weight,
    )