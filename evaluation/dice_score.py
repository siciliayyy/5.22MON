from typing import Optional

import torch

import torch.nn as nn
import torch.nn.functional as F


def dice_score(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute dice score between two images in a given dimension

    :param img1:    first tesnor
    :param img2:    second tensor
    :param dim:     dimension to compute dice score from
                    In this project, 0 is background, 1 is liver, and 2 is tumor.
    """
    diff = img1 * img2
    intersect = diff.sum()

    mag1 = img1.sum()
    mag2 = img2.sum()

    dice = 2 * intersect / (mag1 + mag2 + 1)

    return dice.item()



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
