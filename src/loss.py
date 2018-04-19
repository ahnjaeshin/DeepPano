import torch
import torch.nn as nn
import torch.nn.functional as F

class IOULoss():
    """BCE - log(jaccard index)
    """
    def __init__(self, weight=0):
        self.nll = nn.BCEWithLogitsLoss()
        self.weight = weight
    
    def __call__(self, output, target):
        loss = self.nll(output, target)

        if self.weight != 0:
            eps = 1e-15
            target = (target == 1).float()
            output = F.sigmoid(output)

            intersection = (output * target).sum()
            union = output.sum() + target.sum() - intersection

            loss -= self.weight * torch.log((intersection + eps) / (union + eps))
        return loss