import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def where(cond, x_1, x_2):
    cond = cond.float()    
    return (cond * x_1) + ((1-cond) * x_2)

class BCEIOULoss():
    """BCE - log(jaccard index)
    """
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss
    
class IOULoss():
    """1 - jaccard index
    """
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, output, target):
        batch_size = output.size()[0]

        output = F.sigmoid(output)
        target = (target == 1).float()
        assert output.size()[0] == target.size()[0]
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        assert output.size()[1] == target.size()[1]
        assert (output <= 1).all() and (output >= 0).all()

        eps = 1e-15

        intersection = (output * target).sum(dim=-1)
        union = output.sum(dim=-1) + target.sum(dim=-1)

        assert union.size()[0] == batch_size and intersection.size()[0] == batch_size

        union = union - intersection + eps
        intersection = intersection + eps

        return 1 - torch.mean(torch.div(intersection, union))

class DICELoss():
    """1 - jaccard index
    """
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, output, target):
        batch_size = output.size()[0]

        output = F.sigmoid(output)
        target = (target == 1).float()
        assert output.size()[0] == target.size()[0]
        output = output.view(batch_size, -1)
        target = target.view(batch_size, -1)

        assert output.size()[1] == target.size()[1]
        assert (output <= 1).all() and (output >= 0).all()

        eps = 1e-15

        intersection = (output * target).sum(dim=-1)
        union = output.sum(dim=-1) + target.sum(dim=-1)

        assert union.size()[0] == batch_size and intersection.size()[0] == batch_size

        union = union + eps
        intersection = intersection * 2 + eps

        return 1 - torch.mean(torch.div(intersection, union))