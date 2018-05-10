import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss, _WeightedLoss

class MultiOutputLoss(_WeightedLoss):
    
    def __init__(self, losses, weight=None, size_average=True, reduce=True):
        super(MultiOutputLoss, self).__init__(weight, size_average, reduce)
        """concatenate losses
        
        Arguments:
            losses {iterable} -- list of functions
            weight {iterable} -- weight of loss
            reduce {bool}     -- if true, sum loss by batch
        """

        assert len(set([loss.reduce for loss in losses])) <= 1, "all losses should have same reduce value"

        self.losses = losses

        if weight is None:
            self.weight = torch.ones(1, len(losses))
            # weight : (1, output_len)
        else:
            self.weight = torch.tensor(weight).view(1, -1)

    def __call__(self, outputs, targets):
        
        assert len(self.losses) == len(outputs) == len(targets)

        total_loss = torch.stack([loss(output, target).view(-1) 
                        for loss, output, target in zip(self.losses, outputs, targets)])
        
        # total_loss (output_len) or (output_len, batch_size)

        total_loss = torch.mm(self.weight, total_loss)
        total_loss = total_loss / self.weight.sum()

        # total_loss = (1) or (batch_size)
        return total_loss    
    
class IOULoss(_WeightedLoss):
    """1 - jaccard index
    """

    def __init__(self, weight=None, size_average=True, reduce=True):
        super(IOULoss, self).__init__(weight, size_average, reduce)

    def __call__(self, output, target):
        batch_size = output.size()[0]

        output = F.sigmoid(output).view(batch_size, -1)
        target = (target == 1).float().view(batch_size, -1)
        assert output.size() == target.size()
        assert (output <= 1).all() and (output >= 0).all()

        smooth = 1

        intersection = (output * target).sum(dim=-1)
        union = output.sum(dim=-1) + target.sum(dim=-1)

        assert union.size()[0] == batch_size and intersection.size()[0] == batch_size

        union = union - intersection + smooth
        intersection = intersection + smooth

        loss = (1 - torch.div(intersection, union))

        if self.weight is not None:
            loss = loss * self.weight

        if not self.reduce:
            return loss
        elif self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class DICELoss(_WeightedLoss):
    """ 1 - F1 score
    """

    def __init__(self, weight=None, size_average=True, reduce=True):
        super(DICELoss, self).__init__(weight, size_average, reduce)

    def __call__(self, output, target, reduce=True):
        batch_size = output.size()[0]

        output = F.sigmoid(output).view(batch_size, -1)
        target = (target == 1).float().view(batch_size, -1)
        assert output.size() == target.size()
        assert (output <= 1).all() and (output >= 0).all()

        smooth = 1

        intersection = (output * target).sum(dim=-1)
        union = output.sum(dim=-1) + target.sum(dim=-1)

        assert union.size()[0] == batch_size and intersection.size()[0] == batch_size

        union = union + smooth
        intersection = intersection * 2 + smooth

        loss = (1 - torch.div(intersection, union))

        if self.weight is not None:
            loss = loss * self.weight

        if not self.reduce:
            return loss
        elif self.size_average:
            return loss.mean()
        else:
            return loss.sum()