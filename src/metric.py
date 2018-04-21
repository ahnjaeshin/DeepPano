
import numpy as np
from torch.nn import functional as F
class Metric():
    """data stored as list of tuples
       (output, target)
    """

    def __init__(self, threshold = 0.5):
        self.threshold = threshold

    def eval(self, *measure):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

class IOU(Metric):

    def __init__(self, threshold = 0.5):
        super(IOU, self).__init__(threshold)

    def eval(self, *measure):
        output = measure[0]
        target = measure[1]
        assert (output <= 1).all() and (output >= 0).all()
        assert (set(np.unique(target)) == {0,1} ) or (set(np.unique(target)) == {0}) or (set(np.unique(target)) == {1})
        out = (output > self.threshold).astype(int)
        union = np.logical_or(out, target).sum()
        intersection = np.logical_and(out, target).sum()

        if union == 0 and intersection == 0:
            return 1
        return intersection / union

    def __repr__(self):
        return 'IoU, threshold: {}'.format(self.threshold)

class DICE(Metric):
    
    def __init__(self, threshold = 0.5):
        super(DICE, self).__init__(threshold)

    def eval(self, *measure):
        output = measure[0]
        target = measure[1]
        assert (output <= 1).all() and (output >= 0).all()
        assert (set(np.unique(target)) == {0,1} ) or (set(np.unique(target)) == {0}) or (set(np.unique(target)) == {1})
        out = (output > self.threshold).astype(int)
        union = np.logical_or(out, target).sum()
        intersection = np.logical_and(out, target).sum()

        if union == 0 and intersection == 0:
            return 1
        return (intersection * 2) / (union + intersection)

    def __repr__(self):
        return 'DICE, threshold: {}'.format(self.threshold)


