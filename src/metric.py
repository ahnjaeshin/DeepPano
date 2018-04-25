
import numpy as np
import torch
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
        batch_size = output.shape[0]

        assert output.shape[0] == target.shape[0]
        output = np.reshape(output, (batch_size, -1))
        target = np.reshape(target, (batch_size, -1))
        assert output.shape[1] == target.shape[1]
        assert (output <= 1).all() and (output >= 0).all()
        assert (set(np.unique(target)) == {0,1} ) or (set(np.unique(target)) == {0}) or (set(np.unique(target)) == {1})
    
        out = (output > self.threshold).astype(int)
        union = np.logical_or(out, target).sum(axis=-1)
        intersection = np.logical_and(out, target).sum(axis=-1)

        assert union.shape[0] == batch_size and intersection.shape[0] == batch_size

        zero_indices = np.where(union != 0)
        union = union[zero_indices]
        intersection = intersection[zero_indices]

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'IoU, threshold: {}'.format(self.threshold)

class DICE(Metric):
    
    def __init__(self, threshold = 0.5):
        super(DICE, self).__init__(threshold)

    def eval(self, *measure):
        output = measure[0]
        target = measure[1]
        batch_size = output.shape[0]

        assert output.shape[0] == target.shape[0]
        output = np.reshape(output, (batch_size, -1))
        target = np.reshape(target, (batch_size, -1))
        assert output.shape[1] == target.shape[1]
        assert (output <= 1).all() and (output >= 0).all()
        assert (set(np.unique(target)) == {0,1} ) or (set(np.unique(target)) == {0}) or (set(np.unique(target)) == {1})
    
        out = (output > self.threshold).astype(int)
        union = np.logical_or(out, target).sum(axis=-1)
        intersection = np.logical_and(out, target).sum(axis=-1)

        assert union.shape[0] == batch_size and intersection.shape[0] == batch_size

        zero_indices = np.where(union != 0)
        union = union[zero_indices]
        intersection = intersection[zero_indices]

        union = union + intersection
        intersection = intersection * 2

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'DICE, threshold: {}'.format(self.threshold)

class Accuracy(Metric):
    
    def __init__(self, threshold = 0.5):
        super(Accuracy, self).__init__(threshold)

    def eval(self, *measure):
        output = measure[0]
        target = measure[1]
        batch_size = output.shape[0]

        assert output.shape[0] == target.shape[0]
        assert (output <= 1).all() and (output >= 0).all()
        assert (set(np.unique(target)) == {0,1} ) or (set(np.unique(target)) == {0}) or (set(np.unique(target)) == {1})
    
        out = (output > self.threshold).astype(int)

        return np.mean((out == target).astype(int))

    def __repr__(self):
        return 'DICE, threshold: {}'.format(self.threshold)


class F1BySegment(Metric):
    """[summary]
    
    Arguments:
        Metric {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    
    def __init__(self, threshold = 0.5):
        super(F1BySegment, self).__init__(threshold)

    def eval(self, *measure):
        raise NotImplementedError

    def __repr__(self):
        return 'F1, threshold: {}'.format(self.threshold)

