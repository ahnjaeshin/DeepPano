
import numpy as np
import torch
from torch.nn import functional as F

def isIter(x):
    """return true if x is list or tuple"""
    return type(x) == tuple or type(x) == list

def numpy(func):
    def metric(self, output, target):
        output = [o.numpy() for o in output] if isIter(output) else output
        target = [t.numpy() for t in target] if isIter(target) else target
        return func(self, output, target)
    return metric

def segmentation(func):
    def metric(self, output, target):
        batch_size = output[0].shape[0]
        output = output[0]
        target = target[0]
        return func(self, np.reshape(output, (batch_size, -1)), np.reshape(target, (batch_size, -1)))
    return metric

def classification(func):
    def metric(self, output, target):
        return func(self, output[1], target[1])
    return metric

def render(func):
    def render_output(self, output, target):
        output1, output2 = output
        output2 = (output2 >= 0.5).astype(int)
        output1 = output1 * output2.reshape(output2.shape[0], 1, 1, 1)
        output = output1, output2

        return func(self, output, target)
    return render_output

class Metric():
    """data stored as list of tuples
       (output, target)
    """

    def __init__(self, threshold = 0.5):
        self.threshold = threshold

    def eval(self, output, target):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

class IOU(Metric):

    def __init__(self, threshold = 0.5):
        super(IOU, self).__init__(threshold)

    @numpy
    @segmentation
    def eval(self, output, target):

        assert output.shape == target.shape
        assert (output <= 1).all() and (output >= 0).all()
        assert set(np.unique(target)).issubset({0,1})

        out = (output > self.threshold).astype(int)
        union = np.logical_or(out, target).sum(axis=-1)
        intersection = np.logical_and(out, target).sum(axis=-1)

        zero_indices = np.where(union != 0)
        union = union[zero_indices]
        intersection = intersection[zero_indices]

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'IoU({})'.format(self.threshold)

class DICE(Metric):
    
    def __init__(self, threshold = 0.5):
        super(DICE, self).__init__(threshold)

    @numpy
    @segmentation
    def eval(self, output, target):

        assert output.shape == target.shape
        assert (output <= 1).all() and (output >= 0).all()
        assert set(np.unique(target)).issubset({0,1})
    
        out = (output > self.threshold).astype(int)
        union = np.logical_or(out, target).sum(axis=-1)
        intersection = np.logical_and(out, target).sum(axis=-1)

        zero_indices = np.where(union != 0)
        union = union[zero_indices]
        intersection = intersection[zero_indices]

        union = union + intersection
        intersection = intersection * 2

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'DICE({})'.format(self.threshold)

class NEW_IOU(Metric):
    def __init__(self, threshold = 0.5, target_threshold = 0.5):
        super(NEW_IOU, self).__init__(threshold)
        self.target_threshold = target_threshold

    @numpy
    @render
    @segmentation
    def eval(self, output, target):

        assert output.shape == target.shape
        assert (output <= 1).all() and (output >= 0).all()
        assert set(np.unique(target)).issubset({0,1})

        out = (output > self.threshold).astype(int)
        union = np.logical_or(out, target).sum(axis=-1)
        intersection = np.logical_and(out, target).sum(axis=-1)

        zero_indices = np.where(union != 0)
        union = union[zero_indices]
        intersection = intersection[zero_indices]

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'NEW_IOU({}, target:{})'.format(self.threshold, self.target_threshold)

class NEW_DICE(Metric):
    def __init__(self, threshold = 0.5, target_threshold = 0.5):
        super(NEW_DICE, self).__init__(threshold)
        self.target_threshold = target_threshold

    @numpy
    @render
    @segmentation
    def eval(self, output, target):
        assert output.shape == target.shape
        assert (output <= 1).all() and (output >= 0).all()
        assert set(np.unique(target)).issubset({0,1})
        
        out = (output > self.threshold).astype(int)
        union = np.logical_or(out, target).sum(axis=-1)
        intersection = np.logical_and(out, target).sum(axis=-1)

        zero_indices = np.where(union != 0)
        union = union[zero_indices]
        intersection = intersection[zero_indices]

        union = union + intersection
        intersection = intersection * 2

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'NEW_DICE({}, target:{})'.format(self.threshold, self.target_threshold)

class Accuracy(Metric):
    
    def __init__(self, threshold = 0.5):
        super(Accuracy, self).__init__(threshold)

    @numpy
    @classification
    def eval(self, output, target):

        assert output.shape[0] == target.shape[0]
        assert (output <= 1).all() and (output >= 0).all()
        assert set(np.unique(target)).issubset({0,1})
    
        out = (output > self.threshold).astype(int)

        return np.mean((out == target).astype(int))

    def __repr__(self):
        return 'accuracy({})'.format(self.threshold)


class F1BySegment(Metric):
    """[summary]
    
    Arguments:
        Metric {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    def __init__(self, threshold = 0.5):
        super(F1BySegment, self).__init__(threshold)

    def eval(self, output, target):
        raise NotImplementedError

    def __repr__(self):
        return 'F1({})'.format(self.threshold)

