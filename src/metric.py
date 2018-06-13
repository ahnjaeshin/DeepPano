
import numpy as np
import torch
from torch.nn import functional as F

class Metric():
    """data stored as list of tuples
       (output, target)
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, output, target):
        return self.eval(output, target)

    def eval(self, output, target):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

class SegmentationMetric(Metric):
    """segmentation metrics
        
    Arguments:
        threshold {int} -- 0 ~ 1 float
    
    Keyword Arguments:
        mode {str} -- both, major, minor (default: {'BOTH'})
    """
    
    def __init__(self, threshold, mode='both'):
        super(SegmentationMetric, self).__init__(threshold)
        self.mode = mode
    
    def __call__(self, output, target):
        batch_size = output.size(0)

        # image (batch_size, channel, W, H)
        # do nothing if mode is 'both'

        if self.mode == 'major':
            output = output.narrow(1, 0, 1)
            target = target.narrow(1, 0, 1)
        elif self.mode == 'minor':
            output = output.narrow(1, 1, 1)
            target = target.narrow(1, 1, 1)

        output = output.view(batch_size, -1).numpy()
        target = target.view(batch_size, -1).numpy()

        assert output.shape == target.shape
        assert (output <= 1).all() and (output >= 0).all()
        assert set(np.unique(target)).issubset({0,1})

        output = (output > self.threshold).astype(int)
            
        return self.eval(output, target)

    def __repr__(self):
        return '/{}/{}'.format(self.threshold, self.mode)
        
class ClassificationMetric(Metric):
    """image into 0 (None), 1 (Single), 2 (Double)
        
    Arguments:
        threshold {int} -- 0 ~ 1 float
    
    Keyword Arguments:
        mode {str} -- both, major, minor (default: {'BOTH'})
    """
    
    def __init__(self, threshold, mode='both'):
        super(ClassificationMetric, self).__init__(threshold)
        self.mode = mode
    
    def __call__(self, output, target):
        # image (batch_size, channel, W, H)
        # do nothing if mode is 'both'

        batch_size = output.size(0)
        channel = output.size(1)

        output = output.view(batch_size, channel, -1)
        target = target.view(batch_size, channel, -1)

        if self.mode == 'major':
            output = output.narrow(1, 0, 1)
            target = target.narrow(1, 0, 1)
        elif self.mode == 'minor':
            output = output.narrow(1, 1, 1)
            target = target.narrow(1, 1, 1)

        output = (output > self.threshold).byte()
        # (batch_size, channel, ?) of 0, 1
        output = output.numpy().any(axis=2)
        target = target.numpy().any(axis=2)
        # (batch_size, channel)
        output = output.sum(axis=1)
        target = target.sum(axis=1)
            
        return self.eval(output, target)

    def __repr__(self):
        return '/{}/{}'.format(self.threshold, self.mode)
        
class Accuracy(ClassificationMetric):
    
    def __init__(self, threshold = 0.5, mode = 'both'):
        super(Accuracy, self).__init__(threshold, mode)
    
    def eval(self, output, target):
        # (batch size) => each element is 0, 1, 2
        length = output.shape[0]

        return ((output == target).sum() / length)

    def __repr__(self):
        return 'accuracy' + super(Accuracy, self).__repr__()

class F1(ClassificationMetric):
    
    def __init__(self, threshold = 0.5, mode='major'):
        super(F1, self).__init__(threshold, mode)
        assert (mode is not 'both')

    def eval(self, output, target):
        
        union = np.logical_or(output, target).sum()
        intersection = np.logical_and(output, target).sum()

        if union == 0:
            return 1
        
        return (intersection + intersection) / (union + intersection)

    def __repr__(self):
        return 'f1' + super(F1, self).__repr__()

class IOU(SegmentationMetric):

    def __init__(self, threshold = 0.5, mode='both'):
        super(IOU, self).__init__(threshold, mode)

    def eval(self, output, target):
        
        union = np.logical_or(output, target).sum(axis=-1)
        intersection = np.logical_and(output, target).sum(axis=-1)

        zero_indicies = np.where(union == 0)
        union[zero_indicies] = 1
        intersection[zero_indicies] = 1

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'IoU' + super(IOU, self).__repr__()

class DICE(SegmentationMetric):
    
    def __init__(self, threshold = 0.5, mode='both'):
        super(DICE, self).__init__(threshold, mode)

    def eval(self, output, target):

        union = np.logical_or(output, target).sum(axis=-1)
        intersection = np.logical_and(output, target).sum(axis=-1)

        zero_indicies = np.where(union == 0)
        union[zero_indicies] = 1
        intersection[zero_indicies] = 1

        union = union + intersection
        intersection = intersection * 2

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'DICE' + super(DICE, self).__repr__()

class IOU_SEG(SegmentationMetric):
    
    def __init__(self, threshold = 0.5, mode='both'):
        super(IOU_SEG, self).__init__(threshold, mode)

    def eval(self, output, target):
        
        non_zero_indicies = np.where(target != 0)
        target = target[non_zero_indicies]
        output = output[non_zero_indicies]
        
        union = np.logical_or(output, target).sum(axis=-1)
        intersection = np.logical_and(output, target).sum(axis=-1)

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'IoU_SEG' + super(IOU_SEG, self).__repr__()

class DICE_SEG(SegmentationMetric):
    
    def __init__(self, threshold = 0.5, mode='both'):
        super(DICE_SEG, self).__init__(threshold, mode)

    def eval(self, output, target):
        
        non_zero_indicies = np.where(target != 0)
        target = target[non_zero_indicies]
        output = output[non_zero_indicies]

        union = np.logical_or(output, target).sum(axis=-1)
        intersection = np.logical_and(output, target).sum(axis=-1)

        union = union + intersection
        intersection = intersection * 2

        return np.mean(np.divide(intersection, union))

    def __repr__(self):
        return 'DICE_SEG' + super(DICE_SEG, self).__repr__()
