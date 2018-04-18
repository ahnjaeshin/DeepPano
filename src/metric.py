
import numpy as np
from torch.nn import functional as F
class Metric():
    """data stored as list of tuples
       (output, target)
    """

    def __init__(self, threshold = 0.5, save_output = False):
        self.raw = []
        self.indexes = []
        self.save_output = save_output
        self.threshold = threshold

    def add(self, *measure):
        """add new result
        
        Arguments:
            output -- output of model
            target -- actual label
        """
        index = self.eval(*measure)
        self.indexes.append( index )
        if self.save_output: 
            self.raw.append(measure)
        
        return index

    def zero(self):
        """clear data
        """
        self.indexes = []
        self.raw = []

    def eval(self, *measure):
        raise NotImplementedError

    def mean(self):
        return np.average(self.indexes)

    def max(self):
        return np.max(self.indexes)

    def min(self):
        return np.min(self.indexes)  

    def hist(self):
        return np.array(self.indexes)

    def getRecentScore(self):
        return self.indexes[-1]

    def __repr__(self):
        raise NotImplementedError

class IOU(Metric):

    def __init__(self, threshold = 0.5, save_output = False):
        super(IOU, self).__init__(threshold, save_output)

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
