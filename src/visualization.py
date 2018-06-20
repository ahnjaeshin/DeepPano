"""
visualize into images (confusion matrix) and save into logfile
"""

import torch
import numpy as np
import itertools
import scipy.ndimage


class ConfusionMatrix():
    
    def __init__(self, threshold = 0.5):
        self.threshold = threshold
    
    def __call__(self, output, target):
        # image (batch_size, channel, W, H) of 0 ~ 1
        batch_size = output.size(0)
        channel = output.size(1)

        output = output.view(batch_size, channel, -1)
        target = target.view(batch_size, channel, -1)

        output = (output > self.threshold).byte()
        # (batch_size, channel, ?) of 0, 1
        output = output.numpy().any(axis=2)
        target = target.numpy().any(axis=2)
        # (batch_size, channel)
        output = output.sum(axis=1)
        target = target.sum(axis=1)

        assert (output.shape == target.shape)
        # (batch_size) of 0, 1, 2

        classes = set(np.unique(target))
        classes_len = 3
        matrix = np.zeros((classes_len, classes_len))

        output = list(output)
        target = list(target)

        for o, t in zip(output, target):
            matrix[t][o] += 1

        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        matrix = matrix * 200 + 55
        matrix = scipy.ndimage.zoom(matrix, 10, order=0)

        return torch.from_numpy(matrix)

    def __repr__(self):
        return 'confusion'