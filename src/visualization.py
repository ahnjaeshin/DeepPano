"""
visualize into images (confusion matrix) and save into logfile
"""

import torch
import numpy as np
import seaborn as sns
import itertools
import scipy.ndimage

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig = plt.figure()

    return fig

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
        classes_len = len(classes)
        matrix = np.zeros((classes_len, classes_len))

        output = list(output)
        target = list(target)

        print(output, target)

        for o, t in zip(output, target):
            matrix[t][o] += 1

        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        matrix = matrix * 200 + 55

        print (matrix.shape)

        matrix = scipy.ndimage.zoom(matrix, 10, order=0)

        print (matrix.shape)

        return torch.from_numpy(matrix)

    def __repr__(self):
        return 'confusion'