"""
visualize into images (confusion matrix) and save into logfile
"""

import torch
import numpy as np
import seaborn as sns
import itertools


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
        # image (batch_size, channel, W, H)

        output = output.sum(dim=3).sum(dim=2)
        target = target.sum(dim=3).sum(dim=2)
        # (batch_size, channel)
        output = (output > self.threshold).float().sum(dim=1).numpy().astype(int)
        target = (target > self.threshold).float().sum(dim=1).numpy().astype(int)

        assert (output.shape == target.shape)

        # (batch_size) of 0, 1, 2

        classes = set(np.unique(target))
        matrix = torch.zeros((len(classes), len(classes)))

        output = list(output)
        target = list(target)

        print(output, target)

        for o, t in zip(output, target):
            matrix[o][t] += 1

        print(matrix)
        
        matrix = matrix / matrix.sum(dim=0, keepdim=True)
        matrix = matrix * 255
        
        

        return matrix

    def __repr__(self):
        return 'confusion'