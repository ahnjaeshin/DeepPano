"""



Things to log
1. number of parameters
2. forward propagation, backward propagation
"""

import collections
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision.utils as utils
from torchvision import datasets, transforms


class Trainer(): 
    """trainer class
    1. able to train, test models
    2. able to search for hyper paramters
        
        Arguments:
            model {[type]} -- [description]
            criterion {[type]} -- [description]
            optimizer {[type]} -- [description]
            datasets {[type]} -- [description]
            metrics {[type]} -- [description]

        ensemble!!
    """

    def __init__(self, model, datasets, criterion, optimizer, scheduler):
        
        self.model = model
        self.datasets = datasets
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.use_gpu = torch.cuda.is_available()
        self.use_multi_gpu = torch.cuda.device_count() > 1

        if not self.use_gpu: 
            print("CUDA is unavailable. It'll be very slow...")
        
        self.writer = SummaryWriter()



    def train(self):
        pass

    def search(self):
        pass

    def ensemble(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
