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
from torchvision import datasets


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

    def train(self, batch_size = 16, num_workers = 32, num_epoch = 100):
        dataloaders = {
            x: DataLoader(dataset = self.datasets[x], 
                          batch_size=batch_size, 
                          shuffle = (x=='train'), 
                          pin_memory=True, 
                          num_workers=num_workers) 
                for x in ['train', 'val']}

        for epoch in range(num_epoch):
            for state in ['train', 'val']:
                if state == 'train':
                    self.model.train(True)
                    # print('Learning rate: {}'.format(self.scheduler.get_lr()[0]))
                else:
                    self.model.train(False) # same as self.model.eval()

                total_loss = 0
                start_time = time.time()
                
                for batch_idx, (data, target, filename, index) in enumerate(tqdm(dataloaders[state])):
                    target_indices = target
                    data = Variable(data)
                    target = Variable(target)
                    if self.use_gpu : 
                        data = data.cuda()
                        target = target.cuda(async=True)
                    
                    output = self.model(data)
                    # _, predics = torch.max(outputs.data, 1)
                    loss = self.criterion(output, target)
                    
                    if state == 'train' : 
                        self.optimizer.zero_grad()            
                        loss.backward()
                        self.optimizer.step()
                        # self.scheduler.step()
                        
                    curr_loss = loss.data[0]
                    # # curr_score = self.metric.add_score(output, target_indices)
                    total_loss += curr_loss
                    batch_size = data.size(0)
                    # if batch_idx % self.print_period == 0:
                    tqdm.write('Epoch: {}    Loss: {:.6f}'.format(epoch, curr_loss))

        # avg_loss = total_loss/len(dataloader)
        # avg_score = self.metric.avg_score()
        # end_time = time.time()        
        # print('{dataset_name} set : Epoch {epoch}, {metric_name} : {score:.4f}, '
        #       'time spent : {time}\n '.format(
        #           dataset_name = dataset_name, epoch = self.epoch, 
        #           metric_name = self.metric.name(), 
        #           score = avg_score, time = end_time - start_time))
        # # print(self.metric) # use this if needed
        # self.writer.add_scalar('{}/loss'.format(dataset_name), avg_loss, self.epoch)
        # self.writer.add_scalar('{}/{}'.format(dataset_name, self.metric.name()), avg_score, self.epoch)
        
        # return avg_score

    def search(self):
        pass

    def ensemble(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
