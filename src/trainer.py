"""



Things to log
1. number of parameters
2. forward propagation, backward propagation
"""

import collections
import datetime
import time
import os

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
from torch.autograd import Variable

from utils import slack_message, count_parameters
from torch.nn import functional as F
import tabulate


def cuda(x):
    """for use in gpu"""
    return x.cuda(async=True) if torch.cuda.is_available() else x

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

    def __init__(self, model, datasets, criterion, optimizer, scheduler, metrics):
        
        self.model = model
        self.datasets = datasets
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics

        if not torch.cuda.is_available(): 
            print("CUDA is unavailable. It'll be very slow...")
        
        self.writer = SummaryWriter(comment="UNet")
        dummy_input = Variable(torch.rand(1, 2, 224, 224), requires_grad=True)
        self.writer.add_graph(self.model, (dummy_input, ))
        self.writer.add_scalar('number of parameter', count_parameters(self.model))
        self.writer.add_text('info', 'this is start', 0)

    def train(self, batch_size = 16, num_workers = 32, epochs = 100, log_freq_ratio = 10):
        
        slack_message('train started', '#botlog')
        slack_message(self.model.__repr__, '#botlog')

        dataloaders = { x: DataLoader(dataset = self.datasets[x], 
                                      batch_size = batch_size, 
                                      shuffle = (x == 'train'), 
                                      pin_memory=True, 
                                      num_workers=num_workers) 
                for x in ['train', 'val']}

        if torch.cuda.is_available():
            self.model = cuda(self.model)
            torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        start_time = datetime.datetime.now()

        for epoch in tqdm(range(epochs), desc='epoch'):
            self.train_once(epoch, dataloaders['train'], True, log_freq_ratio)
            self.train_once(epoch, dataloaders['val'], False, log_freq_ratio)
            
            elasped_time = datetime.datetime.now() - start_time
            eta = start_time + ((elasped_time / (epoch + 1)) * epochs)

            log = 'epoch: {}/{}, elasped: {}, eta: {}'.format(epoch, epochs, elasped_time, eta)
            tqdm.write(log)
            slack_message(log, '#botlog')

        slack_message('train ended', '#botlog')

    def train_once(self, epoch, dataloader, train=True, log_freq_ratio = 10):
        
        batch_losses = []
        forward_times = []
        backward_times = []
        data_times = []
        curr_score = 0

        if train:
            self.model.train(True)
        else:
            self.model.train(False) # same as self.model.eval()

        start = time.time()
        for batch_idx, (data, target, filepath, index) in enumerate(tqdm(dataloader, desc='batch')):

            data_time = time.time() - start
            data_times.append(data_time)

            data = cuda(Variable(data))
            target = cuda(Variable(target))
            
            output = self.model(data)
            loss = self.criterion(output, target)
            output = F.sigmoid(output)
            
            forward_time = time.time() - data_time
            forward_times.append(forward_time)

            if train : 
                self.optimizer.zero_grad()            
                loss.backward()
                self.optimizer.step()
                
            batch_loss = loss.cpu().data[0]
            batch_losses.append(batch_loss)
            backward_time = time.time() - forward_time
            backward_times.append(backward_time)
            start = time.time()

            batch_size = data.size(0)
            niter = epoch*len(dataloader) + batch_idx

            for metric in self.metrics:
                curr_score = metric.add(output.data.cpu().numpy(), target.data.cpu().numpy())

            if batch_idx % log_freq_ratio == 0:
                log = [
                    'Epoch: [{0}][{1}/{2}]'.format(epoch, batch_idx, len(dataloader)),
                    'Forward Time {batch_time:.3f} ({batch_time_avg:.3f})'.format(batch_time=forward_time, batch_time_avg=np.mean(forward_times)),
                    'Backward Time {batch_time:.3f} ({batch_time_avg:.3f})'.format(batch_time=backward_time, batch_time_avg=np.mean(backward_times)),
                    'Data {data_time:.3f} ({data_time_avg:.3f})'.format(data_time=data_time, data_time_avg=np.mean(data_times)),
                    'Loss {loss:.4f} ({loss_avg:.4f})'.format(loss=batch_loss, loss_avg=np.mean(batch_losses)),
                ]

                self.writer.add_scalar('time(forward)', forward_time, niter)
                self.writer.add_scalar('time(backward)', backward_time, niter)
                self.writer.add_scalar('time(data)', data_time, niter)
                self.writer.add_scalar('loss', batch_loss, niter)

                if train:
                    lr = [group['lr'] for group in self.optimizer.param_groups][0]
                    tqdm.write('Learning rate: {}'.format(lr))
                    self.writer.add_scalar('learning rate', lr, epoch)

                for metric in self.metrics:
                    log.append('metric-{}: {:.5f} ({:.5f})'.format(metric.__repr__(), metric.getRecentScore(), metric.mean()))
                    self.writer.add_scalar('metric-{}'.format(metric.__repr__()), metric.getRecentScore(), niter)
                    self.writer.add_histogram('metric-{}'.format(metric.__repr__()), metric.hist(), niter, bins='auto')
                log = "\n".join(log)

                self.writer.add_image('input-pano', utils.make_grid(data.data.cpu().narrow(1, 1, 1), normalize=True, scale_each=True), niter)
                self.writer.add_image('input-guideline', utils.make_grid(data.data.cpu().narrow(1, 0, 1), normalize=True, scale_each=True), niter)
                self.writer.add_image('target', utils.make_grid(target.data.cpu(), normalize=True, scale_each=True), niter)
                self.writer.add_image('output', utils.make_grid(output.data.cpu(), normalize=True, scale_each=True), niter)
                self.writer.add_pr_curve('accuracy', target.data.cpu(), output.data.cpu(), niter)

                tqdm.write(log)
                slack_message(log, '#botlog')
                
        if train: # smoothing effect to LR reduce on platue
            self.scheduler.step(np.mean(batch_losses))

        # todo : for every batch, log hard examples

    def search(self):
        pass

    def ensemble(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
