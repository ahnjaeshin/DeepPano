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

from torch.utils.data import DataLoader

from utils import AverageMeter, GeometricMeter, ImageMeter, ClassMeter
import shutil

from visualization import ConfusionMatrix

class Pipeline:
    def __init__(self, *pipe):
        self.pipe = pipe
    
    def __call__(self, data):
        for p in self.pipe:
            data = p(data)
        return data


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

    def __init__(self, model, datasets, metrics, LOG):
        
        self.model = model
        self.datasets = datasets
        self.metrics = metrics
        self.LOG = LOG

        self.epochs = 0
        self.start_epoch = 0
        
    def train(self, checkpoint, batch_size, num_workers, epochs, log_freq):
        
        if checkpoint:
            self.start_epoch = self.model.load(checkpoint)
        self.epochs = epochs
        self.LOG('slack', name='train', values='train started: using {} gpus'.format(torch.cuda.device_count()))

        dataloaders = { x: DataLoader(dataset = self.datasets[x], 
                                      batch_size = batch_size, 
                                      shuffle = True, 
                                      pin_memory=torch.cuda.is_available(), 
                                      num_workers=num_workers) 
                for x in ['train', 'val']}

        start_time = datetime.datetime.now()
        for epoch in range(self.start_epoch, epochs):
            do_log = ((epoch + 1) % log_freq == 0)

            for state in ('train', 'val'):
                self.train_once(epoch, dataloaders[state], state, do_log)                 

            elasped_time = datetime.datetime.now() - start_time
            eta = start_time + ((elasped_time / ((epoch - self.start_epoch) + 1)) * (epochs - self.start_epoch))

            if do_log:
                self.model.checkpoint(epoch, self.LOG.log_dir_base+'checkpoint/')
                self.LOG('epoch: {}/{}, elasped: {}, eta: {}'.format(epoch + 1, epochs, elasped_time, eta), *log)

        self.LOG('slack', name='train ended', values='')

    def train_once(self, epoch, dataloader, mode, do_log):

        losses = AverageMeter()
        data_times = AverageMeter()
        propagate_times = AverageMeter()
        metric_scores = [AverageMeter() for metric in self.metrics]
        metric_geometric_scores = [GeometricMeter() for metric in self.metrics]
        
        input_image = ImageMeter()
        output_image = ImageMeter()
        target_image = ImageMeter()

        confusion = ConfusionMatrix(threshold=0.5)

        start = time.time()
        for input, target, _ in dataloader:
            data_times.update(time.time() - start)
            batch_size = input.size(0)

            # forward & backward
            loss, output = self.model(input, target, mode)

            losses.update(loss, batch_size)
            del loss
            propagate_times.update(time.time() - data_times.val)            
            start = time.time()

            for metric, arith_score, geo_score in zip(self.metrics, metric_scores, metric_geometric_scores):
                arith_score.update(metric(output, target), batch_size)
                geo_score.update(metric(output, target), batch_size)
            
            input_image.update(input)
            output_image.update(output)
            target_image.update(target)

        if mode == 'train':
            self.model.step(epoch, self.LOG)
            self.datasets['train'].step()

        # log ()
        self.LOG('tensorboard', type='scalar', turn=mode, name='time/propagate', epoch=epoch, values=propagate_times.avg)
        self.LOG('tensorboard', type='scalar', turn=mode, name='time/data', epoch=epoch, values=data_times.avg)
        self.LOG('tensorboard', type='scalar', turn=mode, name='loss', epoch=epoch, values=losses.avg)
        
        for metric, arith_score, geo_score in zip(self.metrics, metric_scores, metric_geometric_scores):
            self.LOG('tensorboard', type='scalar', turn=mode, name='arithmetic/{}'.format(metric.__repr__()), epoch=epoch, values=arith_score.avg)
            self.LOG('tensorboard', type='scalar', turn=mode, name='geometric/{}'.format(metric.__repr__()), epoch=epoch, values=geo_score.avg)
            
        if do_log:
            self.LOG('tensorboard', type='image', turn=mode, name='input/box', epoch=epoch, values=input_image.getImages().narrow(1, 0, 1))
            self.LOG('tensorboard', type='image', turn=mode, name='input/pano', epoch=epoch, values=input_image.getImages().narrow(1, 1, 1))
            self.LOG('tensorboard', type='image', turn=mode, name='major/output', epoch=epoch, values=output_image.getImages().narrow(1, 0, 1))
            self.LOG('tensorboard', type='image', turn=mode, name='minor/output', epoch=epoch, values=output_image.getImages().narrow(1, 1, 1))
            self.LOG('tensorboard', type='image', turn=mode, name='major/target', epoch=epoch, values=target_image.getImages().narrow(1, 0, 1))
            self.LOG('tensorboard', type='image', turn=mode, name='minor/target', epoch=epoch, values=target_image.getImages().narrow(1, 1, 1))

            self.LOG('tensorboard', type='image', turn=mode, name='confusion', epoch=epoch, values=confusion(output_image.images, target_image.images))

        self.write_seg_pr(mode, output_image.images, target_image.images, 'major/both', epoch)
        self.write_seg_pr(mode, output_image.images.narrow(1, 0, 1), target_image.images.narrow(1, 0, 1), 'major/seg', epoch)
        self.write_seg_pr(mode, output_image.images.narrow(1, 1, 1), target_image.images.narrow(1, 1, 1), 'minor/seg', epoch)
        self.write_class_pr(mode, output_image.images.narrow(1, 0, 1), target_image.images.narrow(1, 0, 1), 'major/class', epoch)
        self.write_class_pr(mode, output_image.images.narrow(1, 1, 1), target_image.images.narrow(1, 1, 1), 'minor/class', epoch)


    def write_seg_pr(self, turn, output, target, name, epoch):
        output = output.contiguous().view(output.numel())
        target = target.contiguous().view(target.numel())
        self.LOG('tensorboard', type='pr', turn=turn, name=name, epoch=epoch, values=(target, output))

    def write_class_pr(self, turn, output, target, name, epoch):
        batch_size = output.size(0)
        channel = output.size(1)
        assert channel == 1
        output = output.view(batch_size, -1).mean(dim=1)
        target = target.view(batch_size, -1).mean(dim=1)
        self.LOG('tensorboard', type='pr', turn=turn, name=name, epoch=epoch, values=(target, output))

    def search(self):
        pass
