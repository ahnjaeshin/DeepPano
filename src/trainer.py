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
import torch.nn.init as init
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.utils import make_grid
from torchvision import datasets
from torch.autograd import Variable

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

def cuda(x, device, async=False):
    """for use in gpu
    add async=True"""

    if not torch.cuda.is_available():
        return x
    if async:
        return x.cuda(device, non_blocking=True)
    return x.to(device)

def cpu(x):
    return x.detach().cpu()

def write_scalar_log(name, val, epoch, log, writer):
    """write log to both slack and tensorboard
    """

    log.append('{name} : {val:.5f}'.format(name=name, val=val))
    writer.add_scalar(name, val, epoch)


class Init():
    def __init__(self, init="xavier_normal"):
        self.init = init
    
    def __call__(self, m):
        if isinstance(m, nn.Conv2d):
            if self.init is "xavier_normal":
                init.xavier_normal(m.weight)
            elif self.init is "he_uniform":
                init.kaiming_uniform(m.weight)
            elif self.init is "he_normal":
                init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

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

    def __init__(self, model, datasets, criterion, optimizer, scheduler, metrics, 
                writers, LOG, path, checkpoint=None, init=None):
        
        self.model = model
        self.datasets = datasets
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.path = path
        self.checkpoint = checkpoint
        self.writers = writers
        self.LOG = LOG

        self.epochs = 0
        self.start_epoch = 0
        self.best_score = 0
        self.init = Init(init)

        if init:
            self.model.apply(self.init)

        if checkpoint:
            self.load(checkpoint)

        if not torch.cuda.is_available(): 
            print("CUDA is unavailable")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, *args, **kwargs):
        """Call trainer
        Does train, ensemble, etc.. as needed
        
        Keyword Arguments:
            batch_size {int} -- input batch size (default: {16})
            num_workers {int} -- number of worker threads (data loading) (default: {32})
            epochs {int} -- total number of epochs (default: {100})
            log_freq {int} -- frquency of logs (default: {10})
        """

        
    def train(self, batch_size, num_workers, epochs, log_freq):
        
        self.epochs = epochs
        
        self.LOG('train', 'train started: using {} gpus'.format(torch.cuda.device_count()))

        dataloaders = { x: DataLoader(dataset = self.datasets[x], 
                                      batch_size = batch_size, 
                                      shuffle = True, 
                                      pin_memory=torch.cuda.is_available(), 
                                      num_workers=num_workers) 
                for x in ['train', 'val']}

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)
            torch.backends.cudnn.benchmark = True

        start_time = datetime.datetime.now()
        self.best_score = 0

        for epoch in range(self.start_epoch, epochs):
            do_log = ((epoch + 1) % log_freq == 0)

            log = []
            for state in ('train', 'val'):
                log_i, score = self.train_once(epoch, dataloaders[state], state == 'train', self.writers[state], do_log)
                log += log_i

                is_best = self.best_score < score
                self.best_score = max(self.best_score, score)
                self.save_checkpoint(epoch, is_best, do_log)

            elasped_time = datetime.datetime.now() - start_time
            eta = start_time + ((elasped_time / ((epoch - self.start_epoch) + 1)) * (epochs - self.start_epoch))

            print('\n'.join(log))
            if do_log:
                self.LOG('epoch: {}/{}, elasped: {}, eta: {}'.format(epoch + 1, epochs, elasped_time, eta), *log)

        self.LOG('train ended', '')

    def train_once(self, epoch, dataloader, train, writer, do_log=True):
        """one epoch with logging
        
        Arguments:
            epoch {int} -- the current epoch
            dataloader {Dataloader object} -- the loader to iterate through
            train {bool} -- true if train mode
            writer {tensorboardX writer} -- writer log
        
        Keyword Arguments:
            do_log {bool} -- save model if true (default: {True})
        """
        
        losses = AverageMeter()
        data_times = AverageMeter()
        propagate_times = AverageMeter()
        metric_scores = [AverageMeter() for metric in self.metrics]
        metric_geometric_scores = [GeometricMeter() for metric in self.metrics]
        curr_scores = AverageMeter() # average of all metrics

        both_result = ClassMeter()
        major_result = ClassMeter()
        minor_result = ClassMeter()

        input_image = ImageMeter(16)
        output_image = ImageMeter(16)
        target_image = ImageMeter(16)

        confusion = ConfusionMatrix(threshold=0.5)

        start = time.time()
        for input, target, index in dataloader:
            data_times.update(time.time() - start)
            batch_size = input.size(0)

            # forward & backward
            loss, output = self.batch_once(input, target, train)

            losses.update(loss, batch_size)
            del loss
            propagate_times.update(time.time() - data_times.val)            
            start = time.time()

            for metric, arith_score, geo_score in zip(self.metrics, metric_scores, metric_geometric_scores):
                arith_score.update(metric(output, target), batch_size)
                geo_score.update(metric(output, target), batch_size)
                curr_scores.update(arith_score.val, batch_size)
            
            both_result.update(output, target)
            major_result.update(output.narrow(1, 0, 1), target.narrow(1, 0, 1))
            minor_result.update(output.narrow(1, 1, 1), target.narrow(1, 1, 1))

            input_image.update(input)
            output_image.update(output)
            target_image.update(target)

        if train:
            self.scheduler.step()

        # log ()
        log = [ '{0}'.format('train' if train else 'val'), ]
        write_scalar_log('time/propagate', propagate_times.avg, epoch, log, writer)
        write_scalar_log('time/data', data_times.avg, epoch, log, writer)
        write_scalar_log('loss', losses.avg, epoch, log, writer)
        write_scalar_log('score', curr_scores.avg, epoch, log, writer)

        if train:
            lr = [group['lr'] for group in self.optimizer.param_groups][0]
            write_scalar_log('learning_rate', lr, epoch, log, writer)

            model = self.model.module if torch.cuda.device_count() > 1 else self.model

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value, epoch, bins='doane')
                writer.add_histogram(tag + '/grad', value, epoch, bins='doane')

        for metric, arith_score, geo_score in zip(self.metrics, metric_scores, metric_geometric_scores):
            write_scalar_log('arithmetic/{}'.format(metric.__repr__()), arith_score.avg, epoch, log, writer)
            write_scalar_log('geometric/{}'.format(metric.__repr__()), geo_score.avg, epoch, log, writer)
            
        writer.add_image('input/box', make_grid(input_image.images.narrow(1, 0, 1), normalize=True, scale_each=True), epoch)
        writer.add_image('input/pano', make_grid(input_image.images.narrow(1, 1, 1), normalize=True, scale_each=True), epoch)
        writer.add_image('major/output', make_grid(output_image.images.narrow(1, 0, 1), normalize=True, scale_each=True), epoch)
        writer.add_image('major/target', make_grid(target_image.images.narrow(1, 0, 1), normalize=True, scale_each=True), epoch)
        writer.add_image('minor/output', make_grid(output_image.images.narrow(1, 1, 1), normalize=True, scale_each=True), epoch)
        writer.add_image('minor/target', make_grid(target_image.images.narrow(1, 1, 1), normalize=True, scale_each=True), epoch)

        writer.add_image('confusion', confusion(output_image.images, target_image.images), epoch)

        writer.add_pr_curve('both', both_result.targets, both_result.outputs, epoch)
        writer.add_pr_curve('major', major_result.targets, major_result.outputs, epoch)
        writer.add_pr_curve('minor', minor_result.targets, minor_result.outputs, epoch)

        return log, curr_scores.avg

    def batch_once(self, input, target, train):
        assert set(np.unique(target[0])).issubset({0,1})

        input, target = cuda(input, self.device), cuda(target, self.device, True)
        
        if train:
            self.model.train()
            output = self.model(input)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()   
            loss.backward()         
            self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(input)
            loss = self.criterion(output, target)

        output = cpu(output)
        loss = loss.cpu().item()

        return loss, output

    def infer(self):
        pass

    def search(self):
        pass

    def ensemble(self):
        pass        

    def save_checkpoint(self, epoch, is_best, do_log):

        if not do_log and not is_best:
            return

        if torch.cuda.device_count() > 1:
            model = self.model.module
        else:
            model = self.model

        filename = "../result/checkpoint/{}-{}.pth.tar".format(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M'), epoch)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'model': model.__class__.__name__,
            'best_score': self.best_score,
            'optimizer' : self.optimizer.state_dict(),
        }
        if do_log:
            torch.save(state, filename)
            if is_best:
                shutil.copyfile(filename, "../result/checkpoint/best-{}-{}.pth.tar".format(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M'), epoch))
        else:
            torch.save(state, "../result/checkpoint/best-{}-{}.pth.tar".format(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M'), epoch))

    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            self.LOG('checkpointing', "=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path)

            # if checkpoint['model'] is not self.model.__class__.__name__:
            #     raise Exception('model name does not match: checkpoint: {}, model: {}'.format(checkpoint['model'], self.model.__class__.__name__))

            self.start_epoch = checkpoint['epoch']
            self.best_score = checkpoint['best_score']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
            self.LOG('checkpoint', "=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path))
            self.LOG('checkpoint', "=> no checkpoint found at '{}'".format(path))