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
from tqdm import tqdm

from torchvision.utils import make_grid
from torchvision import datasets
from torch.autograd import Variable

from utils import slack_message, AverageMeter, GeometricMeter, ImageMeter, ClassMeter
from torch.nn import functional as F
import shutil
from metric import Accuracy

import collections

def cuda(x, async=False):
    """for use in gpu
    add async=True"""

    if not torch.cuda.is_available():
        return x

    if isinstance(x, collections.Iterable):
        if async:
            return [x_i.cuda(non_blocking=True) for x_i in x]
        else:
            return [x_i.cuda() for x_i in x]
  
    if async:
        return x.cuda(non_blocking=True)
    else:
        return x.cuda()

def render_output(x):
    """sigmoid => cpu
    
    Arguments:
        x {tensor or iterable of tensors}
    """
    if isinstance(x, collections.Iterable):
        return [F.sigmoid(x_i).detach().cpu() for x_i in x]
    else:
        return F.sigmoid(x).detach().cpu()

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

    def __init__(self, model, datasets, criterion, optimizer, scheduler, metrics, writers, checkpoint=None, init=None):
        
        self.model = model
        self.datasets = datasets
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.checkpoint = checkpoint
        self.writers = writers

        self.start_epoch = 0
        self.best_score = 0
        self.init = Init(init)

        self.criterion2 = nn.BCEWithLogitsLoss()
        self.acc = Accuracy(0.5)

        if init:
            self.model.apply(self.init)

        if checkpoint:
            self.load(checkpoint)

        if not torch.cuda.is_available(): 
            print("CUDA is unavailable. It'll be very slow...")

    def __call__(self, batch_size = 16, num_workers = 32, epochs = 100, log_freq = 10):
        """Call trainer
        Does train, ensemble, etc.. as needed
        
        Keyword Arguments:
            batch_size {int} -- input batch size (default: {16})
            num_workers {int} -- number of worker threads (data loading) (default: {32})
            epochs {int} -- total number of epochs (default: {100})
            log_freq {int} -- frquency of logs (default: {10})
        """

        
    def train(self, batch_size, num_workers, epochs, log_freq):
        
        slack_message('train started', '#botlog')

        dataloaders = { x: DataLoader(dataset = self.datasets[x], 
                                      batch_size = batch_size, 
                                      shuffle = True, 
                                      pin_memory=torch.cuda.is_available(), 
                                      num_workers=num_workers) 
                for x in ['train', 'val']}

        if torch.cuda.is_available():
            self.model = cuda(self.model)
            torch.backends.cudnn.benchmark = True
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        start_time = datetime.datetime.now()
        self.best_score = 0

        for epoch in tqdm(range(self.start_epoch, epochs), desc='epoch'):
            do_log = ((epoch + 1) % log_freq == 0)

            for state in ('train', 'val'):
                self.train_once(epoch, dataloaders[state], state == 'train', self.writers[state], do_log)
            
            elasped_time = datetime.datetime.now() - start_time
            eta = start_time + ((elasped_time / ((epoch - self.start_epoch) + 1)) * (epochs - self.start_epoch))

            log = 'epoch: {}/{}, elasped: {}, eta: {}'.format(epoch + 1, epochs, elasped_time, eta)
            tqdm.write(log)
            slack_message(log, '#botlog')

        slack_message('train ended', '#botlog')

    def train_once(self, epoch, dataloader, train, writer, do_log=True):
        """one forward, one backward with logging
        
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

        result_classification = ClassMeter()
        result_segmentation = ClassMeter()

        input_image = ImageMeter(30)
        output_image = ImageMeter(30)
        target_image = ImageMeter(30)

        self.model.train(train)

        start = time.time()
        for batch_idx, (input, target, filepath, index) in enumerate(tqdm(dataloader, desc='batch')):
            data_times.update(time.time() - start)
            batch_size = input.size(0)

            # forward & backward
            loss, output = self.batch_once(input, target, train)

            losses.update(loss, batch_size)
            del loss
            propagate_times.update(time.time() - data_times.val)            
            start = time.time()

            for metric, arith_score, geo_score in zip(self.metrics, metric_scores, metric_geometric_scores):
                arith_score.update(metric.eval(output, target), batch_size)
                geo_score.update(metric.eval(output, target), batch_size)
                curr_scores.update(arith_score.val, batch_size)
            
            result_classification.update(output[1], target[1])
            result_segmentation.update(output[0], target[0])

            input_image.update(input)
            output_image.update(output[0])
            target_image.update(target[0])

        log = [
            '[{0}] Epoch: [{1}][{2}/{3}]'.format('train' if train else 'val', epoch, batch_idx + 1, len(dataloader)),
        ]
        write_scalar_log('time/propagate', propagate_times.avg, epoch, log, writer)
        write_scalar_log('time/data', data_times.avg, epoch, log, writer)
        write_scalar_log('loss', losses.avg, epoch, log, writer)

        if train:
            lr = [group['lr'] for group in self.optimizer.param_groups][0]
            self.scheduler.step()
            write_scalar_log('learning_rate', lr, epoch, log, writer)

            for tag, value in self.model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('model/(train)' + tag, value, epoch, bins='doane')
                writer.add_histogram('model/(train)' + tag + '/grad', value, epoch, bins='doane')
        else:
            is_best = self.best_score < curr_scores.avg
            self.best_score = max(self.best_score, curr_scores.avg)
            self.save_checkpoint(epoch, is_best)

        for metric, arith_score, geo_score in zip(self.metrics, metric_scores, metric_geometric_scores):
            write_scalar_log('metric/arithmetic/{}'.format(metric.__repr__()), arith_score.avg, epoch, log, writer)
            write_scalar_log('metric/geometric/{}'.format(metric.__repr__()), geo_score.avg, epoch, log, writer)

        tqdm.write("\n".join(log))

        if do_log:
            slack_message("\n".join(log), '#botlog')

        writer.add_image('input/pano', make_grid(input_image.images.narrow(1, 1, 1), normalize=True, scale_each=True), epoch)
        writer.add_image('input/guideline', make_grid(input_image.images.narrow(1, 0, 1), normalize=True, scale_each=True), epoch)
        writer.add_image('target/segmentation', make_grid(target_image.images, normalize=True, scale_each=True), epoch)
        writer.add_image('output', make_grid(output_image.images, normalize=True, scale_each=True), epoch)
        
        writer.add_pr_curve('accuracy/segmentation', result_classification.targets, result_classification.outputs, epoch)
        writer.add_pr_curve('accuracy/classification', result_segmentation.targets, result_segmentation.outputs, epoch)

    def batch_once(self, input, target, train):
        assert set(np.unique(target[0])).issubset({0,1})

        input, target = cuda(input), cuda(target)

        output = self.model(input)
        loss = self.criterion(outputs=output, targets=target)

        if train: 
            self.optimizer.zero_grad()   
            loss.backward()         
            self.optimizer.step()

        return loss.cpu().item(), render_output(output)

    def search(self):
        pass

    def ensemble(self):
        pass        

    def save_checkpoint(self, epoch, is_best):

        if torch.cuda.device_count() > 1:
            model = self.model.module
        else:
            model = self.model

        filename = "../result/checkpoint-{}-{}.pth.tar".format(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M'), epoch)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'model': model.__class__.__name__,
            'best_score': self.best_score,
            'optimizer' : self.optimizer.state_dict(),
        }
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, "../result/checkpoint-best-{}-{}.pth.tar".format(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M'), epoch))

    def load(self, path):
        if os.path.isfile(path):
            tqdm.write("=> loading checkpoint '{}'".format(path))
            slack_message("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path)
            self.start_epoch = checkpoint['epoch']
            self.best_score = checkpoint['best_score']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            tqdm.write("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
            slack_message("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
        else:
            tqdm.write("=> no checkpoint found at '{}'".format(path))
            slack_message("=> no checkpoint found at '{}'".format(path))