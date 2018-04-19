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

from torchvision.utils import make_grid
from torchvision import datasets
from torch.autograd import Variable

from utils import slack_message, count_parameters, AverageMeter
from torch.nn import functional as F
import shutil


def cuda(x, async=True):
    """for use in gpu
    add async=True"""
    return x.cuda() if torch.cuda.is_available() else x

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

    def __init__(self, model, datasets, criterion, optimizer, scheduler, metrics, checkpoint=None):
        
        self.model = model
        self.datasets = datasets
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.checkpoint = checkpoint

        self.start_epoch = 0
        self.best_score = 0

        if checkpoint:
            self.load(checkpoint)

        if not torch.cuda.is_available(): 
            print("CUDA is unavailable. It'll be very slow...")
        
        self.trainWriter = SummaryWriter(comment="-train")
        self.valWriter = SummaryWriter(comment="-val")
        dummy_input = Variable(torch.rand(1, 2, 224, 224), requires_grad=True)

        for writer in [self.trainWriter, self.valWriter]:
            writer.add_graph(self.model, (dummy_input, ))
            writer.add_scalar('number of parameter', count_parameters(self.model))
            writer.add_text('info', 'this is start', 0)

    def train(self, batch_size = 16, num_workers = 32, epochs = 100, log_freq = 10):
        
        slack_message('train started', '#botlog')
        # slack_message(self.model.__repr__, '#botlog')

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
        self.best_score = 0

        for epoch in tqdm(range(self.start_epoch, epochs), desc='epoch'):
            checkpoint = (epoch % log_freq == 0)
            self.train_once(epoch, dataloaders['train'], True, checkpoint)
            self.train_once(epoch, dataloaders['val'], False, checkpoint)
            
            elasped_time = datetime.datetime.now() - start_time
            eta = start_time + ((elasped_time / (epoch + 1)) * epochs)

            log = 'epoch: {}/{}, elasped: {}, eta: {}'.format(epoch, epochs, elasped_time, eta)
            tqdm.write(log)
            slack_message(log, '#botlog')

        slack_message('train ended', '#botlog')

    def train_once(self, epoch, dataloader, train, checkpoint=True):
        
        losses = AverageMeter()
        forward_times = AverageMeter()
        backward_times = AverageMeter()
        data_times = AverageMeter()
        curr_scores = AverageMeter()

        best_score = 0

        self.model.train(train)
        state = 'train' if train else 'val'
        writer = self.trainWriter if train else self.valWriter

        start = time.time()
        for batch_idx, (input, target, filepath, index) in enumerate(tqdm(dataloader, desc='batch')):
            
            data_times.update(time.time() - start)

            input = cuda(Variable(input))
            target = cuda(Variable(target))
            output = self.model(input)
            loss = self.criterion(output, target)
            output = F.sigmoid(output)

            losses.update(loss.cpu().data[0], input.cpu().size(0))
            forward_times.update(time.time() - data_times.val)

            if train : 
                self.optimizer.zero_grad()            
                loss.backward()
                self.optimizer.step()
                
            backward_times.update(time.time() - forward_times.val)
            start = time.time()

            for metric in self.metrics:
                curr_score = 0
                curr_score = curr_score + metric.add(output.data.cpu().numpy(), target.data.cpu().numpy())
                curr_scores.update(curr_score)

            if batch_idx == len(dataloader) - 1:
                log = [
                    '[{}]'.format(state),
                    'Epoch: [{0}][{1}/{2}]'.format(epoch, batch_idx, len(dataloader)),
                    'Forward Time {time.val:.3f} ({time.avg:.3f})'.format(time=forward_times),
                    'Backward Time {time.val:.3f} ({time.avg:.3f})'.format(time=backward_times),
                    'Data {time.val:.3f} ({time.avg:.3f})'.format(time=data_times),
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses),
                ]

                writer.add_scalar('time/forward', forward_times.val, epoch)
                writer.add_scalar('time/backward', backward_times.val, epoch)
                writer.add_scalar('time/data', data_times.val, epoch)
                writer.add_scalar('loss', losses.val, epoch)

                if train:
                    lr = [group['lr'] for group in self.optimizer.param_groups][0]
                    tqdm.write('Learning rate: {}'.format(lr))
                    writer.add_scalar('learning rate', lr, epoch)

                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('model/(train)' + tag, value.data.cpu().numpy(), epoch, bins='doane')
                        writer.add_histogram('model/(train)' + tag + '/grad', value.grad.data.cpu().numpy(), epoch, bins='doane')

                for metric in self.metrics:
                    log.append('metric/{}: {:.5f} ({:.5f})'.format(metric.__repr__(), metric.getRecentScore(), metric.mean()))
                    writer.add_scalar('metric/{}'.format(metric.__repr__()), metric.getRecentScore(), epoch)
                log = "\n".join(log)

                writer.add_image('input/pano', make_grid(input.data.cpu().narrow(1, 1, 1), normalize=True, scale_each=True), epoch)
                writer.add_image('input/guideline', make_grid(input.data.cpu().narrow(1, 0, 1), normalize=True, scale_each=True), epoch)
                writer.add_image('target', make_grid(target.data.cpu(), normalize=True, scale_each=True), epoch)
                writer.add_image('output', make_grid(output.data.cpu(), normalize=True, scale_each=True), epoch)
                
                writer.add_pr_curve('accuracy', target.data.cpu(), output.data.cpu(), epoch)

                tqdm.write(log)
                slack_message(log, '#botlog')

                is_best = self.best_score < curr_score
                self.best_score = max(self.best_score, curr_score)

                if checkpoint:
                    self.save_checkpoint(epoch, is_best)
                
        if train: # smoothing effect to LR reduce on platue
            self.scheduler.step(losses.avg)

        # todo : for every batch, log hard examples

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

