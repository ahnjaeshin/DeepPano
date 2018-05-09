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

from utils import slack_message, AverageMeter
from torch.nn import functional as F
import shutil
from metric import Accuracy


def cuda(x, async=False):
    """for use in gpu
    add async=True"""
    if async:
        return x.cuda(non_blocking=True) if torch.cuda.is_available() else x
    else:
        return x.cuda() if torch.cuda.is_available() else x

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
        
    def train(self, batch_size = 16, num_workers = 32, epochs = 100, log_freq = 10):
        
        slack_message('train started', '#botlog')
        # slack_message(self.model.__repr__, '#botlog')

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
            checkpoint = ((epoch + 1) % log_freq == 0)

            for state in ('train', 'val'):
                self.train_once(epoch, dataloaders[state], state == 'train', self.writers[state], checkpoint)
            
            elasped_time = datetime.datetime.now() - start_time
            eta = start_time + ((elasped_time / (epoch + 1)) * epochs)

            log = 'epoch: {}/{}, elasped: {}, eta: {}'.format(epoch + 1, epochs, elasped_time, eta)
            tqdm.write(log)
            slack_message(log, '#botlog')

        slack_message('train ended', '#botlog')

    def train_once(self, epoch, dataloader, train, writer, checkpoint=True):
        
        losses = AverageMeter()
        losses2 = AverageMeter()
        forward_times = AverageMeter()
        backward_times = AverageMeter()
        data_times = AverageMeter()
        curr_scores = AverageMeter()
        
        metric_scores = [AverageMeter() for metric in self.metrics]
        acc_score = AverageMeter()

        self.model.train(train)

        start = time.time()
        for batch_idx, (input, target, filepath, index) in enumerate(tqdm(dataloader, desc='batch')):
            assert (set(np.unique(target)) == {0,1} ) or (set(np.unique(target)) == {0}) or (set(np.unique(target)) == {1})
            data_times.update(time.time() - start)

            batch_size = input.size(0)

            t = target.view(batch_size, -1)
            t = t.sum(dim=1)
            target2 = (t != 0).float() # 1 is segmentable

            input = cuda(Variable(input))
            target = cuda(Variable(target))
            target2 = cuda(Variable(target2))
            output, output2 = self.model(input)
            loss = self.criterion(output, target)
            loss2 = self.criterion2(output2, target2)
            output = F.sigmoid(output)
            output2 = F.sigmoid(output2)

            output2 = (output2 >= 0.5).float()
            # print(output.size(), output2.size())
            output = output * output2.view(batch_size, 1, 1, 1)          

            losses.update(loss.cpu().item(), batch_size)
            losses2.update(loss2.cpu().item(), batch_size)
            forward_times.update(time.time() - data_times.val)
        
            if train : 
                self.optimizer.zero_grad()            
                loss.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()
                
            backward_times.update(time.time() - forward_times.val)
            start = time.time()

            for metric, score in zip(self.metrics, metric_scores):
                score.update(metric.eval(output.data.cpu().numpy(), target.data.cpu().numpy()), batch_size)
                curr_scores.update(score.val, batch_size)

            acc_score.update(self.acc.eval(output2.data.cpu().numpy(), target2.data.cpu().numpy()), batch_size)

            if batch_idx == len(dataloader) - 1: 
                log = [
                    '[{}]'.format('train' if train else 'val'),
                    'Epoch: [{0}][{1}/{2}]'.format(epoch, batch_idx + 1, len(dataloader)),
                    'Forward Time {time.val:.3f} ({time.avg:.3f})'.format(time=forward_times),
                    'Backward Time {time.val:.3f} ({time.avg:.3f})'.format(time=backward_times),
                    'Data {time.val:.3f} ({time.avg:.3f})'.format(time=data_times),
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses),
                    'Loss_acc {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses2),
                ]

                writer.add_scalar('time/forward', forward_times.avg, epoch)
                writer.add_scalar('time/backward', backward_times.avg, epoch)
                writer.add_scalar('time/data', data_times.avg, epoch)
                writer.add_scalar('loss/iou', losses.avg, epoch)
                writer.add_scalar('loss/acc', losses2.avg, epoch)

                if train:
                    lr = [group['lr'] for group in self.optimizer.param_groups][0]
                    tqdm.write('Learning rate: {}'.format(lr))
                    writer.add_scalar('learning rate', lr, epoch)

                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('model/(train)' + tag, value.data.cpu().numpy(), epoch, bins='doane')
                        writer.add_histogram('model/(train)' + tag + '/grad', value.grad.data.cpu().numpy(), epoch, bins='doane')

                for metric, score in zip(self.metrics, metric_scores):
                    log.append('metric/{name}: {metric.val:.5f} ({metric.avg:.5f})'.format(name=metric.__repr__(), metric=score))
                    writer.add_scalar('metric/{name}'.format(name = metric.__repr__()), score.avg, epoch)
                log.append('acc: {acc.avg}'.format(acc=acc_score))
                writer.add_scalar('metric/accuracy', acc_score.avg, epoch)
                log = "\n".join(log)

                writer.add_image('input/pano', make_grid(input.data.cpu().narrow(1, 1, 1), normalize=True, scale_each=True), epoch)
                writer.add_image('input/guideline', make_grid(input.data.cpu().narrow(1, 0, 1), normalize=True, scale_each=True), epoch)
                writer.add_image('target', make_grid(target.data.cpu(), normalize=True, scale_each=True), epoch)
                writer.add_image('output', make_grid(output.data.cpu(), normalize=True, scale_each=True), epoch)
                
                writer.add_pr_curve('accuracy/iou', target.data.cpu(), output.data.cpu(), epoch)
                writer.add_pr_curve('accuracy/acc', target2.data.cpu(), output2.data.cpu(), epoch)

                # writer.add_embedding(embed.data.cpu().view(input.size(0), -1), metadata=None, label_img=output.data.cpu(), global_step=epoch, tag='output')

                tqdm.write(log)

                is_best = self.best_score < curr_scores.avg
                self.best_score = max(self.best_score, curr_scores.avg)

                if checkpoint:
                    slack_message(log, '#botlog')
                    self.save_checkpoint(epoch, is_best)
                
        if train: # smoothing effect to LR reduce on platue
            self.scheduler.step()

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