import datetime
import logging
import math
import os
import socket
import time
from collections import OrderedDict
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from slacker import Slacker
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# define a class to log values during training
class AverageMeter():
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GeometricMeter():
    """Borrowed ideas from AverageMeter Above

    because of over/under flow problems, log first
    """

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.product = 0
        self.count = 0
        self.avg = 0
    
    def update(self, val, n=1):
        self.val = val
        self.product += math.log(val + 1e-15) * n
        self.count += n
        self.avg = math.exp(self.product / self.count)

class ImageMeter():
    
    def __init__(self):
        """images = 4d tensor of images
        
        Keyword Arguments:
            limit {int} -- maximum number of images to store (default: {20})
        """

        self.images = None

    def update(self, image):
        if self.images is None:
            self.images = image
        self.images = torch.cat([self.images, image], dim=0)
        
    def getImages(self, k=32):
        if self.images.size(0) < k:
            return self.images
        else:
            return self.images.narrow(0, 0, k)

class ClassMeter():
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.outputs = None
        self.targets = None

    def update(self, output, target):
        
        if self.outputs is None:
            self.outputs = output.numpy().flatten()
            self.targets = output.numpy().flatten()
        else:
            self.outputs = np.append(self.outputs, output.numpy())
            self.targets = np.append(self.targets, target.numpy())
        

def broadcast(func):
    """ 
    decorator for broadcasting
    """
    def broadcast_func(x, *args, **kwargs):
        if type(x) is list or type(x) is tuple:
            return [func(x_i, *args, **kwargs) for x_i in x]
        else:
            return func(x, *args, **kwargs)
    return broadcast_func


"""
Code borrowed & modified from https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
"""
def model_summary(model, input_size):

    def calculate_depth(module):
        
        assert isinstance(module, nn.Module)

        if len(list(module.children())) == 0:
            return 0
        
        depth = reduce(lambda max_d, m: max(max_d, calculate_depth(m) + 1), module.children(), -1)

        return depth

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__.__name__)
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['depth'] = calculate_depth(module)
            if isinstance(output, (list,tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params
            
        if (not isinstance(module, nn.Sequential) and 
            not isinstance(module, nn.ModuleList) and 
            not (module == model)):
            hooks.append(module.register_forward_hook(hook))
            
    dtype = torch.FloatTensor
    
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.rand(1,*in_size).type(dtype) for in_size in input_size]
    else:
        x = torch.rand(1,*input_size).type(dtype)
        
    log = []
    max_depth = 0
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    log.append('----------------------------------------------------------------')
    line_new = '{:>20}  {:>20} {:>10}'.format('Layer (type)', 'Output Shape', 'Param #')
    log.append(line_new)
    log.append('================================================================')
    total_params = 0
    trainable_params = 0

    max_depth = max({summary[layer]['depth'] for layer in summary})

    for layer in summary:
        # input_shape, output_shape, trainable, nb_params

        indent = '-' * (max_depth - summary[layer]['depth']) * 2

        if summary[layer]['nb_params'] == 0:
            line_new = '{:<25} {:>20} {:>10}'.format(indent + layer, str(summary[layer]['output_shape']), 'None')
        else:
            line_new = '{:<25} {:>20} {:>10}'.format(indent + layer, str(summary[layer]['output_shape']), summary[layer]['nb_params'])
        
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        log.append(line_new)
    log.append('================================================================')
    log.append('Total params: ' + str(total_params))
    log.append('Trainable params: ' + str(trainable_params))
    log.append('Non-trainable params: ' + str(total_params - trainable_params))
    log.append('----------------------------------------------------------------')

    log = slack_message_chunker(log)

    return log, trainable_params

def slack_message_chunker(log):
    """divide list into list of logs (each log less than limi)
    
    Arguments:
        message {list} -- [list of logs]
    """

    log_segment = []
    block = ['```']
    total = 0
    SLACK_LIMIT = 7500 #8000
    for line in log:
        line_length = len(line)
        if total + line_length >= SLACK_LIMIT:
            block.append('```')
            log_segment.append('\n'.join(block))
            block = ['```', line]
            total = 0
        else:
            block.append(line)
            total = total + line_length
    
    block.append('```')
    log_segment.append('\n'.join(block))
    return log_segment

@broadcast
def slack_message(msg, header, slack, channel, host):
    log = {}
    log['title'] = header
    log['text'] = msg
    try:
        slack.chat.post_message(channel, text=None, attachments=[log], as_user=False, username=host)
    except Exception as e:
        print("slack error occured: {}".format(e))


class Logger:
    
    def __init__(self, title, log=False, channel=None, start_time=None, trial=None, logdir=None):
        slack_token = os.environ["SLACK_API_TOKEN"]
        self.slack = Slacker(slack_token)
        self.host = '{host}_{name}@bot'.format(host=socket.gethostname(), name=title)
        self.log = log
        self.channel = channel if channel else 'C9ZKLPGBV' # channel id of #botlog channel
        self.time = time if time else datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        self.trial = trial if trial else 0

        self.log_dir_base = '../result/runs/{title}/{time}_{trial}/'.format(title=title, time=start_time, trial=trial)
        self.writers = {x : SummaryWriter(log_dir=self.log_dir_base + x) for x in ('train', 'val')}

    def __call__(self, mode, **content):
        lookup = {
            'slack': self.send_slack_message,
            'print': self.print_log,
            'tensorboard': self.tensorboard,
            'model': self.add_model,
        }
        lookup[mode](**content)

    def print_log(self, name, values):
        print('{name}: {val}'.format(name=name, val=values))

    def tensorboard(self, type, turn, name, epoch, values):
        assert turn in ('train', 'val')
        if type == 'scalar':
            self.writers[turn].add_scalar(name, values, epoch)
        elif type == 'image':
            self.writers[turn].add_image(name, make_grid(values, normalize=True, scale_each=True), epoch)
        elif type == 'histogram':
            self.writers[turn].add_histogram(name, values, epoch, bins='doane')
        elif type == 'pr':
            target, output = values
            self.writers[turn].add_pr_curve(name, target, output, epoch)
        else:
            raise NotImplementedError

    def add_model(self, title, model, input_size):
        model = model.cpu()
        dummy_input = torch.rand(1, *input_size)
        self.writers['train'].add_graph(model, (dummy_input, ))
        model_sum, trainable_param = model_summary(model, input_size=input_size)
        self.writers['train'].add_scalar('number of parameter/w_input', count_parameters(model))
        self.writers['train'].add_scalar('number of parameter/wo_input', trainable_param)
        self.send_slack_message(title, model_sum, merge=False)
        self.print_log(title, '\n'.join(model_sum))
        self.print_log(title, model.__repr__())

    def send_slack_message(self, name, values, merge=True):
        if type(values) is list or type(values) is tuple:
            if merge:
                values = '\n'.join(values)
        slack_message(values, name, slack=self.slack, channel=self.channel, host=self.host)

    def finish(self):
        self.writers['train'].close()
        self.writers['val'].close()

class TypeParser:
    
    def __init__(self, types):
        self.types = types

    def __call__(self, type, param=None):
        return self.lookup(type)() if param is None else self.lookup(type)(**param)

    def lookup(self, type):
        if type not in self.types:
            print("types: {}".format(self.types))
            raise NotImplementedError
        return self.types[type]
