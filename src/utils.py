import argparse
import socket
import os
import datetime
import time
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from functools import reduce
import torch.nn.functional as F

import math
import numpy as np

from slacker import Slacker

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
        if self.image.size(0) < k:
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
            
    # if torch.cuda.is_available():
    #     dtype = torch.cuda.FloatTensor
    # else: #TODO : change to support cuda
    dtype = torch.FloatTensor
    
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1,*in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(1,*input_size)).type(dtype)
        
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
    return "```" + '\n'.join(log) + "```"


def timer(func):
    """decorator to time functions (for profiling)
    
    Arguments:
        func {function} -- function you want to time
    """
    pass

class slack_message:
    """send slack message
    
    Arguments:
        text {string} -- any string want to send
    
    Keyword Arguments:
        channel {string} -- the name of channel to send to (default: channel #botlog)
    """

    def __init__(self, name, channel=None):
        slack_token = os.environ["SLACK_API_TOKEN"]
        self.slack = Slacker(slack_token)
        self.host = socket.gethostname() + ' ' + name + '@bot'

        self.channel = channel    
        if not channel:
            self.channel = 'C9ZKLPGBV' # channel id of #botlog channel

    def __call__(self, header, *msg):
        log = {}
        log['title'] = header
        log['text'] = '\n'.join(msg)
        try:
            self.slack.chat.post_message(self.channel, text=None, attachments=[log], as_user=False, username=self.host)
        except Exception as e:
            print("slack error occured: {}".format(e))
