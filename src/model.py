from models import *
from utils import TypeParser
import torch
import torch.nn as nn
import torch.nn.init as init
import loss
import os
import numpy as np

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


class Init():
    def __init__(self, init="xavier_normal"):
        self.init = init
    
    def __call__(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if self.init is "xavier_normal":
                init.xavier_normal(m.weight)
            elif self.init is "he_uniform":
                init.kaiming_uniform(m.weight)
            elif self.init is "he_normal":
                init.kaiming_normal(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

def getOptimizer(type, param, module_params):
    optimParser = TypeParser(types = {
        "SGD": torch.optim.SGD,
        "ADAM": torch.optim.Adam,
    })
    param['params'] = module_params
    return optimParser(type, param)

def getScheduler(type, param, optimizer):
    schedParser = TypeParser(types = {
        "Step": torch.optim.lr_scheduler.StepLR,
        "Plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    })
    param["optimizer"] = optimizer
    return schedParser(type, param)

def getLoss(type, param):
    lossParser = TypeParser(types = {
        "IOU": loss.IOULoss,
        "DICE": loss.DICELoss,
        "CE": nn.CrossEntropyLoss,
    })
    return lossParser(type, param)

def getModule(type, param):
    moduleParser = TypeParser(types={
        'UNET': unet.UNet, 
        'WNet': unet.WNet,
        'FCDenseNetSmall': tiramisu.FCDenseNetSmall,
        'FCDenseNet57': tiramisu.FCDenseNet57,
        'FCDenseNet67': tiramisu.FCDenseNet67,
        'FCDenseNet103': tiramisu.FCDenseNet103,
        'RecurNet': unet.RecurNet,
        'RecurNet2': unet.RecurNet2,
    })
    return moduleParser(type, param)

def getModel(category, param):
    modelParser = TypeParser(types = {
        "Vanilla": VanillaModel,
    })
    MODEL = modelParser(category, param)

    return MODEL

class BasicModule:
    def __init__(self, type, title, optimizer, scheduler,
                checkpoint="", weight_init="he_uniform"):
        pass

    def modelSummary(self, input_size, LOG):
        pass

class VanillaModel():
    
    def __init__(self, module, weight_init, optimizer, scheduler, loss, ensemble=False):
        
        self.module = getModule(**module)
        self.optimizer = getOptimizer(**optimizer, module_params=self.module.parameters())
        self.scheduler = getScheduler(**scheduler, optimizer=self.optimizer)
        self.criterion = getLoss(**loss)
        self.ensemble = ensemble

        if init:
            self.module.apply(Init(init))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input, target, turn):
        assert set(np.unique(target[0])).issubset({0,1})
        input, target = cuda(input, self.device), cuda(target, self.device, True)

        if turn == 'train':
            loss, output = self.train(input, target)
        elif turn == 'val':
            loss, output = self.validate(input, target)

        return loss.cpu().item(), cpu(output)


    def train(self, input, target):
        
        self.module.train()
        output = self.module(input)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()   
        loss.backward()         
        self.optimizer.step()

        return loss, output
    
    def validate(self, input, target):

        self.module.eval()
        with torch.no_grad():
            output = self.module(input)
        loss = self.criterion(output, target)

        return loss, output

    def step(self, epoch, LOG):
        
        lr = [group['lr'] for group in self.optimizer.param_groups][0]
        LOG('tensorboard', type='scalar', turn='train', 
            name='learning_rate', epoch=epoch, values=lr)

        module = self.module.module if torch.cuda.device_count() > 1 else self.module

        for tag, value in module.named_parameters():
            tag = tag.replace('.', '/')
            LOG('tensorboard', type='histogram', turn='train', 
                name=tag, epoch=epoch, values=value.data.cpu().numpy())
            LOG('tensorboard', type='histogram', turn='train', 
                name=tag+'/grad', epoch=epoch, values=value.grad.cpu().numpy())
        
        self.scheduler.step()

    def modelSummary(self, input_size, LOG):
        LOG('model', title=self.module.__class__.__name__, model=self.module, input_size=input_size)

    def getLog(self):
        pass

    def gpu(self):
        if torch.cuda.device_count() > 1:
            self.module = torch.nn.DataParallel(self.module)

        if torch.cuda.is_available():
            self.module = self.module.to(self.device)
            self.criterion = self.criterion.to(self.device)
            torch.backends.cudnn.benchmark = True
        else:
            print("CUDA is unavailable")

    def checkpoint(self, epoch, path):

        module = self.module.module if torch.cuda.device_count() > 1 else self.module

        os.makedirs(os.path.dirname(path), exist_ok=True)
        filename = "{}{}.pth.tar".format(path, epoch)

        state = {
            'epoch': epoch,
            'state_dict': module.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        }
        torch.save(state, filename)

    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path)
            epoch = checkpoint['epoch'] + 1
            self.module.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(path, epoch))
        else:
            epoch = 0
            print("=> no checkpoint found at '{}'".format(path))

        return epoch