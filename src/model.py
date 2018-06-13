from models import *
from utils import TypeParser
import torch
import torch.nn as nn
import torch.nn.init as init
import loss as L
import os
import numpy as np
import torch.nn.functional as F
import scheduler

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
        "RMS": torch.optim.RMSprop,
    })
    param['params'] = module_params
    return optimParser(type, param)

def getScheduler(type, param, optimizer):
    schedParser = TypeParser(types = {
        "Step": torch.optim.lr_scheduler.StepLR,
        "Plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "Restart": scheduler.CosineAnnealingLR,
    })
    param["optimizer"] = optimizer
    return schedParser(type, param)

def getLoss(type, param):
    lossParser = TypeParser(types = {
        "IOU": L.IOULoss,
        "DICE": L.DICELoss,
        "CE": nn.CrossEntropyLoss,
        "GAN": L.GANLoss,
        "WDICE": L.DICEWeightLoss,
    })
    param['reduce'] = False
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
        'Split': dna.SplitNet,
        'Cross': dna.CrossNet,
        'DNA': dna.DNANet,
    })
    return moduleParser(type, param)

def getGANModule(type, param):
    moduleParser = TypeParser(types = {
        'UNET': unet.StableUNet,
        'SimpleClassify': unet.SimpleClassify,
        'Split': dna.SplitNet,
        'Cross': dna.CrossNet,
        'DNA': dna.DNANet,
    })
    return moduleParser(type, param)

def getModel(category, param):
    modelParser = TypeParser(types = {
        "Vanilla": VanillaModel,
        "GAN": GANModel,
        "Recur": RecurModel,
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
        
        self.ensemble = ensemble

        if self.ensemble:
            module_params = []
            self.module = []
            for i in range(3):
                m = getModule(**module)
                if init:
                    m.apply(Init(init))
                self.module.append(m)
                module_params = module_params + list(m.parameters())
        else:
            self.module = getModule(**module)
            if init:
                self.module.apply(Init(init))
            module_params = self.module.parameters()

        self.optimizer = getOptimizer(**optimizer, module_params=module_params)
        self.scheduler = getScheduler(**scheduler, optimizer=self.optimizer)
        
        self.criterion = getLoss(**loss)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input, target, turn):
        assert set(np.unique(target[0])).issubset({0,1})
        input, target = cuda(input, self.device), cuda(target, self.device, True)

        if turn == 'train':
            loss, output = self.train(input, target)
        elif turn == 'val':
            loss, output = self.validate(input, target)

        return loss.mean().cpu().item(), cpu(output)


    def train(self, input, target):
        
        if self.ensemble:
            output = []
            for m in self.module:
                m.train()
                out = m(input)
                out = F.sigmoid(out)
                output.append(out)
            output = torch.stack(output, dim=0)
            output = torch.mean(output, dim=0)
        else:
            self.module.train()
            output = self.module(input)
            output = F.sigmoid(output)

        loss = self.criterion(output, target)
        self.optimizer.zero_grad()   
        loss.sum().backward()         
        self.optimizer.step()

        return loss, output
    
    def validate(self, input, target):

        
        with torch.no_grad():
            if self.ensemble:
                output = []
                for m in self.module:
                    m.eval()
                    out = m(input)
                    output.append(out)
                output = torch.stack(output, dim=0)
                output = torch.mean(output, dim=0)
            else:
                self.module.eval()
                output = self.module(input)
            output = F.sigmoid(output)
        loss = self.criterion(output, target)

        return loss, output

    def test(self, input, target):
        with torch.no_grad():
            if self.ensemble:
                output = []
                for m in self.module:
                    m.eval()
                    out = m(input)
                    output.append(out)
                output = torch.stack(output, dim=0)
                output = torch.mean(output, dim=0)
            else:
                self.module.eval()
                output = self.module(input)
            output = F.sigmoid(output)
        loss = self.criterion(output, target)

        return loss, output

    def step(self, epoch, LOG):
        
        lr = [group['lr'] for group in self.optimizer.param_groups][0]
        LOG('tensorboard', type='scalar', turn='train', 
            name='learning_rate', epoch=epoch, values=lr)

        if self.ensemble:
            for i, m in enumerate(self.module):
                m = m.module if torch.cuda.device_count() > 1 else m
                for tag, value in m.named_parameters():
                    tag = str(i) + '_' + tag.replace('.', '/')
                    LOG('tensorboard', type='histogram', turn='train', 
                        name=tag, epoch=epoch, values=value.data.cpu().numpy())
                    if value.grad is not None:
                        LOG('tensorboard', type='histogram', turn='train', 
                            name=tag+'/grad', epoch=epoch, values=value.grad.cpu().numpy())
        else:  
            module = self.module.module if torch.cuda.device_count() > 1 else self.module
            for tag, value in module.named_parameters():
                tag = tag.replace('.', '/')
                LOG('tensorboard', type='histogram', turn='train', 
                    name=tag, epoch=epoch, values=value.data.cpu().numpy())
                if value.grad is not None:
                    LOG('tensorboard', type='histogram', turn='train', 
                        name=tag+'/grad', epoch=epoch, values=value.grad.cpu().numpy())
        
        self.scheduler.step()

    def modelSummary(self, input_size, LOG):
        module = self.module[0] if self.ensemble else self.module
        
        LOG('model', title=module.__class__.__name__, model=module, input_size=input_size)

    def gpu(self):
        if self.ensemble:
            if torch.cuda.device_count() > 1:
                self.module = [torch.nn.DataParallel(m) for m in self.module]

            if torch.cuda.is_available():
                self.module = [m.to(self.device) for m in self.module]
                self.criterion = self.criterion.to(self.device)
                torch.backends.cudnn.benchmark = True
            else:
                print("CUDA is unavailable")

        else:
            if torch.cuda.device_count() > 1:
                self.module = torch.nn.DataParallel(self.module)

            if torch.cuda.is_available():
                self.module = self.module.to(self.device)
                self.criterion = self.criterion.to(self.device)
                torch.backends.cudnn.benchmark = True
            else:
                print("CUDA is unavailable")

    def cpu(self):
        if self.ensemble:
            if torch.cuda.device_count() > 1:
                self.module = [(m.module) for m in self.module]

            if torch.cuda.is_available():
                self.module = [m.cpu() for m in self.module]
                self.criterion = self.criterion.cpu()

        else:
            if torch.cuda.device_count() > 1:
                self.module = self.module.module

            if torch.cuda.is_available():
                self.module = self.module.cpu()
                self.criterion = self.criterion.cpu()

    def checkpoint(self, epoch, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)
        filename = "{}{}.pth.tar".format(path, epoch)

        state = {
            'epoch': epoch,
        }

        if self.ensemble:
            state_dicts = []
            for m in self.module:
                m = m.module if torch.cuda.device_count() > 1 else m
                state_dicts.append(m.state_dict())
            state['state_dict'] = state_dicts
            state['optimizer'] = self.optimizer.state_dict()

        else:
            module = self.module.module if torch.cuda.device_count() > 1 else self.module
            state['state_dict'] = module.state_dict()
            state['optimizer'] = self.optimizer.state_dict()

        torch.save(state, filename)

    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path)
            epoch = checkpoint['epoch'] + 1
            if self.ensemble:
                for module, state_dict in zip(self.module, checkpoint['state_dict']):
                    module.load_state_dict(state_dict)
            else:
                self.module.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(path, epoch))
        else:
            epoch = 0
            print("=> no checkpoint found at '{}'".format(path))

        return epoch

class GANModel():

    def __init__(self, module, weight_init, optimizer, scheduler, loss, ensemble=False):
        self.G = getGANModule(**module)
        self.D = unet.SimpleClassify(1, 1, module["param"]["unit"])
        if init:
            init_func = Init(init)
            self.G.apply(init_func)
            self.D.apply(init_func)

        self.optimizer_G = torch.optim.Adam(params=self.G.parameters(), lr=0.001)
        self.optimizer_D = torch.optim.SGD(params=self.G.parameters(), lr=0.01)

        self.scheduler_G = getScheduler(**scheduler, optimizer=self.optimizer_G)
        self.scheduler_D = getScheduler(**scheduler, optimizer=self.optimizer_D)

        self.criterion = getLoss(**loss, reduce=False)
        self.ganLoss = L.GANLoss(lsgan=True, reduce=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input, target, turn):
        assert set(np.unique(target[0])).issubset({0,1})
        batch_size = input.size(0)
        real_label = cuda(torch.ones((batch_size)), self.device)
        fake_label = cuda(torch.zeros((batch_size)), self.device)
        input, target = cuda(input, self.device), cuda(target, self.device, True)

        if turn == 'train':
            loss, output = self.train(input, target, fake_label, real_label)
        elif turn == 'val':
            loss, output = self.validate(input, target, fake_label, real_label)

        return loss.mean().cpu().item(), cpu(output)

    def train(self, input, target, fake_label, real_label):
        # D: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        self.G.train()
        self.D.train()
        ## Real
        real_pair = target
        real_pred_major = self.D(real_pair[:,0:1,:,:])
        real_pred_minor = self.D(real_pair[:,1:2,:,:])
        real_D_loss = self.ganLoss(real_pred_major, real_label) + self.ganLoss(real_pred_minor, real_label)
        ## Fake
        output_org = self.G(input)
        output = F.tanh(output_org)
        fake_pair = output.detach()
        fake_pred_major = self.D(fake_pair[:,0:1,:,:])
        fake_pred_minor = self.D(fake_pair[:,1:2,:,:])
        fake_D_loss = self.ganLoss(fake_pred_major, fake_label) + self.ganLoss(fake_pred_minor, fake_label)

        self.optimizer_D.zero_grad()
        loss_D = real_D_loss * 0.5 + fake_D_loss * 0.5
        loss_D.sum().backward()
        self.optimizer_D.step()

        # G: maximize log(D(x,G(x))) + DICE(y,G(x))
        fake_pair = output
        fake_pred_major = self.D(fake_pair[:,0:1,:,:])
        fake_pred_minor = self.D(fake_pair[:,1:2,:,:])
        fake_G_loss = self.ganLoss(fake_pred_major, fake_label) + self.ganLoss(fake_pred_minor, fake_label)

        seg_loss = self.criterion(F.sigmoid(output_org), target) * 10
        self.optimizer_G.zero_grad()
        loss_G = (fake_G_loss + seg_loss) / 2
        loss_G.sum().backward()
        self.optimizer_G.step()

        loss = loss_D + loss_G
        return loss, F.sigmoid(output_org)

    def validate(self, input, target, fake_label, real_label):
        with torch.no_grad():
            self.G.eval()
            output = self.G(input)
            out = F.tanh(output)
            pred_fake_major = self.D(out[:,0:1,:,:])
            pred_fake_minor = self.D(out[:,1:2,:,:])
            pred_real_major = self.D(target[:,0:1,:,:])
            pred_real_minor = self.D(target[:,1:2,:,:])

            D_fake_loss = self.ganLoss(pred_fake_major, fake_label) + self.ganLoss(pred_fake_minor, fake_label)
            D_real_loss = self.ganLoss(pred_real_major, real_label) + self.ganLoss(pred_real_minor, real_label)
            D_loss = (D_fake_loss + D_real_loss) / 8
            G_loss = self.criterion(output, target) /2
            loss = D_loss + G_loss

            pred = torch.cat([pred_major, pred_minor], dim=1)
            output = output + pred
            output = F.sigmoid(output)
            
        return loss, output

    def test(self, input, target):
        batch_size = input.size(0)
        fake_label = torch.zeros(batch_size)
        with torch.no_grad():
            self.G.eval()
            output = self.G(input)
            out = F.tanh(output)
            pred_major = self.D(out[:,0:1,:,:])
            pred_minor = self.D(out[:,1:2,:,:])

            D_fake_loss = self.ganLoss(pred_major, fake_label) + self.ganLoss(pred_minor, fake_label)
            D_real_loss = self.ganLoss(target[:,0:1,:,:], fake_label) + self.ganLoss(target[:,1:2,:,:], fake_label)
            D_loss = (D_fake_loss + D_real_loss) / 8
            G_loss = self.criterion(output, target) /2
            loss = D_loss + G_loss

            pred = torch.cat([pred_major, pred_minor], dim=1)
            output = output + pred
            output = F.sigmoid(output)
            
        return loss, output

    def step(self, epoch, LOG, histo=True):
        lr_G = [group['lr'] for group in self.optimizer_G.param_groups][0]
        LOG('tensorboard', type='scalar', turn='train', 
            name='learning_rate/generator', epoch=epoch, values=lr_G)
        lr_D = [group['lr'] for group in self.optimizer_D.param_groups][0]
        LOG('tensorboard', type='scalar', turn='train', 
            name='learning_rate/discriminator', epoch=epoch, values=lr_D)

        # torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1)
        # torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1)
        clip = 1
        if histo:
            G = self.G.module if torch.cuda.device_count() > 1 else self.G
            D = self.D.module if torch.cuda.device_count() > 1 else self.D
            for tag, value in G.named_parameters():
                tag = tag.replace('.', '/')
                # value.grad.data.clamp_(-clip,clip)
                LOG('tensorboard', type='histogram', turn='train', 
                    name='G/'+tag, epoch=epoch, values=value.data.cpu().numpy())
                LOG('tensorboard', type='histogram', turn='train', 
                    name='G/'+tag+'/grad', epoch=epoch, values=value.grad.data.cpu().numpy())
            for tag, value in D.named_parameters():
                tag = tag.replace('.', '/')
                # value.grad.data.clamp_(-clip,clip)
                LOG('tensorboard', type='histogram', turn='train', 
                    name='D/'+tag, epoch=epoch, values=value.data.cpu().numpy())
                LOG('tensorboard', type='histogram', turn='train', 
                    name='D/'+tag+'/grad', epoch=epoch, values=value.grad.cpu().numpy())

        self.scheduler_D.step()
        self.scheduler_G.step()
    
    def modelSummary(self, input_size, LOG):
        c, h, w = input_size
        G_size = (2, h, w)
        D_size = (4, h, w)
        LOG('model', title='Generator/'+self.G.__class__.__name__, model=self.G, input_size=G_size)
        # LOG('model', title='Discriminator/'+self.D.__class__.__name__, model=self.D, input_size=D_size)

    def gpu(self):
        if torch.cuda.device_count() > 1:
            self.G = torch.nn.DataParallel(self.G)
            self.D = torch.nn.DataParallel(self.D)

        if torch.cuda.is_available():
            self.G = self.G.to(self.device)
            self.D = self.D.to(self.device)
            self.criterion = self.criterion.to(self.device)
            self.ganLoss = self.ganLoss.to(self.device)
            torch.backends.cudnn.benchmark = True
        else:
            print("CUDA is unavailable")

    def cpu(self):
        if torch.cuda.device_count() > 1:
            self.G = self.G.module
            self.D = self.D.module

        if torch.cuda.is_available():
            self.G = self.G.cpu()
            self.D = self.D.cpu()
            self.criterion = self.criterion.cpu()
            self.ganLoss = self.ganLoss.cpu()

    def checkpoint(self, epoch, path):
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        filename = "{}{}.pth.tar".format(path, epoch)

        state = {
            'epoch': epoch,
        }
        state_dicts = []
        G = self.G.module if torch.cuda.device_count() > 1 else self.G
        D = self.D.module if torch.cuda.device_count() > 1 else self.D

        state['optim_G'] = self.optimizer_G.state_dict()
        state['optim_D'] = self.optimizer_D.state_dict()

        state['G'] = G.state_dict()
        state['D'] = D.state_dict()

        torch.save(state, filename)

    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path)
            epoch = checkpoint['epoch'] + 1
            self.G.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})".format(path, epoch))
        else:
            epoch = 0
            print("=> no checkpoint found at '{}'".format(path))

        return 0

class RecurModel():
    
    def __init__(self, module, weight_init, optimizer, scheduler, loss, ensemble=False):
        self.ensemble = ensemble
        self.loop = 4
        self.module = getModule(**module)
        if init:
            self.module.apply(Init(init))
        module_params = self.module.parameters()
        self.optimizer = getOptimizer(**optimizer, module_params=module_params)
        self.scheduler = getScheduler(**scheduler, optimizer=self.optimizer)
        
        self.criterion = getLoss(**loss)

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
        
        for i in range(self.loop):
            self.module.train()
            output = self.module(input)
            output = F.sigmoid(output)

        loss = self.criterion(output, target)
        self.optimizer.zero_grad()   
        loss.backward()         
        self.optimizer.step()

        return loss, output
    
    def validate(self, input, target):
        
        with torch.no_grad():
            self.module.eval()
            output = self.module(input)
            output = F.sigmoid(output)
        loss = self.criterion(output, target)

        return loss, output

    def test(self, input, target):
        input, target = cuda(input, self.device), cuda(target, self.device, True)
        with torch.no_grad():
            self.module.eval()
            output = self.module(input)
            output = F.sigmoid(output)
        loss = self.criterion(output, target)

        return loss, output

    def step(self, epoch, LOG):
        
        lr = [group['lr'] for group in self.optimizer.param_groups][0]
        LOG('tensorboard', type='scalar', turn='train', 
            name='learning_rate', epoch=epoch, values=lr)

        if self.ensemble:
            for i, m in enumerate(self.module):
                m = m.module if torch.cuda.device_count() > 1 else m
                for tag, value in m.named_parameters():
                    tag = str(i) + '_' + tag.replace('.', '/')
                    LOG('tensorboard', type='histogram', turn='train', 
                        name=tag, epoch=epoch, values=value.data.cpu().numpy())
                    LOG('tensorboard', type='histogram', turn='train', 
                        name=tag+'/grad', epoch=epoch, values=value.grad.cpu().numpy())
        else:  
            module = self.module.module if torch.cuda.device_count() > 1 else self.module
            for tag, value in module.named_parameters():
                tag = tag.replace('.', '/')
                LOG('tensorboard', type='histogram', turn='train', 
                    name=tag, epoch=epoch, values=value.data.cpu().numpy())
                LOG('tensorboard', type='histogram', turn='train', 
                    name=tag+'/grad', epoch=epoch, values=value.grad.cpu().numpy())
        
        self.scheduler.step()

    def modelSummary(self, input_size, LOG):
        module = self.module[0] if self.ensemble else self.module
        
        LOG('model', title=module.__class__.__name__, model=module, input_size=input_size)

    def gpu(self):
        if self.ensemble:
            if torch.cuda.device_count() > 1:
                self.module = [torch.nn.DataParallel(m) for m in self.module]

            if torch.cuda.is_available():
                self.module = [m.to(self.device) for m in self.module]
                self.criterion = self.criterion.to(self.device)
                torch.backends.cudnn.benchmark = True
            else:
                print("CUDA is unavailable")

        else:
            if torch.cuda.device_count() > 1:
                self.module = torch.nn.DataParallel(self.module)

            if torch.cuda.is_available():
                self.module = self.module.to(self.device)
                self.criterion = self.criterion.to(self.device)
                torch.backends.cudnn.benchmark = True
            else:
                print("CUDA is unavailable")

    def checkpoint(self, epoch, path):

        os.makedirs(os.path.dirname(path), exist_ok=True)
        filename = "{}{}.pth.tar".format(path, epoch)

        state = {
            'epoch': epoch,
        }

        if self.ensemble:
            state_dicts = []
            for m in self.module:
                m = m.module if torch.cuda.device_count() > 1 else m
                state_dicts.append(m.state_dict())
            state['state_dict'] = state_dicts
            state['optimizer'] = self.optimizer.state_dict()

        else:
            module = self.module.module if torch.cuda.device_count() > 1 else self.module
            state['state_dict'] = module.state_dict()
            state['optimizer'] = self.optimizer.state_dict()

        torch.save(state, filename)

    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))

            checkpoint = torch.load(path)
            epoch = checkpoint['epoch'] + 1
            if self.ensemble:
                for module, state_dict in zip(self.module, checkpoint['state_dict']):
                    module.load_state_dict(state_dict)
            else:
                self.module.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(path, epoch))
        else:
            epoch = 0
            print("=> no checkpoint found at '{}'".format(path))

        return epoch