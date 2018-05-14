
import random
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch

def resetSeed(seed):
    """seed: seed value of random packages"""
    random.seed(seed)

class Augment():

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input_pano, input_box, target_major, target_minor):
        raise NotImplementedError


class TripleAugment(Augment):
    
    def __init__(self, transform, size, box_mean, box_std, pano_mean, pano_std):
        super(TripleAugment, self).__init__(transform)

        self.transform += [
            ToAll(T.Resize(size)),
            TargetOnly(Threshold()),
            ToAll(T.ToTensor()),
            PanoOnly(T.Normalize((pano_mean, ), (pano_std, ))),
            BoxOnly(T.Normalize((box_mean, ), (box_std, ))),
        ]
    
    def __call__(self, input_pano, input_box, target_major, target_minor):
        for t in self.transform:
            input_pano, input_box, target_major, target_minor = t(input_pano, input_box, target_major, target_minor)

        return input_pano, input_box, target_major, target_minor


class ToAll(Augment):
    
    def __call__(self, input_pano, input_box, target_major, target_minor, seed=None):
        if not seed:
            seed = random.randint(0,2**32)

        resetSeed(seed)
        input_pano = self.transform(input_pano)
        resetSeed(seed)
        input_box = self.transform(input_box)
        resetSeed(seed)
        target_major = self.transform(target_major)
        resetSeed(seed)
        target_minor = self.transform(target_minor)
        
        return input_pano, input_box, target_major, target_minor


class InputOnly(Augment):

    def __call__(self, input_pano, input_box, target_major, target_minor, seed=None):
        if not seed:
            seed = random.randint(0,2**32)

        resetSeed(seed)
        input_pano = self.transform(input_pano)
        resetSeed(seed)
        input_box = self.transform(input_box)
        return input_pano, input_box, target_major, target_minor

class TargetOnly(Augment):
    
    def __call__(self, input_pano, input_box, target_major, target_minor, seed=None):
        if not seed:
            seed = random.randint(0,2**32)

        resetSeed(seed)
        target_major = self.transform(target_major)
        resetSeed(seed)
        target_minor = self.transform(target_minor)

        return input_pano, input_box, target_major, target_minor


class PanoOnly(Augment):
    
    def __call__(self, input_pano, input_box, target_major, target_minor):
        input_pano = self.transform(input_pano)
        return input_pano, input_box, target_major, target_minor


class BoxOnly(Augment):
    
    def __call__(self, input_pano, input_box, target_major, target_minor):
        input_box = self.transform(input_box)
        return input_pano, input_box, target_major, target_minor


##############


class Threshold():

    def __init__(self, threshhold = 50):
        self.threshhold = threshhold

    def __call__(self, img):
        img = img.point(lambda p: 255 if p > self.threshhold else 0)
        return img
