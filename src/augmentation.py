
import random
import numpy as np
import torchvision.transforms.functional as F
import torch

def resetSeed(seed):
    """seed: seed value of random packages"""
    random.seed(seed)

class Augment():

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input_pano, input_box, target):
        raise NotImplementedError


class TripleAugment(Augment):
    
    def __call__(self, input_pano, input_box, target):
        for t in self.transform:
            input_pano, input_box, target = t(input_pano, input_box, target)

        return input_pano, input_box, target


class ToAll(Augment):
    
    def __call__(self, input_pano, input_box, target, seed=None):
        if not seed:
            seed = random.randint(0,2**32)

        resetSeed(seed)
        input_pano = self.transform(input_pano)
        resetSeed(seed)
        input_box = self.transform(input_box)
        resetSeed(seed)
        target = self.transform(target)
        return input_pano, input_box, target


class InputOnly(Augment):

    def __call__(self, input_pano, input_box, target, seed=None):
        if not seed:
            seed = random.randint(0,2**32)

        resetSeed(seed)
        input_pano = self.transform(input_pano)
        resetSeed(seed)
        input_box = self.transform(input_box)
        return input_pano, input_box, target


class PanoOnly(Augment):
    
    def __call__(self, input_pano, input_box, target):
        input_pano = self.transform(input_pano)
        return input_pano, input_box, target


class BoxOnly(Augment):
    
    def __call__(self, input_pano, input_box, target):
        input_box = self.transform(input_box)
        return input_pano, input_box, target


class TargetOnly(Augment):
    
    def __call__(self, input_pano, input_box, target):
        target = self.transform(target)
        return input_pano, input_box, target

class Threshhold():

    def __init__(self, threshhold = 50):
        self.threshhold = threshhold

    def __call__(self, img):
        img = img.point(lambda p: 255 if p > self.threshhold else 0)
        return img