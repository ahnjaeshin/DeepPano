
import random
import numpy as np

class DualAugment:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask

class ToBoth:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x, target=None):
        seed = random.randint(0,2**32)
        random.seed(seed)
        np.random.seed(seed)
        x = self.transform(x)
        random.seed(seed)
        np.random.seed(seed)
        target = self.transform(target)
        return x, target

class TargetOnly:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x, target=None):
        return x, self.transform(target)

class ImageOnly:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x, mask=None):
        return self.transform(x), mask