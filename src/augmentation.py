
import random
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torch
from PIL import Image
from imgaug import augmenters as iaa
import imgaug as ia

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

        self.transform = [TargetOnly(Threshold())] + self.transform
    
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

class PanoTarget(Augment):
    
    def __call__(self, input_pano, input_box, target_major, target_minor):
        input_pano, target_major, target_minor = self.transform(input_pano, target_major, target_minor)
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

class Cutout():

    def __init__(self, size=12, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, img, target_major, target_minor):
        if random.random() > self.p:
            return img, target_major, target_minor
        img = np.array(img)
        t_major_guide = np.array(target_major)
        t_minor_guide = np.array(target_minor)
        h, w = img.shape
        pad = (self.size // 2)

        for i in range(1000):
            x, y = random.randrange(pad, h - pad), random.randrange(pad, w - pad)
            roi_major = t_major_guide[x-pad:x+pad, y-pad:y+pad]
            roi_minor = t_minor_guide[x-pad:x+pad, y-pad:y+pad]
            if np.all(roi_major != 0) or np.all(roi_minor != 0):
                img[x-pad:x+pad, y-pad:y+pad] = 0
                break

        img = Image.fromarray(img)
        return img, target_major, target_minor
class RandomAug:
    def __init__(self):
        # from imgaug example
        self.aug = iaa.Sequential([
            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.Sometimes(0.5, [
                iaa.SomeOf((0, 3),
                [
                    ##########################
                    # 1.     BLUR            #
                    ##########################
                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 5.0)),
                        iaa.AverageBlur(k=(2, 9)),
                    ]),

                    ##########################
                    # 2~3    emboss, sharpen #
                    ##########################
                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    ##########################
                    # 4    dropout           #
                    ##########################
                    iaa.Dropout((0.01, 0.1)),

                    ##########################
                    # 5~7 light                #
                    ##########################
                    # Add a value of -10 to 10 to each pixel.
                    # iaa.Add((-10, 10)),
                    # # # Change brightness of images (50-150% of original value).
                    # iaa.Multiply((0.5, 1.5)),
                    # Improve or worsen the contrast of images.
                    iaa.ContrastNormalization((0.5, 2.0)),

                    ##########################
                    # 8    salt & pepper     #
                    ##########################
                    iaa.SaltAndPepper(p=0.3),

                ], random_order=True
                ),
            ])
            
        ])

    def __call__(self, img):
        img = np.array(img)
        img = np.expand_dims(img, axis=2)
        img = self.aug.augment_image(img)
        assert (img.shape[2] == 1)
        img = img[:, :, 0]
        img = Image.fromarray(img)
        return img

class RandomAffine:
    pass