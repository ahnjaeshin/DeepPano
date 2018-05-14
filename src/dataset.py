"""

open, load, augment dataset

do multiprocessing
bottleneck이 되지 않도록 시간 재기
GPU utilization
"""

import os
import pickle
import torch
import numpy as np
from PIL import Image
from typing import NamedTuple
from torch.utils.data import Dataset
import pandas as pd

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def isImageFile(filename):
    """Check if a file is an slide
    
    Arguments:
        filename {string} -- path or file name 
    """
    return filename.lower().endswith(IMG_EXTENSIONS)


def getAbsoluteAddress(filedir):
    if filedir == '0':
        return None
    return os.path.join(os.path.dirname(__file__), filedir)
    

def createImage(size):
    """create blank PIL image of size
    
    Arguments:
        size {tuple} -- size of image
    """
    img = np.zeros(size)
    img = Image.fromarray(img, mode='L')
    return img


class Patch(NamedTuple):
    pano_path: str
    box_path: str
    major_input_path: str
    minor_input_path: str
    major_target_path: str
    minor_target_path: str
    size: tuple

class PanoSet(Dataset):
    """
    extended Dataset class for pytorch
    
    """

    def __init__(self, meta_data_path, filter_func, transform = None):
        
        self.path = meta_data_path
        df = pd.read_csv(self.path)
        data = []

        for idx, row in df.iterrows():
            if filter_func(row):
                pano_path = getAbsoluteAddress(row['Cropped.Pano.Img'])
                box_path = getAbsoluteAddress(row['Cropped.Box.Img'])
                major_input_path = getAbsoluteAddress(row['Cropped.Major.Annot.Img'])
                minor_input_path = getAbsoluteAddress(row['Cropped.Minor.Annot.Img'])
                major_target_path = getAbsoluteAddress(row['Major.Target.Img'])
                minor_target_path = getAbsoluteAddress(row['Minor.Target.Img'])
                size = eval(row['Cropped.Img.Size'])

                data.append(Patch(
                    pano_path, box_path,  
                    major_input_path,  minor_input_path,
                    major_target_path, minor_target_path,
                    size))
        
        self.data = data
        self.meta_data = df
        self.transform = transform

    def __getitem__(self, index, doTransform=True):
        """
        
        Arguments:
            index {int} -- index of slide
        Returns:
            tuple: (image, target, path, index)
                target is class_index
        """
        patch = self.data[index]

        input_pano = Image.open(patch.pano_path)
        input_box = Image.open(patch.box_path)

        if patch.major_target_path is None:
            target_major = createImage(patch.size)
        else:
            target_major = Image.open(patch.major_target_path)

        if patch.minor_target_path is None:
            target_minor = createImage(patch.size)
        else:
            target_minor = Image.open(patch.minor_target_path)

        target_major = target_major.point(lambda p: 255 if p > 50 else 0)
        target_minor = target_minor.point(lambda p: 255 if p > 50 else 0)

        target = Image.merge("LA",(target_major,target_minor))

        if self.transform is not None and doTransform:
            input_pano, input_box, target = self.transform(input_pano, input_box, target)
        input = torch.cat([input_box, input_pano], dim=0)

        assert set(np.unique(target)).issubset({0,1})
      
        return (input, target, index)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return 'Dataset: {} [size: {}]'.format(self.__class__.__name__, len(self.data))

if __name__ == '__main__':
    print(__doc__)

