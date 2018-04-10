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
from torch.utils.data import Dataset, DataLoader

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def isImageFile(filename):
    """Check if a file is an slide
    
    Arguments:
        filename {string} -- path or file name 
    """
    return filename.lower().endswith(IMG_EXTENSIONS)

class Patch(NamedTuple):
    filename: str
    input_path: str
    segment_path: str
    target_path: str

class PanoSet(Dataset):
    """
    extended Dataset class for pytorch
    
    Arguments:
        data_path {str} -- path of dataset
        meta_data_path {str} -- path of metadata
        filter_func {[type]} -- [description]
        strategy {callable} -- [description]
        transform {[type]} -- [description]
        target_transform {[type]} -- [description]
    """

    def __init__(self, data_path, transform = None, target_transform = None):

        data_list = [(file, os.path.join(root, file)) 
                        for root, _, files in os.walk(data_path) 
                            for file in files 
                                if isImageFile(file)]
        assert len(data_list) % 3 == 0
        data_list.sort()

        # to be changed to reading metadata
        data = []
        for i in range(len(data_list) // 3):
            filename = data_list[3 * i][0]
            input_path = data_list[3 * i][1]
            segment_path = data_list[3 * i + 1][1]
            target_path = data_list[3 * i + 2][1]
            data.append(Patch(filename, input_path, segment_path, target_path))
        
        self.data = data
        self.meta_data = None
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index, doTransform=True):
        """
        
        Arguments:
            index {int} -- index of slide
        Returns:
            tuple: (image, target, path, index)
                target is class_index
        """
        patch = self.data[index]

        img = Image.merge("LA", (Image.open(patch.input_path), Image.open(patch.segment_path)))

        target = Image.open(patch.target_path)
        filename = patch.filename

        if self.transform is not None and doTransform:
            img = self.transform(img)
        if self.target_transform is not None and doTransform:
            target = self.target_transform(target)
        
        return (img, target, filename, index)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return 'Dataset: {} \n '.format(self.__class__.__name__, )

if __name__ == '__main__':
    print(__doc__)