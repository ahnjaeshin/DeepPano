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
    return os.path.join(os.path.dirname(__file__), filedir)

class Patch(NamedTuple):
    pano_path: str
    box_path: str
    input_path: str
    target_path: str

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
                input_path = getAbsoluteAddress(row['Cropped.Input.Img'])
                target_path = getAbsoluteAddress(row['Target.Img'])

                data.append(Patch(pano_path, box_path, input_path, target_path))
        
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
        target_segmentation = Image.open(patch.target_path)
        filepath = patch.input_path

        target_segmentation = target_segmentation.point(lambda p: 255 if p > 50 else 0 )
        assert set(np.unique(target_segmentation)).issubset({0,255})

        if self.transform is not None and doTransform:
            input_pano, input_box, target_segmentation = self.transform(input_pano, input_box, target_segmentation)
        input = torch.cat([input_box, input_pano], dim=0)
        assert set(np.unique(target_segmentation)).issubset({0,1})

        target_classification = (target_segmentation.sum() != 0).float()
        
        return (input, (target_segmentation, target_classification), filepath, index)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return 'Dataset: {} \n '.format(self.__class__.__name__, )

if __name__ == '__main__':
    print(__doc__)

