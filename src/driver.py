
import argparse
import json

import random
import numpy as np
import torch
import torch.nn as nn
import signal
import traceback
import metric as M
from dataset import PanoSet, PretrainPanoSet
from model import getModel
from torchvision import transforms
from trainer import Trainer
import augmentation as AUG
from utils import Logger, TypeParser
from typing import NamedTuple
import pandas as pd

import imgaug as ia

from inference import Inference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", "-c", type=str, required=True, help="path to config file")

FRONT = (11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43)
writers = {}

class Filter():
    
    def __init__(self, filters):
        lookup = {
            "train" : lambda row: row['Train.Val'] == 'train' and int(row['Segmentable.Type']) > 0,
            "val" : lambda row: row['Train.Val'] == 'val' and int(row['Segmentable.Type']) > 0,
            "easy": lambda row: int(row['Segmentable.Type']) <= 10,
            "segmentable": lambda row: int(row['Segmentable.Type']) < 10,
            "unsegmentable": lambda row: int(row['Segmentable.Type']) >= 10,
            "front-teeth": lambda row: int(row['Tooth.Num.Annot']) in FRONT,
        }
        self.func = []
        self.func += [lookup[f] for f in filters]

    def __call__(self, row):
        for f in self.func:
            if not f(row):
                return False
        return True

def getAugmentation(augments, param):
    
    def lookupAugment(category, type, param=None):
        types = {
            "HFlip": transforms.RandomHorizontalFlip,
            "VFlip": transforms.RandomVerticalFlip,
            "Rotate": transforms.RandomRotation,
            "Crop": transforms.RandomCrop,
            "Cutout": AUG.Cutout,
            "Random": AUG.RandomAug
        }

        categories = {
            "All": AUG.ToAll,
            "Input": AUG.InputOnly,
            "Target": AUG.TargetOnly,
            "Pano": AUG.PanoOnly,
            "Box": AUG.BoxOnly,
            "PanoTarget": AUG.PanoTarget,
        }

        aug_func = types[type]() if param is None else types[type](**param)
        return categories[category](aug_func)

    return AUG.TripleAugment([lookupAugment(**a) for a in augments], **param)


class Data(NamedTuple):
    metadata_path: str
    pano_mean: float
    pano_std: float
    box_mean: float
    box_std : float

def main(experiment, logging, augmentation, dataset, model, metric, training):

    if experiment["reproducible"]:
        print('fix seed on')
        seed = 0
        random.seed(seed) # augmentation
        np.random.seed(seed) # numpy
        ia.seed(seed) #  imgaug library
        torch.manual_seed(seed) # cpu
        torch.cuda.manual_seed(seed) # gpu
        torch.cuda.manual_seed_all(seed) # multi gpu
        torch.backends.cudnn.enabled = False # cudnn library 
        torch.backends.cudnn.deterministic = True

    ##################
    #     logging    #
    ##################
    LOG = Logger(**logging)    
    LOG('print', name='config', values=json.dumps(config))
    
    ##################
    #     dataset    #
    ##################
    config_data_path = dataset["data-dir"]
    config_data_name = dataset["name"]
    config_dataset_filter = dataset["filter"]
    config_dataset_pretrain = dataset["pretrain"]

    df = pd.read_csv(config_data_path)
    data = df.loc[df['DataSet.Title'] == config_data_name].iloc[0]
    data = Data(data['Csv.File'], data['Pano.Mean'], data['Pano.Stdev'], data['Box.Mean'], data['Box.Stdev'])    

    data_filter = { x : Filter(config_dataset_filter[x]) for x in ('train', 'val')}

    ##################
    #  augmentation  #
    ##################

    augmentation_param = {
        'size': augmentation['size'],
        'box_mean': float(data.box_mean), 'pano_mean': float(data.pano_mean),
        'box_std': float(data.box_std), 'pano_std': float(data.pano_std),
    }

    LOG('print', name='augmentation', values=augmentation_param)
    LOG('slack', name='augmentation', values=augmentation_param)

    
    D = PretrainPanoSet if config_dataset_pretrain else PanoSet

    augmentations = {x : getAugmentation(augmentation[x], augmentation_param) for x in ('train', 'val')}    

    datasets = { x: D(data.metadata_path, data_filter[x], transform=augmentations[x])
                    for x in ('train', 'val')}

    LOG('slack', name='dataset', values=[str(datasets['train']), str(datasets['val'])])

    ##################
    #      model     #
    ##################

    MODEL = getModel(**model)
    input_size = [augmentation['channel']] + augmentation['size']
    MODEL.modelSummary(input_size, LOG)

    ##################
    #   metric   #
    ##################

    metricParser = TypeParser(types = {
        "IOU": M.IOU, 
        "DICE": M.DICE,
        "accuracy": M.Accuracy,
        "f1": M.F1,
    })
    metrics = [metricParser(**m) for m in metric]

    ##################
    #    training    #
    ##################
    trainer = Trainer(model=MODEL, 
                        datasets=datasets, 
                        metrics=metrics, 
                        LOG=LOG)

    try: 
        trainer.train(**config["training"])
    except Exception as e:
        LOG('slack', name='warning', values='abrupt end, {}'.format(e))
        print('abrupt end, {}'.format(e))
        print(traceback.format_exc())

    exit(0)

    infer = Inference(
        model=model,
        datasets=datasets,
        LOG=LOG,
        metrics=metrics,
        visualizations=None,
        writers=writers,
    )

    infer(
        
    )

    LOG.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(**config)
