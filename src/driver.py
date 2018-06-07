"""
timer interrupt
"""


import argparse
import datetime
import json
import socket
import random
import numpy as np
import torch
import torch.nn as nn
import signal
from tensorboardX import SummaryWriter
import traceback
import datetime
import loss
import metric
from dataset import PanoSet, PretrainPanoSet
from model import *
from torchvision import transforms
from trainer import Trainer
import augmentation as AUG
from utils import count_parameters, slack_message, model_summary
import torch.multiprocessing as mp
from typing import NamedTuple
import pandas as pd

import imgaug as ia

from inference import Inference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", "-c", type=str, required=True, help="path to config file")
parser.add_argument("--title", "-t", type=str, help="title of experiment")

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

class TypeParser:
    
    def __init__(self, table):
        self.table = table

    def __call__(self, type, param=None):
        return self.lookup(type)() if param is None else self.lookup(type)(**param)

    def lookup(self, type):
        if type not in self.table:
            print("table: {}".format(self.table))
            raise NotImplementedError
        return self.table[type]

class Data(NamedTuple):
    metadata_path: str
    pano_mean: float
    pano_std: float
    box_mean: float
    box_std : float

def main(config, title):
    
    ##################
    #     logging    #
    ##################
    config_logging = config["logging"]
    config_logging_start_time = config_logging["start_time"]
    config_logging_title = config_logging["title"]
    config_logging_trial = config_logging["trial"]

    if config_logging["reproducible"]:
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

    if title is None:
        title = config_logging_title

    if config_logging_start_time == '':
        config_logging_start_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    log_dir = '../result/runs/{title}/{name}/{time}_{trial}/'.format(title=title, 
                                                            name=config["dataset"]["name"],
                                                           time=config_logging_start_time,
                                                           trial=config_logging_trial)

    writers = {x : SummaryWriter(log_dir=log_dir + x) for x in ('train', 'val')}
    
    # need better way
    LOG = slack_message(title)
    LOG('config', json.dumps(config))
    
    ##################
    #     dataset    #
    ##################
    config_dataset = config["dataset"]
    config_data_path = config_dataset["data-dir"]
    config_data_name = config_dataset["name"]
    config_dataset_filter = config_dataset["filter"]
    config_dataset_pretrain = config_dataset["pretrain"]

    df = pd.read_csv(config_data_path)
    data = df.loc[df['DataSet.Title'] == config_data_name].iloc[0]
    data = Data(data['Csv.File'], data['Pano.Mean'], data['Pano.Stdev'], data['Box.Mean'], data['Box.Stdev'])    

    data_filter = { x : Filter(config_dataset_filter[x]) for x in ('train', 'val')}

    ##################
    #  augmentation  #
    ##################
    config_augmentation = config["augmentation"]
    augmentation_param = {
        'size': config_augmentation['size'],
        'box_mean': float(data.box_mean), 'pano_mean': float(data.pano_mean),
        'box_std': float(data.box_std), 'pano_std': float(data.pano_std),
    }

    print(augmentation_param)

    
    dataset = PretrainPanoSet if config_dataset_pretrain else PanoSet

    augmentations = {x : getAugmentation(config_augmentation[x], augmentation_param) for x in ('train', 'val')}    

    datasets = { x: dataset(data.metadata_path, data_filter[x], transform=augmentations[x])
                    for x in ('train', 'val')}

    LOG('dataset', str(datasets['train']), str(datasets['val']))

    ##################
    #      model     #
    ##################
    config_model = config["model"]
    config_model_type = config_model["type"]

    MODEL = {
        'UNET': unet.UNet, 'WNet': unet.WNet,
        'FCDenseNetSmall': tiramisu.FCDenseNetSmall,
        'FCDenseNet57': tiramisu.FCDenseNet57,
        'FCDenseNet67': tiramisu.FCDenseNet67,
        'FCDenseNet103': tiramisu.FCDenseNet103,
        'RecurNet': unet.RecurNet,
        'RecurNet2': unet.RecurNet2,
    }[config_model_type]

    model = MODEL(2, 2, **config_model["param"])
    dummy_input = torch.rand(2, 2, *config_augmentation['size'])
    writers['train'].add_graph(model, (dummy_input, ))

    model_sum, trainable_param = model_summary(model, input_size=(2, 224, 224))
    writers['train'].add_scalar('number of parameter/ver1', count_parameters(model))
    writers['train'].add_scalar('number of parameter/ver2', trainable_param)
    LOG('model', model.__repr__())
    LOG('model summary', *model_sum)
    print(model_sum)
    print(model.__repr__())

    ##################
    #   evaluation   #
    ##################
    config_evaluation = config["evaluation"]
    # IOU, DICE, F1
    config_metrics = config_evaluation["metrics"]
    # IOU, DICE
    config_loss = config_evaluation["loss"]

    metricParser = TypeParser(table = {
        "IOU": metric.IOU, 
        "DICE": metric.DICE,
        "accuracy": metric.Accuracy,
        "f1": metric.F1,
    })
    metrics = [metricParser(**m) for m in config_metrics]

    lossParser = TypeParser(table = {
        "IOU": loss.IOULoss,
        "DICE": loss.DICELoss,
        "CE": nn.CrossEntropyLoss,
    })
    criterion = lossParser(**config_loss)

    ##################
    #    learning    #
    ##################
    config_learning = config["learning"]
    
    # xavier_uniform, xavier_normal, he_uniform, he_normal
    config_learning_weightinit = config_learning["weight_init"]
    # give path to load checkpoint or null
    config_learning_checkpoint = config_learning["checkpoint"]

    # optimizer, default : 0.9, False, 1-e4
    config_optimizer = config_learning["optimizer"]
    config_optimizer["param"]["params"] = model.parameters()
    optimParser = TypeParser(table = {
        "SGD": torch.optim.SGD,
        "ADAM": torch.optim.Adam,
    })
    optimizer = optimParser(**config_optimizer)

    config_learning_scheduler = config_learning["lr_schedule"]
    config_learning_scheduler["param"]["optimizer"] = optimizer
    schedParser = TypeParser(table = {
        "Step": torch.optim.lr_scheduler.StepLR,
        "Plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    })
    scheduler = schedParser(**config_learning_scheduler)

    ##################
    #    training    #
    ##################
    trainer = Trainer(model=model, 
                        datasets=datasets, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        metrics=metrics, 
                        writers=writers,
                        LOG=LOG,
                        path=log_dir + 'train',
                        checkpoint=config_learning_checkpoint,
                        init=config_learning_weightinit)

    try: 
        trainer.train(**config["training"])
    except Exception as e:
        LOG('warning', 'abrupt end, {}'.format(e))
        print('abrupt end, {}'.format(e))
        print(traceback.format_exc())

    infer = Inference(
        model=model,
        datasets=datasets,
        criterion=criterion,
        LOG=LOG,
        metrics=metrics,
        path=log_dir,
        visualizations=None,
        writers=writers,
    )

    infer(
        
    )

    writers['train'].close()
    writers['val'].close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(json.load(open(args.config)), args.title)
