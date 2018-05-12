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
from tensorboardX import SummaryWriter

import loss
import metric
from dataset import PanoSet
from model import UNet
from torchvision import transforms
from trainer import Trainer
import augmentation as AUG
from utils import count_parameters, slack_message, model_summary
import torch.multiprocessing as mp
from typing import NamedTuple
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", "-c", type=str, required=True, help="path to config file")

FRONT = (11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43)

class Filter():
    
    def __init__(self, filters):
        lookup = {
            "train" : lambda row: row['Train.Val'] == 'train',
            "val" : lambda row: row['Train.Val'] == 'val',
            "all": lambda row: int(row['Segmentable.Type']) <= 10,
            "segmentable": lambda row: int(row['Segmentable.Type']) < 10,
            "unsegmentable": lambda row: int(row['Segmentable.Type']) == 10,
            "front-teeth": lambda row: int(row['Tooth.Num.Annot']) in FRONT,
        }
        self.func = [lambda row: row['Target.Img'] != '-1']
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
        }

        categories = {
            "All": AUG.ToAll,
            "Input": AUG.InputOnly,
            "Target": AUG.TargetOnly,
            "Pano": AUG.PanoOnly,
            "Box": AUG.BoxOnly,
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

def main(config):
    
    ##################
    #     logging    #
    ##################
    config_logging = config["logging"]
    config_logging_debug = config_logging["debug"]
    config_logging_channel = config_logging["slack_channel"]
    # write title in hierarchical way e.g. UNET/HE/SGD
    config_logging_title = config_logging["title"]
    config_logging_trial = config_logging["trial"]

    if config_logging["reproducible"]:
        print('fix seed on')
        seed = 0
        random.seed(seed) # augmentation
        np.random.seed(seed) # numpy
        torch.manual_seed(seed) # cpu
        torch.cuda.manual_seed(seed) # gpu
        torch.cuda.manual_seed_all(seed) # multi gpu
        torch.backends.cudnn.enabled = False # cudnn library 
        torch.backends.cudnn.deterministic = True

    log_dir = 'runs/{title}/{time}_{host}_{trial}/'.format(title=config_logging_title, 
                                                           time=datetime.datetime.now().strftime('%b%d_%H-%M'), 
                                                           host=socket.gethostname(), 
                                                           trial=config_logging_trial)
    writers = {x : SummaryWriter(log_dir=log_dir + x) for x in ('train', 'val')}
    
    # need better way
    log = []
    slack_message(json.dumps(config), config_logging_channel)
    
    ##################
    #     dataset    #
    ##################
    config_dataset = config["dataset"]
    config_data_path = config_dataset["data-dir"]
    config_data_name = config_dataset["name"]
    config_dataset_filter = config_dataset["filter"]

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
        'box_mean': data.box_mean, 'pano_mean': data.pano_mean,
        'box_std': data.box_std, 'pano_std': data.pano_std,
    }
    augmentations = {x : getAugmentation(config_augmentation[x], augmentation_param) for x in ('train', 'val')}    

    datasets = { x: PanoSet(data.metadata_path, data_filter[x], transform=augmentations[x])
                    for x in ('train', 'val')}

    log.append(str(datasets['train']))
    log.append(str(datasets['val']))

    ##################
    #      model     #
    ##################
    config_model = config["model"]

    model = UNet(2, 1, **config_model["param"])
    # dummy_input = torch.rand(1, 2, 128, 128)
    # writers['train'].add_graph(model, (dummy_input, ))
    # torch.onnx.export(model, dummy_input, "graph.proto", verbose=True)
    # writers['train'].add_graph_onnx("graph.proto")
    model_sum = model_summary(model, input_size=(2, 224, 224))
    writers['train'].add_scalar('number of parameter', count_parameters(model))
    slack_message(model.__repr__(), config_logging_channel)
    slack_message(model_sum, config_logging_channel)

    ##################
    #   evaluation   #
    ##################
    config_evaluation = config["evaluation"]
    # IOU, DICE, F1
    config_metrics = config_evaluation["metrics"]
    # IOU, DICE
    config_loss = config_evaluation["loss"]
    config_segmentation_loss = config_loss["segmentation"]
    config_classification_loss = config_loss["classification"]

    metricParser = TypeParser(table = {
        "IOU": metric.IOU, 
        "DICE": metric.DICE,
        "accuracy": metric.Accuracy,
    })
    metrics = [metricParser(**m) for m in config_metrics]

    lossParser = TypeParser(table = {
        "IOU": loss.IOULoss,
        "DICE": loss.DICELoss,
        "BCE": nn.BCEWithLogitsLoss,
    })
    criterion = loss.MultiOutputLoss(
        [lossParser(**config_segmentation_loss), lossParser(**config_classification_loss)],
        **config_loss["param"])

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

    try: 
        trainer = Trainer(model=model, 
                        datasets=datasets, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        metrics=metrics, 
                        writers=writers,
                        checkpoint=config_learning_checkpoint,
                        init=config_learning_weightinit)
    except KeyboardInterrupt:
        slack_message('abrupt end', '#botlog')


    ##################
    #    training    #
    ##################
    try:
        trainer.train(**config["training"])
    except KeyboardInterrupt:
        slack_message("abupt end", config_logging_channel)
    
    writers['train'].close()
    writers['val'].close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(json.load(open(args.config)))
    # mp = mp.set_start_method("spawn")

# json.dump(config, fp, sort_keys=True, indent=4)
