"""


timer interrupt
"""


import argparse
import json

import torch
import torch.nn as nn

from model import UNet
from preprocess import PanoSet
from torchvision import transforms
from trainer import Trainer
import metric
from loss import IOULoss, DICELoss, BCEIOULoss

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=str, required=True, help="path to config file")

class Filter():
    
    def __init__(self, filters):
        lookup = {
            "train" : lambda row: row['Train.Val'] == 'train',
            "val" : lambda row: row['Train.Val'] == 'val',
        }
        self.func = [lambda row: row['Target.Img'] != '-1']
        self.func += [lookup[f] for f in filters]

    def __call__(self, row):
        for f in self.func:
            if not f(row):
                return False
        return True



def main(config):
    
    ##################
    #     logging    #
    ##################
    config_logging = config["logging"]
    
    ##################
    #  augmentation  #
    ##################
    config_augmentation = config["augmentation"]


    augmentations = {
        'train' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),         
            # transforms.Normalize(mean=(0,0), std=(255,255)),
        ]),
        'val'  : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),         
            # transforms.Normalize(mean=(0,0), std=(255,255)),
        ])
    }

    target_augmentations = {
        'train' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.point(lambda p: 255 if p > 50 else 0 )),
            transforms.ToTensor(),         
            # transforms.Normalize(mean=(0,), std=(1,)),
            
        ]),
        'val'  : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.point(lambda p: 255 if p > 50 else 0 )),
            transforms.ToTensor(),      
            # transforms.Normalize(mean=(0,0), std=(255,255)),
        ])
    }


    ##################
    #     dataset    #
    ##################
    config_dataset = config["dataset"]
    config_dataset_path = config_dataset["data-dir"]

    train_filter = Filter(['train'])
    val_filter = Filter(['val'])

    datasets = {
        'train': PanoSet(config_dataset_path, train_filter, transform=augmentations['train'], target_transform=target_augmentations['train']),
        'val': PanoSet(config_dataset_path, val_filter, transform=augmentations['val'], target_transform=target_augmentations['val']),
    }


    ##################
    #      model     #
    ##################
    config_model = config["model"]

    model = UNet(2, 1, bilinear=False)

    ##################
    #   evaluation   #
    ##################
    config_evaluation = config["evaluation"]
    config_metrics = config_evaluation["metrics"]
    config_loss = config_evaluation["loss"]

    metric_lookup = {"IOU": metric.IOU, "DICE": metric.DICE}
    metrics = [metric_lookup[m["type"]](m["threshold"]) for m in config_metrics]

    # criterion = nn.BCEWithLogitsLoss()
    # criterion = IOULoss()
    # criterion = DICELoss()
    criterion = BCEIOULoss(jaccard_weight=1)

    ##################
    #    learning    #
    ##################
    config_learning = config["learning"]
    
    # xavier_uniform, xavier_normal, he_uniform, he_normal
    config_learning_weightinit = config_learning["weight_init"]
    # give path to load checkpoint or null
    config_learning_checkpoint = config_model["checkpoint"]

    optimizer = torch.optim.SGD(model.parameters(), lr=config_learning['lr_init'], momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 130, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    trainer = Trainer(model=model, 
                      datasets=datasets, 
                      criterion=criterion, 
                      optimizer=optimizer, 
                      scheduler=scheduler, 
                      metrics=metrics, 
                      checkpoint=config_learning_checkpoint,
                      init=config_learning_weightinit)


    ##################
    #    training    #
    ##################
    config_training = config["training"]
    # print out log every log_feq #
    config_learning_logfreq = config_training["log_freq"]
    # batch size
    config_learning_batchsize = config_training['batch_size']
    config_learning_numworker = config_training['num_workers']
    config_learning_epoch = config_training['epochs']

    trainer.train(batch_size=config_learning_batchsize,
                  num_workers=config_learning_numworker,
                  epochs=config_learning_epoch,
                  log_freq=config_learning_logfreq)

if __name__ == "__main__":
    args = parser.parse_args()
    main(json.load(open(args.config)))

# json.dump(config, fp, sort_keys=True, indent=4)
