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
from metric import IOU
from loss import IOULoss

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=str, required=True, help="path to config file")

MODE = ('train', 'val', 'test')

def main(config):
    config_dataset = config["dataset"]
    assert 'data-dir' in config_dataset
    
    config_augmentation = config["augmentation"]
    config_model = config["model"]
    config_training = config["training"]
    config_learning = config["learning"]
    config_evaluation = config["evaluation"]
    config_logging = config["logging"]

    augmentations = {
        'train' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),         
            transforms.Normalize(mean=(0,0), std=(255,255)),
        ]),
        'val'  : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),         
            transforms.Normalize(mean=(0,0), std=(255,255)),
        ]),
        'test' : None
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
        ]),
        'test' : None,
    }

    assert all(m in augmentations for m in MODE)

    datasets = {
        'train': PanoSet(config_dataset['data-dir'], (lambda row: row['Train.Val'] == 'train' and row['Target.Img'] != '-1'), transform=augmentations['train'], target_transform=target_augmentations['train']),
        'val': PanoSet(config_dataset['data-dir'], (lambda row: row['Train.Val'] == 'val' and row['Target.Img'] != '-1'), transform=augmentations['val'], target_transform=target_augmentations['val']),
        'test': None
    }

    model = UNet(2, 1, bilinear=False)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = IOULoss(weight=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=config_learning['lr_init'], momentum=0.9, nesterov=True, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 20, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    metrics = [IOU(threshold=0.5), IOU(threshold=0.3), IOU(threshold=0.8)]

    checkpoint = config_model["checkpoint"]
    log_freq = config_training["log_freq"]

    trainer = Trainer(model, datasets, criterion, optimizer, scheduler, metrics, checkpoint)

    trainer.train(batch_size=config_training['batch_size'],
                  num_workers=config_training['num_workers'],
                  epochs=config_training['epochs'],
                  log_freq=log_freq)

if __name__ == "__main__":
    args = parser.parse_args()
    main(json.load(open(args.config)))

# json.dump(config, fp, sort_keys=True, indent=4)
