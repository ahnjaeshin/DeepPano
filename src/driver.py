"""


timer interrupt
"""


import argparse
import json

import torch
import torch.nn as nn
import socket
import datetime

from model import UNet
from preprocess import PanoSet
from torchvision import transforms
from trainer import Trainer
from transform import DualAugment, ToBoth, ImageOnly, TargetOnly
import metric
import loss
from utils import slack_message, count_parameters
from tensorboardX import SummaryWriter
from torch.autograd import Variable

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=str, required=True, help="path to config file")

FRONT = (11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43)

class Filter():
    
    def __init__(self, filters):
        lookup = {
            "train" : lambda row: row['Train.Val'] == 'train',
            "val" : lambda row: row['Train.Val'] == 'val',
            "segmentable": lambda row: int(row['Segmentable.Type']) < 10,
            "unsegmentable": lambda row: int(row['Segmentable.Type']) >= 10,
            "front-teeth": lambda row: int(row['Tooth.Num.Annot']) in FRONT,
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
    config_logging_debug = config_logging["debug"]
    config_logging_channel = config_logging["slack_channel"]
    # write title in hierarchical way e.g. UNET/HE/SGD
    config_logging_title = config_logging["title"]
    config_logging_trial = config_logging["trial"]

    log_dir = 'runs/{title}/{time}_{host}_{trial}/'.format(title=config_logging_title, 
                                                           time=datetime.datetime.now().strftime('%b%d_%H-%M'), 
                                                           host=socket.gethostname(), 
                                                           trial=config_logging_trial)
    writers = {x : SummaryWriter(log_dir=log_dir + x) for x in ('train', 'val')}
    
    ##################
    #  augmentation  #
    ##################
    config_augmentation = config["augmentation"]

    augmentations = {
        'train' : DualAugment([
            ToBoth(transforms.RandomHorizontalFlip()),
            ToBoth(transforms.RandomVerticalFlip()),

            ToBoth(transforms.Resize((224, 224))),
            TargetOnly(transforms.Lambda(lambda img: img.point(lambda p: 255 if p > 50 else 0 ))),
            ToBoth(transforms.ToTensor()), 
        ]),
        'val' : DualAugment([
            ToBoth(transforms.Resize((224, 224))),
            TargetOnly(transforms.Lambda(lambda img: img.point(lambda p: 255 if p > 50 else 0 ))),
            ToBoth(transforms.ToTensor()),
        ])
    }

    ##################
    #     dataset    #
    ##################
    config_dataset = config["dataset"]
    config_dataset_path = config_dataset["data-dir"]
    config_dataset_filter = config_dataset["filter"]

    data_filter = { x : Filter(config_dataset_filter[x]) for x in ('train', 'val')}

    datasets = { x: PanoSet(config_dataset_path, data_filter[x], transform=augmentations[x])
                    for x in ('train', 'val')}

    ##################
    #      model     #
    ##################
    config_model = config["model"]

    model = UNet(2, 1, bilinear=False)

    dummy_input = Variable(torch.rand(1, 2, 128, 128), requires_grad=True)
    writers['train'].add_graph(model, (dummy_input, ))
    writers['train'].add_scalar('number of parameter', count_parameters(model))


    ##################
    #   evaluation   #
    ##################
    config_evaluation = config["evaluation"]
    # IOU, DICE, F1
    config_metrics = config_evaluation["metrics"]
    # BCE, IOU, DICE, BCEIOU
    config_loss = config_evaluation["loss"]

    metric_lookup = {"IOU": metric.IOU, "DICE": metric.DICE}
    metrics = [metric_lookup[m["type"]](m["threshold"]) for m in config_metrics]

    if config_loss["type"] == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif config_loss["type"] == "IOU":
        criterion = loss.IOULoss()
    elif config_loss["type"] == "DICE":
        criterion = loss.DICELoss()
    elif config_loss["type"] == "BCEIOU":
        loss_param = config_loss["param"]
        criterion = loss.BCEIOULoss(jaccard_weight=loss_param["weight"])
    else:
        raise NotImplementedError

    ##################
    #    learning    #
    ##################
    config_learning = config["learning"]
    
    # xavier_uniform, xavier_normal, he_uniform, he_normal
    config_learning_weightinit = config_learning["weight_init"]
    # give path to load checkpoint or null
    config_learning_checkpoint = config_learning["checkpoint"]

    optimizer = torch.optim.SGD(model.parameters(), lr=config_learning['lr_init'], momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 130, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
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
    config_training = config["training"]
    # print out log every log_feq #
    config_learning_logfreq = config_training["log_freq"]
    # batch size
    config_learning_batchsize = config_training['batch_size']
    config_learning_numworker = config_training['num_workers']
    config_learning_epoch = config_training['epochs']

    try:
        trainer.train(batch_size=config_learning_batchsize,
                    num_workers=config_learning_numworker,
                    epochs=config_learning_epoch,
                    log_freq=config_learning_logfreq)
    except KeyboardInterrupt:
        slack_message("abupt end", config_logging_channel)
    
    writers['train'].close()
    writers['val'].close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(json.load(open(args.config)))

# json.dump(config, fp, sort_keys=True, indent=4)
