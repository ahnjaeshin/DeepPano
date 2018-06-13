
import argparse
import json
import random
import numpy as np
import torch
import traceback
import metric as M
from dataset import getDataset
from model import getModel
from trainer import Trainer
from utils import Logger, TypeParser
import imgaug as ia

from inference import Inference

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", "-c", type=str, required=True, help="path to config file")

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
    datasets = getDataset(**dataset, **augmentation)

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
        LOG.finish()
        print('abrupt end, {}'.format(e))
        print(traceback.format_exc())

    infer = Inference(
        model=MODEL,
        datasets=datasets,
        LOG=LOG,
        metrics=metrics,
        visualizations=None,
    )

    infer(
        
    )

    LOG.finish()

if __name__ == "__main__":
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(**config)
