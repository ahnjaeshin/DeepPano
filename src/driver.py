"""


timer interrupt
"""


import argparse
import json
from preprocess import PanoSet
from trainer import Trainer
from torchvision import models, transforms

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=str, required=True, help="path to config file")

MODE = ('train', 'val', 'test')

def main(config):
    config_dataset = config["dataset"]
    assert all(m+'-dir' in config_dataset for m in MODE)
    print (config_dataset)

    augmentations = {
        'train' : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ]),
        'val'  : transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ]),
        'test' : None
    }
    assert all(m in augmentations for m in MODE)

    datasets = {
        x: PanoSet(config_dataset[x + '-dir'], transform=augmentations[x], target_transform=augmentations[x])
            for x in MODE
    }

    t = Trainer(None, datasets, None, None, None)

if __name__ == "__main__":
    args = parser.parse_args()
    main(json.load(open(args.config)))

# json.dump(config, fp, sort_keys=True, indent=4)