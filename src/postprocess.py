from model import UNet
import argparse
import os
import torch
from torchvision import transforms
from torch.nn import functional as F
import torchvision.transforms.functional as F_trans
from PIL import Image
import numpy as np
from torch.autograd import Variable

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=str, required=True, help="path to model checkpoint")

class Inference():
    def __init__(self, model, path):
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),         
        ])
        self.path = path
        if path:
            self.load(self.path)

    def load(self, path):
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))

            if torch.cuda.is_available():
                checkpoint = torch.load(path)
            else:
                checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['state_dict'], )
        else:
            print("=> no checkpoint found at '{}'".format(path))

    def infer(self, pano_path, segment_path):
        img = Image.merge("LA", (Image.open(segment_path), Image.open(pano_path)))
        img = self.transform(img)
        img = Variable(img.view(1, 2, 224, 224))
        output, segmentable = self.model(img)
        output = F.sigmoid(output)
        output = output.data.view(224, 224)
        output = output.numpy()
        output = (output > 0.5).astype(int) * 255
        output = Image.fromarray(np.uint8(output))

        output.save('./example.png')

def main(path):
    model = UNet(2, 1, False)
    i = Inference(model, path)
    i.infer('../data/metadata/FirstSet-20180415182244-2/cropPanoImg-T1-Pano-042-11.jpg', '../data/metadata/FirstSet-20180415182244-2/targetImg-T1-Pano-042-11.jpg')

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.model)