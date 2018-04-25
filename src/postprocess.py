from model import UNet
from trainer import Trainer
import argparse
import os
import torch
#from preprocess import PanoSet
from torchvision import transforms
from torch.nn import functional as F
import torchvision.transforms.functional as F_trans
from PIL import Image
import numpy as np
import cv2
from torch.autograd import Variable
import pandas as pd
import time
import datetime
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parameterPath", type=str, required=True, help="path to model checkpoint")
parser.add_argument("--dataCsv", type=str, required=True, help="path to dataset csv file")

class PanoWithOutputImgs():
    def __init__(self, path):
        self.imgDict = {}
        self.path = path
        self.teethColor = [(0,0,255), (0,127,255), (0,255,255), (0,255,0), (255,255,0), (255,0,0),
                (255,0,127), (255,0,255)] #TODO: change it

    def getImg(self, fileName):
        if fileName in self.imgDict:
            return self.imgDict[fileName]
        else:
            panoImg = cv2.imread(self.path + fileName + '.jpg', cv2.IMREAD_COLOR)
            self.imgDict[fileName] = panoImg
            return panoImg

    def updateImg(self, fileName, outputImg, toothNum, leftUpmostCoord):
        print('This teeth is {}'.format(toothNum))
        panoImg = self.getImg(fileName)
        color = self.getColor(toothNum)
        coloredOutputImg = self.oneBlobAndColorImg(outputImg, color)
        updatedPanoImg = self.addImg(panoImg, coloredOutputImg, leftUpmostCoord)

        if fileName in self.imgDict:
            self.imgDict[fileName] = updatedPanoImg
            return 0
        else:
            return 1

    def getColor(self, toothNum):
        return self.teethColor[(toothNum % 10 - 1)] # TODO: change it

    def oneBlobAndColorImg(self, img, color):

       # Assume img in grayscale, threshold applied
        h, w = img.shape[:2]
        imgDraw = np.zeros((h+2, w+2, 3), np.uint8) # img to draw contour
        imgFrame = np.zeros((h+2, w+2), np.uint8) # img with padding
        imgFrame[1:(h+1), 1:(w+1)] = img

        th, imgTh = cv2.threshold(imgFrame, 200, 255, cv2.THRESH_BINARY_INV) # threshold just in case...
        tempImg, contours, hierarchy = cv2.findContours(imgTh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contoursNum = len(contours) - 1 # except the whole img
        maxArea = 0
        maxIndex = -1

        for i in range(contoursNum):
            area = cv2.contourArea(contours[i])
            if area > maxArea:
                maxArea = area
                maxIndex = i

        maxContour = contours[maxIndex]
        cv2.drawContours(imgDraw, [maxContour], 0, color, cv2.FILLED) # draw filled contour
        cv2.drawContours(imgDraw, [maxContour], 0, (0, 0, 0), 1) # erase contour border line

        resultImg = imgDraw[1:(h+1), 1:(w+1), 0:3].copy() # remove padding

        return resultImg

    def addImg(self, panoImg, outputImg, leftUpmostCoord):
        x, y = leftUpmostCoord
        x1 = x
        x2 = (x1 + outputImg.shape[1])
        y1 = (panoImg.shape[0] - y)
        y2 = (y1 - outputImg.shape[0])

        print('outputImg h: {}, w: {}'.format(outputImg.shape[0], outputImg.shape[1]))
        print('X:{}, Y:{}, x: {}, y:{}, x1: {}, x2: {}, y1: {}, y2: {}'
                .format(panoImg.shape[1], panoImg.shape[0], x, y, x1, x2, y1, y2))
        augOutImg = cv2.copyMakeBorder(panoImg, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        augOutImg[y2:y1, x1:x2] = outputImg

        cv2.addWeighted(augOutImg, 0.5, panoImg, 0.5, 0, panoImg)

        return panoImg

    def saveImg(self, path):
        rows = []
        for fileName, panoImg in self.imgDict.items():
            cv2.imwrite(path + 'PanoWithOutputImg-' + fileName + '.jpg', panoImg)
            rows.append([fileName])
        return rows

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
        w, h = img.size
        img = self.transform(img)
        img = Variable(img.view(1, 2, 224, 224))
        output, segmentable = self.model(img)
        output = F.sigmoid(output)
        output = output.data.view(224, 224)
        output = output.numpy()
        output = cv2.resize(output, (w, h))
        output = (output > 0.5).astype(int) * 255

        return output


def __main__(parameterPath, dataCsv):

    timestamp = datetime.datetime.fromtimestamp(time.mktime(time.localtime())).strftime('%Y%m%d%H%M%S')
    parameterFileName = re.sub('(\S*/)*(\S+).pth.tar', '\\2', parameterPath)

    model = UNet(2, 1, False)
    inference = Inference(model, parameterPath)

    try:
        inputDf = pd.read_csv(dataCsv)
    except IOError:
        print('cannot read input file')
        return

    # Intentionally skip colNum check

    outCols = ['Pano.With.Output.Img']

    panoImgPath = '../data/rawdata/panoImg/'
    panoWithOutputImgs = PanoWithOutputImgs(panoImgPath)

    saveImgPath = '../result/panoWithOutputImg/' + parameterFileName + '-' + timestamp + '/'
    if (os.path.exists(saveImgPath)):
        print('directory already exists for saveImgPath {}'.format(saveImgPath))
        return
    else:
        os.mkdir(saveImgPath)

    for idx, row in inputDf.iterrows():
        panoFileName = re.sub('(\S*/)*cropPanoImg-(\S+)-([0-9][0-9]).jpg', '\\2', row['Cropped.Pano.Img'])
        toothNum = int(row['Tooth.Num.Annot'])
        leftUpmostCoord = eval(row['Left.Upmost.Coord'])

        outputImg = inference.infer(row['Cropped.Pano.Img'], row['Cropped.Box.Img'])
        panoWithOutputImgs.updateImg(panoFileName, outputImg, toothNum, leftUpmostCoord)

        panoWithOutputImgs.saveImg(saveImgPath)

    rows = panoWithOutputImgs.saveImg(saveImgPath)

    return

if __name__ == "__main__":
    args = parser.parse_args()
    __main__(args.parameterPath, args.dataCsv)
