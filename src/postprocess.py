from model import *
from trainer import Trainer
import argparse
import os
import torch
from torchvision import transforms
from torch.nn import functional as F
import torchvision.transforms.functional as F_trans
from PIL import Image
import numpy as np
import cv2
from torch.autograd import Variable
import pandas as pd
import re
import sys

#parser = argparse.ArgumentParser(description=__doc__)
#parser.add_argument("--modelPath", type=str, required=True, help="path to model checkpoint")
#parser.add_argument("--model", type=str, required=True, help="path to model checkpoint")
#parser.add_argument("--dataCsv", type=str, required=True, help="path to model checkpoint")

class PanoWithOutputImgs():
    def __init__(self, path):
        self.thresList = [0.3, 0.5, 0.8, 0.9]
        self.imgTypeList = ['MajorOnly', 'MinorSameCol', 'MinorGray']
        self.imgDictArray = [[{} for i in range(1, len(self.imgTypeList))] for i in range(1, len(self.thresList))]

        self.path = path
        self.teethColor = [(0,0,255), (0,127,255), (0,255,255), (0,255,0), (255,255,0), (255,0,0),
                (255,0,127), (255,0,255)] #TODO: change it
        
        patternImg = cv2.imread('../data/fillpattern.jpg', cv2.IMREAD_GRAYSCALE)
        ret, patternThres = cv2.threshold(patternImg, 150, 255, cv2.THRESH_BINARY)
        self.patternMask = cv2.bitwise_not(patternThres)
 
    def getImg(self, fileName, thres, imgType):
        imgDict = self.imgDictArray[self.thresList.index(thres)][self.imgTypeList.index(imgType)]
        if fileName in imgDict:
            return imgDict[fileName]
        else:
            panoImg = cv2.imread(self.path + fileName + '.jpg', cv2.IMREAD_COLOR)
            imgDict[fileName] = panoImg
            return panoImg

    def updateImg(self, fileName, majorImg, minorImg, toothNum, leftUpmostCoord):

        for thres in self.thresList:
            _updateImg(fileName, majorImg, minorImg, toothNum, leftUpmostCoord, thres)

        return

    def _updateImg(self, fileName, majorImg, minorImg, toothNum, leftUpmostCoord, thres):
        
        print('This teeth: {}, thres: {}'.format(toothNum, thres))

        thresMajorImg = (majorImg > thres).astype(int) * 255
        thresMinorImg = (minorImg > thres).astype(int) * 255

        for imgType in self.imgTypeList:
            panoImg = self.getImg(fileName, thres, imgType)
            coloredMajorImg, coloredMinorImg = self.oneBlobAndColorImg(majorImg, minorImg, toothNum, imgType)
            updatedPanoImg = self.addImg(panoImg, coloredMajorImg, leftUpmostCoord)
            updatedPanoImg = self.addImg(updatedPanoImg, coloredMinorImg, leftUpmostCoord)
            __updateImg(fileName, updatedPanoImg, thres, imgType)

    def __updateImg(self, fileName, result, thres, imgType):
        imgDict = self.imgDictArray[self.thresList.index(thres)][self.imgTypeList.index(imgType)]
        if fileName in imgDict:
            imgDict[fileName] = result
            return 0
        else:
            return 1

    def getColor(self, toothNum):
        return self.teethColor[(toothNum % 10 - 1)] # TODO: change it

    def oneBlobAndColorImg(self, majorImg, minorImg, toothNum, imgType):

        color = self.getColor(toothNum)

        majorOutput = _oneBlobAndColorImg(majorImg, color)[0]
        if imgType == 'MajorOnly':
            return (majorOutput, None)
        elif imgType == 'MinorSameCol':
            return (majorOutput, _oneBlobAndColorImg(minorImg, color)[1])
        elif imgType == 'MinorGray':
            return (majotOutput, _oneBlobAndColorImg(minorImg, (150, 150, 150))[1])
        else:
            return (None, None)
 

    def _oneBlobAndColorImgMajor(self, img, color):

        # Assume img in grayscale, threshold applied
        h, w = img.shape[:2]
        imgDraw = np.zeros((h+2, w+2, 3), np.uint8) # img to draw contour
        imgFrame = np.zeros((h+2, w+2), np.uint8) # img with padding
        imgFrame[1:(h+1), 1:(w+1)] = img

        #th, imgTh = cv2.threshold(imgFrame, 200, 255, cv2.THRESH_BINARY_INV) # threshold just in case...
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

        patternCrop = self.patternMask[0:(h+2), 0:(w+2)]
        imgInv = cv2.bitwise_not(img)
        imgPattern = cv2.bitwise_and(imgDraw, imgDraw, mask = patternCrop)
        cv2.drawContours(imgPattern, [maxContour], 0, color, 7)
        imgPattern = cv2.bitwise_and(imgPattern, imgPattern, mask = imgInv)
        resultPattern = imgPattern[1:(h+1), 1:(w+1), 0:3].copy()
    
        return (resultImg, resultPattern)


    def addImg(self, panoImg, outputImg, leftUpmostCoord):
        x, y = leftUpmostCoord
        x1 = x
        x2 = (x1 + outputImg.shape[1])
        y1 = (panoImg.shape[0] - y)
        y2 = (y1 - outputImg.shape[0])

        print('outputImg h: {}, w: {}'.format(outputImg.shape[0], outputImg.shape[1]))
        print('X:{}, Y:{}, x: {}, y:{}, x1: {}, x2: {}, y1: {}, y2: {}'
                .format(panoImg.shape[1], panoImg.shape[0], x, y, x1, x2, y1, y2))
        augOutImg = panoImg.copy()
        
        outputGray = cv2.cvtColor(outputImg, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(outputGray, 10, 255, cv2.THRESH_BINARY)
        maskInv = cv2.bitwise_not(mask)

        panoROI= panoImg[y2:y1, x1:x2]
        imgBack = cv2.bitwise_and(panoROI, panoROI, mask = maskInv)
        imgFront = cv2.bitwise_and(outputImg, outputImg, mask=mask)

        imgAdd = cv2.add(imgBack, imgFront)
        augOutImg[y2:y1, x1:x2] = imgAdd
        cv2.addWeighted(augOutImg, 0.5, panoImg, 0.5, 0, panoImg)

        return panoImg


    def saveImg(self, path):
        rows = []
        for thres in self.thresList:
            for imgType in self.imgTypeList:
                imgDict = self.imgDictArray[self.thresList.index(thres)][self.imgTypeList.index(imgType)]
                for fileName, panoImg in imgDict.items():
                    cv2.imwrite(path + 'PanoWithOutputImg-' + fileName + '-' + str(thres) + '-' + imgtype + '.jpg', panoImg)
            rows.append([fileName])
        return rows


class Inference():

    def __init__(self, checkpointPath, configPath):

        pano_mean = 0.45443202007
        pano_std = 0.19530903237
        box_mean = 0.130272261
        box_std = 0.33651706451

        config = json.load(open(configPath))
        self.model_config = config["model"]
        self.size = config["augmentation"]["size"]
        self.model = getModel(**self.model_config)
    
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),       
            transforms.Normalize((box_mean, pano_mean), (pano_std, box_std)),
        ])
        self.configPath = configPath
        self.model.restore(checkpointPath)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            self.model.gpu()

    def toInput(self, boxImg, panoImg):

        img = Image.merge("LA", (boxImg, panoImg))
        img = self.transform(img)
        return img

    def infer(self, box_paths, pano_paths, use_gpu=True):
        assert len(pano_paths) == len(box_paths)

        input = [
            self.toInput(Image.open(box), Image.open(pano)) for pano, box in zip(box_paths, pano_paths)
        ]
        input = torch.stack(input, dim=0)

        if torch.cuda.is_available():
            input = cuda(input, self.device)
            output = self.model.test(input)
            cpu(output)
        else:
            self.model.cpu()
            output = self.model.test(input)

        return output.numpy()


def postprocess(path, config, dataCsv):

    print('postprocess: path:{}, config: {}, dataCsv: {}'.format(path, config, dataCsv))
    parameterFileName = re.sub('(\S*/)*(\S+).pth.tar', '\\2', path)
    parameterFolderName = re.sub('(\S*/)*(\S+)[/](\S+[/])(\S+).pth.tar', '\\2', path)
    print('           : fileName: {}, folderName: {}'.format(parameterFileName, parameterFolderName))

    try:
        inputDf = pd.read_csv(dataCsv)
    except IOError:
        print('cannot read input file')
        return

    # Intentionally skip colNum check

    outCols = ['Pano.With.Output.Img']

    panoImgPath = '../data/rawdata/panoImg/'
    panoWithOutputImgs = PanoWithOutputImgs(panoImgPath)

    saveImgPath = '../result/panoWithOutputImg/' + parameterFolderName + '-' + parameterFileName + '/'
   
    if not os.path.exists('../result/panoWithOutputImg/'):
        os.mkdir('../result/panoWithOutputImg/')

    if (os.path.exists(saveImgPath)):
        print('directory already exists for saveImgPath {}'.format(saveImgPath))
        #return
    else:
        os.mkdir(saveImgPath)

    inference = Inference(path, config)
    
    batchsize = 32
    currsize = 1
    panoImgList = []
    boxImgList = []
    panoToothNumDic = {}
    panoCoorDic = {}

    for idx, row in inputDf.iterrows():

        if not(currsize > bathsize):
            panoImgList.append(panoImgPath + row['Cropped.Pano.Img'])
            boxImgList.append(panoImgPath + row['Cropped.Box.Img'])
            panoToothNumDic[row['Cropped.Pano.Img']] = int(row['Tooth.Num.Panoseg'])
            panoCoorDic[row['Cropped.Pano.Img']] = eval(row['Left.Upmost.Coord'])
            currsize += 1
            continue

        output = inference.infer(panoImgList, boxImgList)

        for i in range(len(panoImgList)):
            panoPath = panoImgList[i]
            majorImg = output[i][0] 
            minorImg = output[i][1]
            panoFileName = re.sub('(\S*/)*cropPanoImg-(\S+)-([0-9][0-9]).jpg', '\\2', panoPath)
            toothNum = panoToothNumDic[panoFileName]
            leftUpmostCoord = panoCoorDic[panoFileName]

            panoWithOutputImgs.updateImg(panoFileName, majorImg, minorImg, toothNum, leftUpmostCoord)

            panoWithOutputImgs.saveImg(saveImgPath)

        # reinitialize
        currsize = 1
        panoimgList = []
        boximgList = []
        panoToothNumDic = {}
        panoCoorDic = {}

    if not (currsize == 1):
        output = inference.infer(panoImgList, boxImgList)

        for i in range(len(panoImgList)):
            panoFileName = panoImgList[i]
            majorImg = output[i][0] 
            minorImg = output[i][1]
            panoFileName = re.sub('(\S*/)*cropPanoImg-(\S+)-([0-9][0-9]).jpg', '\\2', panoFileName)
            toothNum = panoToothNumDic[panoFileName]
            leftUpmostCoord = panoCoorDic[panoFileName]

            panoWithOutputImgs.updateImg(panoFileName, majorImg, minorImg, toothNum, leftUpmostCoord)

            panoWithOutputImgs.saveImg(saveImgPath)

    rows = panoWithOutputImgs.saveImg(saveImgPath)
    return


def main(path, dataCsv):
    if (path == '-1'):
        checkpointDir = '../result/checkpoint/'
        flag1 = False
        flag2 = False
        for dirpath, dirnames, files in os.walk(checkpointDir):
            if (flag1 == True):
                break
            flag1 = True
            for direc in dirnames:
                if (flag2 == True):
                    break
                #flag2 = True
                flag3 = False
                print('direc: {}'.format(direc))
                dirpath = checkpointDir + direc + '/'
                for dirpath2, dirnames2, files2 in os.walk(dirpath):
                    if (flag3 == True):
                        break
                    flag3 = True
                    print('{}, {}, {}'.format(dirpath2, dirnames2, files2))
                    assert(len(files2) == 1)
                    config = dirpath + files2[0]
                    assert(len(dirnames2) == 1)
                    flag4 = False
                    print('config: {}'.format(config))
                    for dirpath3, dirnames3, files3, in os.walk(dirpath + dirnames2[0]): 
                        if (flag4 == True):
                            break
                        flag4 = True
                        for f in files3:
                            print('checkpoint: {}'.format(f))
                            postprocess(dirpath + dirnames2[0] + '/' + f, config, dataCsv)

    else:
        flag1 = False
        for dirpath, dirnames, files in os.walk(path):
            if (flag1 == True):
                break
            flag1 = True
            assert(len(files) == 1)
            config = dirpath + files[0]
            assert(len(dirnames) == 1)
            flag4 = False
            print('direc: {}, config: {}'.format(path, config))
            for dirpath3, dirnames3, files3, in os.walk(path + dirnames[0]): 
                if (flag4 == True):
                    break
                flag4 = True
                for f in files3:
                    print('checkpoint: {}'.format(f))
                    postprocess(path + dirnames[0] + '/' + f, config, dataCsv)

    return


if __name__ == "__main__":
    #args = parser.parse_args()
    main(sys.argv[1], sys.argv[2])
