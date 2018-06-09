'''

python3 manage.py genData (marginType) (inputFileName)

python3 manage.py genStat (fileName)


'''

import sys
import os
import pandas as pd
import xml.etree.ElementTree as et
import numpy as np
import cv2
from psd_tools import PSDImage
import time
import datetime
import copy
import re
import math


def __main__():
    
    if len(sys.argv) < 2:
        print('need to input commands\n',
                'genData (marginType) (inputFileName), or\n',
                'genStat (fileName)')
        return
    
    command = str(sys.argv[1])
    
    if command == "genData":
        if len(sys.argv) != 4:
            print('need to input marginType and inputFileName\n')
            return
        marginType = int(sys.argv[2])
        inputFileName = str(sys.argv[3])
        generateDataset(marginType, inputFileName)
    elif command == "genStat":
        if len(sys.argv) != 3:
            print('need to input fileName\n')
            return
        fileName = str(sys.argv[2])
        calcStat(fileName)
    else:
        print('need to type in command')

    return


#######################
#   GenerateDataset   #
#######################


def generateDataset(marginType, inputFileName):
    
    try:
        inputDf = pd.read_csv(inputFileName)
    except IOError:
        print('cannot read input file')
        return

    rowNum, colNum = inputDf.shape
    
#    if colNum != 5: # TODO: temporary
#        print('wrong number of columns')
#        return

    # TODO: columns check?

    # timestamp = datetime.datetime.fromtimestamp(time.mktime(time.localtime())).strftime('%Y%m%d%H%M%S')
    # TODO: should extract inputFileName from route using regex
    outputFormat = inputFileName[17:-4] + '-' + str(marginType)
    outImgPath = '../data/metadata/' + outputFormat + '/'
    outCsvFileName = '../data/metadata/' + outputFormat + '.csv'
    if (os.path.exists(outImgPath)):
        print('directory already exists for outImgPath' + outImgPath)
        # return
    else:
        os.mkdir(outImgPath)

    print('outputRoute: {}'.format(outputFormat))

    outCols = ['Name', 'Cropped.Pano.Img', 'Cropped.Box.Img', 'Cropped.Major.Annot.Img', 'Cropped.Minor.Annot.Img',
            'Left.Upmost.Coord', 'Cropped.Img.Size', 'Tooth.Num.Panoseg', 'Tooth.Num.Major.Annot', 'Tooth.Num.Minor.Annot',
            'Max.Teeth.IOU', '2nd.Max.Teeth.IOU', 'Max.Box.IOU', '2nd.Max.Box.IOU', 'Margin.Type', 'Segmentable.Type',
            'Major.Target.Img', 'Minor.Target.Img', 'All.Img', 'Train.Val']
    rows = []

    for idx, row in inputDf.iterrows():
        rows.extend(generateDatasetForEachFile(marginType, outImgPath, row))
   
    outputDf = pd.DataFrame(rows, columns=outCols)
    outputDf.to_csv(outCsvFileName, encoding='utf-8')

    return


def generateDatasetForEachFile(marginType, outImgPath, row):

    outRows = []

    print('row: {}'.format(row))
    imageTitle = str(row['Image.Title'])
    panoFileName = row['Pano.File']
    xmlFileName = row['Xml.File']
    annotFileName = row['Annot.File']

    doAnnot = False if annotFileName == -1 else True
    isAnnotDir = False
    if doAnnot and not (annotFileName[-3:] == 'psd'):
        isAnnotDir = True

    panoImg = cv2.flip(cv2.imread(panoFileName, cv2.IMREAD_GRAYSCALE), 0)
    imgShape = panoImg.shape

    annotImgs = None
    if doAnnot and not isAnnotDir:
        annotPsd = PSDImage.load(annotFileName)
        annotImgs = extractImgsFromPsd(annotPsd, imgShape) # flipped
    elif isAnnotDir:
        annotImgs = extractImgsFromDir(annotFileName, imgShape)

    # XML Parsing
    root = et.parse(xmlFileName).getroot()

    for tooth in root.iter('Tooth'):

        print(tooth.attrib)
        toothNum = str(tooth.attrib['Number'])
        thisTitle = imageTitle + '-' + str(toothNum)
        coords = genCoordsFromTooth(tooth)
        
        cropPanoImg = cv2.copyMakeBorder(panoImg, 0, 0, 0, 0, cv2.BORDER_REPLICATE) # flipped
        leftMostCoor, cropPanoImg = cropImageWithMargin(cropPanoImg, coords, marginType, imgShape)
        cropPanoImg = cv2.flip(cropPanoImg, 0) # unflip

        print('img size: {} for coords: {}'.format(cropPanoImg.shape, coords))

        boxImg = np.zeros(imgShape, dtype=np.uint8)
        boxImg = genBoxImage(boxImg, coords) # flipped
        leftMostCoor, cropBoxImg = cropImageWithMargin(boxImg, coords, marginType, imgShape)
        cropBoxImg = cv2.flip(cropBoxImg, 0) # unflip
        
        # Leave it for debugging usage
        inputImg = cv2.copyMakeBorder(cropPanoImg, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        cv2.addWeighted(cv2.add(cropPanoImg, cropBoxImg), 0.2, inputImg, 0.8, 0, inputImg)

        if not doAnnot:

            cpiName = outImgPath + 'cropPanoImg' + '-' + thisTitle + '.jpg'
            cbiName = outImgPath + 'cropBoxImg' + '-' + thisTitle + '.jpg'

            cv2.imwrite(cpiName, cropPanoImg)
            cv2.imwrite(cbiName, cropBoxImg)

            newRow = [thisTitle, cpiName, cbiName, -1, -1, leftMostCoor, cropPanoImg.shape, toothNum,
                    -1, -1, -1, -1, -1, -1, marginType, -1, -1, -1, -1, row['Train.Val']]
            outRows.append(newRow)

            continue

        maxIOU, sndMaxIOU, fstBoxIOU, sndBoxIOU, majorToothNum, majorAnnotImg, minorToothNum, minorAnnotImg = genAnnotImages(annotImgs, boxImg, imgShape) # flipped
        leftMostCoor, cropMajorAnnotImg = cropImageWithMargin(majorAnnotImg, coords, marginType, imgShape)
        cropMajorAnnotImg = cv2.flip(cropMajorAnnotImg, 0) # unflip
        leftMostCoor, cropMinorAnnotImg = cropImageWithMargin(minorAnnotImg, coords, marginType, imgShape)
        cropMinorAnnotImg = cv2.flip(cropMinorAnnotImg, 0) # unflip

        segType = decideSegType(maxIOU, sndMaxIOU, fstBoxIOU)
        majorTargetFlag = decideTargetFlag(maxIOU)
        minorTargetFlag = decideTargetFlag(sndMaxIOU)
        majorTargetImg = np.zeros(cropMajorAnnotImg.shape, dtype = np.uint8) if not majorTargetFlag else cropMajorAnnotImg
        minorTargetImg = np.zeros(cropMinorAnnotImg.shape, dtype = np.uint8) if not minorTargetFlag else cropMinorAnnotImg

        allImg = cv2.copyMakeBorder(inputImg, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        cv2.addWeighted(cv2.add(inputImg, cv2.add(majorTargetImg, minorTargetImg)), 0.6, allImg, 0.4, 0, allImg)

        # TODO: Wrong tooth number check?

        # TODO: calculate imageTitle from panoFileName and delete imageTitle column from .csv
        cpiName = outImgPath + 'cropPanoImg' + '-' + thisTitle + '.jpg'
        cbiName = outImgPath + 'cropBoxImg' + '-' + thisTitle + '.jpg'
        macaiName = outImgPath + 'cropAnnotMajorImg' + '-' + thisTitle + '.jpg'
        micaiName = outImgPath + 'cropAnnotMinorImg' + '-' + thisTitle + '.jpg'
        matiName = 0 if not majorTargetFlag else re.sub('cropPanoImg', 'targetMajorImg', cpiName)
        mitiName = 0 if not minorTargetFlag else re.sub('cropPanoImg', 'targetMinorImg', cpiName)
        aiName = re.sub('cropPanoImg', 'allImg', cpiName)

        # export images
        cv2.imwrite(cpiName, cropPanoImg)
        cv2.imwrite(cbiName, cropBoxImg)
        cv2.imwrite(macaiName, cropMajorAnnotImg)
        cv2.imwrite(micaiName, cropMinorAnnotImg)
        if majorTargetFlag:
            cv2.imwrite(matiName, cropMajorAnnotImg)
        if minorTargetFlag:
            cv2.imwrite(mitiName, cropMinorAnnotImg)
        cv2.imwrite(aiName, allImg)

        # write row for .csv
        newRow = [thisTitle, cpiName, cbiName, macaiName, micaiName, leftMostCoor, cropPanoImg.shape, toothNum,
                majorToothNum, minorToothNum, maxIOU, sndMaxIOU, fstBoxIOU, sndBoxIOU, marginType,
                segType, matiName, mitiName, aiName, row['Train.Val']]
        outRows.append(newRow)

    return outRows


# This function returns flipped imgs
def extractImgsFromPsd(annotPsd, imgShape):

    annotImgs = {}

    for layer in annotPsd.layers:

        if layer is None or layer.bbox == (0, 0, 0, 0):
            continue

        layerImg = np.array(layer.as_PIL())

        # get alpha channel from png and convert to grayscale
        if (layerImg.shape[-1] != 4):
            # Then this is background image
            continue

        r, g, b, a = cv2.split(layerImg)
        layerImg = cv2.merge([a, a, a])
        layerImg = cv2.cvtColor(layerImg, cv2.COLOR_BGR2GRAY)
        ret, layerImg = cv2.threshold(layerImg, 200, 255, cv2.THRESH_BINARY)

        annotImg = np.zeros(imgShape, dtype=np.uint8)
        b1, b2 = layer.bbox.y1, layer.bbox.y2
        b3, b4 = layer.bbox.x1, layer.bbox.x2
        annotImg[b1:b2, b3:b4] = layerImg
        annotImg = cv2.flip(annotImg, 0) # flip
        annotImgs[layer.name.strip()] = annotImg

    return annotImgs


# This function returns flipped imgs
def extractImgsFromDir(annotDir, imgShape):

    annotImgs = {}

    for subdir, dirs, files in os.walk(annotDir):
        for fileName in files:
            print(fileName)
            annotImg = cv2.flip(cv2.imread(annotDir + fileName, cv2.IMREAD_GRAYSCALE), 0)
            name = re.sub('Target-(\d+).jpg', '\\1', fileName)
            annotImgs[name] = annotImg

    return annotImgs


def genBoxImage(img, coords):

    x1, y1, x2, y2 = 5000, 5000, 0, 0

    for i in range(len(coords)):
        cv2.fillPoly(img, [np.array([coords[i], coords[(i+1)%len(coords)], coords[(i+2)%len(coords)]])], 255)
        #cv2.line(img, coords[i], coords[(i+1)%4], 255, 1)
        x, y = coords[i]
        if x < x1:
            x1 = x
        if y < y1:
            y1 = y
        if x > x2:
            x2 = x
        if y > y2:
            y2 = y

    #h, w = img.shape[:2]
    #mask = np.zeros((h+2, w+2), dtype=np.uint8)
    print('x1: {}, y1: {}, x2: {}, y2: {}'.format(x1, y1, x2, y2))
    #cv2.floodFill(img, mask, (int((x1+x2)/2), int((y1+y2)/2)), 255)
    
    return img


def genAnnotImages(annotImgs, boxImg, imgShape):

    maxIOU = 0
    sndMaxIOU = 0
    fstBoxIOU = 0
    sndBoxIOU = 0

    maxIOULayerName = 0
    maxIOULayerImg = np.zeros(imgShape, dtype=np.uint8)
    sndMaxIOULayerName = 0
    sndMaxIOULayerImg = np.zeros(imgShape, dtype=np.uint8)

    for name, annotImg in annotImgs.items():

        intersectionImg = cv2.bitwise_and(annotImg, boxImg)
        intersectionArea = np.sum(intersectionImg == 255)
        teethArea = np.sum(annotImg == 255)
        boxArea = np.sum(boxImg == 255)
        thisIOU = intersectionArea / teethArea
        thisBoxIOU = intersectionArea / boxArea

        if thisIOU > 0:
            print ('layerName: {}, thisIOU: {}'.format(name, thisIOU))

        if (maxIOU <  thisIOU):

            sndMaxIOU = maxIOU
            sndBoxIOU = fstBoxIOU
            sndMaxIOULayerName = maxIOULayerName
            sndMaxIOULayerImg = maxIOULayerImg

            maxIOU = thisIOU
            fstBoxIOU = thisBoxIOU
            maxIOULayerName = name
            maxIOULayerImg = annotImg

        elif (sndMaxIOU < thisIOU):

            sndMaxIOU = thisIOU
            sndBoxIOU = thisBoxIOU
            sndMaxIOULayerName = name
            sndMaxIOULayerImg = annotImg

    return (maxIOU, sndMaxIOU, fstBoxIOU, sndBoxIOU, maxIOULayerName, maxIOULayerImg, sndMaxIOULayerName, sndMaxIOULayerImg)


def decideTargetFlag(maxIOU):
    # intersectionImg = cv2.bitwise_and(cropAnnotImg, cropBoxImg)
    # intersectionArea = np.sum(intersectionImg == 255)
    # teethArea = np.sum(cropAnnotImg == 255)
    # newMaxIOU = intersectionArea / teethArea
    # if maxIOU < 0.05 or (maxIOU / newMaxIOU) < 0.67:
    if maxIOU < 0.08:
        return False
    return True


def decideSegType(maxIOU, sndMaxIOU, boxIOU):
    if maxIOU < 0.3 and boxIOU < 0.3:
        return 10 # no teeth
    if maxIOU > 0.67:
        if (maxIOU * 0.67) < sndMaxIOU:
            return 12 # two teeth
        return 4 # inclusive
    if (maxIOU * 0.67) < sndMaxIOU:
        return 11 # half half
    return 5 # half inclusive


def genCoordsFromTooth(tooth):
    coords = []
    for coord in tooth:
        coords.append([int(coord.attrib['X']), int(coord.attrib['Y'])])
    return coords
    # return [[int(coord.attrib['X']), int(coord.attrib['Y'])]] for coord in tooth]


def cropImageWithMargin(img, coords, marginType, imgShape):
    marginFuns = {1: _cropImageWithMargin1, 2: _cropImageWithMargin2, 3: _cropImageWithMargin3,
            4: _cropImageWithMargin4, 5: _cropImageWithMargin5, 6: _cropImageWithMargin6,
            7: _cropImageWithMargin7, 8: _cropImageWithMargin8, 9: _cropImageWithMargin9,
            10: _cropImageWithMargin10}
    return marginFuns[marginType](img, coords, imgShape)


def __cropImageWithSimpleMargin(img, coords, marginList, imgShape):

    x1, x2, y1, y2 = 5000, 0, 5000, 0

    for coord in coords:
        x = coord[0]
        y = coord[1]

        if x < x1:
            x1 = x
        if y < y1:
            y1 = y
        if x > x2:
            x2 = x
        if y > y2:
            y2 = y

    x1 = (x1 - marginList[0]) if ((x1 - marginList[0]) > 0) else 0
    x2 = (x2 + marginList[1]) if ((x2 + marginList[0]) < imgShape[1]) else imgShape[1]
    y1 = (y1 - marginList[2]) if ((y1 - marginList[2]) > 0) else 0
    y2 = (y2 + marginList[3]) if ((y2 + marginList[3]) < imgShape[0]) else imgShape[0]

    return ((x1, y1), img[y1:y2, x1:x2])


def _cropImageWithMargin1(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [40, 40, 40, 40], imgShape)

def _cropImageWithMargin2(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [50, 50, 80, 80], imgShape)

def _cropImageWithMargin3(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [80, 80, 80, 80], imgShape)

def _cropImageWithMargin4(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [60, 60, 100, 100], imgShape)

def _cropImageWithMargin5(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [100, 100, 100, 100], imgShape)

def _cropImageWithMargin6(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [200, 200, 200, 200], imgShape)

def _cropImageWithMargin7(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [400, 400, 400, 400], imgShape)

def _cropImageWithMargin8(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [60, 60, 120, 120], imgShape)

def _cropImageWithMargin9(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [70, 70, 150, 150], imgShape)

    
def _cropImageWithMargin10(img, coords, imgShape):
    return __cropImageWithSimpleMargin(img, coords, [100, 100, 150, 150], imgShape)


################
#   CalcStat   #
################


def calcStat(fileName):

    try:
        inputDf = pd.read_csv(fileName)
    except IOError:
        print('cannot read input file')
        return

    sumPixelNum = 0
    sumPanoPixelValue = 0
    sumBoxPixelValue = 0
    varPano = 0
    varBox = 0

    totalCount = 0
    noneCount = 0
    noneNSingleCount = 0
    
    for idx, row in inputDf.iterrows():

        cropPanoImg = cv2.imread(row['Cropped.Pano.Img'], cv2.IMREAD_GRAYSCALE)
        cropBoxImg = cv2.imread(row['Cropped.Box.Img'], cv2.IMREAD_GRAYSCALE)
        h, w = cropBoxImg.shape

        sumPixelNum += h * w
        sumPanoPixelValue += np.sum(cropPanoImg) / 255
        sumBoxPixelValue += np.sum(cropBoxImg) / 255

        totalCount += 1
        noneCount += 1 if (row['Major.Target.Img'] == str(0)) else 0
        noneNSingleCount +=1 if (row['Minor.Target.Img'] == str(0)) else 0
           
    meanPano = sumPanoPixelValue / sumPixelNum
    meanBox = sumBoxPixelValue / sumPixelNum

    singleCount = noneNSingleCount - noneCount
    doubleCount = totalCount - noneNSingleCount

    for idx, row in inputDf.iterrows():

        cropPanoImg = cv2.imread(row['Cropped.Pano.Img'], cv2.IMREAD_GRAYSCALE)
        cropBoxImg = cv2.imread(row['Cropped.Box.Img'], cv2.IMREAD_GRAYSCALE)

        cropPanoImgVar = np.square(cropPanoImg / 255 - meanPano)
        cropBoxImgVar = np.square(cropBoxImg / 255 - meanBox)

        varPano += np.sum(cropPanoImgVar)
        varBox += np.sum(cropBoxImgVar)


    varPano /= sumPixelNum
    varBox /= sumPixelNum

    stdPano = math.sqrt(varPano)
    stdBox = math.sqrt(varBox)

    print("Pano(Mean, Std) = ({}, {})".format(meanPano, stdPano))
    print("Box(Mean, Std) = ({}, {})".format(meanBox, stdBox))
    print("None: {}, Single: {}, Double: {}".format(noneCount, singleCount, doubleCount))

    return


if __name__ == '__main__':
    __main__()
