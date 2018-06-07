'''
coco related functions from
https://github.com/nightrome/cocoapi,
https://github.com/matterport/Mask_RCNN
'''

from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image
import skimage.io as io
import pylab
import sys
import os
import xml.etree.ElementTree as ET
import random
import pandas as pd
import time


def __main__():
    
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    if len(sys.argv) < 2:
        return
    
    howMany = int(sys.argv[1])

    makePretrainData(howMany)
    
    return


def makePretrainData(howMany):

    makePretrainDataTrainVal(int(howMany * 0.9), 'train2017', 'train')
    makePretrainDataTrainVal(howMany - int(howMany * 0.9), 'val2017', 'val')

    return


def makePretrainDataTrainVal(howMany, dataType, trainVal):

    dataDir = '../data/rawdata/Coco'
    annFile = '{}/instances_{}.json'.format(dataDir, dataType)

    if not os.path.exists(dataDir):
        os.mkdir(dataDir)

    if not os.path.exists(dataDir + '/Annot'):
        os.mkdir(dataDir + '/Annot')

    if not os.path.exists(dataDir + '/Xml'):
        os.mkdir(dataDir + '/Xml')

    if not os.path.exists(dataDir + '/Input'):
        os.mkdir(dataDir + '/Input')

    coco = COCO(annFile)

    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    outputFileName = '../data/metadata/Coco.csv'

    cols = ['Image.Title', 'Pano.File', 'Xml.File', 'Annot.File', 'Train.Val']
    rows = []

    # if exists, append
    if os.path.isfile(outputFileName):
        inputDf = pd.read_csv(outputFileName)
        for idx, row in inputDf.iterrows():
            rows.append([row['Image.Title'], row['Pano.File'], row['Xml.File'], row['Annot.File'], row['Train.Val']])

    howMany = len(imgIds) if howMany < 0 else howMany

    for i in range(howMany):

        imgId = 0
        inputImgName = ''

        while True:
            imgId = imgIds[np.random.randint(0, len(imgIds))]
            inputImgName = dataDir + '/Input/InputImg-' + str(imgId) + '.jpg'
            if not os.path.isfile(inputImgName):
                break

        img = coco.loadImgs(imgId)[0]
        print('img: {}'.format(img))
        inputImg = io.imread(img['coco_url'])
        io.imsave(inputImgName, inputImg)

        xmlName = dataDir + '/Xml/Xml-' + str(imgId) + '.xml'
        xmlRoot = ET.Element("root")
        xmlToothList = ET.SubElement(xmlRoot, "ToothList")

        annotDir = dataDir + '/Annot/Annot-' + str(imgId) + '/'
        os.mkdir(annotDir)

        pngs = cocoSegmentationToPng(coco, imgId)

        for j in range(len(pngs)):

            png = pngs[j]
            annotImg = np.array(png)
            annotName = dataDir + '/Annot/Annot-' + str(imgId) + '/TargetImg-' + str(j) + '.jpg'
            cv2.imwrite(annotName, annotImg)
            xmlTooth = ET.SubElement(xmlToothList, "Tooth", Number=str(j))
            coords = makeRandomBoundingBox(annotImg)
            for i in range(len(coords)):
                (x, y) = coords[i]
                ET.SubElement(xmlTooth, 'P' + str(i), Y=str(y), X=str(x))

        xmlTree = ET.ElementTree(xmlRoot)
        xmlTree.write(xmlName)

        row = [imgId, inputImgName, xmlName, annotDir, trainVal]
        rows.append(row)

        # write every time so that there will be result to use even after an abrupt error
        outputDf = pd.DataFrame(rows, columns=cols)
        outputDf.to_csv(outputFileName)

    return


def makeRandomBoundingBox(img):

    height, width = img.shape
    res, thr = cv2.threshold(img, 127, 255, 0)
    temp, contours, hier = cv2.findContours(thr, 1, 2)

    xmin, ymin, xmax, ymax = (10000000, 10000000, 0, 0)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if (x < xmin):
            xmin = x
        if (y < ymin):
            ymin = y
        if ((x + w) > xmax):
            xmax = x + w
        if ((y + h) > ymax):
            ymax = y + h

    x, y, w, h = (xmin, ymin, xmax - xmin, ymax - ymin)

    print('height:{}, width:{}'.format(height, width))
    p1 = (x, height - y)
    p2 = (x + w, height - y)
    p3 = (x + w, height - y - h)
    p4 = (x, height - y - h)

    print(p1, p2, p3, p4)

    coords = [p1, p2, p3, p4]

    y1 = min(max(int(height - y + (np.random.normal(0, 5) * 0.5 * h)), 0), height)
    y2 = min(max(int(height - y + (np.random.normal(0, 5) * 0.5 * h)), 0), height)
    y3 = min(max(int(height - y - h + (np.random.normal(0, 5) * 0.5 * h)), 0), height)
    y4 = min(max(int(height - y - h + (np.random.normal(0, 5) * 0.5 * h)), 0), height)

    x1 = min(max(int(x + (np.random.normal(0, 5) * 0.5 * w)), 0), width)
    x2 = min(max(int(x + w + (np.random.normal(0, 5) * 0.5 * w)), 0), width)
    x3 = min(max(int(x + w + (np.random.normal(0, 5) * 0.5 * w)), 0), width)
    x4 = min(max(int(x + (np.random.normal(0, 5) * 0.5 * w)), 0), width)

    coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    print(coords)

    # no sort needed since manage.py has changed

    return coords if boxIsBigEnough(img, coords) else makeRandomBoundingBox(img)


def boxIsBigEnough(annotImg, coords):

    shape = annotImg.shape
    print('shape: {}'.format(shape))
    boxImg = np.zeros((shape[0], shape[1], 1), dtype=np.uint8)
    for i in range(len(coords)):
        cv2.fillPoly(boxImg, [np.array([coords[i], coords[(i+1)%len(coords)], coords[(i+2)%len(coords)]])], 255)
        print('\tfor i: {}, boxImg: {}'.format(i, np.sum(boxImg == 255)))

    print('boxImg: {}, annotImg: {}'.format(np.sum(boxImg == 255), np.sum(annotImg == 255)))
    if np.sum(boxImg == 255) <= (0.1 * np.sum(annotImg == 255)):
        return False

    return True


def cocoSegmentationToSegmentationMap(coco, imgId, checkUniquePixelLabel=True, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the id of the COCO image
    :param checkUniquePixelLabel: (optional) whether every pixel can have at most one label
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: labelMap - [h x w] segmentation map that indicates the label of each pixel
    '''

    # Init
    curImg = coco.imgs[imgId]
    imageSize = (curImg['height'], curImg['width'])

    # Get annotations of the current image (may be empty)
    imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
    catIds = coco.getCatIds(catNms=['person'])
    if includeCrowd:
        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds)
    else:
        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=False)

    imgAnnots = coco.loadAnns(annIds)

    labelMaps = []
    # Combine all annotations of this image in labelMap
    for a in range(0, len(imgAnnots)):
        labelMask = coco.annToMask(imgAnnots[a]) == 1
        newLabel = 255 #imgAnnots[a]['category_id']

        labelMap = np.zeros(imageSize)
        labelMap[labelMask] = newLabel
        labelMaps.append(labelMap)

    return labelMaps


def cocoSegmentationToPng(coco, imgId, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map and write it to disk.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the COCO id of the image (last part of the file name)
    :param pngPath: the path of the .png file
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: None
    '''

    pngs = []

    # Create label map
    labelMaps = cocoSegmentationToSegmentationMap(coco, imgId, includeCrowd=includeCrowd)

    for i in range(len(labelMaps)):
        labelMap = labelMaps[i]
        labelMap = labelMap.astype(np.int8)

        # Write to png file
        png = Image.fromarray(labelMap).convert('P')
        pngs.append(png)

    return pngs


if __name__ == '__main__':
    __main__()
