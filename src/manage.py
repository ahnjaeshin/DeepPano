'''

python3 manage.py genData (marginType) (inputFileName)

or

python3 manage.py decideTarget (updateFileName)

'''

import sys
import os
import pandas as pd
import xml.etree.ElementTree as et
import numpy as np
import cv2
from psd_tools import PSDImage

def __main__():
    
    if len(sys.argv) < 2:
        print('need to input commands\n',
                'genData (marginType) (inputFileName), or\n',
                'decideTarget (updateFileName)')
        return
    
    command = str(sys.argv[1])
    
    if command == "genData":
        if len(sys.argv) != 4:
            print('need to input marginType and inputFileName\n')
            return
        marginType = int(sys.argv[2])
        inputFileName = str(sys.argv[3])
        generateDataset(marginType, inputFileName)
    elif command == "decideTarget":
        if len(sys.argv) != 3:
            print('need to input updateFileName\n')
            return
        updateFileName = str(sys.argv[2])
        decideTarget(updateFileName)

    return


def generateDataset(marginType, inputFileName):
    
    try:
        inputDf = pd.read_csv(inputFileName)
    except IOError:
        print('cannot read input file')
        return

    rowNum, colNum = inputDf.shape
    
    if colNum != 4: # TODO: temporary
        print('wrong number of columns')
        return

    # TODO: columns check?

    outCsvFileName = '../data/metadata/tentative.csv' # TODO: autogenerate with timestamp & inputFileName & marginType 
    outImgPath = '../data/metadata/testOutImg/' # TODO: autogenerate
    if (os.path.exists(outImgPath)):
        print('directory already exists for outImgPath' + outImgPath)
        # skip here now for debugging purpose but later should output error
    else:
        os.mkdir(outImgPath)
    outCols = ['Cropped.Pano.Img', 'Cropped.Box.Img', 'Cropped.Input.Img', 'Cropped.Annot.Img',
            'Left.Upmost.Coord', 'Tooth.Num.Panoseg', 'Tooth.Num.Annot', 'Margin.Type']
    rows = []

    for idx, row in inputDf.iterrows():
        rows.extend(generateDatasetForEachFile(marginType, outImgPath, row))
   
    outputDf = pd.DataFrame(rows, columns=outCols)
    outputDf.to_csv(outCsvFileName, encoding='utf-8')

    return


def generateDatasetForEachFile(marginType, outImgPath, row):

    outRows = []

    print('row: {}'.format(row))
    imageTitle = row['Image.Title']
    panoFileName = row['Pano.File']
    xmlFileName = row['Xml.File']
    annotFileName = row['Annot.File']

    panoImg = cv2.flip(cv2.imread(panoFileName, cv2.IMREAD_GRAYSCALE), 0)
    annotPsd = PSDImage.load(annotFileName)
    
    #if panoImg.shape != annotImg.shape:
    #    print('panoFile and annotFile sizes do not match')
    #    return

    imgShape = panoImg.shape
    
    
    # XML Parsing
    root = et.parse(xmlFileName).getroot()

    for tooth in root.iter('Tooth'):

        print(tooth.attrib)
        toothNum = int(tooth.attrib['Number'])
        coords = genCoordsFromTooth(tooth)
        # TODO: Better make it marginedCoords
        marginedXYs = calculateCoordsWithMargin(marginType, coords)
        leftMostCoor = (marginedXYs[0], marginedXYs[2])
        
        cropPanoImg = cv2.copyMakeBorder(panoImg, 0, 0, 0, 0, cv2.BORDER_REPLICATE) # flipped
        cropPanoImg = cropImage(cropPanoImg, marginedXYs) # flipped
        cropPanoImg = cv2.flip(cropPanoImg, 0) # unflip

        boxImg = np.zeros(imgShape, dtype=np.uint8)
        boxImg = genBoxImage(boxImg, coords) # flipped
        cropBoxImg = cropImage(boxImg, marginedXYs)
        cropBoxImg = cv2.flip(cropBoxImg, 0) # unflip
        
        # Leave it for debugging usage
        inputImg = cv2.add(cropPanoImg, cropBoxImg)
        
        annotToothNum, annotImg = genAnnotImage(annotPsd, boxImg, imgShape) # flipped
        cropAnnotImg = cropImage(annotImg, marginedXYs)
        cropAnnotImg = cv2.flip(cropAnnotImg, 0) # unflip

        # TODO: Wrong tooth number check?

        # TODO: calculate imageTitle from panoFileName and delete imageTitle column from .csv
        cpiName = outImgPath + 'cropPanoImg' + '-' + imageTitle + '-' + str(toothNum) + '.jpg'
        cbiName = outImgPath + 'cropBoxImg' + '-' + imageTitle + '-' + str(toothNum) + '.jpg'
        iiName = outImgPath + 'inputImg' + '-' + imageTitle + '-' + str(toothNum) + '.jpg'
        caiName = outImgPath + 'cropAnnotImg' + '-' + imageTitle + '-' + str(toothNum) + '.jpg'

        # export images
        cv2.imwrite(cpiName, cropPanoImg)
        cv2.imwrite(cbiName, cropBoxImg)
        cv2.imwrite(iiName, inputImg)
        cv2.imwrite(caiName, cropAnnotImg)

        # write row for .csv
        newRow = [cpiName, cbiName, iiName, caiName, (marginedXYs[0], marginedXYs[2]),
                toothNum, annotToothNum, marginType]
        outRows.append(newRow)

    return outRows


# TODO: Better make it cropImageWithMargin(img, coords, marginType)
def cropImage(img, xYs):
    x1, x2, y1, y2 = xYs[0], xYs[1], xYs[2], xYs[3]
    img = img[y1:y2, x1:x2]
    return img


def genBoxImage(img, coords):

    x1, y1, x2, y2 = 5000, 5000, 0, 0

    for i in range(4):
        cv2.line(img, coords[i], coords[(i+1)%4], 255, 1)
        x, y = coords[i]
        if x < x1:
            x1 = x
        if y < y1:
            y1 = y
        if x > x2:
            x2 = x
        if y > y2:
            y2 = y
    
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), dtype=np.uint8)
    print('x1: {}, x2: {}, x3: {}, x4: {}'.format(x1, y1, x2, y2))
    cv2.floodFill(img, mask, (int((x1+x2)/2), int((y1+y2)/2)), 255)
    
    return img


def genAnnotImage(annotPsd, boxImg, imgShape):
    
    maxIOU = 0
    maxIOULayer = None
    maxIOULayerImg = np.zeros(imgShape, dtype=np.uint8)

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
        annotImg = cv2.flip(annotImg, 0)

        intersectionImg = cv2.bitwise_and(annotImg, boxImg)
        intersectionArea = np.sum(intersectionImg == 255)
        teethArea = np.sum(annotImg == 255)
        thisIOU = intersectionArea / teethArea
        print ('layerName: {}, thisIOU: {}'.format(layer.name, thisIOU))
        if (maxIOU <  thisIOU):
            maxIOU = thisIOU
            maxIOULayer = layer
            maxIOULayerImg = annotImg

    #if maxIOU < 0.8:
    #    maxIOULayer = None
    #    maxIOULayerImg = np.zeros(imgShape, dtype=np.uint8)

    return ((0, maxIOULayerImg) if maxIOULayer is None else (maxIOULayer.name, maxIOULayerImg))


def genCoordsFromTooth(tooth):
    return [(int(coord.attrib['X']), int(coord.attrib['Y'])) for coord in tooth]


def calculateCoordsWithMargin(marginType, coords):
    if marginType == 1:
        return _calculateCoordsWithMargin1(coords)
    return []


def _calculateCoordsWithMargin1(coords):

    marginValue = 40

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

    x1 -= marginValue
    x2 += marginValue
    y1 -= marginValue
    y2 += marginValue

    return (x1, x2, y1, y2)


if __name__ == '__main__':
    __main__()
