import sys
import pandas as pd
import xml.etree.ElementTree as et
import numpy as np
import cv2
from psd_tools import PSDImage

'''

python3 manage.py genData (marginType) (outputFileName)

or

python3 manage.py decideTarget (updateFileName)

'''

def __main__():
    
    if len(sys.argv) < 2:
        print('need to input commands\n'
                + 'genData (marginType) (outputFileName), or\n'
                + 'decideTarget (updateFileName)')
        return
    
    command = str(sys.argv[1])
    
    if command == "genData":
        if len(sys.argv) != 4:
            print('need to input marginType and outputFileName\n')
            return
        marginType = int(sys.argv[2])
        outputFileName = str(sys.argv[3])
        generateDataset(marginType, outputFileName)
    elif command == "decideTarget":
        if len(sys.argv) != 3:
            print('need to input updateFileName\n')
            return
        updateFileName = str(sys.argv[2])
        decideTarget(updateFileName)

    return


def generateDataset(marginType, outputFileName):
    
    try:
        inputDf = pd.read_csv('../data/metadata/MouthRawData.csv') # TODO: decide file name
    except IOError:
        print('cannot read MouthRawData file')
        return

    rowNum = inputDf.shape[0]
    colNum = inputDf.shape[1]
    
    if colNum != 7:
        print('wrong number of columns')
        return

    # TODO: columns check?

    outCols = ['1'] # TODO: decide column names
    rows = []

    for idx, row in inputDf.iterrows():
        rows.extend(generateDatasetForEachFile(marginType, row))
    
    outputDf = pd.DataFrame(rows, columns=outCols)
    outputDf.to_csv(outputFileName, encoding='utf-8')

    return


def generateDatasetForEachFile(marginType, row):

    outRows = []

    # TODO: decide column names
    print(row)
    panoFileName = row['pano.File']
    xmlFileName = row['xml.File']
    annotFileName = row['annot.File']

    panoImg = cv2.flip(cv2.imread(panoFileName, cv2.IMREAD_GRAYSCALE), 0)
    annotPsd = PSDImage.load(annotFileName)
    
    if panoImg.shape != annotImg.shape:
        print('panoFile and annotFile sizes do not match')
        return

    imgShape = panoImg.shape
    
    
    # XML Parsing
    tree = et.parse(xmlFileName)
    root = tree.getroot()

    for tooth in root.iter('Tooth'):
 
        toothNum = int(tooth.attrib)
        coords = genCoordsFromTooth(tooth)
        # TODO: Better make it marginedCoords
        marginedXYs = calculateCoordsWithMargin(marginType, coords)
        leftMostCoor = (marginedXYs[0], marginedXYs[2])
        
        cropPanoImg = cv2.copyMakeBorder(panoImg, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        cropImage(cropPanoImg, marginedXYs)

        cropBoxImg = np.zeros(imgShape, NP.unit8)
        genBoxImage(cropBoxImg, coords)
        cropImage(cropBoxImg, marginedXYs)
        
        # TODO: How to add channel info?
        inputImg = cv2.add(cropPanoImg, cropBoxImg)
        
        # TODO: Better make it to gen annotImg then crop, just like boxImg
        # Not sure if current code works
        cropAnnotImg = genAnnotImage(annotPsd, toothNum, marginedXYs, imgShape)
    
        # TODO: Wrong tooth number check?

        # export images
        # TODO: name convention needed
        cv2.imwrite('A', cropPanoImg)
        cv2.imwrite('B', cropBoxImg)
        cv2.imwrite('C', inputImg)
        cv2.imwrite('D', cropAnnotImg)

        # write row for .csv
        newRow = ['A'] # TODO: decide naming convention
        outRows.append(newRow)

    return outRows


# TODO: Better make it cropImageWithMargin(img, coords, marginType)
def cropImage(img, xYs):
    x1, x2, y1, y2 = xYs[0], xYs[1], xYs[2], xYs[3]
    panoImg[y1:y2, x1:x2]
    return


def genBoxImage(img, coords):

    for i in range(4):
        cv2.line(img, coors[i], coors[(i+1)%4], 255, 1)
    
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), NP.uint8)
    cv2.floodFill(img, mask, (int((x1+x2)/2), int(y1+y2)/2), 255)
    
    return


# TODO: Better separate it into two parts
def getAnnotImage(annotPsd, toothNum, marginedXYs, imgShape):
    
    # need a way to translate toothNum to layerNum or layer name
    psdLayer = psd.layers[toothNum] # TODO: this assumes many things...
    annotImg = psdLayer.as_PIL()
    
    b1, b2 = psdLayer.bbox.y1, psdLayer.bbox.y2
    b3, b4 = psdLayer.bbox.x1, psdLayer.bbox.x2
    r, g, b, a = cv2.split(np.array(annotImg))
    x1, x2, y1, y2 = marginedXYs[0], marginedXYs[1], marginedXYs[2], marginedXYs[3]
    imgY = imgShape[0]

    annotImg = cv2.merge([a, a, a])
    annotImg = cv2.copyMakeBorder(annotImg, b1+y2-imgY, imgY-b2-y1, b3-x1, x2-b4,
            cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return annotImg


def genCoordsFormTooth(tooth):
    coords = []
    for coord in tooth:
        coords.append((int(coord.attrib['X']), int(coord.attrib['Y'])))
    return coords


def calculateCoordsWithMargin(marginType, coords):
    if marginType == 1:
        return _calculateCoordsWithMargin1(coords)
    return []


def _calculateCoordsWithMargin1(coords):

    marginValue = 40

    x1 = 5000
    x2 = 0
    y1 = 5000
    y2 = 0
    
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

    return [x1, x2, y1, y2]


if __name__ == '__main__':
    __main__()
