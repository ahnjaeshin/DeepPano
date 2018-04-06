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
        inputDf = pd.read_csv('../data/metadata/MouthRawData.csv') # TODO: decide file name
    except IOError:
        print('cannot read input file')
        return

    rowNum, colNum = inputDf.shape
    
    if colNum != 7:
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
            'Left.Upmost.Coord', 'Tooth.Num.Panoseg', 'Margin.Type'] # TODO: decide column names
    rows = []

    for idx, row in inputDf.iterrows():
        rows.extend(generateDatasetForEachFile(marginType, outImgPath, row))
   
    outputDf = pd.DataFrame(rows, columns=outCols)
    outputDf.to_csv(outCsvFileName, encoding='utf-8')

    return


def generateDatasetForEachFile(marginType, outImgPath, row):

    outRows = []

    # TODO: decide column names
    print(row)
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
        
        cropPanoImg = cv2.copyMakeBorder(panoImg, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        cropPanoImg = cropImage(cropPanoImg, marginedXYs)
        cropPanoImg = cv2.flip(cropPanoImg, 0)

        cropBoxImg = np.zeros(imgShape, dtype=np.uint8)
        genBoxImage(cropBoxImg, coords)
        cropBoxImg = cropImage(cropBoxImg, marginedXYs)
        cropBoxImg = cv2.flip(cropBoxImg, 0)
        
        # Leave it for debugging usage
        inputImg = cv2.add(cropPanoImg, cropBoxImg)
        
        cropAnnotImg = genAnnotImage(annotPsd, toothNum, marginedXYs, imgShape)
        cropAnnotImg = cv2.flip(cropImage(cv2.flip(cropAnnotImg, 0), marginedXYs), 0)

        # TODO: Wrong tooth number check?

        #TODO: add Pano number
        cpiName = outImgPath + 'cropPanoImg' + str(toothNum) + '.jpg'
        cbiName = outImgPath + 'cropBoxImg' + str(toothNum) + '.jpg'
        iiName = outImgPath + 'inputImg' + str(toothNum) + '.jpg' 
        caiName = outImgPath + 'cropAnnotImg' + str(toothNum) + '.jpg' 

        # export images
        cv2.imwrite(cpiName, cropPanoImg)
        cv2.imwrite(cbiName, cropBoxImg)
        cv2.imwrite(iiName, inputImg)
        cv2.imwrite(caiName, cropAnnotImg)

        # write row for .csv
        newRow = [cpiName, cbiName, iiName, caiName, (marginedXYs[0], marginedXYs[2]),
                toothNum, marginType]
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
    print (x1, y1, x2, y2)
    cv2.floodFill(img, mask, (int((x1+x2)/2), int((y1+y2)/2)), 255)
    
    return


def genAnnotImage(annotPsd, toothNum, marginedXYs, imgShape):
    
    # TODO: we will take another method
    psdLayer = None
    for layer in annotPsd.layers:
        if str(layer.name) == 'teeth_'+str(toothNum): # TODO: need to change
            psdLayer = layer
            break
    layerImg = np.zeros(imgShape, dtype=np.uint8)
    if psdLayer is None or psdLayer.bbox == (0, 0, 0, 0):
        return layerImg
    layerImg = psdLayer.as_PIL()
    
    b1, b2 = psdLayer.bbox.y1, psdLayer.bbox.y2
    b3, b4 = psdLayer.bbox.x1, psdLayer.bbox.x2
    r, g, b, a = cv2.split(np.array(layerImg))

    # get alpha channel from png and convert to grayscale
    layerImg = cv2.merge([a, a, a])
    layerImg = cv2.cvtColor(layerImg, cv2.COLOR_BGR2GRAY)
    annotImg = np.zeros(imgShape, dtype=np.uint8)
    annotImg[b1:b2, b3:b4] = layerImg
       
    return annotImg


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
