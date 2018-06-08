import sys
import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from psd_tools import PSDImage
import time
import datetime
import copy
import re
import math
import random
    
def __main__():
    
        
    if len(sys.argv) < 3:
        print('need to input numbers for double, singe none')
        return
    
    #command = str(sys.argv[1])
    '''
    if command == "XML":
        if len(sys.argv) != 3:
            print('need to input image title\n')
            return
        imageTitle = str(sys.argv[2])
        createBoxXml(imageTitle)
    elif command == "CSV":
        if len(sys.argv) != 3:
            print('need to input ()\n')
            return
        buildCsvFile()
    else:
        print('need to type in command')
    '''

    psdDir = '../data/rawdata/psdFile/'

    for subdir, dirs, files in os.walk(psdDir):
        for fileName in files:
            print(fileName) # e.g. T1-Pano-002.psd
            name = fileName[:-4] # 확장자 뺀 이름 e.g. T1-Pano-002
            createBoxXml(fileName, sys.argv[1], sys.argv[2], sys.argv[3])

    return
   
 
    ####################
    ### Box Creation ###
    ####################
    
    
def extractFromPsd(annotPsd, imgShape):
    
    annotImgs = {}
    imgsBoundary = {}
    
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
        #annotImg = cv2.flip(annotImg, 0) # flip
    
        annotImgs[layer.name.strip()] = annotImg
        imgsBoundary[layer.name.strip()] = [b1, b2, b3, b4]
    
    return (annotImgs, imgsBoundary)
    
    
def createBoxXml(imageTitle, doubleNum, singleNum, noneNum):
    
    panoDir = '../data/rawdata/panoImg/'
    psdDir = '../data/rawdata/psdFile/'
    xmlDir = '../data/rawdata/xmlFile/'   
    xmlName = xmlDir + imageTitle + '.xml'
    panoImg = cv2.imread(panoDir + imageTitle + '.jpg', cv2.IMREAD_GRAYSCALE)
    imgShape = panoImg.shape
    annotPsd = PSDImage.load(psdDir + imageTitle + '.psd')
    annotImgs, imgsBoundary = extractFromPsd(annotPsd, imgShape) 
    
    doubleBoxList = {}
    singleBoxList = {}
    noneBoxList = {} 
    hollowBoxList = {}
    teethNum = 10
    
    while teethNum < 49:
            
        teethNum += 1
        if teethNum % 10 == 0:
            continue
            
        name = str(teethNum)
       
        teethType = 'real'
        neighborTeethKeys = findNeighborTeeth(imgsBoundary, name, imgShape)
    
        if name not in annotImgs:
            if neighborTeethKeys[0] == 0:
                if neighborTeethKeys[1] == 0:
                    # impossible... but still
                    teethType = 'absense'
                else:
                    newBoundary = list(imgsBoundary[neighborTeethKeys[1]])
                    shiftAmount = newBoundary[3] - newBoundary[2]
                    newBoundary[2] -= shiftAmount
                    newBoundary[3] -= shiftAmount
                    teethType = 'hollow_end'
            elif neighborTeethKeys[1] == 0:
                newBoundary = list(imgsBoundary[neighborTeethKeys[0]])
                shiftAmount = newBoundary[3] - newBoundary[2]
                newBoundary[2] += shiftAmount
                newBoundary[3] += shiftAmount
                teethType = 'hollow_end'
            else:
                teethType = 'hollow'

    
        if teethType == 'hollow_end':
            if newBoundary[0] < 0:
                newBoundary[0] = 0
            if newBoundary[1] >= imgShape[0]:
                newBoundary[1] = imgShape[0]-1
            if newBoundary[2] < 0:
                newBoundary[2] = 0
            if newBoundary[3] >= imgShape[1]:
                newBoundary[3] = imgShape[1]-1
            # add hollow teeth as if it's real
            newAnnotImg = np.zeros(imgShape, dtype=np.uint8)
            newAnnotImg[newBoundary[0]:newBoundary[1],newBoundary[2]:newBoundary[3]] = 255
            annotImgs[name] = newAnnotImg 
            imgsBoundary[name] = newBoundary
    
        if teethType == 'hollow':
            newBoundary = []
            newBoundary.append(max(imgsBoundary[neighborTeethKeys[0]][0], imgsBoundary[neighborTeethKeys[1]][0]))
            newBoundary.append(min(imgsBoundary[neighborTeethKeys[0]][1], imgsBoundary[neighborTeethKeys[1]][1]))
            newBoundary.append(int((imgsBoundary[neighborTeethKeys[0]][2]+imgsBoundary[neighborTeethKeys[0]][3])/2))
            newBoundary.append(int((imgsBoundary[neighborTeethKeys[1]][2]+imgsBoundary[neighborTeethKeys[1]][3])/2))

            hollowBoxImg = np.zeros(imgShape, dtype=np.uint8)
            hollowBoxImg[newBoundary[0]:newBoundary[1],newBoundary[2]:newBoundary[3]] = 255

            neighborMask = cv2.bitwise_or(annotImgs[neighborTeethKeys[0]], annotImgs[neighborTeethKeys[1]])
            neighborMask = cv2.bitwise_not(neighborMask)
            hollowBoxImg = cv2.bitwise_and(hollowBoxImg, neighborMask)                       
 
            newAnnotImg = np.zeros(imgShape, dtype=np.uint8)
            img, contours, hierachy = cv2.findContours(hollowBoxImg, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #newAnnotImg = cv2.drawContours(newAnnotImg, contours, 0, 255, -1)
            contoursNum = len(contours)  # except the whole img
            maxArea = 0
            maxIndex = -1
            for i in range(contoursNum):
                area = cv2.contourArea(contours[i])
                if area > maxArea:
                    maxArea = area
                    maxIndex = i   

            maxContour = contours[maxIndex]
            cv2.drawContours(newAnnotImg, [maxContour], 0, 255, cv2.FILLED)

            newAnnotArea = np.sum(newAnnotImg == 255)
            leftAnnotArea = np.sum(annotImgs[neighborTeethKeys[0]] == 255)
            rightAnnotArea = np.sum(annotImgs[neighborTeethKeys[1]] == 255)

            if newAnnotArea < leftAnnotArea * 0.1 and newAnnotArea < rightAnnotArea * 0.1:
                teethType = 'absense'                
            else:     
                x,y,w,h = cv2.boundingRect(maxContour)

                newBoundary[0] = y
                newBoundary[1] = y + h
                newBoundary[2] = x
                newBoundary[3] = x + w
                 
                annotImgs[name] = newAnnotImg
                imgsBoundary[name] = newBoundary
    
        print("start for main = ", name, ", type = ", teethType)            
    
        if teethType == 'absense':
            continue
    
        print(neighborTeethKeys)
        print(imgsBoundary[name])

        ### Double Logic ###
        i = 0 
        while i < doubleNum: # how many doubel boxes?
            if teethType == 'real':
                createdBox = createDoubleBox(annotImgs, imgsBoundary, name, imgShape, 'double')
            else:    
                #createdBox = createDoubleBox(annotImgs, imgsBoundary, name, imgShape, 'triple')
                break
                
            if createdBox == 0:
                continue
                    
            doubleBoxList[name + '-' + str(i)] = createdBox
            i += 1

        print("double  done!")
        
        ### Single Logic ###
        i = 0
        while i < singleNum: # how many single boxes?
            if teethType == 'real':
                createdBox = createSingleOrNoneBox(annotImgs,imgsBoundary,name,neighborTeethKeys,imgShape,'single')
            else:
                #createdBox = createDoubleBox(annotImgs, imgsBoundary, name, imgShape, 'double')
                break

            if createdBox == 0:
                continue

            singleBoxList[name + '-' + str(i)] = createdBox
            i += 1

        print("single done!")
            
        ###  None Logic ###
        i = 0
        while i < noneNum: # how many none boxes?
            if teethType == 'real':
                createdBox = createSingleOrNoneBox(annotImgs,imgsBoundary,name,neighborTeethKeys,imgShape,'none')
            else:
                #createdBox = createSingleOrNoneBox(annotImgs,imgsBoundary,name,neighborTeethKeys,imgShape,'single')
                break
                
            if createdBox ==0:
                continue

            noneBoxList[name + '-' + str(i)] = createdBox
            i += 1
                 
        print("none done!")
        
        
        # hollow logic
        i = 0
        while i < doubleNum/2: # how many hollow boxes?
            if teethType == 'hollow' or teethType == 'hollow_end':
                createdBox1 = createDoubleBox(annotImgs,imgsBoundary,name,imgShape,'double')
                createdBox2 = createSingleOrNoneBox(annotImgs,imgsBoundary,name,neighborTeethKeys,imgShape,'single')
            else:
                break

            if createdBox ==0:
                continue

            hollowBoxList[name + '-' + str(2*i)] = createdBox1
            hollowBoxList[name + '-' + str(2*i+1)] = createdBox2
            i += 1

        print('hollow done!')

        if teethType == 'hollow' or teethType == 'hollow_end':
            del annotImgs[name]
            del imgsBoundary[name]

        

    # build XML 
    #xmlRoot = ET.Element("root")
    xmlRoot = ET.parse(xmlName).getroot()
    #xmlToothList = ET.SubElement(xmlRoot, "ToothList")
    xmlToothList = xmlRoot.find("ToothList")


    for name, points in doubleBoxList.items():

        #img = np.zeros(imgShape, dtype=np.uint8)
        #img = fillBox(img, points)
        #pimg = panoImg.copy()
        #cv2.addWeighted(cv2.add(panoImg, img), 0.2, pimg, 0.8, 0, pimg)
        #cv2.imwrite(dataDir + "boxtest/Double-" + name + ".jpg", pimg)

        xmlTooth = ET.SubElement(xmlToothList, "Tooth", Number = name + 'double')

        for i in range(4):           
            x = points[i][0]
            y = imgShape[0] - points[i][1] - 1
            ET.SubElement(xmlTooth, 'P' + str(i), Y = str(y), X = str(x))

    for name, points in singleBoxList.items():

        #img = np.zeros(imgShape, dtype=np.uint8)
        #img = fillBox(img, points)
        #pimg = panoImg.copy()
        #cv2.addWeighted(cv2.add(panoImg, img), 0.2, pimg, 0.8, 0, pimg)
        #cv2.imwrite(dataDir + "boxtest/Single-" + name + ".jpg", pimg)

        xmlTooth = ET.SubElement(xmlToothList, "Tooth", Number = name + 'single')

        for i in range(4):
            x = points[i][0]
            y = imgShape[0] - points[i][1] - 1
            ET.SubElement(xmlTooth, 'P' + str(i), Y = str(y), X = str(x))    

    for name, points in noneBoxList.items():

        #img = np.zeros(imgShape, dtype=np.uint8)
        #img = fillBox(img, points)
        #pimg = panoImg.copy()
        #cv2.addWeighted(cv2.add(panoImg, img), 0.2, pimg, 0.8, 0, pimg)
        #cv2.imwrite(dataDir + "boxtest/None-" + name + ".jpg", pimg)

        xmlTooth = ET.SubElement(xmlToothList, "Tooth", Number = name + 'none')

        for i in range(4):
            x = points[i][0]
            y = imgShape[0] - points[i][1] - 1
            ET.SubElement(xmlTooth, 'P' + str(i), Y = str(y), X = str(x))

    for name, points in hollowBoxList.items():

        #img = np.zeros(imgShape, dtype=np.uint8)
        #img = fillBox(img, points)
        #pimg = panoImg.copy()
        #cv2.addWeighted(cv2.add(panoImg, img), 0.2, pimg, 0.8, 0, pimg)
        #cv2.imwrite(dataDir + "boxtest/Hollow-" + name + ".jpg", pimg)

        xmlTooth = ET.SubElement(xmlToothList, "Tooth", Number = name + 'hollow')

        for i in range(4):
            x = points[i][0]
            y = imgShape[0] - points[i][1] - 1
            ET.SubElement(xmlTooth, 'P' + str(i), Y = str(y), X = str(x))

    xmlTree = ET.ElementTree(xmlRoot)
    xmlTree.write(xmlDir + imageTitle + '.xml')

    return

    
def findNeighborTeeth(imgsBoundary, mainTeethKey, imgShape):
    
    neighborTeethKeys = []
    teethKeys = [['18','17','16','15','14','13','12','11','21','22','23','24','25','26','27','28'],
                ['48','47','46','45','44','43','42','41','31','32','33','34','35','36','37','38']]
    isBot = 1 if mainTeethKey[0] > '2' else 0
   
    if mainTeethKey[1] == '9':
        if mainTeethKey[0] == '1' or mainTeethKey[0] == '4':
            mainTeethIdx = -1
        else:
            mainTeethIdx = 16
    else:
        for i in range(16):
            if teethKeys[isBot][i] == mainTeethKey:
                break
        mainTeethIdx = i
   
    leftEnd = 0
    rightEnd = imgShape[1] - 1
        
        # find left   
    tempIdx = mainTeethIdx - 1 
    
    while True:
        if tempIdx < 0:
            neighborTeethKeys.append(0)
            break
        tempKey = teethKeys[isBot][tempIdx]
        if tempKey in imgsBoundary:
            neighborTeethKeys.append(tempKey)
            leftEnd = imgsBoundary[tempKey][2]
            break
    
        tempIdx -= 1
    
        # find right
    tempIdx = mainTeethIdx + 1
    
    while True:
        if tempIdx > 15:
            neighborTeethKeys.append(0)
            break
        tempKey = teethKeys[isBot][tempIdx]
        if tempKey in imgsBoundary:
            neighborTeethKeys.append(tempKey)
            rightEnd = imgsBoundary[tempKey][3]
            break
    
        tempIdx += 1
    
    print(leftEnd, rightEnd, tempKey)
    # find up / down
    tempIdx = 0
    for tempIdx in range(16):
       
        tempKey = teethKeys[1-isBot][tempIdx]
        if tempKey not in imgsBoundary:
            continue

        tempLeft = imgsBoundary[tempKey][2]
        tempRight = imgsBoundary[tempKey][3]
        if tempLeft < leftEnd and tempRight < leftEnd:
            continue
        if tempLeft > rightEnd and tempRight > rightEnd:
            continue

        neighborTeethKeys.append(tempKey)

    return neighborTeethKeys

'''
def findCommonPoint(line1, line2, imgShape):
        
    if line1[0] == line1[2]:
        if line2[0] == line2[2]:
            if line1[0] == line2[0]:
                return (2, (0,0))
            else:
                return (0, (0,0))
        else:
            a2 = (line2[1]-line2[3]) / (line2[0]-line2[2])
            b2 = (line2[0]*line2[3] - line2[1]*line2[2]) / (line2[0]-line2[2])
            x = line1[0]
            y = a2 * x + b2
    else:
        a1 = (line1[1]-line1[3]) / (line1[0]-line1[2])
        b1 = (line1[0]*line1[3] - line1[1]*line1[2]) / (line1[0]-line1[2])

    if line2[0] == line2[2]:
        x = line2[0]
        y = a2 * x + b2  
    else:
        a2 = (line2[1]-line2[3]) / (line2[0]-line2[2])
        b2 = (line2[0]*line2[3] - line2[1]*line2[2]) / (line2[0]-line2[2])
        
        if a1 == a2:
            if b1 == b2:
                return (2, (0,0))
            else:
                return (0, (0,0))
        else:
            x = (b2 - b1) / (a1 - a2)
                y = (a1*b2 - a2*b1) / (a1 - a2)

    if x < 0 or y < 0 or x >= imgShape[1] or y >= imgShape[0]:
        return (0, (x,y))    

    return (1, (int(x),int(y)))
'''

def getLineCutImg(annotImg, criticalLine, angle, imgShape):

    lineCutImg = np.zeros(imgShape, dtype=np.uint8)
    mask = np.zeros((imgShape[0]+2,imgShape[1]+2), dtype=np.uint8)

    p1 = [0, 0]
    p2 = [0, imgShape[0]-1]
    p3 = [imgShape[1]-1, imgShape[0]-1]
    p4 = [imgShape[1]-1, 0]
    q1 = [criticalLine[0], criticalLine[1]]
    q2 = [criticalLine[2], criticalLine[3]]

    if angle < 90:
        pts = np.array([q1, q2, p4, p1, p2], np.int32)
        fillPoint = (1,1)
    elif angle < 180:
        pts = np.array([q1, q2, p1, p4, p3], np.int32)
        fillPoint = (imgShape[1]-2, 1)
    elif angle < 270:
        pts = np.array([q1, q2, p4, p3, p2], np.int32)
        fillPoint = (imgShape[1]-2, imgShape[0]-2)
    else:
        pts = np.array([q1, q2, p1, p2, p3], np.int32)
        fillPoint = (1, imgShape[0]-2)

    pts = pts.reshape((-1,1,2))
    cv2.polylines(lineCutImg, [pts], True, 255)
    cv2.floodFill(lineCutImg, mask, fillPoint, 255)
    intersectionImg = cv2.bitwise_and(annotImg, lineCutImg)

    return intersectionImg


def findCriticalLine(annotImg, topLeft, topRight, imgShape, inputAngle):

    criticalLine = []
    p1 = [0, 0]
    p2 = [0, imgShape[0]-1]
    p3 = [imgShape[1]-1, imgShape[0]-1]
    p4 = [imgShape[1]-1, 0]   
    teethArea = np.sum(annotImg == 255)
    angle = inputAngle % 180 # maybe random between 0 ~ 9?
    prev_q = [0,0,0,0]
    q1 = [0,0]
    q2 = [0,0]
    found = 0

    if angle < 90:
        rad = math.radians(angle)
        fillPoint = (1,1)        
    else:
        rad = math.radians(180-angle)
        fillPoint = (p4[0]-1,1)        
 
    tan = math.tan(rad)

    if tan == 0:
        q1[0] = topLeft[0]
        q1[1] = p2[1]        
    elif angle == 0:
        q1[0] = p4[0]
        q1[1] = topRight[1]
    elif angle < 90:
        q1[1] = int((topLeft[0]+1)/tan) + topLeft[1] - 1
        if q1[1] <= p2[1]:
            q1[0] = 0 
        else:
            q1[0] = int((q1[1]+1-p2[1])*tan) - 1
            q1[1] = p2[1]
    else:
        q1[1] = int((p4[0]-topRight[0]+1)/tan) + topRight[1] - 1
        if q1[1] <= p2[1]:
            q1[0] = p3[0]
        else:
            q1[0] = p3[0] - int((q1[1]+1-p3[1])*tan) + 1
            q1[1] = p2[1]


    while(q1[0] <= p3[0] and q1[0] >= 0):

        if tan == 0:
            q2[0] = q1[0]
            q2[1] = 0
        elif angle == 0:
            q2[0] = 0
            q2[1] = q1[1]
        elif angle < 90:
            q2[0] = int((q1[1]+1) * tan) + q1[0] - 1
            if q2[0] <= p3[0]:
                q2[1] = 0
            else:
                q2[1] = int((q2[0]+1-p4[0])/tan) - 1
                q2[0] = p4[0]
        else:
            q2[0] = q1[0] - int((q1[1]+1) * tan) + 1
            if q2[0] >= 0:
                q2[1] = 0
            else:
                q2[1] = int(-q2[0]/tan) -1
                q2[0] = 0

        lineCutImg = getLineCutImg(annotImg, [q1[0],q1[1],q2[0],q2[1]], angle, imgShape)
        lineCutArea = np.sum(lineCutImg == 255)
        lineCutIOU = lineCutArea / teethArea            

        if found == 0:
            if lineCutIOU >= 0.08:
                found = 1
                if inputAngle < 180:
                    criticalLine = [prev_q[0], prev_q[1], prev_q[2], prev_q[3]]
                    break
            prev_q = [q1[0], q1[1], q2[0], q2[1]]
        elif found == 1:
            if lineCutIOU >= 0.92:
                found = 2
                if inputAngle >= 180:
                    criticalLine = [q1[0], q1[1], q2[0], q2[1]]
                break
        else:
            break 

        if q1[1] < p2[1]:
            q1[1] += 2
            if q1[1] > p2[1]:
                q1[1] = p2[1]
        elif angle < 90:
            q1[0] += 2
        else:
            q1[0] -= 2


    return criticalLine
   

def findSurvivedKeys(annotImgs, survivedKeys, criticalLine, angle, imgShape):

    newSurvivedKeys = []

    for key in survivedKeys:

        if key == 0:
            continue
        teethImg = annotImgs[key]
        teethArea = np.sum(teethImg == 255)
        lineCutImg = getLineCutImg(teethImg, criticalLine, angle, imgShape)
        lineCutArea = np.sum(lineCutImg == 255)
        thisIOU = lineCutArea / teethArea
        
        if thisIOU >= 0.08:
            newSurvivedKeys.append(key)

    return newSurvivedKeys


def createBoxInsideLines(criticalLineList, angleList, mainBoundary, imgShape):

    h = mainBoundary[1] - mainBoundary[0]
    w = mainBoundary[3] - mainBoundary[2]
    
    rangeTop = int(mainBoundary[0] - h)
    rangeBot = int(mainBoundary[1] + h)
    rangeLeft = int(mainBoundary[2] - w)
    rangeRight = int(mainBoundary[3] + w)

    if rangeTop < 0:
        rangeTop = 0
    if rangeLeft < 0:
        rangeLeft = 0
    if rangeBot >= imgShape[0]:
        rangeBot = imgShape[0] - 1
    if rangeRight >= imgShape[1]:
        rangeRight = imgShape[1] - 1

    insideLinesImg = np.ones(imgShape, dtype=np.uint8) * 255
    
    for i in range(len(angleList)):
    
        line = criticalLineList[i]
        angle = angleList[i]
        insideLinesImg = getLineCutImg(insideLinesImg, line, angle, imgShape)
        

    n = 0
    boxPoints = []

    while True:

        print("creation trying...")
        i=0 
        while n < 4:
            if i > 999:
                return 0
            i+=1
            x = random.randrange(rangeLeft, rangeRight)
            y = random.randrange(rangeTop, rangeBot)
    
            if insideLinesImg[y, x] == 255:
                n += 1
                boxPoints.append([x,y])

        '''
        boxImg = np.zeros(imgShape, dtype=np.uint8)        
        boxImg = fillBox(boxImg, boxPoints)
        boxArea = np.sum(boxImg == 255)
        if boxArea >= minBoxSize:
            print("success!")
            break
        '''

        break
        
    return boxPoints


def createSingleOrNoneBox(annotImgs, imgsBoundary, mainTeethKey, neighborTeethKeys, imgShape, boxType):

    '''
    neighborTeethKeys = findNeighborTeeth(imgsBoundary, mainTeethKey, imgShape)

    if mainTeethKey in annotImgs:
        mainTeethImg = annotImgs[mainTeethKey]
        mainBoundary = imgsBoundary[mainTeethKey]
    else:
        if neighborTeethKeys[0] == 0:
            if neighborTeethKeys[1] == 0: 
                # impossible... but still
                return -1
            
            mainBoundary = imgsBoundary[neighborTeethKeys[1]]
            shiftAmount = mainBoundary[3] - mainBoundary[2] 
            mainBoundary[2] -= shiftAmount
            mainBoundary[3] -= shiftAmount
        elif neighborTeethKeys[1] == 0:
            mainBoundary = imgsBoundary[neighborTeethKeys[0]]
            shiftAmount = mainBoundary[3] - mainBoundary[2]
            mainBoundary[2] += shiftAmount
            mainBoundary[3] += shiftAmount
        else:
            mainBoundary = [0,0,0,0]
            mainBoundary[0] = int((imgsBoundary[neighborTeethKeys[0]][0]+imgsBoundary[neighborTeethKeys[1]][0])/2)
            mainBoundary[1] = int((imgsBoundary[neighborTeethKeys[0]][1]+imgsBoundary[neighborTeethKeys[1]][1])/2)
            mainBoundary[2] = imgsBoundary[neighborTeethKeys[0]][3]
            mainBoundary[3] = imgsBoundary[neighborTeethKeys[1]][2]

            if mainBoundary[2] > mainBoundary[3]:
                return -1 

        pts = np.array([[mainBoundary[2], int((mainBoundary[0]+mainBoundary[1])/2)],
                        [int((mainBoundary[2]+mainBoundary[3])/2), mainBoundary[1]],
                        [mainBoundary[3], int((mainBoundary[0]+mainBoundary[1])/2)],
                        [int((mainBoundary[2]+mainBoundary[3])/2), mainBoundary[0]]], np.int32)
        pts = pts.reshape((-1,1,2))
        fillPoint = (int((mainBoundary[2]+mainBoundary[3])/2), int((mainBoundary[0]+mainBoundary[1])/2))
        mainTeethImg = np.zeros(imgShape, dtype=np.uint8)
        mask = np.zeros((imgShape[0]+2,imgShape[1]+2), dtype=np.uint8)
        cv2.polylines(mainTeethImg, [pts], True, 255)
        cv2.floodFill(mainTeethImg, mask, fillPoint, 255)
    '''

    mainTeethImg = annotImgs[mainTeethKey]
    mainBoundary = imgsBoundary[mainTeethKey]
    mainTeethArea = np.sum(mainTeethImg == 255)
    criticalLineList = []
    angleList = []       
    survivedKeys = list(neighborTeethKeys)

    while len(survivedKeys) > 0:

        if survivedKeys[0] == 0:
            survivedKeys = survivedKeys[1:]
            continue

        tempTeethImg = annotImgs[survivedKeys[0]]
        tempBoundary = imgsBoundary[survivedKeys[0]]
        topLeft = (tempBoundary[2], tempBoundary[0])
        topRight = (tempBoundary[3], tempBoundary[0])
        
        print("add line for ", survivedKeys[0])
        angle = random.randrange(0,360)
        criticalLine = findCriticalLine(tempTeethImg, topLeft, topRight, imgShape, angle)
        
        checkMainArea = np.sum(getLineCutImg(mainTeethImg, criticalLine, angle, imgShape) == 255)
        if checkMainArea < mainTeethArea * 0.55:
            continue

        survivedKeys = survivedKeys[1:]
        survivedKeys = findSurvivedKeys(annotImgs, survivedKeys, criticalLine, angle, imgShape)
        
        criticalLineList.append(criticalLine)
        angleList.append(angle)


    if boxType == 'none':
        topLeft = (mainBoundary[2], mainBoundary[0])
        topRight = (mainBoundary[3], mainBoundary[0])
        angle = random.randrange(0,360)
        criticalLine = findCriticalLine(mainTeethImg, topLeft, topRight, imgShape, angle)

        angleList.append(angle)
        criticalLineList.append(criticalLine)

        minBoxSize = mainTeethArea / 10       
    else:
        minBoxSize = mainTeethArea / 10


    while True:

        boxPoints = createBoxInsideLines(criticalLineList, angleList, mainBoundary, imgShape)
        
        if boxPoints == 0:
            return 0            

        boxImg = np.zeros(imgShape, dtype=np.uint8)
        boxImg = fillBox(boxImg, boxPoints)
        boxArea = np.sum(boxImg == 255)

        if boxArea < minBoxSize:
            continue

        if boxType == 'none':
            break
        
        intersectionImg = cv2.bitwise_and(boxImg, mainTeethImg)
        intersectionArea = np.sum(intersectionImg == 255)
        thisIOU = intersectionArea / mainTeethArea

        if thisIOU >= 0.08:
            break

    print("success!")

    return boxPoints


def createDoubleBox(annotImgs, imgsBoundary, mainTeethKey, imgShape, boxType):

    if boxType == 'double':
        findNum = 1
    else:
        findNum = 2

    mainTeethImg = annotImgs[mainTeethKey]
    mainBoundary = imgsBoundary[mainTeethKey]
    mainTeethArea = np.sum(mainTeethImg == 255)
    main_h = mainBoundary[1] - mainBoundary[0]
    main_w = mainBoundary[3] - mainBoundary[2]

    boxPoints = []

    print("area = ", mainTeethArea)

    while True:

        rand_w = random.randrange(int(main_w), int(3 * main_w))
        rand_h = random.randrange(int(main_h), int(1.8 * main_h))
        rand_y = (mainBoundary[0]+mainBoundary[1])/2 + random.randrange(int(-0.2*main_h),int(0.2*main_h))
        rand_x = (mainBoundary[2]+mainBoundary[3])/2 + random.randrange(int(-0.2*main_w),int(0.2*main_w))

        box_x1 = int(rand_x - rand_w/2)
        box_x2 = int(rand_x + rand_w/2)
        box_y1 = int(rand_y - rand_h/2)
        box_y2 = int(rand_y + rand_h/2)


        if box_x1 < 0:
            box_x1 = 0 
        if box_y1 < 0:
            box_y1 = 0 
        if box_x2 >= imgShape[1]:
            box_x2 = imgShape[1]-1 
        if box_y2 >= imgShape[0]:
            box_x2 = imgShape[0]-1 

        for i in range(50):

            r1 = random.randrange(box_x1, box_x2)
            r2 = random.randrange(box_x1, box_x2)
            r3 = random.randrange(box_y1, box_y2)
            r4 = random.randrange(box_y1, box_y2)

            boxPoints = [[r1,box_y1],[box_x2,r3],[r2,box_y2],[box_x1,r4]]
            pts = np.array(boxPoints, np.int32)
            pts = pts.reshape((-1,1,2))
            fillPoint = (int((box_x1+box_x2)/2),int((r3+r4)/2))
            
            boxImg = np.zeros(imgShape, dtype=np.uint8)
            mask = np.zeros((imgShape[0]+2,imgShape[1]+2), dtype=np.uint8)
            cv2.polylines(boxImg, [pts], True, 255)
            cv2.floodFill(boxImg, mask, fillPoint, 255)
            
            intersectionImg = cv2.bitwise_and(mainTeethImg, boxImg)
            intersectionArea = np.sum(intersectionImg == 255)
            mainIOU = intersectionArea / mainTeethArea
            if mainIOU < 0.08:
                print("try again...1")
                continue

            found = 0 
            for name, annotImg in annotImgs.items():

                if name == mainTeethKey:
                    continue

                intersectionImg = cv2.bitwise_and(annotImg, boxImg)
                intersectionArea = np.sum(intersectionImg == 255)
                annotArea = np.sum(annotImg == 255)
                thisIOU = intersectionArea / annotArea

                if thisIOU >= 0.08:
                    found += 1

                if found == findNum:
                    break

            if found == findNum:
                break        
    
        if found == findNum:
            break
    
        print("try again...2")

    return boxPoints


def fillBox(img, pts):

    l = len(pts)
    for i in range(3):
        cv2.fillPoly(img, [np.array([pts[i], pts[(i+1)%l], pts[(i+2)%l]])], 255)

    return img


######################
### Build CSV File ###
######################

def buildCsvFile():

    cols = ['Image.Title', 'Pano.File', 'Xml.File', 'Annot.File', 'Train.Val']
    rows = []

    return


## MAIN ###
if __name__ == '__main__':
    __main__()

