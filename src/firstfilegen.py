import os
import sys
import pandas as pd

def __main__():

    if len(sys.argv) < 2:
        print('need to input mode: semi or not')
        return

    semiOrNot = str(sys.argv[1])
    makeFirstFile(semiOrNot)
    return

def makeFirstFile(semiOrNot):
    panoDir = '../data/rawdata/panoImg/'
    psdDir = '../data/rawdata/psdFile/'
    xmlDir =  '../data/rawdata/xmlFile/'

    outFileName = '../data/metadata/DataSet.csv'
    if semiOrNot == 'semi':
        outFileName = '../data/metadata/SemiSet.csv'

    valNames = ['T1-Pano-042', 'T1-Pano-062', 'T1-Pano-081', 'T1-Pano-107', 'T1-Pano-111', 'T1-Pano-126', 'T1-Pano-136']
    cols = ['Image.Title', 'Pano.File', 'Xml.File', 'Annot.File', 'Train.Val']
    rows = []

    for subdir, dirs, files in os.walk(panoDir):
        for fileName in files:
            print(fileName)
            name = fileName[:-4]
            panoName = panoDir + fileName
            psdName = psdDir + name + '.psd'
            xmlName = xmlDir + name + '.xml'
            trainVal = 'val' if (name in valNames) else 'train'
            if not os.path.exists(xmlName):
                print('error: xml does not exist for {}'.format(name))
                continue
            if not os.path.exists(psdName):
                if (semiOrNot == 'semi'):
                    psdName = -1
                    trainVal = 'semi'
                else:
                    continue
            rows.append([name, panoName, xmlName, psdName, trainVal])

    outputDf = pd.DataFrame(rows, columns=cols)
    outputDf.to_csv(outFileName)

    return

if __name__ == '__main__':
    __main__()
