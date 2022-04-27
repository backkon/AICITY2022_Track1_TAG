"""
@version: 0.1
@author: queqi
@discription:
@time: 2022/4/18 12:19 上午

exmaple:

"""
"""
@version: 0.1
@author: queqi
@discription:
@time: 2022/3/25 11:36 上午

exmaple:

"""
from xml.dom.minidom import Document
import os
import shutil

def trainGen(rootPath, savePath):

    imageTrainPath = savePath + '/image_train'
    if not os.path.exists(imageTrainPath):
        os.makedirs(imageTrainPath)

    doc = Document()
    root = doc.createElement('TrainingImages')
    root.setAttribute('Version', '1.0')
    doc.appendChild(root)
    itemsInfo = doc.createElement('Items')
    itemsInfo.setAttribute('number', '')
    root.appendChild(itemsInfo)
    imageNum = 0
    sceneList = os.listdir(rootPath)
    sceneList = [sn for sn in sceneList if sn != '.DS_Store']
    for sceneName in sceneList:
        print('start to process {}'.format(sceneName))
        IDList = os.listdir(rootPath + '/' + sceneName)
        IDList = [idn for idn in IDList if idn != '.DS_Store']
        for IDName in IDList:
            imageList = os.listdir(rootPath + '/' + sceneName + '/' + IDName)
            imageList = [imn for imn in imageList if imn != '.DS_Store']
            for imageName in imageList:
                shutil.copyfile(rootPath + '/' + sceneName + '/' + IDName + '/' + imageName, imageTrainPath + '/' + imageName)
                cameraName = imageName.split('_')[0]
                nodeManager = doc.createElement('Item')
                nodeManager.setAttribute('imageName', imageName)
                nodeManager.setAttribute('vehicleID', '{:04d}'.format(int(IDName)))
                nodeManager.setAttribute('cameraID', cameraName)
                nodeManager.setAttribute('sceneID', sceneName.split('_')[0])
                itemsInfo.appendChild(nodeManager)
                imageNum += 1
        print('{} done'.format(sceneName))

    itemsInfo.setAttribute('number', str(imageNum))

    fp = open(savePath + '/train_label.xml', 'w')
    doc.writexml(fp, indent="  ", addindent='\t', newl="\n", encoding='gb2312')
    fp.close()

if __name__ == '__main__':
    rootPath = './AIC22/AIC22_ReID_DATA'
    savePath = './AIC22/AIC22_Track1_ReID'
    # rootPath = '/Users/queqi/Documents/项目目录/AICity/datasets/AIC22_ReID_DATA_tmp'
    # savePath = '/Users/queqi/Documents/项目目录/AICity/datasets/AIC22_Track1_ReID_tmp'
    if os.path.exists(savePath):
        os.makedirs(savePath)
    trainGen(rootPath, savePath)