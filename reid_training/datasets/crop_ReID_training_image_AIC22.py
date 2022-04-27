"""
@version: 0.1
@author: queqi
@discription:
@time: 2022/4/17 11:32 下午

exmaple:

"""
import os
import cv2
import tqdm


def main(dataRootPath, savePath):
    dataNameList = ['train', 'validation']
    for dataName in dataNameList:
        dataPath = dataRootPath + '/' + dataName
        SList = os.listdir(dataPath)
        SList = [s for s in SList if s != '.DS_Store']
        for SName in SList:
            SPath = dataPath + '/' + SName
            CList = os.listdir(SPath)
            CList = [c for c in CList if c != '.DS_Store']
            for CName in CList:
                print('start to process {} {}'.format(SName, CName))
                CPath = SPath + '/' + CName
                videoPathName = CPath + '/vdo.avi'
                gtPathName = CPath + '/gt/gt.txt'
                cap = cv2.VideoCapture(videoPathName)
                with open(gtPathName, 'r') as gtt:
                    for line in gtt:
                        words = line.split(',')
                        frameOrd = int(words[0]) - 1
                        IDName = int(words[1])
                        print('\rframe: {:010d} ID:{:04}'.format(frameOrd, IDName), end='')
                        BBox = [int(words[2]), int(words[3]), int(words[4]), int(words[5])]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frameOrd)
                        res, frame = cap.read()
                        savePathNew = savePath + '/{}_imgs/{}'.format(SName, IDName)
                        if not os.path.exists(savePathNew):
                            os.makedirs(savePathNew)
                        saveName = 'c0{}_{}_{}.jpg'.format(int(CName.replace('c', '')), IDName, frameOrd + 1)
                        cropImg = frame[BBox[1]:BBox[1]+BBox[3], BBox[0]:BBox[0]+BBox[2], :]
                        cv2.imwrite(savePathNew + '/' + saveName, cropImg)
                print('\n{} {} done'.format(SName, CName))







if __name__ == '__main__':
    dataRootPath = './AIC22/AIC22_Track1_MTMC_Tracking'
    savePath = './AIC22/AIC22_ReID_DATA'
    # dataRootPath = '/Users/queqi/Documents/项目目录/AICity/datasets/AIC22_Track1_MTMC_Tracking'
    # savePath = '/Users/queqi/Documents/项目目录/AICity/datasets/AIC22_ReID_DATA_tmp'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    main(dataRootPath, savePath)