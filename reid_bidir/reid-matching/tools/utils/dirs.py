import os
from os.path import join as opj
import cv2
import numpy as np

BBOX_B = 10/15


class dir():
    def __init__(self):
        dirs = {}
        dirs_path = "./dirs"
        cams = os.listdir(dirs_path)
        for cam in cams:
            cid = int(cam[-3:])
            dirs[cid] = {}
            for trackzone in os.listdir(opj(dirs_path, cam)):
                trackdir_img = cv2.imread(opj(dirs_path, cam, trackzone))
                dirname=trackzone.split(".")[0]
                dirs[cid][dirname] = trackdir_img

        self.dirs = dirs
        self.current_cam = 0

    def set_cam(self, cam):
        self.current_cam = cam

    def get_dir(self,zs,ze,bbox):
        ''' 角度 R	G	B  return
            右前 255 0 255 1
            右侧 0  255	0  2
            右后 255 0  0  3
            左前 0  0  255 -1
            左侧 0  255 255 -2
            左后 255 255 0 -3
            '''
        stoestr=str(zs)+"to"+str(ze)
        if stoestr not in self.dirs[self.current_cam]:
            return 0
        img = self.dirs[self.current_cam][stoestr]
        cx=int((bbox[0]+bbox[2])/2)
        cy=int((bbox[1]+bbox[3])/2)
        pix = img[cy, cx, :]
        dirnum=0
        # opencv顺序BGR
        if np.sum(np.abs(pix -np.array([255,0,255])))<5:
            dirnum= 1
        if np.sum(np.abs(pix -np.array([0,255,0])))<5:
            dirnum= 2
        if np.sum(np.abs(pix -np.array([0,0,255])))<5:
            dirnum= 3
        if np.sum(np.abs(pix -np.array([255,0,0])))<5:
            dirnum= -1
        if np.sum(np.abs(pix -np.array([255,255,0])))<5:
            dirnum= -2
        if np.sum(np.abs(pix -np.array([0,255,255])))<5:
            dirnum= -3
        return dirnum

    def tag_dir(self, mot_dic):
        for tracklet in mot_dic:
            tracklet_dict = mot_dic[tracklet]
            frame_list = list(tracklet_dict.keys())
            frame_list.sort()
            zone_list = [tracklet_dict[f]['zone'] for f in frame_list]
            bbox_list = [tracklet_dict[f]['bbox'] for f in frame_list]
            zs = zone_list[0]
            ze = zone_list[-1]
            for f in frame_list:
                bbox=tracklet_dict[f]['bbox']
                mot_dic[tracklet][f]["dir"]=self.get_dir(zs,ze,bbox)

        print(f"Dirs tagged")
        return mot_dic
