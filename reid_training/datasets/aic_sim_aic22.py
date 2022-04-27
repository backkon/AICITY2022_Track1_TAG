# encoding: utf-8

import glob
import re
import xml.dom.minidom as XD
import os.path as osp
from .bases import BaseImageDataset
import os
import logging

logger = logging.getLogger("reid_baseline.train")

class AIC_SIM_AIC22(BaseImageDataset):
    dataset_dir = 'AIC22/AIC22_Track2_ReID'
    sim_dataset_dir = 'AIC21/AIC21_Track2_ReID_Simulation'

    def __init__(self, root='../data', verbose=True,crop_test=False, **kwargs):
        super(AIC_SIM_AIC22, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.sim_dataset_dir = osp.join(root, self.sim_dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)

        self.train = train

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.sim_train_dir = osp.join(self.sim_dataset_dir, 'sys_image_train')
        sim_train = self._process_sim(self.sim_train_dir, begin_id= self.num_train_pids, relabel=True)
        self.train = self.train + sim_train
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        self.query = self.train
        self.gallery = self.train
        

        if verbose:
            logger.info("=> AIC_SIM loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, relabel=False, if_track=False):
        xml_dir = osp.join(self.dataset_dir, 'train_label.xml')
        with open(xml_dir, "r", encoding="utf-8") as f:
            datasource = f.read()
            f.close()
        info = XD.parseString(datasource).documentElement.getElementsByTagName('Item')

        pid_container = set()
        for element in range(len(info)):
            pid = int(info[element].getAttribute('vehicleID'))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []

        for element in range(len(info)):
            pid, camid = map(int, [info[element].getAttribute('vehicleID'), info[element].getAttribute('cameraID')[1:]])
            image_name = str(info[element].getAttribute('imageName'))
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((osp.join(dir_path, image_name), pid, camid, -1))
        return dataset

    def _process_sim(self,dir_path,begin_id, relabel=False):
        img_path = os.listdir(dir_path)
        train = []
        pid_container = set()
        for img in img_path:
            pid_container.add(int(img[:5]))

        pid2label = {pid: (label+begin_id) for label, pid in enumerate(pid_container)}
        for img in sorted(img_path):
            camid = int(img[7:10])
            pid = int(img[:5])
            if relabel: pid = pid2label[pid]
            train.append((osp.join(dir_path, img), pid, camid,-1))
        return train

if __name__ == '__main__':
    aic = AIC(root='/workspace/Projects/AIC22/AICITY2021_Track2_DMT-main/datasets')
