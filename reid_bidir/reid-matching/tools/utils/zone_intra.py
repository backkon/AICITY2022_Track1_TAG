import os
from os.path import join as opj
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering,DBSCAN
from torch import zero_

BBOX_B = 10/15

class zone():
    def __init__(self):
        # 0: b 1: g 3: r 123:w
        # w r 非高速
        # b g 高速
        zones = {}
        zone_path = "./zone"
        for img_name in os.listdir(zone_path):
            camnum = int(img_name.split('.')[0][-3:])
            zone_img = cv2.imread(opj(zone_path, img_name))
            zones[camnum] = zone_img
        self.zones = zones

        sub_zones = {}
        zone_path = "./zone_bidir"
        for img_name in os.listdir(zone_path):
            camnum = int(img_name.split('.')[0][-3:])
            zone_img = cv2.imread(opj(zone_path, img_name))
            sub_zones[camnum] = zone_img
        self.sub_zones = sub_zones

        self.current_cam = 0

    def set_cam(self,cam):
        self.current_cam = cam

    def get_zone(self,bbox):
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)

        pix = self.zones[self.current_cam][cy, cx, :]
        zone_num = 0
        if pix[0] > 50 and pix[1] > 50 and pix[2] > 50:  # w
            zone_num = 1
        if pix[0] < 50 and pix[1] < 50 and pix[2] > 50:  # r
            zone_num = 2
        if pix[0] < 50 and pix[1] > 50 and pix[2] < 50:  # g
            zone_num = 3
        if pix[0] > 50 and pix[1] < 50 and pix[2] < 50:  # b
            zone_num = 4

        pix = self.sub_zones[self.current_cam][cy, cx, :]
        sub_zone_num = -1
        if pix[0] < 50 and pix[1] < 50 and pix[2] > 50:  # r   # 5 一定是从cid小的相机进来
            sub_zone_num = 5
        if pix[0] < 50 and pix[1] > 50 and pix[2] < 50:  # g   # 6 一定是去向cid小的相机
            sub_zone_num = 6
        if pix[0] > 50 and pix[1] < 50 and pix[2] < 50:  # b   # 7 一定是去向cid大的相机
            sub_zone_num = 7
        if pix[0] > 50 and pix[1] > 50 and pix[2] > 50:  # w   # 8 一定是从cid大的相机进来
            sub_zone_num = 8

        return zone_num, sub_zone_num


    def is_ignore(self,zone_list,frame_list, cid):
        # 0 不在任何路口 1 白色 2 红色 3 绿色 4 蓝色
        zs, ze = zone_list[0], zone_list[-1]
        fs, fe = frame_list[0],frame_list[-1]
        if zs == ze:
            # 如果一直在一个区域里，排除
            if ze in [1,2]:
                return 2
            if zs!=0 and 0 in zone_list:
                return 0
            if fe-fs>1500:
                return 2
            if fs<2:
                if cid in [45]:
                    if ze in [3,4]:
                        return 1
                    else:
                        return 2
            if fe > 1999:
                if cid in [41]:
                    if ze not in [3]:
                        return 2
                    else:
                        return 0
            if fs<2 or fe>1999:
              if ze in [3,4]:
                return 0
            if ze in [3,4]:
                return 1
            return 2
        else:
            # 如果区域发生变化
            if cid in [41, 42, 43, 44, 45, 46]:
                # 如果从支路进支路出，排除
                if zs == 1 and ze == 2:
                    return 2
                if zs == 2 and ze == 1:
                    return 2
            if cid in [41]:
                # 在41相机，车辆没有进出42相机
                if (zs in [1, 2]) and ze == 4:
                    return 2
                if zs == 4 and (ze in [1, 2]):
                    return 2
            if cid in [46]:
                # 在46相机，车辆没有进出45相机
                if (zs in [1, 2]) and ze == 3:
                    return 2
                if zs == 3 and (ze in [1, 2]):
                    return 2
            return 0

    def filter_mot(self,mot_list, cid):
        new_mot_list = dict()
        sub_mot_list = dict()
        for tracklet in mot_list:
            # if cid == 45 and tracklet==207:
            #     print(tracklet)
            tracklet_dict = mot_list[tracklet]
            frame_list = list(tracklet_dict.keys())
            frame_list.sort()
            # zone_list = []
            sub_zone_dict = {-1:0,5:0,6:0,7:0,8:0}
            for f in frame_list:
                sub_zone_dict[tracklet_dict[f]['sub_zone']] += 1

            if sub_zone_dict[5] > 1 or sub_zone_dict[6] > 1 or sub_zone_dict[7] > 1 or sub_zone_dict[8] > 1:
                for f in frame_list:  
                    if sub_zone_dict[5] > sub_zone_dict[6] and sub_zone_dict[6] > 0 and tracklet_dict[f]['sub_zone'] == 6:
                        tracklet_dict[f]['sub_zone'] = 5
                    elif sub_zone_dict[6] >= sub_zone_dict[5] and sub_zone_dict[5] > 0 and tracklet_dict[f]['sub_zone'] == 5:
                        tracklet_dict[f]['sub_zone'] = 6
                    if sub_zone_dict[7] > sub_zone_dict[8] and sub_zone_dict[8] > 0 and tracklet_dict[f]['sub_zone'] == 8:
                        tracklet_dict[f]['sub_zone'] = 7
                    elif sub_zone_dict[8] >= sub_zone_dict[7] and sub_zone_dict[7] > 0 and tracklet_dict[f]['sub_zone'] == 7:
                        tracklet_dict[f]['sub_zone'] = 8
                        
                zones_sum = len(frame_list)
                if sub_zone_dict[5] == zones_sum and frame_list[-1] <= 1999:
                    continue
                elif sub_zone_dict[8] == zones_sum and frame_list[-1] <= 1999:
                    continue
                elif sub_zone_dict[6] == zones_sum and frame_list[0] >= 2:
                    continue
                elif sub_zone_dict[7] == zones_sum and frame_list[0] >= 2:
                    if zones_sum < 3:
                        continue                
                            
                new_mot_list[tracklet] = tracklet_dict  

                          
            # if self.is_ignore(zone_list,frame_list, cid)==0:
            #     new_mot_list[tracklet] = tracklet_dict
            # if self.is_ignore(zone_list,frame_list, cid)==1:
            #     sub_mot_list[tracklet] = tracklet_dict
        return new_mot_list

    def filter_bbox(self,mot_list,cid):
        new_mot_list = dict()
        yh = self.zones[cid].shape[0]
        for tracklet in mot_list:
            # if tracklet==1812:
            #     print(tracklet)
            tracklet_dict = mot_list[tracklet]
            frame_list = list(tracklet_dict.keys())
            frame_list.sort()
            bbox_list = []
            for f in frame_list:
                bbox_list.append(tracklet_dict[f]['bbox'])
            bbox_x = [b[0] for b in bbox_list]
            bbox_y = [b[1] for b in bbox_list]
            bbox_w = [b[2]-b[0] for b in bbox_list]
            bbox_h = [b[3]-b[1] for b in bbox_list]
            new_frame_list = list()
            if 0 in bbox_x or 0 in bbox_y:
                b0 = [i for i, f in enumerate(frame_list) if bbox_x[i]<5 or bbox_y[i]+bbox_h[i]>yh-5]
                if len(b0)==len(frame_list):
                    if cid in [41,42,44,45,46]:
                        continue
                    max_w = max(bbox_w)
                    max_h = max(bbox_h)
                    for i,f in enumerate(frame_list):
                        if bbox_w[i] > max_w * BBOX_B and bbox_h[i] > max_h * BBOX_B:
                            new_frame_list.append(f)
                else:
                    l_i,r_i = 0,len(frame_list)-1
                    if b0[0]==0:
                        for i in range(len(b0)-1):
                            if b0[i]+1==b0[i+1]:
                                l_i=b0[i+1]
                            else:
                                break
                    if b0[-1]==len(frame_list)-1:
                        for i in range(len(b0) - 1):
                            i = len(b0)-1-i
                            if b0[i]-1==b0[i-1]:
                                r_i=b0[i-1]
                            else:
                                break

                    max_lw, max_lh = bbox_w[l_i], bbox_h[l_i]
                    max_rw, max_rh = bbox_w[r_i], bbox_h[r_i]
                    for i,f in enumerate(frame_list):
                        if i < l_i:
                            if bbox_w[i] > max_lw * BBOX_B and bbox_h[i] > max_lh * BBOX_B:
                                new_frame_list.append(f)
                        elif i > r_i:
                            if bbox_w[i] > max_rw * BBOX_B and bbox_h[i] > max_rh * BBOX_B:
                                new_frame_list.append(f)
                        else:
                            new_frame_list.append(f)
                new_tracklet_dict=dict()
                for f in new_frame_list:
                    new_tracklet_dict[f]=tracklet_dict[f]
                new_mot_list[tracklet] = new_tracklet_dict
            else:
                new_mot_list[tracklet] = tracklet_dict
        return new_mot_list

    def break_mot(self,mot_list,cid):
        # if not cid in [41,44,45,46]:
        #     return mot_list
        new_mot_list = dict()
        new_num_tracklets = max(mot_list)+1
        for tracklet in mot_list:
            tracklet_dict = mot_list[tracklet]
            frame_list = list(tracklet_dict.keys())
            frame_list.sort()
            zone_list = []
            back_tracklet = False
            new_zone_f = 0
            pre_frame = frame_list[0]
            time_break = False
            # if tracklet == 714:
            #     print(tracklet)
            for f in frame_list:
                if f-pre_frame>100:
                    if cid in [44,45]:
                        time_break = True
                        break
                if not cid in [41, 44, 45, 46]:
                    break
                pre_frame=f
                new_zone = tracklet_dict[f]['zone']
                if len(zone_list)>0 and zone_list[-1] == new_zone:
                    continue
                if new_zone_f>1:
                    if len(zone_list) > 1 and new_zone in zone_list:
                        back_tracklet = True
                    zone_list.append(new_zone)
                    new_zone_f=0
                else:
                    new_zone_f+=1
            if back_tracklet:
                new_tracklet_dict = dict()
                pre_bbox = -1
                pre_arrow = 0
                have_break = False
                for f in frame_list:
                    now_bbox = tracklet_dict[f]['bbox']
                    if pre_bbox == -1:
                        pre_bbox = now_bbox
                    now_arrow = now_bbox[0] - pre_bbox[0]
                    if pre_arrow*now_arrow<0 and len(new_tracklet_dict)>15 and not have_break:
                        new_mot_list[tracklet] = new_tracklet_dict
                        new_tracklet_dict = dict()
                        have_break = True
                    if have_break:
                        tracklet_dict[f]['id'] = new_num_tracklets
                    new_tracklet_dict[f] = tracklet_dict[f]
                    pre_bbox,pre_arrow = now_bbox,now_arrow
                if have_break:
                    new_mot_list[new_num_tracklets] = new_tracklet_dict
                    new_num_tracklets += 1
                else:
                    new_mot_list[tracklet] = new_tracklet_dict
            elif time_break:
                new_tracklet_dict = dict()
                have_break = False
                pre_frame = frame_list[0]
                for f in frame_list:
                    if f-pre_frame>100:
                        new_mot_list[tracklet] = new_tracklet_dict
                        new_tracklet_dict = dict()
                        have_break = True
                    new_tracklet_dict[f] = tracklet_dict[f]
                    pre_frame=f
                if have_break:
                    new_mot_list[new_num_tracklets] = new_tracklet_dict
                    new_num_tracklets += 1
                else:
                    new_mot_list[tracklet] = new_tracklet_dict
            else:
                new_mot_list[tracklet] = tracklet_dict
        print("old:{} new:{}".format(len(mot_list),len(new_mot_list)))
        return new_mot_list

    def intra_matching(self,mot_list,sub_mot_list):
        sub_zone_dict = dict()
        new_mot_list = dict()
        new_mot_list,new_sub_mot_list = self.do_intra_matching2(mot_list,sub_mot_list)
        # for tracklet in sub_mot_list:
        #     frame_list = list(sub_mot_list[tracklet])
        #     frame_zone = sub_mot_list[tracklet][frame_list[0]]['zone']
        #     if frame_zone in sub_zone_dict:
        #         sub_zone_dict[frame_zone][tracklet]=sub_mot_list[tracklet]
        #     else:
        #         sub_zone_dict[frame_zone]={tracklet:sub_mot_list[tracklet]}
        # for tracklet in mot_list:
        #     frame_list = list(mot_list[tracklet])
        #     frame_list.sort()
        #     frame_zone = mot_list[tracklet][frame_list[0]]['zone']
        #     if frame_zone in sub_zone_dict:
        #         sub_zone_dict[frame_zone][tracklet]=mot_list[tracklet]
        #     frame_zone = mot_list[tracklet][frame_list[-1]]['zone']
        #     if frame_zone in sub_zone_dict:
        #         sub_zone_dict[frame_zone][tracklet]=mot_list[tracklet]
        # mot_zone_dict = dict()
        # for sub_zone in sub_zone_dict:
        #     mot_zone_list = sub_zone_dict[sub_zone]
        #     mot_zone_list = self.do_intra_matching(mot_zone_list, sub_zone)
        #     mot_zone_list = self.do_intra_matching(mot_zone_list, sub_zone)
        #     mot_zone_list = self.do_intra_matching(mot_zone_list, sub_zone)
        #     mot_zone_list = self.do_intra_matching(mot_zone_list, sub_zone)
        #     mot_zone_dict[sub_zone] = mot_zone_list

        return new_mot_list

    def do_intra_matching2(self,mot_list,sub_list):
        new_zone_dict = dict()
        def get_trac_info(tracklet1):
            t1_f = list(tracklet1)
            t1_f.sort()
            t1_fs = t1_f[0]
            t1_fe = t1_f[-1]
            t1_zs = tracklet1[t1_fs]['zone']
            t1_ze = tracklet1[t1_fe]['zone']
            t1_boxs = tracklet1[t1_fs]['bbox']
            t1_boxe = tracklet1[t1_fe]['bbox']
            t1_boxs = [(t1_boxs[2] + t1_boxs[0]) / 2, (t1_boxs[3] + t1_boxs[1]) / 2]
            t1_boxe = [(t1_boxe[2] + t1_boxe[0]) / 2, (t1_boxe[3] + t1_boxe[1]) / 2]
            return t1_fs,t1_fe,t1_zs,t1_ze,t1_boxs,t1_boxe

        for t1id in sub_list:
            tracklet1 = sub_list[t1id]
            if tracklet1 == -1:
                continue
            t1_fs,t1_fe,t1_zs,t1_ze,t1_boxs,t1_boxe = get_trac_info(tracklet1)
            sim_dict = dict()
            for t2id in mot_list:
                tracklet2 = mot_list[t2id]
                t2_fs,t2_fe,t2_zs,t2_ze,t2_boxs,t2_boxe = get_trac_info(tracklet2)
                if t1_ze == t2_zs:
                    if abs(t2_fs-t1_fe)<5 and abs(t2_boxe[0]-t1_boxs[0])<50 and abs(t2_boxe[1]-t1_boxs[1])<50:
                        t1_feat = tracklet1[t1_fe]['feat']
                        t2_feat = tracklet2[t2_fs]['feat']
                        sim_dict[t2id] = np.matmul(t1_feat,t2_feat)
                if t1_zs == t2_ze:
                    if abs(t2_fe-t1_fs)<5 and abs(t2_boxs[0]-t1_boxe[0])<50 and abs(t2_boxs[1]-t1_boxe[1])<50:
                        t1_feat = tracklet1[t1_fs]['feat']
                        t2_feat = tracklet2[t2_fe]['feat']
                        sim_dict[t2id] = np.matmul(t1_feat,t2_feat)
            if len(sim_dict)>0:
                max_sim = 0
                max_id = 0
                for t2id in sim_dict:
                    if sim_dict[t2id]>max_sim:
                        sim_dict[t2id]=max_sim
                        max_id = t2id
                if max_sim>0.5:
                    t2 = mot_list[max_id]
                    for t1f in tracklet1:
                        if t1f not in t2:
                            tracklet1[t1f]['id'] = max_id
                            t2[t1f] = tracklet1[t1f]
                    mot_list[max_id] = t2
                    sub_list[t1id] = -1
        return mot_list,sub_list

    def do_intra_matching(self,sub_zone_dict,sub_zone):
        new_zone_dict = dict()
        id_list = list(sub_zone_dict)
        id2index = dict()
        for index,id in enumerate(id_list):
            id2index[id] = index
        def get_trac_info(tracklet1):
            t1_f = list(tracklet1)
            t1_f.sort()
            t1_fs = t1_f[0]
            t1_fe = t1_f[-1]
            t1_zs = tracklet1[t1_fs]['zone']
            t1_ze = tracklet1[t1_fe]['zone']
            t1_boxs = tracklet1[t1_fs]['bbox']
            t1_boxe = tracklet1[t1_fe]['bbox']
            t1_boxs = [(t1_boxs[2] + t1_boxs[0]) / 2, (t1_boxs[3] + t1_boxs[1]) / 2]
            t1_boxe = [(t1_boxe[2] + t1_boxe[0]) / 2, (t1_boxe[3] + t1_boxe[1]) / 2]
            return t1_fs,t1_fe,t1_zs,t1_ze,t1_boxs,t1_boxe

        sim_matrix = np.zeros([len(id_list),len(id_list)])

        for t1id in sub_zone_dict:
            tracklet1 = sub_zone_dict[t1id]
            t1_fs,t1_fe,t1_zs,t1_ze,t1_boxs,t1_boxe = get_trac_info(tracklet1)
            t1_feat = tracklet1[t1_fe]['feat']
            for t2id in sub_zone_dict:
                if t1id == t2id:
                    continue
                tracklet2 = sub_zone_dict[t2id]
                t2_fs,t2_fe,t2_zs,t2_ze,t2_boxs,t2_boxe = get_trac_info(tracklet2)
                if t1_zs!=t1_ze and t2_ze!=t2_zs or t1_fe>t2_fs:
                    continue
                if abs(t1_boxe[0]-t2_boxs[0])>50 or abs(t1_boxe[1]-t2_boxs[1])>50:
                    continue
                if t2_fs-t1_fe>5:
                    continue
                t2_feat = tracklet2[t2_fs]['feat']
                sim_matrix[id2index[t1id],id2index[t2id]] = np.matmul(t1_feat, t2_feat)
                sim_matrix[id2index[t2id],id2index[t1id]] = np.matmul(t1_feat, t2_feat)
        sim_matrix = 1-sim_matrix
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.7,
                                                 affinity='precomputed',
                                                 linkage='complete').fit_predict(sim_matrix)
        new_zone_dict = dict()
        label2id = dict()
        for index,label in enumerate(cluster_labels):
            tracklet = sub_zone_dict[id_list[index]]
            if label not in label2id:
                new_id = tracklet[list(tracklet)[0]]
                new_tracklet = dict()
            else:
                new_id = label2id[label]
                new_tracklet = new_zone_dict[label2id[label]]
            for tf in tracklet:
                tracklet[tf]['id'] = new_id
                new_tracklet[tf]=tracklet[tf]
            new_zone_dict[label] = new_tracklet

        return new_zone_dict