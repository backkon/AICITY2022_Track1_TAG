
import os
from os.path import join as opj
import numpy as np
import pickle
from utils.zone_intra import zone
from utils.dirs import dir
import sys

sys.path.append('../../../')
from config import cfg
# tid tracklet id
# cid camera id


def parse_pt(pt_file, zones):
    if not os.path.isfile(pt_file):
        return dict()
    with open(pt_file, 'rb') as f:
        lines = pickle.load(f)
    mot_list = dict()
    for line in lines:
        fid = int(lines[line]['frame'][3:])
        tid = lines[line]['id']
        bbox = list(map(lambda x: int(float(x)), lines[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = lines[line]
        out_dict['zone'], out_dict['sub_zone'] = zones.get_zone(bbox)
        mot_list[tid][fid] = out_dict
    return mot_list


def parse_bias(timestamp_dir, scene_name):
    cid_bias = dict()
    for sname in scene_name:
        with open(opj(timestamp_dir, sname + '.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                cid = int(line[0][2:])
                bias = float(line[1])
                if cid not in cid_bias:
                    cid_bias[cid] = bias
    return cid_bias


def out_new_mot(mot_dic, mot_path):
    # 输出筛选过后的轨迹每帧的特征
    pickle.dump(mot_dic, open(mot_path, 'wb'))


def filter_low_conf(mot_dic):
    filtered_mot_dic = mot_dic.copy()
    old_len = len(mot_dic)
    for tid in mot_dic:
        hasfeat = False
        for fid in mot_dic[tid]:
            if ("feat" in mot_dic[tid][fid]):
                hasfeat = True
                break
        if not hasfeat:
            filtered_mot_dic.pop(tid)
    new_len = len(filtered_mot_dic)
    print(f"filter_low_conf:  old : {old_len} new:{new_len}")
    return filtered_mot_dic


if __name__ == '__main__':
    # 将轨迹筛选（单相机间，轨迹方向等）后进行特征平均
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    scene_name = ['S06']
    data_dir = cfg.DATA_DIR
    save_dir = './exp/viz/test/S06/mytrajectory/'
    cid_bias = parse_bias(cfg.CID_BIAS_DIR, scene_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    cams = os.listdir(data_dir)
    cams = list(filter(lambda x: 'c' in x, cams))
    cams.sort()
    zones = zone()
    dirs = dir()

    for cam in cams:
        print('processing {}...'.format(cam))
        cid = int(cam[-3:])
        if cid not in [41, 42, 43, 44, 45, 46]:
            break
        f_w = open(opj(save_dir, '{}.pkl'.format(cam)), 'wb')
        cur_bias = cid_bias[cid]
        mot_path = opj(data_dir, cam, '{}_mot_feat.pkl'.format(cam))
        new_mot_path = opj(data_dir, cam, '{}_mot_feat_break.pkl'.format(cam))
        print(new_mot_path)
        zones.set_cam(cid)
        dirs.set_cam(cid)
        mot_dic = parse_pt(mot_path, zones)
        # mot_dic = filter_low_conf(mot_dic)
        mot_dic = zones.break_mot(mot_dic, cid)

        # mot_list = zones.comb_mot(mot_list, cid)
        mot_dic = zones.filter_mot(mot_dic, cid)  # filter by zone
        mot_dic = zones.filter_bbox(mot_dic, cid)  # filter bbox
        mot_dic = dirs.tag_dir(mot_dic)
        # mot_dic = filter_low_conf(mot_dic) #确保截断后tracklet有特征
        out_new_mot(mot_dic, new_mot_path)

        tid_data = dict()
        for tid in mot_dic:
            tracklet = mot_dic[tid]
            frame_list = list(tracklet.keys())
            frame_list.sort()
            # if tid==11 and cid==44:
            #     print(tid)
            zone_list = [tracklet[f]['zone'] for f in frame_list]
            sub_zone_list = [tracklet[f]['sub_zone'] for f in frame_list]
            feature_list = [tracklet[f]['feat'] for f in frame_list if
                            (tracklet[f]['bbox'][3] - tracklet[f]['bbox'][1]) * (
                tracklet[f]['bbox'][2] - tracklet[f]['bbox'][0]) > 2000 and "feat" in tracklet[f]]
            if len(feature_list) < 2:
                feature_list = [tracklet[f]['feat']
                                for f in frame_list if "feat" in tracklet[f]]

            dir_feature_list = {}
            dir_mean_feat = {}
            for d in [1, 2, 3, -1, -2, -3]:
                dir_feature_list[d] = [tracklet[f]['feat'] for f in frame_list if
                                       (tracklet[f]['bbox'][3] - tracklet[f]['bbox'][1]) * (
                    tracklet[f]['bbox'][2] - tracklet[f]['bbox'][0]) > 2000 and "feat" in tracklet[f] and tracklet[f]["dir"] == d]+\
                    [tracklet[f]['feat_flip'] for f in frame_list if(tracklet[f]['bbox'][3] - tracklet[f]['bbox'][1]) * (
                        tracklet[f]['bbox'][2] - tracklet[f]['bbox'][0]) > 2000 and "feat_flip" in tracklet[f] and tracklet[f]["dir"] == -d]
                if len(dir_feature_list[d]) < 2:
                    dir_feature_list[d] = [tracklet[f]['feat']
                                           for f in frame_list if "feat" in tracklet[f] and tracklet[f]["dir"] == d]+[tracklet[f]['feat_flip']
                                           for f in frame_list if "feat_flip" in tracklet[f] and tracklet[f]["dir"] == -d]
                if dir_feature_list[d]:
                    all_feat = np.array([feat for feat in dir_feature_list[d]])
                    mean_feat = np.mean(all_feat, axis=0)
                    dir_mean_feat[d] = mean_feat

            io_time = [cur_bias + frame_list[0] / 10.,cur_bias + frame_list[-1] / 10.]  # 通行时间
            all_feat = np.array([feat for feat in feature_list])
            mean_feat = np.mean(all_feat, axis=0)  # 对所有特征求平均
            tid_data[tid] = {
                'cam': cid,
                'tid': tid,
                "dir_mean_feat": dir_mean_feat,
                'mean_feat': mean_feat,
                'zone_list': zone_list,
                'sub_zone_list':sub_zone_list,
                'frame_list': frame_list,
                'tracklet': tracklet,
                'io_time': io_time
            }

        pickle.dump(tid_data, f_w)
        f_w.close()
