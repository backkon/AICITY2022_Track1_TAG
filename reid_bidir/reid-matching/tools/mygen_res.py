import os
from os.path import join as opj
import cv2
import pickle
import sys
sys.path.append('../../../')
from config import cfg
    
def parse_pt(pt_file):

    with open(pt_file, 'rb') as f:
        mot_dic = pickle.load(f)

    return mot_dic

def show_res(map_tid):
    show_dict = dict()
    for cid_tid in map_tid:
        iid = map_tid[cid_tid]
        if iid in show_dict:
            show_dict[iid].append(cid_tid)
        else:
            show_dict[iid] = [cid_tid]
    for i in show_dict:
        print('ID{}:{}'.format(i,show_dict[i]))

if __name__ == '__main__':
    # 整理数据格式并输出
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    data_dir = cfg.DATA_DIR
    roi_dir = cfg.ROI_DIR

    map_tid = pickle.load(open('test_cluster.pkl', 'rb'))['cluster']
    show_res(map_tid)
    f_w = open(cfg.MCMT_OUTPUT_TXT, 'w')
    cams = os.listdir(data_dir)
    cams = list(filter(lambda x: 'c04' in x, cams))
    cams.sort()
    for cam in cams:
        cid = int(cam[-3:])
        
        roi = cv2.imread(opj(roi_dir, '{}/roi.jpg'.format(cam)), 0)
        height, width = roi.shape
        mot_dic = parse_pt(opj(data_dir, cam,'{}_mot_feat_break.pkl'.format(cam)))
        for tid in mot_dic:
            tracklet = mot_dic[tid]
            for fid in tracklet:
                bbox=tracklet[fid]["bbox"]
                cx = 0.5*bbox[0] + 0.5*bbox[2]
                cy = 0.5*bbox[1] + 0.5*bbox[3]
                w = bbox[2] - bbox[0]
                w = min(w*1.2,w+40)
                h = bbox[3] - bbox[1]
                h = min(h*1.2,h+40)
                x1, y1 = max(0, cx - 0.5*w), max(0, cy - 0.5*h)
                x2, y2 = min(width, cx + 0.5*w), min(height, cy + 0.5*h)
                w , h = x2-x1 , y2-y1
                new_bbox = list(map(int, [x1, y1, w, h]))
                # new_rect = rect # 使用原bbox
                if (cid, tid) in map_tid:
                    new_tid = map_tid[(cid, tid)]
                    f_w.write(str(cid) + ' ' + str(new_tid) + ' ' + str(fid+1) + ' ' + ' '.join(map(str, new_bbox)) + ' -1 -1' '\n')

    f_w.close()
