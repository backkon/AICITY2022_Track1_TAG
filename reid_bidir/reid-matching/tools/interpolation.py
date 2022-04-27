import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../../../')
from config import cfg

def write_results_score(filename, results):
    save_format = '{camera} {id} {frame} {x1} {y1} {w} {h} -1 -1\n'
    with open(filename, 'w') as f:
        for i in range(results.shape[0]):
            frame_data = results[i]
            camera_id = int(frame_data[0])
            track_id = int(frame_data[1])
            frame_id = int(frame_data[2])
            x1, y1, w, h = frame_data[3:7]
            line = save_format.format(camera = camera_id, id=track_id, frame=frame_id, x1=int(x1), y1=int(y1), w=int(w), h=int(h))
            f.write(line)

#〈camera_id〉〈obj_id〉〈frame_id〉〈xmin〉〈ymin〉〈width〉〈height〉〈xworld〉〈yworld〉
def dti(txt_path, save_path, n_min=25, n_dti=20):
    inter_boxes = []
    log = open('log.txt','w')
    seq_info = np.loadtxt(txt_path, dtype=np.float64, delimiter=' ')
    camera_pools = {}
    for ind in range(41, 47):
        index = (seq_info[:, 0] == ind)
        camera_pools[ind] = seq_info[index]
    
    seq_results = np.empty((0, 9), dtype=np.float64)
    for cid, seq_data in tqdm(camera_pools.items()):
        tracklet_pools = np.empty((0, 9), dtype=np.float64)
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        for track_id in range(min_id, max_id + 1):
            index = (seq_data[:, 1] == track_id)
            tracklet = seq_data[index]
            if tracklet.shape[0] == 0:
                continue
            tracklet = tracklet[tracklet[:, 2].argsort()]
            tracklet_dti = tracklet
            n_frame = tracklet.shape[0]
            if n_frame > n_min:
                frames = tracklet[:, 2]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 3:7]
                        left_bbox = tracklet[i - 1, 3:7]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / \
                                        (right_frame - left_frame) + left_bbox
                            if curr_bbox[2]*curr_bbox[3] > 10000:
                                inter_boxes.append(10000)
                            else:
                                inter_boxes.append(curr_bbox[2]*curr_bbox[3])
                            # if curr_bbox[2]*curr_bbox[3] < 1000:
                            #     inter_boxes.append(curr_bbox[2]*curr_bbox[3])
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                
                if num_dti > 0:
                    log.write('相机:{}, id:{}, 第{}-{}帧缺失, 共插值{}帧\n'.format(cid, track_id, int(list(frames_dti.keys())[0]), int(list(frames_dti.keys())[-1]), num_dti))
                    # log.write('{},{},{},{},{}\n'.format(cid, track_id, int(list(frames_dti.keys())[0]), int(list(frames_dti.keys())[-1]), str(list(frames_dti.keys()))))
                    # print('相机:{}, id:{}, 第{}-{}帧缺失, 共插值{}帧\n'.format(cid, track_id, int(list(frames_dti.keys())[0]), int(list(frames_dti.keys())[-1]), num_dti))
                    data_dti = np.zeros((1, 9), dtype=np.float64)
                    for k, v in frames_dti.items():
                        data_dti[0, 0] = cid
                        data_dti[0, 1] = track_id
                        data_dti[0, 2] = k
                        data_dti[0, 3:7] = v
                        data_dti[0, 7:] = [-1, -1]
                        tracklet_dti = np.vstack((tracklet_dti, data_dti))
                    tracklet_pools = np.vstack((tracklet_pools, tracklet_dti)) 
                else:
                    tracklet_pools = np.vstack((tracklet_pools, tracklet))   
            else:
                tracklet_pools = np.vstack((tracklet_pools, tracklet))       
        tracklet_pools = tracklet_pools[tracklet_pools[:, 2].argsort()]
        seq_results = np.vstack((seq_results, tracklet_pools))
    write_results_score(save_path, seq_results)
    log.close()
    return inter_boxes


if __name__ == '__main__':
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    data_dir = cfg.DATA_DIR
    txt_path = cfg.MCMT_OUTPUT_TXT
    save_path = cfg.MCMT_Interpo_OUTPUT_TXT
    inter_boxes = dti(txt_path, save_path, n_min=5, n_dti=300)
