# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import pickle

import cv2
import numpy as np

from application_util import visualization
from application_util.visualization import create_unique_color_uchar
import torch
from torchvision.ops import nms
# from fm_tracker.loader import load_txt
from fm_tracker.multitracker_aic import FairTracker
import sys
sys.path.append('../../../')
# from config import cfg


def gather_sequence_info(sequence_dir, detection_file, max_frame):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
            int(os.path.splitext(f)[0][3:]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    if max_frame > 0:
        max_frame_idx  = max_frame
    
    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = 2048 # default
    det_feat_dic = pickle.load(open(detection_file, 'rb'))
    bbox_dic = {}
    feat_dic = {}
    for image_name in det_feat_dic:
        frame_index = image_name.split('_')[0]
        frame_index = int(frame_index[3:])
        det_bbox = np.array(det_feat_dic[image_name]['bbox']).astype('float32')
        det_feat = det_feat_dic[image_name]['feat']
        score = det_feat_dic[image_name]['conf']
        score = np.array((score,))
        det_bbox = np.concatenate((det_bbox, score)).astype('float32')
        if frame_index not in bbox_dic:
            bbox_dic[frame_index] = [det_bbox]
            feat_dic[frame_index] = [det_feat]
        else:
            bbox_dic[frame_index].append(det_bbox)
            feat_dic[frame_index].append(det_feat)
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": [bbox_dic, feat_dic],
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms,
        "frame_rate": 10
         # "frame_rate": int(info_dict["frameRate"])
    }
    return seq_info

def run(sequence_dir, detection_file,min_box_area = 750):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file, -1)
    tracker = FairTracker(0.1, seq_info["frame_rate"])
    # iou params
    # if seq_info['sequence_name'] == 'c04':
    #     tracker.reid_thr = 0.85
    # else:
    #     seq_info['sequence_name'] == 'c041':
    #     tracker.reid_thr = 0.85

    results = []

    def frame_callback(vis, frame_idx):
        #print("Processing frame %05d" % frame_idx)
        img_path = seq_info['image_filenames'][frame_idx]
        save_path = img_path.replace('img1', 'img2')
        if not os.path.exists(save_path.split('img2')[0]+ 'img2'):
            os.makedirs(save_path.split('img2')[0]+ 'img2')
        img = cv2.imread(img_path)
        # h,w,_ = img.shape

        # if frame_idx == 0:
        #     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
        #     out = cv2.VideoWriter(os.path.join('/home/AIC21-MTMC/datasets/videos', 'result.avi'), fourcc, 25.0, (w,h))

        # Load image and generate detections.
        # detections, feats = load_txt(seq_info["detections"], frame_idx)
        [bbox_dic, feat_dic] = seq_info['detections']
        if frame_idx not in bbox_dic:
            print(f'empty for {frame_idx}')
            return
        # detections = bbox_dic[frame_idx]
        # feats = feat_dic[frame_idx]

        # Run area filter
        detections_ori = bbox_dic[frame_idx]
        feats_ori = feat_dic[frame_idx]
        detections, feats= [], []
        for i, det in enumerate(detections_ori):
            if (det[2]-det[0])*(det[3]-det[1]) > 750:
                detections.append(det)
                feats.append(feats_ori[i])

        # Run non-maxima suppression.
        boxes = np.array([d[:4] for d in detections], dtype=float)
        scores = np.array([d[4] for d in detections], dtype=float)
        nms_keep = nms(torch.from_numpy(boxes),
                                torch.from_numpy(scores),
                                iou_threshold=0.99).numpy()
        detections = np.array([detections[i] for i in nms_keep], dtype=float)
        feats = np.array([feats[i] for i in nms_keep], dtype=float)
        #print(detections)

        # Update tracker.
        if frame_idx == 1572:
            print("t")
        online_targets = tracker.update(detections, feats, frame_idx)
        # Store results.
        cv2.putText(img, str(frame_idx), (10,60), cv2.FONT_HERSHEY_PLAIN, 4, (0,0, 255), 2)
        for t in online_targets:
            tlwh = t.det_tlwh
            # tlwh = t.tlwh
            tid = t.track_id
            score = t.score
            trk_color = create_unique_color_uchar(tid)
            if tlwh[2] * tlwh[3] > min_box_area:
                cv2.putText(img, str(tid) + '|' + str(round(score,2)), (int(tlwh[0]), int(tlwh[1]) - 4), cv2.FONT_HERSHEY_PLAIN, 1, trk_color, 2)
                cv2.rectangle(img, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), trk_color, 2)
        
        cv2.imwrite(save_path, img)
        # vis.out.write(img)

    visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)



if __name__ == "__main__":
    sequence_dir = '/home/backkon/datasets/detection/images/test/S06/c041'
    detection_file = '/home/backkon/datasets/detect_fusion/c041/c041_dets_feat.pkl'

    run(sequence_dir, detection_file)
