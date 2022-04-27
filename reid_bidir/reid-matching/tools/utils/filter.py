import os
from os.path import join as opj
import scipy.io as scio
import cv2
import numpy as np
import pickle
from scipy import spatial
import copy
import multiprocessing
from math import *
from sklearn import preprocessing
import tqdm

CAM_DIST = [[  0, 40, 55,100,120,145],
            [ 40,  0, 15, 60, 80,105],
            [ 55, 15,  0, 40, 65, 90],
            [100, 60, 40,  0, 20, 45],
            [120, 80, 65, 20,  0, 25],
            [145,105, 90, 45, 25,  0]]


CAM_K = [[  0, 0.0033712388044681483,205, 345, 405,495],
                [0.0033712388044681483,  0, 0.005445360509108404, 215, 275,365],
                [205, 0.005445360509108404,  0, 0.0034192450306972217, 200, 290],
                [345, 215, 0.0034192450306972217,  0, 0.002630456226880395, 150],
                [405, 275, 200,  0.002630456226880395,  0, 0.002853138452297527],
                [495,365, 290,  150, 0.002853138452297527,  0]]


CAM_MEAN = [[  0, 81.31194029850745,205, 345, 405,495],
                [81.31194029850745,  0,  30.107228915662645, 215, 275,365],
                [205, 30.107228915662645,  0, 55.06133333333334, 200, 290],
                [345, 215, 55.06133333333334,  0, 31.17, 150],
                [405, 275, 200,  31.17,  0, 49.85263157894737],
                [495,365, 290,  150, 49.85263157894737,  0]]


ka =             [[  0, 0.39,205, 345, 405,495],
                [0.39,  0,  1.05, 215, 275,365],
                [205, 1.05,  0, 0.94, 200, 290],
                [345, 215, 0.94,  0, 1.75, 150],
                [405, 275, 200,  1.75,  0, 0.66],
                [495,365, 290,  150, 0.66,  0]]

kb =             [[  0, 0.82,205, 345, 405,495],
                [0.82,  0,  2.19, 215, 275,365],
                [205, 2.19,  0, 1.39, 200, 290],
                [345, 215, 1.39,  0, 2.44, 150],
                [405, 275, 200,  2.44,  0, 1.39],
                [495,365, 290,  150, 1.39,  0]]

kc =             [[  0, 1.88,205, 345, 405,495],
                [1.88,  0,  4.99, 215, 275,365],
                [205, 4.99,  0, 2.49, 200, 290],
                [345, 215, 2.49,  0, 4.11, 150],
                [405, 275, 200,  4.11,  0, 3.19],
                [495,365, 290,  150, 3.19,  0]]

kd =             [[  0, 2.32,205, 345, 405,495],
                [2.32,  0,  6.12, 215, 275,365],
                [205, 6.12,  0, 2.94, 200, 290],
                [345, 215, 2.94,  0, 4.79, 150],
                [405, 275, 200,  4.79,  0, 3.92],
                [495,365, 290,  150, 3.92,  0]]   


def v_distribution(t, arg_a, arg_b, arg_c, arg_d):
    v = 100.0 / t
    f = 0.7
    if arg_a < v and v < arg_b:
        f = 1 - 0.3 * (v - arg_b) / (arg_a - arg_b)
    elif arg_b < v and v < arg_c:
        f = 1
    elif arg_c < v and v < arg_d:
        f = 1 - 0.3 * (v - arg_c) / (arg_d - arg_c)
    return f



def intracam_ignore(st_mask, cid_tids):
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][0] == cid_tids[j][0]:
                st_mask[i, j] = 0.
    return st_mask

def get_dire(zone_list,cid):
    zs,ze = zone_list[0],zone_list[-1]
    return (zs,ze)

def st_filter(st_mask, cid_tids,cid_tid_dict):
    count = len(cid_tids)
    for i in range(count):
        i_tracklet = cid_tid_dict[cid_tids[i]]
        i_cid = i_tracklet['cam']
        i_subzone = i_tracklet['sub_zone_list']
        i_iot = i_tracklet['io_time']

        i_boxes = i_tracklet['tracklet']
        i_area_list = []
        for k in i_boxes:
            b = i_boxes[k]['bbox']
            i_area_list.append((b[2]-b[0]) * (b[3] - b[1]))
        i_area_max = np.array(i_area_list).max()

        for j in range(count):
            j_tracklet = cid_tid_dict[cid_tids[j]]
            j_cid = j_tracklet['cam']
            j_subzone = j_tracklet['sub_zone_list']
            j_iot = j_tracklet['io_time']

            j_boxes = j_tracklet['tracklet']
            j_area_list = []
            for k in j_boxes:
                b = j_boxes[k]['bbox']
                j_area_list.append((b[2]-b[0]) * (b[3] - b[1]))
            j_area_max = np.array(j_area_list).max()

            match_dire = False
            cam_dist = CAM_DIST[i_cid-41][j_cid-41]
            cam_mean = CAM_MEAN[i_cid-41][j_cid-41]
            cam_k = CAM_K[i_cid-41][j_cid-41]

            cam_a = ka[i_cid-41][j_cid-41]
            cam_b = kb[i_cid-41][j_cid-41]
            cam_c = kc[i_cid-41][j_cid-41]
            cam_d = kd[i_cid-41][j_cid-41]



            if i_cid < j_cid:
                if 7 in i_subzone and 5 in j_subzone and j_iot[0] - cam_dist > i_iot[1]:
                    match_dire = True
                    st_mask[i, j] = 1 - cam_k * abs(abs(j_iot[0] - i_iot[1]) - cam_mean)
                    st_mask[j, i] = 1 - cam_k * abs(abs(j_iot[0] - i_iot[1]) - cam_mean)
                elif 8 in i_subzone and 6 in j_subzone and i_iot[0] > j_iot[1] + cam_dist:
                    match_dire = True
                    st_mask[i, j] = 1 - cam_k * abs(abs(j_iot[1] - i_iot[0]) - cam_mean)
                    st_mask[j, i] = 1 - cam_k * abs(abs(j_iot[1] - i_iot[0]) - cam_mean)
            elif i_cid > j_cid:
                if 5 in i_subzone and 7 in j_subzone and i_iot[0] > j_iot[1] + cam_dist:
                    match_dire = True
                    st_mask[i, j] = 1 - cam_k * abs(abs(j_iot[1] - i_iot[0]) - cam_mean)
                    st_mask[j, i] = 1 - cam_k * abs(abs(j_iot[1] - i_iot[0]) - cam_mean)
                elif 6 in i_subzone and 8 in j_subzone and j_iot[0] - cam_dist > i_iot[1]:
                    match_dire = True
                    st_mask[i, j] = 1 - cam_k * abs(abs(j_iot[0] - i_iot[1]) - cam_mean)
                    st_mask[j, i] = 1 - cam_k * abs(abs(j_iot[0] - i_iot[1]) - cam_mean)

            if 6 in i_subzone and 6 in j_subzone:
                if j_area_max > 200000 and i_area_max > 200000:
                    match_dire = True
                elif (j_area_max > 200000 and i_area_max > 90000 and i_cid == 44) or (j_area_max > 90000 and i_area_max > 200000 and j_cid == 44):
                    match_dire = True     



            if not match_dire:
                st_mask[i, j] = 0.0
                st_mask[j, i] = 0.0
    return st_mask

def subcam_list(cid_tid_dict,cid_tids):
    sub_3_4 = dict()
    sub_4_3 = dict()
    for cid_tid in cid_tids:
        cid,tid = cid_tid
        tracklet = cid_tid_dict[cid_tid]
        if 8 in tracklet['sub_zone_list'] and cid not in [46]:
            if not cid+1 in sub_4_3:
                sub_4_3[cid+1] = []
            sub_4_3[cid + 1].append(cid_tid)
        if 6 in tracklet['sub_zone_list'] and cid not in [41]:
            if not cid in sub_4_3:
                sub_4_3[cid] = []
            sub_4_3[cid].append(cid_tid)
        if 5 in tracklet['sub_zone_list'] and cid not in [41]:
            if not cid-1 in sub_3_4:
                sub_3_4[cid-1] = []
            sub_3_4[cid - 1].append(cid_tid)
        if 7 in tracklet['sub_zone_list'] and cid not in [46]:
            if not cid in sub_3_4:
                sub_3_4[cid] = []
            sub_3_4[cid].append(cid_tid)
    sub_cid_tids = dict()
    for i in sub_3_4:
        sub_cid_tids[(i,i+1)]=sub_3_4[i]
    for i in sub_4_3:
        sub_cid_tids[(i,i-1)]=sub_4_3[i]
    return sub_cid_tids

def subcam_list2(cid_tid_dict,cid_tids):
    sub_dict = dict()
    for cid_tid in cid_tids:
        cid, tid = cid_tid
        if cid not in [41]:
            if not cid in sub_dict:
                sub_dict[cid] = []
            sub_dict[cid].append(cid_tid)
        if cid not in [46]:
            if not cid+1 in sub_dict:
                sub_dict[cid+1] = []
            sub_dict[cid+1].append(cid_tid)
    return sub_dict