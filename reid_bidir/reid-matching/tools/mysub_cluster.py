
from utils.filter import *
from utils.visual_rr import visual_rerank
from sklearn.cluster import AgglomerativeClustering
import sys
sys.path.append('../../../')
from config import cfg





def computeP_neg(cid_tid_dict, cid_tids,la=3.0):
    """"
    使用两相机间所有轨迹的平均特征计算所有相机的P,neg_vector
    """
    neg_vectors = {}
    P={}
    cams = np.unique(np.array(cid_tids)[:,0])
    for cam in cams:
        feats = np.vstack([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(len(cid_tids)) if cid_tids[i][0]==cam])
        neg_vectors[cam] = np.mean(feats,axis=0)
        P[cam] = np.linalg.inv(feats.T.dot(feats)+feats.shape[0]*la*np.eye(feats.shape[1]))
    return P,neg_vectors


def get_sim_matrix(_cfg, cid_tid_dict, cid_tids):
    count = len(cid_tids)
    print('count: ', count)
    dirs = [1, 2, 3, -1, -2, -3]
    total_sim_matrix = np.zeros((count, count), dtype='float32')
    addcount = np.zeros((count, count))
    P,neg_vectors=computeP_neg(cid_tid_dict, cid_tids)
    for dir in dirs:
        diridx = [i for i in range(
            count) if dir in cid_tid_dict[cid_tids[i]]['dir_mean_feat']] #存在这个方向特征的轨迹编号
        if not diridx:
            continue

        q_arr = np.array([cid_tid_dict[cid_tids[i]]['dir_mean_feat'][dir]
                         for i in diridx])
        g_arr = np.array([cid_tid_dict[cid_tids[i]]['dir_mean_feat'][dir]
                         for i in diridx])
        q_arr = normalize(q_arr, axis=1)
        g_arr = normalize(g_arr, axis=1)
        visual_sim_matrix = visual_rerank(q_arr, g_arr,diridx, cid_tids,P,neg_vectors, _cfg)
        visual_sim_matrix = visual_sim_matrix.astype('float32')
        for x, dir_x in enumerate(diridx):  # TODO:用numpy改写
            for y, dir_y in enumerate(diridx):
                total_sim_matrix[dir_x][dir_y] += visual_sim_matrix[x][y]
                addcount[dir_x][dir_y] += 1
    # total_sim_matrix = total_sim_matrix/6 #不需要除以六
    vacant_idx = list(np.unique(np.argwhere(total_sim_matrix == 0)[:, 0]))
    if vacant_idx:
        q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat']
                         for i in vacant_idx])
        g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat']
                         for i in vacant_idx])
        q_arr = normalize(q_arr, axis=1)
        g_arr = normalize(g_arr, axis=1)
        visual_sim_matrix = visual_rerank(q_arr, g_arr,vacant_idx, cid_tids,P,neg_vectors, _cfg)
        visual_sim_matrix = visual_sim_matrix.astype('float32')
        for x, vacant_x in enumerate(vacant_idx):  # TODO:用numpy改写
            for y, vacant_y in enumerate(vacant_idx):
                total_sim_matrix[vacant_x][vacant_y] += visual_sim_matrix[x][y]
                addcount[vacant_x][vacant_y] += 1
    total_sim_matrix = total_sim_matrix/addcount
    # sim_matrix = np.matmul(q_arr, g_arr.T)

    # st mask
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)  # TODO 去除这部分避免掉头车辆无法匹配
    st_mask = st_filter(st_mask, cid_tids, cid_tid_dict)

    # visual rerank

    # print(visual_sim_matrix)
    # merge result
    np.set_printoptions(precision=3)
    sim_matrix = total_sim_matrix * st_mask

    # sim_matrix[sim_matrix < 0] = 0
    np.fill_diagonal(sim_matrix, 0)  # 置对角线为0
    return sim_matrix


def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray


def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster


def get_cid_tid(cluster_labels, cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster


def combin_cluster(sub_labels, cid_tids):
    # 合并所有相邻轨迹成完整轨迹
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster) < 1:
            cluster = sub_labels[sub_c_to_c]
            continue

        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set)) > 0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr += 1
        label_list.sort()
        labels.append(label_list)
    print("new tricklets:{}".format(num_tr))
    return labels, cluster


def combin_feature(cid_tid_dict, sub_cluster):
    dirs = [1, 2, 3, -1, -2, -3]
    for sub_ct in sub_cluster:
        if len(sub_ct) < 2:
            continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
        
        for dir in dirs:
            dir_mean_feat = np.array([cid_tid_dict[i]['dir_mean_feat'][dir] for i in sub_ct if dir in cid_tid_dict[i]['dir_mean_feat']])
            if dir_mean_feat.size!=0:
                for i in sub_ct:
                    cid_tid_dict[i]['dir_mean_feat'][dir] = dir_mean_feat.mean(axis=0)
    return cid_tid_dict


def get_labels(_cfg, cid_tid_dict, cid_tids, score_thr):
    # 两阶段聚类
    # 返回所有摄像机号-轨迹号集合的聚类结果，label是一个列表，存储了同车辆轨迹在cid_tid中的下标
    # 1st cluster
    sub_cid_tids = subcam_list(cid_tid_dict, cid_tids)
    sub_labels = dict()
    score_thr = 0.3
    dis_thrs = [0.7, 0.5, 0.5, 0.5, 0.5,
                0.7, 0.5, 0.5, 0.5, 0.5]
    for i, sub_c_to_c in enumerate(sub_cid_tids):
        # if sub_c_to_c != (44,43):
        #     continue
        sim_matrix = get_sim_matrix(
            _cfg, cid_tid_dict, sub_cid_tids[sub_c_to_c])
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                                 linkage='complete').fit_predict(1 - sim_matrix)  # 对相邻相机间每个轨迹进行聚类
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels, sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels, sub_cluster = combin_cluster(sub_labels, cid_tids)

    # 2ed cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids)
    sub_labels = dict()
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c])
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)
    
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)  # 合并轨迹的特征，用平均值替代
    sub_cid_tids = subcam_list2(cid_tid_dict_new, cid_tids)
    sub_labels = dict()
    for i, sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(
            _cfg, cid_tid_dict_new, sub_cid_tids[sub_c_to_c])
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-0.1, affinity='precomputed',
                                                 linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels, sub_cid_tids[sub_c_to_c])
        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels, sub_cluster = combin_cluster(sub_labels, cid_tids)

    # 3rd cluster
    # cid_tid_dict_new = combin_feature(cid_tid_dict,sub_cluster)
    # sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new, cid_tids)
    # cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - 0.2, affinity='precomputed',
    #                                          linkage='complete').fit_predict(1 - sim_matrix)
    # labels = get_match(cluster_labels)
    return labels


if __name__ == '__main__':
    cfg.merge_from_file(f'../../../config/{sys.argv[1]}')
    cfg.freeze()
    scene_name = ['S06']
    scene_cluster = [[41, 42, 43, 44, 45, 46]]
    fea_dir = './exp/viz/test/S06/mytrajectory/'
    cid_tid_dict = dict()

    for pkl_path in os.listdir(fea_dir):
        cid = int(pkl_path.split('.')[0][-3:])
        with open(opj(fea_dir, pkl_path), 'rb') as f:
            lines = pickle.load(f)
        for line in lines:
            tracklet = lines[line]
            tid = tracklet['tid']
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet

    cid_tids = sorted([key for key in cid_tid_dict.keys()
                      if key[0] in scene_cluster[0]])
    clu = get_labels(cfg, cid_tid_dict, cid_tids, score_thr=cfg.SCORE_THR)
    print('all_clu:', len(clu))
    new_clu = list()
    # 筛掉没有匹配的轨迹
    for c_list in clu:
        if len(c_list) <= 1:
            continue
        cam_list = [cid_tids[c][0] for c in c_list]
        if len(cam_list) != len(set(cam_list)):
            continue
        new_clu.append([cid_tids[c] for c in c_list])
    print('new_clu: ', len(new_clu))

    all_clu = new_clu
    # 存储所有相机-轨迹号对应的label编号
    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    pickle.dump({'cluster': cid_tid_label}, open('test_cluster.pkl', 'wb'))
