"""Merge reid feature from different models."""

import os
import pickle
import sys
sys.path.append('../')
from sklearn import preprocessing
import numpy as np
from config import cfg


def merge_feat(_cfg):
    """Save feature."""

    # NOTE: modify the ensemble list here
    # ensemble_list = os.listdir(_cfg.FEATURES_DIR)
    # all_feat_dir = _cfg.FEATURES_DIR
    ensemble_list = ['detect_reid1', 'detect_reid2', 'detect_reid3']
    all_feat_dir = _cfg.DATA_DIR.split('detect')[0]
    datasets_dir = _cfg.DATA_DIR.split('detect')[0]

    for cam in ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']:
        feat_dic_list = []
        for feat_mode in ensemble_list:
            feat_pkl_file = os.path.join(all_feat_dir, feat_mode, cam,
                                         f'{cam}_dets_feat.pkl')
            feat_mode_dic = pickle.load(open(feat_pkl_file, 'rb'))
            img_names = list(feat_mode_dic.keys())
            if (feat_mode_dic[img_names[0]]['feat'].size != 2048):
                continue
            feat_dic_list.append(feat_mode_dic)
        merged_dic = feat_dic_list[0].copy()

        for patch_name in merged_dic:
            patch_feature_list = []
            patch_feature_flip_list = []
            for feat_mode_dic in feat_dic_list:
                patch_feature_list.append(feat_mode_dic[patch_name]['feat'])
                patch_feature_flip_list.append(feat_mode_dic[patch_name]['feat_flip'])

            patch_feature_array = np.array(patch_feature_list)
            patch_feature_array = preprocessing.normalize(patch_feature_array,norm='l2', axis=1)
            patch_feature_mean = np.mean(patch_feature_array, axis=0)
            patch_feature_mean = preprocessing.normalize(patch_feature_mean,norm='l2', axis=1)

            patch_feature_flip_array = np.array(patch_feature_flip_list)
            patch_feature_flip_array = preprocessing.normalize(patch_feature_flip_array,
                                                          norm='l2', axis=1)
            patch_feature_flip_mean = np.mean(patch_feature_flip_array, axis=0)
            patch_feature_flip_mean = preprocessing.normalize(patch_feature_flip_mean,norm='l2', axis=1)
            merged_dic[patch_name]['feat'] = patch_feature_mean
            merged_dic[patch_name]['feat_flip'] = patch_feature_flip_mean
            merged_dic[patch_name]['feat'] = patch_feature_mean
            merged_dic[patch_name]['feat_flip'] = patch_feature_flip_mean

        merge_dir = os.path.join(datasets_dir, 'detect', cam)
        if not os.path.exists(merge_dir):
            os.makedirs(merge_dir)

        merged_pkl_file = os.path.join(merge_dir, f'{cam}_dets_feat.pkl')
        pickle.dump(merged_dic, open(merged_pkl_file, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        print('save pickle in %s' % merged_pkl_file)


def main():
    """Main method."""
    # 将reid特征融合并放到所有轨迹的字典中保存
    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    merge_feat(cfg)


if __name__ == "__main__":
    main()
