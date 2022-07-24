MCMT_CONFIG_FILE="aic.yml"

cd ./detector/yolov5
# sh gen_det.sh ${MCMT_CONFIG_FILE}

cd ../..
cd ./reid_bidir/
# python extract_myimage_feat.py "aic_reid1.yml"
# python extract_myimage_feat.py "aic_reid2.yml"
# python extract_myimage_feat.py "aic_reid3.yml"
# python merge_myreid_feat.py ${MCMT_CONFIG_FILE}

#### MOT ####
cd ../tracker/MOTBaseline
# sh run_aic.sh ${MCMT_CONFIG_FILE}

# #### MCMVT ####
cd ../../reid_bidir/reid-matching/tools
python mytrajectory_fusion.py ${MCMT_CONFIG_FILE}
python mysub_cluster.py ${MCMT_CONFIG_FILE}
python mygen_res.py ${MCMT_CONFIG_FILE}
python interpolation.py ${MCMT_CONFIG_FILE}


