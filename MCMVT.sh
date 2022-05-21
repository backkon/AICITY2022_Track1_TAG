MCMT_CONFIG_FILE="aic.yml"

cd ./reid_bidir/
python extract_image_feat.py "aic_reid1.yml"
python extract_image_feat.py "aic_reid2.yml"
python extract_image_feat.py "aic_reid3.yml"
python merge_reid_feat.py ${MCMT_CONFIG_FILE}

#### MOT ####
cd ../tracker/MOTBaseline
sh run_aic.sh ${MCMT_CONFIG_FILE}

#### MCMVT ####
cd ../../reid_bidir/reid-matching/tools
python trajectory_fusion.py ${MCMT_CONFIG_FILE}
python sub_cluster.py ${MCMT_CONFIG_FILE}
python gen_res.py ${MCMT_CONFIG_FILE}
python interpolation.py ${MCMT_CONFIG_FILE}


