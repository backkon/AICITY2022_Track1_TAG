cd ./datasets

# crop ReID images
python crop_ReID_training_image_AIC22.py

# copy images and make xml
python gen_ReID_trainning_xml_AIC22.py