# AICITY2022_Track1_TAG
The code is modified from [AIC21-MTMC](https://github.com/LCFractal/AIC21-MTMC).
## Install dependencies

`cd ./detector/yolov5/ && pip install -r requirements.txt`


## Data & Models Preparation 

1. Download dataset from [here](https://www.aicitychallenge.org/2022-track1-download/), unzip it and put it in `datasets` folder. Then```cd detector ``` and run
     ```python gen_images_aic.py aic.yml``` to generate images.
2. ```cd ./reid_training``` and follow the ```README_ReID.md``` to train the ReID models. Already-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1trYAwgsnB414IHcDfkqGSOTJzet0vkvx?usp=sharing). Put ReID models into the folder `./reid_bidir/reid_model/`. 
3. Download [yolov5x model](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5x.pt) (pretrained on COCO), and put it into the folder `./detector/yolov5/`. Then `cd ./detector/yolov5/`, run:`bash gen_det.sh aic.yml`. Already-generated detection results are provided [here](https://drive.google.com/file/d/1uDGIeImgDrE12Du6Oqxpi4EH1cWTi0Xx/view?usp=sharing).

## MCMVT Results Generation

Run `bash MCMVT.sh` to generate MCMVT results. The result file is located in `./reid_bidir/reid-matching/tools/`, the filename is defined in `aic.yml`.

You can modify params of yml files in `./config`, ensure all paths are correct and can be found.

