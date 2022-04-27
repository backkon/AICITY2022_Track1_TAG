The code is modified from [AICITY2021_Track2_DMT](https://github.com/michuanhaohao/AICITY2021_Track2_DMT).

## Install dependencies

`pip install requirements.txt`

We use cuda 11.0 / python 3.7 / torch 1.6.0 / torchvision 0.7.0 for training and testing.


## Prepare Datasets

1. Download AIC21 Track2 dataset: [real_dataset](https://drive.google.com/file/d/1bxNjs_KZ_ocnhpsZmdMsIut93z8CqgBN/view?usp=sharing) and [SPGAN_dataset](https://drive.google.com/file/d/1nPOTrK9WUEK38mwei9yAOCMlNiF1UJXV/view?usp=sharing).
        
2. Download AIC22 Track1 original dataset: [AIC22_Track1_MTMC_Tracking]((https://www.aicitychallenge.org/2022-track1-download/)), then crop images to generate the training data according to GroundTruth:
`bash genReIDDataAIC22.sh`. Already-generated data can be download from [here](https://drive.google.com/file/d/1d92XzAwi0HhKBvCpf9Vw2fQIucopQN1s/view?usp=sharing).

The dataset structure should be like:
```bash
    ├── AIC21/
    │   ├── AIC21_Track2_ReID/
    │   	├── image_train/
    │   	├── image_test/
    │   	├── image_query/
    │   	├── train_label.xml
    │   	├── ...
    │   	├── training_part_seg/
    │   	    ├── cropped_patch/
    │   	├── cropped_aic_test
    │   	    ├── image_test/
    │   	    ├── image_query/		
    │   ├── AIC21_Track2_ReID_Simulation/
    │   	├── sys_image_train/
    │   	├── sys_image_train_tr/
	
	
    ├── AIC22/
    │   ├── AIC22_Track1_MTMC_Tracking/
    │   	├── train/
    │   	├── validation/
    │   ├── AIC22_ReID_DATA/
    │   	├── S01_imgs/
    │   	    ├── 1/
    │   	    ├── ...
    │   	├── S02_imgs/
    │   ├── AIC22_Track1_ReID/
    │   	├── image_train/
    │   	├── train_label.xml
```

3. Put pre-trained models into folder `./pretrained/`:
    -  resnet101_ibn_a-59ea0ac6.pth, densenet169_ibn_a-9f32c161.pth, resnext101_ibn_a-6ace051d.pth and se_resnet101_ibn_a-fabed4e2.pth can be downloaded from [here](https://github.com/XingangPan/IBN-Net)
    -  resnest101-22405ba7.pth can be downloaded from [here](https://github.com/zhanghang1989/ResNeSt)
    -  jx_vit_base_p16_224-80ecf9dd.pth can be downloaded from [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)

## Training and Test

We utilize 1 Tesla V100 (32 GB) GPU for training.

You can train resnext backbone on AIC21 ReID dataset as follows:
```bash
python train.py --config_file configs/stage1/resnext101a_384.yml MODEL.DEVICE_ID "('0')"
```
You can train resnext backbone on AIC22 ReID dataset as follows:
```bash
python train_AIC22.py --config_file configs/stage1/resnext101a_384_AIC22.yml MODEL.DEVICE_ID "('0')"
```

The details about training and testing, please refer to [AICITY2021_Track2_DMT](https://github.com/michuanhaohao/AICITY2021_Track2_DMT)

## Comparison of different datasets
|backbone|IDF1|IDP|IDR|
|---|---|---|---|
|ResNeXt101-IBN-a*|0.7813|0.8498|0.7230|
|**ResNeXt101-IBN-a**|0.7851|0.8481|0.7309|

ResNeXt101-IBN-a is trained on AIC21 and ResNeXt101-IBN-a* is trained on AIC22. After the official confirmation that AIC21 data can be used, we finally adopt the model which is trained on AIC21.
