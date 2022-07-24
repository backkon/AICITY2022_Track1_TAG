import cv2
import os
dir="./datasets/detect"
cams = os.listdir(dir)
for cam in cams:
    imgnames=os.listdir(os.path.join(dir,cam,"dets"))
    print(f"Processing {cam}")
    for imgname in imgnames:
        name=imgname.split(".")[0]
        if name.split("_")[-1]=="flip":
            continue
        img = cv2.imread(os.path.join(dir,cam,"dets",imgname))
        img1 = cv2.flip(img,1)  
        cv2.imwrite(os.path.join(dir,cam,"dets",name+"_flip"+".png"),img1)
