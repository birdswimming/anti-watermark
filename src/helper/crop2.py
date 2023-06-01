from glob import glob
import os
import numpy as np
import cv2
from PIL import Image

resize_size = 1024
pad_size = 300
crop_size = 256

src_path = os.path.join('./data/training_data/mix/data_uncrop.txt')
out = os.path.join('./data/training_data/mix/cropped2')
out_txt = os.path.join('./data/training_data/mix/crop2.txt')

txt_file = open(out_txt, 'w')

with open(src_path, 'r') as f:
    src_files = f.readlines()

for f in src_files:
    f = f.split(' ')
    img_path = f[0]
    print(img_path)

    bbox = [float(b) for b in f[1:5]]
    img = cv2.imread(img_path)
    size0 = img.shape
    minsize = min(size0[0], size0[1]) 
    if minsize > resize_size:
        if minsize == size0[0]:
            # print(resize_size, (size0[0],int(size0[1]*resize_size/minsize)))
            img = cv2.resize(
                img, (int(size0[1]*resize_size/minsize), resize_size),
                interpolation=cv2.INTER_CUBIC
            )
        else:
            # print(size0, (int(size0[0]*resize_size/minsize),resize_size))
            img = cv2.resize(
                img, (resize_size,int(size0[0]*resize_size/minsize)),
                interpolation=cv2.INTER_CUBIC
            )
        size0 = img.shape

    ret = cv2.copyMakeBorder(
        img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, 
        value=(0,0,0)
    )
    cx = (bbox[0]+bbox[2])/2
    cy = (bbox[1]+bbox[3])/2

    pix_cx = int(cx * size0[1])
    pix_cy = int(cy * size0[0])

    left = int(pix_cx-crop_size/2+pad_size)
    right = int(pix_cx+crop_size/2+pad_size)
    up = int(pix_cy-crop_size/2+pad_size)
    down = int(pix_cy+crop_size/2+pad_size)
    
    cropped_img = ret[up:down,left:right,:]
    out_path = os.path.join(out, img_path.split('/')[-1])

    cv2.imwrite(out_path, cropped_img)
    
    left = 0.5 + (bbox[0]-cx) * size0[1] / crop_size
    right =  0.5 + (bbox[2]-cx) * size0[1] / crop_size
    up =  0.5 + (bbox[1]-cy) * size0[0] / crop_size
    down =  0.5 + (bbox[3]-cy) * size0[0] / crop_size

    txt_file.write(out_path+f' {left} {up} {right} {down}\n')
    
