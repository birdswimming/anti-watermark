from glob import glob
import os
import numpy as np
import cv2
from PIL import Image

resize_size = 1024
pad_size = 300
crop_size = 256
bbox_enlarge_ratio = 1.1

src_path = os.path.join('./data/training_data/mix/data_uncrop.txt')
out_real = os.path.join('./data/training_data/mix/crop_original')
out_mask = os.path.join('./data/training_data/mix/crop_mask')
out_txt1 = os.path.join('./data/training_data/mix/crop2.txt')
out_txt2 = os.path.join('./data/training_data/mix/train_partialConv.txt')

txt_file1 = open(out_txt1, 'w')
txt_file2 = open(out_txt2, 'w')

with open(src_path, 'r') as f:
    src_files = f.readlines()

for f in src_files:
    f = f.split(' ')
    img_path = f[0]
    img_path = img_path.split('/')
    img_path[-2] = 'original'
    img_path = '/'.join(img_path)
    print(img_path)

    bbox = [float(b) for b in f[1:5]]
    img = cv2.imread(img_path)
    size0 = img.shape
    minsize = min(size0[0], size0[1]) 
    if minsize > resize_size:
        if minsize == size0[0]:
            img = cv2.resize(
                img, (int(size0[1]*resize_size/minsize), resize_size),
                interpolation=cv2.INTER_CUBIC
            )
        else:
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
    out_path_real = os.path.join(out_real, img_path.split('/')[-1])
    cv2.imwrite(out_path_real, cropped_img)

    left = 0.5 + (bbox[0]-cx) * size0[1] / crop_size
    right =  0.5 + (bbox[2]-cx) * size0[1] / crop_size
    up =  0.5 + (bbox[1]-cy) * size0[0] / crop_size
    down =  0.5 + (bbox[3]-cy) * size0[0] / crop_size

    txt_file1.write(out_path_real+f' {left} {up} {right} {down}\n')

    
    left  = int((0.5 - bbox_enlarge_ratio*(0.5-left)) * crop_size)
    right = int((0.5 - bbox_enlarge_ratio*(0.5-right)) * crop_size)
    up    = int((0.5 - bbox_enlarge_ratio*(0.5-up))  * crop_size)
    down  = int((0.5 - bbox_enlarge_ratio*(0.5-down))  * crop_size)
    if left < 0:
        left = 0
    if right >= crop_size:
        right = crop_size-1
    if up < 0:
        up = 0
    if down >= crop_size:
        down = crop_size-1
        
    masked_img = cropped_img.copy()
    masked_img[:,:,:] = 255
    masked_img[up:down,left:right,:] = 0 

    out_path_mask = os.path.join(out_mask, img_path.split('/')[-1])
    cv2.imwrite(out_path_mask, masked_img)
    

    txt_file2.write(out_path_real+' '+out_path_mask+'\n')
    
