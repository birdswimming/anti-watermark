from random import random
import os
import numpy as n
import cv2

img_path    = os.path.join('./data/', 'testing_data', 'data.txt')
bbox_path   = os.path.join('./data/', 'testing_data', 'bbox_result.txt')
output_path = os.path.join('./data/', 'testing_data', 'mask')
extension_ratio = 1.5
with open(bbox_path, 'r') as f:
    bboxs = f.readlines()

with open(img_path, 'r') as f:
    lines = f.readlines()
    for i, l in enumerate(lines):
        bboxs[i] = bboxs[i].strip().split(' ') 
        l = l.split(' ')
        img = cv2.imread(l[0])
        size = img.shape
        
        width = size[1]
        height = size[0]
        
        mask = img.copy()
        x_min = int(width  * float(bboxs[i][0]))
        y_min = int(height * float(bboxs[i][1]))
        x_max = int(width  * float(bboxs[i][2]))
        y_max = int(height * float(bboxs[i][3]))
        x_mid = int((x_min + x_max) / 2)
        y_mid = int((y_min + y_max) / 2)
        x_half  = int((x_max - x_min)*extension_ratio / 2)
        y_half  = int((y_max - y_min)*extension_ratio / 2)
        x_min = x_mid - x_half
        x_max = x_mid + x_half
        y_min = y_mid - y_half
        y_max = y_mid + y_half
        if x_min < 0:
            x_min = 0
        if x_max >= width:
            x_max = width-1
        if y_min < 0:
            y_min = 0
        if y_max >= height:
            y_max = height-1
        
        mask[:,:,:] = 255
        mask[y_min:y_max, x_min:x_max] = 0
        img_name = os.path.join(output_path, l[0].split('/')[-1])
        print(img_name)
        cv2.imwrite(img_name, mask)