from random import random
import os
import numpy as np
import cv2

data_path = os.path.join('./data/', 'training_data', 'mix', 'crop2.txt')
output_path = os.path.join('./data/', 'training_data', 'tmp')

with open(data_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        l = l.strip()
        l = l.split(' ')
        img_path = l[0]
        bbox = l[1:5]
        img = cv2.imread(img_path)
        size = img.shape
        img = cv2.rectangle(
            img, (int(size[1]*float(bbox[0])), int(size[0]*float(bbox[1]))), 
                 (int(size[1]*float(bbox[2])), int(size[0]*float(bbox[3]))),
                 (0,255,0), 1
            )
        img_name = os.path.join(output_path, img_path.split('/')[-1])
        print(img_name)
        cv2.imwrite(img_name, img)
