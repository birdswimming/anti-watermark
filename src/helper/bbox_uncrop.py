import os
import numpy as np
import cv2

data_path   = os.path.join('./data', 'training_data', 'mix', 'data.txt')
output_path = os.path.join('./data', 'training_data', 'mix', 'data_uncrop.txt')

with open(data_path, 'r') as f:
    lines = f.readlines()

    for i, l in enumerate(lines):
        l = l.split(' ')
        img = cv2.imread(l[0])
        bbox = l[1:5]
        size = img.shape
        img_dir = l[0].split('/')
        img_uncrop = img_dir.copy()
        img_uncrop[-2] = 'watermark'
        img_name = os.path.join(output_path, img_dir[-1])
        print(img_dir)
        #cv2.imwrite(img_name, img)