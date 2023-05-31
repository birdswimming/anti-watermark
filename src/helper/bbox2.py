from random import random
import os
import numpy as np
import cv2

img_path = os.path.join('./data/', 'testing_data', 'cropped', 'test.txt')
bbox_path = os.path.join('./data/', 'testing_data', 'cropped', 'bbox_result.txt')
output_path = os.path.join('./data/', 'testing_data', 'bbox_infer')

with open(bbox_path, 'r') as f:
    bboxs = f.readlines()

with open(img_path, 'r') as f:
    lines = f.readlines()
    for i, l in enumerate(lines):
        l = l.split(' ')
        img = cv2.imread(l[0])
        bboxs[i] = bboxs[i].strip().split(' ')
        size = img.shape
        img = cv2.rectangle(
            img, (int(size[1]*float(l[1])), int(size[0]*float(l[2]))), 
                 (int(size[1]*float(l[3])), int(size[0]*float(l[4]))),
                 (0,255,0), 1
            )
        img = cv2.rectangle(
            img, (int(size[1]*float(bboxs[i][0])), int(size[0]*float(bboxs[i][1]))), 
                 (int(size[1]*float(bboxs[i][2])), int(size[0]*float(bboxs[i][3]))),
                 (0,0,255), 1
            )
        img_name = os.path.join(output_path, l[0].split('/')[-1])
        print(img_name)
        cv2.imwrite(img_name, img)
