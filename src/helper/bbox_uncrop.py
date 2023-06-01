import os
import numpy as np
import cv2

original_path = os.path.join('./data', 'training_data', 'mix', 'imgs.txt')
cropped_path  = os.path.join('./data', 'training_data', 'mix', 'data.txt')
output_path   = os.path.join('./data', 'training_data', 'mix', 'data_uncrop.txt')

with open(original_path, 'r') as f:
    original_imgs = f.readlines()
with open(cropped_path, 'r') as f:
    cropped_data = f.readlines()
output_file = open(output_path, 'w')

for i, l in enumerate(original_imgs):
    
    img_path = l.split(' ')[0] 
    img1 = cv2.imread(img_path)

    bbox = cropped_data[i].split(' ')[1:5]

    bbox = [float(b) for b in bbox]
    bbox[1] = 1 - bbox[1]
    bbox[3] = 1 - bbox[3]
    
    img_path = cropped_data[i].split(' ')[0]
    img2 = cv2.imread(img_path)
    
    size1 = img1.shape
    size2 = img2.shape

    minSize = min(size1[0], size1[1])
    assert(size2[0] == size2[1])

    if minSize == size1[0]:
        bbox[0] = bbox[0] * minSize / size1[1]
        bbox[2] = bbox[2] * minSize / size1[1]
    else:
        bbox[1] = bbox[1] * minSize / size1[0]
        bbox[3] = bbox[3] * minSize / size1[0]

    bbox[1] = 1 - bbox[1]
    bbox[3] = 1 - bbox[3]        
    
    img_path = img_path.split('/')
    img_path[-2] = 'watermark'
    img_path = '/'.join(img_path)
    print(img_path)
    output_file.write(img_path + f' {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')