from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from glob import glob
import os
import numpy as np
import cv2

def Add_Watermark(src_path, out_path, font_size_ratio = 0.04, distance_ratio = 0.05, random = 0):
    fp1 = open(src_path, 'rb')
    im = Image.open(fp1)
    fp2 = open('src/add_watermark/watermark.png', 'rb')
    bitmap = Image.open(fp2)
    
    draw = ImageDraw.Draw(im)
    text = "曾祥乐"
    # 设置字体、字体大小等等
    font_size = int(font_size_ratio * (im.width + im.height)/2)
    bitmap = bitmap.resize((font_size,font_size))
    font = ImageFont.truetype('src/add_watermark/font.ttf', font_size)
    # 添加水印
    distance = (im.width + im.height) * distance_ratio/2 
    
    import random
    rx = (random.random()*random.random()*( 10.0) - 0.9) * distance
    ry = (random.random()*random.random()*(-10.0) + 0.9) * distance

    cvPic = np.array(cv2.imread(src_path))
    fill = (255, 255, 255)
    if (np.average(
        cvPic[int(ry+im.height - 2.1*distance):int(ry+im.height - 0.9*distance),\
        int(rx+distance + font_size * 1.1):int(rx+distance + font_size * 4.4)]  \
    ) > 180):
        fill = (50, 50, 50) 

    draw.text((rx + distance + font_size * 1.25, ry + im.height - 2*distance), text, font=font, fill=fill)
    draw.bitmap((rx + distance, ry + im.height - 1.90*distance), bitmap, '#4A484C')
    
    left = rx + distance
    upper = ry + im.height - 2.0*distance
    right = rx + distance + font_size*4.25
    lower = ry + im.height - 1.9*distance + font_size

    im.save(out_path)
    fp1.close()
    fp2.close()
    return [left, upper, right, lower]
    

def Image_Cut(src_path, out_path, raw_label, picture_size = 256):
    im = Image.open(src_path)
    size = min(im.width, im.height)
    raw_label[1] = im.height - raw_label[1]
    raw_label[3] = im.height - raw_label[3]
    im2 = im.crop((0, im.height - size, size, im.height))
    label= []
    for i in range(len(raw_label)):
        label.append(raw_label[i]/size)
    im2 = im2.resize((picture_size, picture_size))
    im2.save(out_path)
    label[1] = 1 - label[1]
    label[3] = 1 - label[3]
    return label
    
    
src_path_list = ['./data/source_data/birds/*.jpg',
                 './data/source_data/wild/*.jpg']
out_root = 'data/training_data/mix/'

files = []
for src_path in src_path_list:
    files += glob(src_path)
data = open(os.path.join(out_root,'data.txt'), 'w')

cnt = 1
for f in files:
    fname = '{:0>5d}'.format(cnt)
    img_path = os.path.join(f)
    print(img_path)
    mid_path = os.path.join(out_root, 'watermark', fname + '_wm.jpg')
    out_path = os.path.join(out_root, 'cropped', fname+ '_wm_crop.jpg')
    cnt += 1
    label = Add_Watermark(img_path, mid_path)
    label = Image_Cut(mid_path, out_path, label)
    data.write(out_path + " {} {} {} {}\n".format(label[0],label[1],label[2],label[3]))
