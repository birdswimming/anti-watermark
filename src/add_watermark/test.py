from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from random import random

def Add_Watermark(src_path, out_path, font_size_ratio = 0.04, distance_ratio = 0.05, random = 0):
    im = Image.open(src_path)
    bitmap = Image.open('src/add_watermark/watermark.png')
    draw = ImageDraw.Draw(im)
    text = "曾祥乐"
    # 设置字体、字体大小等等
    font_size = int(font_size_ratio * (im.width + im.height)/2)
    bitmap = bitmap.resize((font_size,font_size))
    font = ImageFont.truetype('src/add_watermark/font.ttf', font_size)
    # 添加水印
    distance = (im.width + im.height) * distance_ratio/2 
    draw.text((distance + font_size * 1.25, im.height - 2*distance), text, font=font)
    draw.bitmap((distance, im.height - 1.90*distance), bitmap, '#4A484C')
    
    
    
    left = distance
    upper = im.height - 1.90*distance
    right = distance + font_size*4.25
    lower = im.height - 2*distance - font_size
    im.save(out_path)
    return [left, upper, right, lower]
    

def Image_Cut(src_path, out_path, raw_label, picture_size = 256):
    im = Image.open(src_path)
    size = min(im.width, im.height)
    im2 = im.crop((0, im.height - size, size, im.height))
    label= []
    for i in range(len(raw_label)):
        label.append(raw_label[i]/size)
    im2 = im2.resize((picture_size, picture_size))
    im2.save(out_path)
    return label
    
    

src_path = 'data/training_data/origin/001.jpg'
mid_path = 'data/training_data/with_watermark/001.jpg'
out_path = 'data/training_data/locate_dataset/001.jpg'
label = Add_Watermark(src_path, mid_path)
label = Image_Cut(mid_path, out_path, label)
data = open('data/training_data/locate_dataset/data.txt', 'w')
data.write(out_path + " {} {} {} {}\n".format(label[0],label[1],label[2],label[3]))
