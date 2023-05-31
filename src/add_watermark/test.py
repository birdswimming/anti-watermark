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
    im.save(out_path)

src_path = 'data/source_data/birds/00001.jpg'
out_path = 'data/source_data/birds_wm/00001_wm.jpg'
Add_Watermark(src_path, out_path)