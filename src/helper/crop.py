from glob import glob
import os
import numpy as np
import cv2
from PIL import Image
    
def Image_Cut(src_path, out_path, picture_size = 256):
    im = Image.open(src_path)
    size = min(im.width, im.height)
    im2 = im.crop((0, im.height - size, size, im.height))
    im2 = im2.resize((picture_size, picture_size))
    im2.save(out_path)
    return 0

path = os.path.join('./data/testing_data/renamed/*.jpg')
out = os.path.join('./data/testing_data/cropped')
files = glob(path)
for f in files:
    print(f)
    Image_Cut(f, os.path.join(out, f.split('/')[-1]))

    
