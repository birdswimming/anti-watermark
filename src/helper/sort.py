from glob import glob
import os
path = os.path.join('./data/testing_data/test/*.jpg')
files = glob(path)
for i,f in enumerate(files):
    a = f.split('/')
    b = a[-1].split('.')
    all_path = os.path.join('./data/testing_data/test/', "{:0>5d}".format(i+1) + '.' + b[-1])
    os.rename(f, all_path)