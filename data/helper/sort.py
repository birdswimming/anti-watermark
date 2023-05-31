from glob import glob
import os
path = os.path.join('./wild/*.jpg')
files = glob(path)
for i,f in enumerate(files):
    a = f.split('/')
    b = a[-1].split('.')
    all_path = os.path.join('./wild/', str(i+1) + '.' + b[-1])
    os.rename(f, all_path)