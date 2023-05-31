from glob import glob
import os.path as osp
import os

root = './data/testing_data/cropped'
path = osp.join(root, '*.jpg')
files = glob(path)
fp = open(osp.join(root, 'test.txt'), 'w')
for i,f in enumerate(files):
    a = f.split('/')
    b = a[-1].split('.')
    all_path = osp.join(root, "{:0>5d}".format(i+1) + '.' + b[-1])
    os.rename(f, all_path)
    fp.write(all_path + f' {0} {0} {0} {0}\n')