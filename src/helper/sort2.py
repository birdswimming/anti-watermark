from glob import glob
import os.path as osp
import os
import cv2

root = ['./data/source_data/wild',
        './data/source_data/birds']
out_root = './data/training_data/mix'
files = []
for r in root:
    path = osp.join(r, '*.jpg')
    files += glob(path)

fp = open(osp.join(out_root, 'imgs.txt'), 'w')

for i,f in enumerate(files):
    img = cv2.imread(f)
    print(f)
    new_dir = osp.join(out_root,'original', '{:0>5d}'.format(i+1) + '.jpg')
    fp.write(new_dir + f' {0} {0} {0} {0}\n')
    cv2.imwrite(new_dir, img)