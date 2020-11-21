#!/usr/bin/env python3

import numpy as np
from shutil import copyfile
import cv2
import os
from mot_utils import *

np.set_printoptions(suppress=True)

seq = 'seq05'
existing_im_dir = '/home/ed/Data/frames/images/color/'
existing_ann_dir = '/home/ed/Data/frames/labels/color/'
write_im_dir = '/home/ed/Data/frames/lenet-app/seq05/'
trk_suffix = '_trk'

for root, dirs, files in os.walk(os.path.join(existing_im_dir, seq), topdown=False):
    files = sorted(files)
    for f in files:
        # Save the image dimensions
        im = cv2.imread(os.path.join(root, f))
        try:
            h, w, c = im.shape
        except:
            h, w = im.shape

        # Open the corresponding annotation text file
        file_txt = replace_suffix(f, '.txt')
        with open(os.path.join(os.path.join(existing_ann_dir, seq + trk_suffix), file_txt), 'r') as myfile:
            for line in myfile.readlines():
                bb = BoundingBox(line)
                class_id = bb.track_id
                tlbr = bb.to_tlbr()
                tl_x = int(tlbr[0] * w)
                tl_y = int(tlbr[1] * h)
                br_x = int(tlbr[2] * w)
                br_y = int(tlbr[3] * h)
                crop = im[tl_y:br_y, tl_x:br_x]
                out_filename = replace_suffix(replace_suffix(file_txt, '') + '_cid' + f'{class_id:02}', '.jpg')
                cv2.imwrite(os.path.join(write_im_dir, out_filename), crop)