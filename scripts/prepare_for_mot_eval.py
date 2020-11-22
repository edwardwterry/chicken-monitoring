#!/usr/bin/env python3

import numpy as np
from shutil import copyfile
import cv2
import os
from mot_utils import *
from pathlib import Path

np.set_printoptions(suppress=True)

seq = 'seq05'
existing_im_dir = '/home/ed/Data/frames/images/color/'
existing_ann_dir = '/home/ed/Data/frames/labels/color/'
sequence_dir = '/home/ed/Data/MOTeval/'
dataset_type = 'd20xy2wh15-c12fv' #d00xy2wh15 # CHANGE ME
target_im_dir = 'img1'
target_det_dir = 'det'
target_gt_dir = 'gt'
trk_suffix = '_trk'
fv_model = 'chicken12' # 'cifar10' # CHANGE ME      

add_noise = True # CHANGE ME
keep_all_detections = False # CHANGE ME

fn = FeatureNet(dataset=fv_model)

for root, dirs, files in os.walk(os.path.join(existing_im_dir, seq), topdown=False):
    files = sorted(files)
    frame_id = 1
    dets = []
    gts = []
    for f in files:
        dst = f'{frame_id:06}' + '.jpg'
        Path(os.path.join(
            sequence_dir, seq + '-' + dataset_type, target_im_dir)).mkdir(parents=True, exist_ok=True)
        copyfile(os.path.join(root, f), os.path.join(os.path.join(
            sequence_dir, seq + '-' + dataset_type, target_im_dir), dst))

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
                bbox = BoundingBox(line, format='txt', add_noise=add_noise)
                if bbox.included(keep_all_detections):
                    feat = fn.get_feature_vector(im, bbox, w, h)
                    det = bbox.to_mot(frame_id, w, h, 'det', feat=feat)
                    gt = bbox.to_mot(frame_id, w, h, 'gt')
                    dets.append(det)
                    gts.append(gt)

        frame_id += 1

    # Write gt.txt
    Path(os.path.join(sequence_dir, seq + '-' + dataset_type, target_gt_dir)).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join(os.path.join(sequence_dir, seq + '-' + dataset_type,
                                         target_gt_dir), 'gt.txt'), np.asarray(gts), fmt='%d', delimiter=',')

    # Write det.npy
    Path(os.path.join(sequence_dir, seq + '-' + dataset_type, target_det_dir)).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(os.path.join(sequence_dir, seq + '-' + dataset_type,
                                      target_det_dir), seq + '-' + dataset_type + '.npy'), np.asarray(dets), allow_pickle=False)
