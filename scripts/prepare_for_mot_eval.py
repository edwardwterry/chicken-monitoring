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
sequence_dir = '/home/ed/Data/MOTeval/'
dataset_type = 'nominal'
target_im_dir = 'img1'
target_det_dir = 'det'
target_gt_dir = 'gt'
trk_suffix = '_trk'

# Alterations to apply
# # {perturb bb, ignore det}, TODO FP
# alterations = {'bb_noise': False, 'ignore_det': False}


# def corrupt(yolo, method):
#     if method == 'bb_noise':
#         raise NotImplementedError
#     elif method == 'ignore_det':
#         raise NotImplementedError
#     conf = 1.0
#     return yolo, conf


# def alter_yolo_bb(yolo, conf=1.0):
#     for a, do in alterations.items():
#         if do:
#             yolo, conf = corrupt(yolo, a)
#     return yolo, conf

fn = FeatureNet()

for root, dirs, files in os.walk(os.path.join(existing_im_dir, seq), topdown=False):
    files = sorted(files)
    frame_id = 1
    dets = []
    gts = []
    for f in files:
        dst = f'{frame_id:06}' + '.jpg'
        copyfile(os.path.join(root, f), os.path.join(os.path.join(
            sequence_dir, seq, dataset_type, target_im_dir), dst))

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
                bbox = BoundingBox(line, format='txt')
                feat = fn.get_feature_vector(im, bbox, w, h)
                det = bbox.to_mot(frame_id, w, h, 'det', feat=feat)
                gt = bbox.to_mot(frame_id, w, h, 'gt')
                dets.append(det)
                gts.append(gt)

        frame_id += 1

    # Write gt.txt
    np.savetxt(os.path.join(os.path.join(sequence_dir, seq,
                                         dataset_type, target_gt_dir), 'gt.txt'), np.asarray(gts))

    # Write det.npy
    np.save(os.path.join(os.path.join(sequence_dir, seq, dataset_type,
                                      target_det_dir), 'det.npy'), np.asarray(dets), allow_pickle=False)
