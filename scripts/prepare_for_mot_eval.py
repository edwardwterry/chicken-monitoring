#!/usr/bin/env python3

import numpy as np
from shutil import copyfile
import cv2
import os

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
# {perturb bb, ignore det}, TODO FP
alterations = {'bb_noise': False, 'ignore_det': False}


class MOT():
    def __init__(self, frame_id, track_id, tl_x, tl_y, w, h, conf, cls_, feat=None):
        self.frame_id = frame_id
        self.track_id = track_id
        self.tl_x = tl_x
        self.tl_y = tl_y
        self.w = w
        self.h = h
        self.conf = conf
        self.cls = cls_
        self.feat = feat
        self.populate_arr()

    def populate_arr(self):
        self.arr = [self.frame_id, self.track_id, self.tl_x,
                    self.tl_y, self.w, self.h, self.conf, self.cls]
        if self.feat:
            self.arr.append(self.feat)

    # def to_csv(self):
    #     ret = ''
    #     for elm in self.arr:
    #         ret += elm + ','
    #     # take away last comma
    #     ret = ret[:-1]
    #     return ret

    def to_npy(self):
        return np.asarray(self.arr)


def corrupt(yolo, method):
    if method == 'bb_noise':
        raise NotImplementedError
    elif method == 'ignore_det':
        raise NotImplementedError
    conf = 1.0
    return yolo, conf


def alter_yolo_bb(yolo, conf=1.0):
    for a, do in alterations.items():
        if do:
            yolo, conf = corrupt(yolo, a)
    return yolo, conf


def yolo_to_mot(frame_id, yolo, im_w, im_h, det_gt, feat=[], conf=1.0):
    yolo, conf = alter_yolo_bb(yolo)
    tl_x = float(yolo[1]) * im_w - 0.5 * im_w
    tl_y = float(yolo[2]) * im_h - 0.5 * im_h
    w = float(yolo[3]) * im_w
    h = float(yolo[4]) * im_h
    cls_ = yolo[0]
    if det_gt == 'det':
        mot = MOT(frame_id, -1, tl_x, tl_y, w, h, conf, -1, -1, feat)
    elif det_gt == 'gt':
        track_id = yolo[5]
        mot = MOT(frame_id, track_id, tl_x, tl_y, w, h, 1.0, cls_, 1.0)
    else:
        raise NotImplementedError
    return mot


def replace_suffix(infile, suffix):
    # May only be one instance of '.' in the string
    return infile.split('.')[0] + suffix


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
        try:
            h, w, c = cv2.imread(os.path.join(root, f)).shape
        except:
            h, w, _ = cv2.imread(os.path.join(root, f)).shape

        # Open the corresponding annotation text file
        file_txt = replace_suffix(f, '.txt')
        with open(os.path.join(os.path.join(existing_ann_dir, seq + trk_suffix), file_txt), 'r') as myfile:
            for line in myfile.readlines():
                det = yolo_to_mot(frame_id, line, w, h,
                                  'det', feat)  # TODO feat
                gt = yolo_to_mot(frame_id, line, w, h, 'gt')
                dets.append(det.to_npy())
                gts.append(gt.to_npy())

    # Write gt.txt
    np.savetxt(os.path.join(os.path.join(sequence_dir, seq,
                                         dataset_type, target_gt_dir), 'gt.txt'), np.asarray(gts))

    # Write det.npy
    np.save(os.path.join(os.path.join(sequence_dir, seq, dataset_type,
                                      target_det_dir), 'det.npy'), np.asarray(dets), allow_pickle=False)
