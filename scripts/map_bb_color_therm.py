#!/usr/bin/env python3

# This file takes in an edited color bounding box annotation file
# and converts it to the corresponding thermal image

import numpy as np
import yaml
import os
import unittest
import itertools

# Directories
host_data_dir = '/home/ed/Data/frames/'
images_dir = 'images/'
labels_dir = 'labels/'

# read in extrinsics
T_color_therm =  np.array([[ 12.813, 0.112, 548.383],
 [ -0.077, 15.103,  48.662],
 [  0.      ,0.      ,1.   ]])
T_therm_color =  np.array([[ 0.078,  -0.001, -42.77 ],
 [  0.,      0.066,  -3.441],
 [ -0.,      0.,      1.]])

dims = {'color': (1920, 1080), 'thermal': (80, 60)}
times = {'color': [], 'thermal': []}
image_files = {}
label_files = {}

class_id = '0'

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Input: 2 element array or tuple of (floating point) pixel values, image_type
# Output: 2 element tuple of point represented as a fraction of width and height
def pix2fract(pix, image_type):
    return (pix[0] / dims[image_type][0], pix[1] / dims[image_type][1])    

# Input: 2 element array or tuple of (floating point) fraction of width/height, image_type
# Output: 2 element tuple of point represented as (floating point) pixels
def fract2pix(fract, image_type):
    return (fract[0] * dims[image_type][0], fract[1] * dims[image_type][1])

# Calculates the optimal index offset between color and thermal frames
# Input: a, b are each a list of floats, in ascending order
# a, b may be of different lengths
# Output: offset and whether or not the order of arrays was flipped
def align(a, b):
    flip = False
    if len(b) > len(a):
        flip = not flip
        temp = b
        b = a
        a = temp
    window = len(b)
    offset = 0
    a = np.array(a)
    b = np.array(b)
    err_prev = 0.0
    # print (a, b)
    while offset < (len(a) - len(b)):
        err = np.sum(np.absolute(a[offset:window + offset] - b))
        offset += 1
        if err > err_prev:
            break
        err_prev = err
    if flip:
        offset *= -1
    return offset, flip

# Input: filename with format e.g. /path/to/2020-07-02-09-27-00_1593708463_278986665_color.jpg
# Output: floating point time
def filename2time(filename):
    sec, nsec = filename.split('_')[1:3]
    return float(sec) + float(nsec) / 1e9

# Input: top left & bottom right color fractions
# Output: top left & bottom right thermal fractions
def color2therm(color_fract):
    x0 = color_fract[0]
    y0 = color_fract[1]
    x1 = color_fract[2]
    y1 = color_fract[3]
    tl_color_pix = fract2pix((x0, y0), 'color')
    br_color_pix = fract2pix((x1, y1), 'color')
    tl_color_pix = [tl_color_pix[0], tl_color_pix[1], 1.0] # convert to homogeneous form
    br_color_pix = [br_color_pix[0], br_color_pix[1], 1.0]
    tl_therm_pix = np.dot(T_therm_color, np.array(tl_color_pix))
    br_therm_pix = np.dot(T_therm_color, np.array(br_color_pix))
    tl_therm_fract = pix2fract((tl_therm_pix[0], tl_therm_pix[1]), 'thermal')
    br_therm_fract = pix2fract((br_therm_pix[0], br_therm_pix[1]), 'thermal')
    return (tl_therm_fract[0], tl_therm_fract[1], br_therm_fract[0], br_therm_fract[1])

# center and width/height to top left, bottom right
def xywh2tlbr(pt):
    x = pt[0]
    y = pt[1]
    w = pt[2]
    h = pt[3]
    return (x - w/2, y - h/2, x + w/2, y + h/2)

# top left, bottom right to center and width/height
def tlbr2xywh(pt):
    x0 = pt[0]
    y0 = pt[1]
    x1 = pt[2]
    y1 = pt[3]
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0)

def clip(pt):
    if pt < 0.0:
        pt = 0.0
    elif pt > 1.0:
        pt = 1.0
    return pt

# Segment all image files by sequence, then by image type
# https://www.tutorialspoint.com/python/os_walk.htm
for root, dirs, files in os.walk(host_data_dir + images_dir, topdown=False):
    for name in sorted(files):
        seq = root.split('/')[-1]
        image_type = root.split('/')[-2]
        if not seq in image_files.keys():
            image_files[seq] = {}
            label_files[seq] = {}
        if not image_type in image_files[seq]:
            image_files[seq][image_type] = []
            label_files[seq][image_type] = []
        name = name.split('.jpg')[0]
        image_files[seq][image_type].append(name)

# Build the lists for each image_type
for seq, files in image_files.items():
    color_label_files = []
    thermal_label_files = []
    times['color'] = []
    times['thermal'] = []
    # "create" the corresponding .txt file for the given .jpg
    for f in files['color']:
        filename = os.path.join(host_data_dir + labels_dir + 'color/' + seq, f + '.txt')
        label_files[seq]['color'].append(filename)
        times['color'].append(filename2time(filename))
    for f in files['thermal']:
        filename = os.path.join(host_data_dir + labels_dir + 'thermal/' + seq, f + '.txt')
        label_files[seq]['thermal'].append(filename)
        times['thermal'].append(filename2time(filename))
    offset, flip = align(times['color'], times['thermal'])
    print (seq, offset, flip, len(times['color']), len(times['thermal']))

    # print (offset, flip)
# quit()
# read in label file
for seq in label_files.keys():
    for c, t in itertools.zip_longest(label_files[seq]['color'], label_files[seq]['thermal']):
        if c is not None and t is not None:
            with open(c, 'r') as f_color, open(t, 'w') as f_therm:
                print('newfile')
                packed = ''
                for row in f_color:
                    # print ('color' + row)
                    packed += class_id + ' '
                    class_id = row.split(' ')[0]
                    bb_color_fract = [float(x) for x in row.split(' ')[1:]]
                    bb_color_fract_tlbr = xywh2tlbr(bb_color_fract)
                    print ('bb_color_fract_tlbr', bb_color_fract_tlbr)
                    bb_therm_fract_tlbr = color2therm(bb_color_fract_tlbr)
                    print ('bb_therm_fract_tlbr', bb_therm_fract_tlbr)
                    bb_therm_fract_tlbr = tuple([clip(x) for x in bb_therm_fract_tlbr])
                    print ('bb_therm_fract_tlbr', bb_therm_fract_tlbr)
                    bb_therm_fract_xywh = tlbr2xywh(bb_therm_fract_tlbr)
                    print ('bb_therm_fract_xywh', bb_therm_fract_xywh)
                    packed += str(round(bb_therm_fract_xywh[0], 3)) + ' '
                    packed += str(round(bb_therm_fract_xywh[1], 3)) + ' '
                    packed += str(round(bb_therm_fract_xywh[2], 3)) + ' '
                    packed += str(round(bb_therm_fract_xywh[3], 3)) + '\n'
                # print ('thermal' + packed)
                # f_therm.write(packed)