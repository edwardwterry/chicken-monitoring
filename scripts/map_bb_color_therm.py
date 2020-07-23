# This file takes in an edited color bounding box annotation file
# and converts it to the corresponding thermal image

#!/usr/bin/env python3
import numpy as np
import yaml

# Directories
host_data_dir = '/home/ed/Data/label/'

# read in extrinsics
T_color_therm =  np.array([[ 12.813, 0.112, 548.383],
 [ -0.077, 15.103,  48.662],
 [  0.      ,0.      ,1.   ]])
T_therm_color =  np.array([[ 0.078,  -0.001, -42.77 ],
 [  0.,      0.066,  -3.441],
 [ -0.,      0.,      1.]])

dims = {'color': (1920, 1080), 'therm': (80, 60)}

# Collect all the manifest files to be generated
manifest_file_paths = []
with open(host_data_dir + 'output_manifests.yaml') as f:
    data = yaml.safe_load(f)
    image_types = []
    for image_type, seqs in data.items():
        for seq in seqs:
            s = list(seq.keys())[0]
            if seq[s]:
                manifest_file_paths.append(host_data_dir + image_type + '/' + s + '/' + manifest_file)

def pix2fract(pix, image_type):
    return (pix[0] / dims[image_type][0], pix[1] / dims[image_type][1])    

def fract2pix(fract, image_type):
    return (fract[0] * dims[image_type][0], fract[1] * dims[image_type][1])    

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
    br_color_pix = [tl_color_pix[0], tl_color_pix[1], 1.0]
    tl_therm_pix = np.dot(T_therm_color, np.array(tl_color_pix))
    br_therm_pix = np.dot(T_therm_color, np.array(br_color_pix))
    tl_therm_fract = pix2fract((tl_therm_pix[0], tl_therm_pix[1]), 'therm')
    br_therm_fract = pix2fract((br_therm_pix[0], br_therm_pix[1]), 'therm')
    return (tl_therm_fract[0], tl_therm_fract[1], br_therm_fract[0], br_therm_fract[1])

# center and width/height to top left, bottom right
# fractions
def xywh2tlbr(fract):
    x = fract[0]
    y = fract[1]
    w = fract[2]
    h = fract[3]
    return (x - w/2, y - h/2, x + w/2, y + h/2)

# top left, bottom right to center and width/height
# fractions
def tlbr2xywh(fract):
    x0 = fract[0]
    y0 = fract[1]
    x1 = fract[2]
    y1 = fract[3]
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0)

def color2therm_filename(color_filename):
    return color_filename.replace('color', 'therm')

# read in label file
with open(color_label_file, 'r') as f_color, open(therm_label_file, 'w') as f_therm:
    packed = ''
    for row in f_color:
        packed += class_id + ' '
        class_id = row.split(' ')[0]
        bb_color_fract = [float(x) for x in row.split(' ')[1:]]
        bb_therm_fract = []
        for fract in bb_color_fract:
            bb_color_fract_tlbr = xywh2tlbr(fract)
            bb_therm_fract_tlbr = color2therm(bb_color_fract_tlbr)
            bb_therm_fract_xywh = tlbr2xywh(bb_therm_fract_tlbr)
        packed += str(round(bb_therm_fract_xywh[0], 3)) + ' '
        packed += str(round(bb_therm_fract_xywh[1], 3)) + ' '
        packed += str(round(bb_therm_fract_xywh[2], 3)) + ' '
        packed += str(round(bb_therm_fract_xywh[3], 3)) + '\n'
    f_therm.write(packed)