#!/usr/bin/python3
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile

pd.set_option('display.max_rows', None)

parent_dir = '/home/ed/Data/CUB_200_2011/CUB_200_2011/'
per_class = 10

def convert_to_darknet(im_width, im_height, bb_x, bb_y, bb_width, bb_height):
    x_darknet = (bb_x + 0.5 * bb_width) / im_width
    y_darknet = (bb_y + 0.5 * bb_height) / im_height
    width_darknet = float(bb_width) / float(im_width)
    height_darknet = float(bb_height) / float(im_height)
    return (x_darknet, y_darknet, width_darknet, height_darknet)

# Prepare DataFrame elements
df = pd.DataFrame()
includes = []
darknet_boxes = []
labels = []
shapes = []
tts = []
im_paths = []

# Calculate bounding boxes in darknet format
with open(parent_dir + 'images.txt', 'r') as images, \
     open(parent_dir + 'bounding_boxes.txt', 'r') as boxes:
    for image, box in zip(images.readlines(), boxes.readlines()): 
        # Path
        im_path = image.split(' ')[1].split('\n')[0]
        im_paths.append(im_path)
        im_path = parent_dir + 'images/' + im_path

        # Shape
        im = cv2.imread(im_path)
        shape = im.shape
        rows = shape[0]
        cols = shape[1]
        shapes.append(shape)

        # Bounding box in Darknet format
        b = box.split(' ')[1:]
        x = int(float(b[0]))
        y = int(float(b[1]))
        w = int(float(b[2]))
        h = int(float(b[3]))
        darknet = convert_to_darknet(cols, rows, x, y, w, h)
        darknet = [round(x, 3) for x in darknet]
        darknet_boxes.append(darknet)

# Class labels
with open(parent_dir + 'image_class_labels.txt', 'r') as image_class_labels:
    count = 0
    data_labels = image_class_labels.readlines()
    curr = data_labels[0].split(' ')[1].split('\n')[0]
    prev = data_labels[0].split(' ')[1].split('\n')[0]
    includes = []
    for row in data_labels:
        curr = row.split(' ')[1].split('\n')[0]
        if curr != prev:
            count = 0 # assumes ascending labels
        if count < per_class:
            count += 1
            include = True
        else: 
            include = False            
        prev = curr
        labels.append(curr)
        includes.append(include)

# Test/train split
with open(parent_dir + 'train_test_split.txt', 'r') as tt:
    for row in tt.readlines():           
        tts.append(row.split(' ')[1].split('\n')[0])

# Assemble the DataFrame
df['im_path'] = pd.Series(im_paths)
df['shape'] = pd.Series(shapes)
df['box'] = pd.Series(darknet_boxes)
df['label'] = pd.Series(labels)
df['tt'] = pd.Series(tts)
df['include'] = pd.Series(includes)

manif = {'train': [], 'test': []}

# Copy images into train/test directory
for index, row in df.iterrows():
    if row['include']:
        # Image
        f = row['im_path']
        src = parent_dir + 'images/' + f
        f = f.split('/')[1]
        if row['tt'] == '1':
            suffix = 'train/'
        else:
            suffix = 'test/'
        dst = parent_dir + 'upload/images/' + suffix + f
        copyfile(src, dst)
        if row['tt'] == '1':
            manif['train'].append(dst)
        else:
            manif['test'].append(dst)

        # Bounding box
        f = f.split('.')[0] + '.txt'
        with open(parent_dir + 'upload/labels/' + suffix + f, 'w') as txt_file:
            class_label = 0 # only one class, bird
            packed = str(class_label) + ' '
            packed += str(row['box'][0]) + ' '
            packed += str(row['box'][1]) + ' '
            packed += str(row['box'][2]) + ' '
            packed += str(row['box'][3]) + ' '
            txt_file.write(packed)

with open(parent_dir + 'upload/train.txt', 'w') as f:
    for line in manif['train']:
        line = line.split('upload/')[1]
        line = './' + line
        f.write(line + '\n')

with open(parent_dir + 'upload/test.txt', 'w') as f:
    for line in manif['test']:
        line = line.split('upload/')[1]
        line = './' + line
        f.write(line + '\n')