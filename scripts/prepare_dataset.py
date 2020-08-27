#!/usr/bin/python3

'''
This script has two purposes:
1. Relocate image and label files to adhere to the following directory
structure which is used for training
- data
- classes.names
- train.txt
- test.txt
-- images
--- train
--- test
-- labels
--- train
--- test

2. Create the train.txt and test.txt files which 
are used to tell the training script where to find the images
for each category. The training script will look up the corresponding 
*.txt file in the labels directory
'''

import yaml
import os
from shutil import copyfile

with open('/home/ed/sm_ws/src/chicken-monitoring/cfg/prepare_dataset.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Write the files which point to the image locations
for bucket, seqs in params['split'].items():
    with open(params['write_path'] + bucket + '.txt', 'w+') as f:
        for seq in seqs:
            print('walking', params['read_path'] + params['images_path'] + seq)
            for root, dirs, files in os.walk(params['read_path'] + params['images_path'] + params['image_type'] + seq, topdown=False):
                for name in sorted(files):
                    aws_image_path = params['aws_prefix'] + params['images_path'] + bucket + '/' + name
                    txt_name = name.split('/')[-1].split('.')[0] + '.txt'

                    # Check the corresponding text file exists
                    label_src = os.path.join(params['read_path'] + params['labels_path'] + params['image_type'] + seq, txt_name)
                    assert(label_src)

                    # Write the image path
                    f.write(aws_image_path + '\n')

                    # Copy image file
                    img_src = os.path.join(root, name)
                    img_dst = os.path.join(params['write_path'] + params['images_path'] + bucket, name)
                    copyfile(img_src, img_dst)

                    # Copy label file
                    label_dst = os.path.join(params['write_path'] + params['labels_path'] + bucket, txt_name)
                    copyfile(label_src, label_dst)                

# Write the list of classes
with open(params['write_path'] + params['classes_file'], 'w+') as f:
    for c in params['classes']:
        f.write(c)