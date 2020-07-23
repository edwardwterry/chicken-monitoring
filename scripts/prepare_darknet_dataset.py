#!/usr/bin/python3

import json
import yaml
import random
from shutil import copyfile

random.seed(0)

# Files and directories
host_data_dir = '/home/ed/Data/label/'
host_staging_dir = '/home/ed/Data/label/upload/'
s3_label_dir = 's3://chicken-monitoring-2020/label/'
s3_dataset_dir = 'data/'
s3_images_dir = 'images/'
s3_labels_dir = 'labels/'
manifest_file = 'output.manifest'

# Dataset metadata
train_test_split = 0.8
tt = ['train', 'test']
class_id = '0' # only one class
out = {}
for t in tt:
    out[t] = []

# Convert from pixel bounding boxes to darknet format
def convert_to_darknet(im_width, im_height, bb_x, bb_y, bb_width, bb_height):
    x_darknet = (bb_x + 0.5 * bb_width) / im_width
    y_darknet = (bb_y + 0.5 * bb_height) / im_height
    width_darknet = float(bb_width) / float(im_width)
    height_darknet = float(bb_height) / float(im_height)
    return (x_darknet, y_darknet, width_darknet, height_darknet)

def create_label_file_contents(data):
    packed = ''
    job_name = data['source-ref'].split('/')[-3] + '-' + data['source-ref'].split('/')[-2]
    boxes = data[job_name]['annotations']
    im_width = data[job_name]['image_size'][0]['width']
    im_height = data[job_name]['image_size'][0]['height']
    for box in boxes:
        bb = convert_to_darknet(im_width, im_height, box['left'], box['top'], box['width'], box['height'])
        bb = [round(x, 3) for x in bb]
        packed += class_id + ' '
        packed += str(bb[0]) + ' '
        packed += str(bb[1]) + ' '
        packed += str(bb[2]) + ' '
        packed += str(bb[3]) + '\n'
    return packed

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

# Process the selected output.manifest files, each of which contains the bounding box labels
for m in manifest_file_paths:
    with open(m) as f:
        for entry in f:
            data = json.loads(entry)
            # print(data['source-ref'])
            # Prepare filenames
            im_filename = data['source-ref'].split('/')[-1]
            label_filename = im_filename.split('.')[0] + '.txt'

            # Put in train bucket with train_test_split probability
            bucket = tt[random.random() > train_test_split]

            # Write the bounding box label file
            label_contents = create_label_file_contents(data)
            label_write_path = host_staging_dir + s3_labels_dir + bucket + '/' + label_filename
            with open(label_write_path, 'w+') as f:
                f.write(label_contents)

            # Add to the list of image locations
            s3_image_path = s3_dataset_dir + s3_images_dir + bucket + '/' + im_filename
            out[bucket].append(s3_image_path)

            # Move the images to their proper location
            img_src = host_data_dir + data['source-ref'].split(s3_label_dir)[1]
            img_dst = host_staging_dir + s3_images_dir + bucket + '/' + im_filename
            copyfile(img_src, img_dst)                

# Write the files which point to the image locations
for t in tt:
    with open(host_staging_dir + t + '.txt', 'w+') as f:
        for line in out[t]:
            f.write(line + '\n')