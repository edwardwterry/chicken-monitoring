#!/usr/bin/python3

import json
import yaml
import random
from shutil import copyfile

random.seed(0)

# Files and directories
host_data_dir = '/home/ed/Data/label/'
host_staging_dir = '/home/ed/Data/label/upload/'
labels_dir = 'labels/'
manifest_file = 'output.manifest'

# Dataset metadata
tt = ['train', 'test']
out = {}
for t in tt:
    out[t] = []

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