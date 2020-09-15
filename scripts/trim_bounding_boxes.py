#!/usr/bin/env python3

import os
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

labels_path = '/home/ed/Data/frames/upload/labels'

def trim(orig_center, orig_distance):
    if orig_center + 0.5 * orig_distance > 1.0:
        other = orig_center - 0.5 * orig_distance
        new = 1.0
        return 0.5 * (other + new), new - other
    elif orig_center - 0.5 * orig_distance < 0.0:
        other = orig_center + 0.5 * orig_distance
        new = 0.0
        return 0.5 * (other + new), other - new
    else:
        return orig_center, orig_distance


# https://stackoverflow.com/questions/39086/search-and-replace-a-line-in-a-file-in-python
def replace(file_path, out_lines):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh, 'w') as new_file:
        for line in out_lines:
            new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

for root, dirs, files in os.walk(labels_path, topdown=False):
    for f in files:
        path = os.path.join(root, f)
        with open(path, 'r+') as myfile:
            out_lines = []
            for line in myfile.readlines():
                split = line.split(' ')
                x, y, w, h = [float(x) for x in split[1:]]
                cl = int(split[0])
                x_new, w_new = trim(x, w)
                y_new, h_new = trim(y, h)
                # if not x == x_new:
                    # print ('changed from', x, x_new, w, w_new)
                new_line = str(cl) + ' '
                new_line += str(x_new) + ' '
                new_line += str(y_new) + ' '
                new_line += str(w_new) + ' '
                new_line += str(h_new) + ' '
                new_line += '\n'
                out_lines.append(new_line)
            replace(path, out_lines)
