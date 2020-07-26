#!/usr/bin/env python3

# The purpose of this script is to pad leading zeros
# after the mistake during file writing initially

import os

dir = '/home/ed/Data/frames/labels/thermal/'
for root, dirs, files in os.walk(dir):
    for f in files:
        f_ = f.split('_')
        if f.endswith('.jpg') or f.endswith('.txt'):
            nsecs = f_[2]
            if len(nsecs) < 9:
                while len(nsecs) < 9:
                    nsecs = '0' + nsecs
                f_[2] = nsecs
                src = os.path.join(root, f)
                rename = '_'.join(f_)
                dst = os.path.join(root, rename)
                print(dst)
                os.rename(src, dst)