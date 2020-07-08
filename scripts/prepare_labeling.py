#!/usr/bin/env python3

import csv
import rospy
import os
import rosbag
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

# Set stuff up
br = CvBridge()
output_files = []

# Parameters
throttle_rate = 5 # Hz
color_topics = {'rect': 'color/camera/image_rect_color_throttle/compressed', 'raw': '/color/camera/image_rect_color/compressed'}
thermal_topic = 'thermal/camera/image_raw_throttle/compressed'
scale = {'color': 0.4, 'thermal': 7.5}
dest = '/home/ed/Data/label_images/'

# Pre-process parameters
throttle_period = 1.0 / throttle_rate

def get_bag_paths(parent_dir):
    bag_paths = []
    for root, dirs, files in os.walk(parent_dir):
        files = sorted(files)
        for file in files:
            if file.endswith(".bag"):
                bag_paths.append(root + file)
    return bag_paths

def construct_filename(bag_path, stamp, format):
    date = bag_path.split('/')[4].split('_test')[0] # assumes this sort of structure /home/ed/Data/2020-06-24-11-33-00_test
    secs = str(stamp.secs)
    nsecs = str(stamp.nsecs)
    return date + '_' + secs + '_' + nsecs + '_' + source + '.jpg'

def ros_time_to_float(ros_time):
    return ros_time.secs + float(ros_time.nsecs) / 1e9

def save_frames_from_bag(bag_path, source, start, end, format):
    bag = rosbag.Bag(bag_path)
    if source == 'color':   
        topic = color_topics[format]
    elif source == 'thermal':
        topic = thermal_topic
    t_prev = 0.0        
    for _, msg, t in bag.read_messages(topics=topic):
        t = t.to_sec()
        if t >= start and t <= end and (t - t_prev) >= throttle_period:
            im = br.compressed_imgmsg_to_cv2(msg)
            im = cv2.resize(im, (int(im.shape[1] * scale[source]), int(im.shape[0] * scale[source])))
            filename = construct_filename(bag_path, msg.header.stamp, format)
            cv2.imwrite(dest + filename, im)
            output_files.append(filename)
            t_prev = t 
    bag.close()


# read input files
# bag path, start time, end time
with open('/home/ed/sm_ws/src/chicken_monitoring/cfg/prepare_labels.csv', 'r') as f:
    rows = csv.reader(f)
    for row in rows:
        bag_dir = row[2]
        bags = get_bag_paths(bag_dir)
        for bag in bags:
            # for source in ['thermal']:
            for source in ['color']:
                print('Processing', bag, source, row[0], row[1], row[3])
                start = float(row[0])
                end = float(row[1])
                format = row[3]
                save_frames_from_bag(bag, source, start, end, format)

with open('/home/ed/sm_ws/src/chicken_monitoring/cfg/output_files.csv', 'w') as f:
    writer = csv.writer(f)
    for file in output_files:
        writer.writerow(file)
        