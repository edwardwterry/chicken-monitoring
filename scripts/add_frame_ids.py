#!/usr/bin/env python

import rosbag
from vision_msgs.msg import Detection2DArray

bag = rosbag.Bag('/home/ed/Data/frames/rosbag/seq09_det_thermal.bag', 'r')
outbag = rosbag.Bag('/home/ed/Data/frames/rosbag/seq09_det_thermal_mod.bag', 'w')

for topic, msg, t in bag.read_messages(topics=['thermal_det']):
    out_msg = Detection2DArray()
    out_msg = msg
    out_msg.header.frame_id = 'thermal'
    # outbag.write('thermal_det', out_msg)