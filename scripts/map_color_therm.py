#!/usr/bin/env python3

import sys
import numpy as np
import rospy
import yaml
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import BoundingBox2D
import datetime

class Map():
    def __init__(self):
        # Camera extrinsics
        extr = rospy.get_param('extrinsics') # date, T_color_therm
        for k in extr.keys():
            extr[k] = np.array(extr[k]) # convert each one to np.array

        # Pub/sub
        self.pub = rospy.Publisher('thermal_bb', Detection2DArray, queue_size=1)
        self.sub = rospy.Subscriber('thermal_bb_mapped', Detection2DArray, self.clbk)
        
    def month_day_to_str(m, d):
        if m < 10:
            m = '0' + str(m)
        else: 
            m = str(m)
        if d < 10:
            d = '0' + str(d)
        else: 
            d = str(d)
        return m + '_' + d

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

    def map(in_box, T):
        '''
        in_box is a BoundingBox2D
        T is 3x3 np.array
        '''
        tlbr = self.xywh2tlbr(in_box.center.x, in_box.center.y, in_box.size_x, in_box.size_y)
        tl = [tlbr[0], tlbr[1], 1.0] # convert to homogeneous form
        br = [tlbr[2], tlbr[3], 1.0] # convert to homogeneous form
        tl_T = np.dot(T, np.array(tl))
        br_T = np.dot(T, np.array(br))
        tlbr_T = [tl_T[0], tl_T[1], br_T[0], br_T[1]]
        xywh = self.tlbr2xywh(tlbr_T)
        out_box = BoundingBox2D()
        out_box.center.x = xywh[0]
        out_box.center.y = xywh[1]
        out_box.size_x = xywh[2]
        out_box.size_y = xywh[3]
        return out_box

    def therm_det_clbk(self, msg):
        # Find out which extrinsics to look up (they change from one day to another)
        date = datetime.datetime.from_timestamp(msg.header.stamp.secs)
        md = month_day_to_str(date.month, date.day)
        T = extr[md]

        # Modify bounding boxes in place
        for det in msg.detections:
            det.bbox = map(det.bbox, T)

        # Republish
        self.pub.publish(msg)

def main(args):
    rospy.init_node('map_color_therm', anonymous=True)
    m = Map()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main(sys.argv)    