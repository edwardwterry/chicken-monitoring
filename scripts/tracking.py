#!/usr/bin/env python3

import sys
import numpy as np
import rospy
import yaml
import cv2
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image

br = CvBridge()

class Tracking():
    def __init__(self):

        # Pub/sub
        self.color_all_bb_pub = rospy.Publisher('color_bb_all', Image, queue_size=1)
        self.color_sub = rospy.Subscriber('color', Image, self.im_clbk)
        self.thermal_det_mapped_sub = rospy.Subscriber('thermal_det_mapped', Detection2DArray, self.clbk)
        self.im = None

    def overlay_bb(self, im, x1, y1, x2, y2, conf):
        return cv2.rectangle(im, (x1, y1), (x2, y2), (255 * conf, 0, 0), 3)

    # center and width/height to top left, bottom right
    def xywh2tlbr(self, pt):
        x = pt[0]
        y = pt[1]
        w = pt[2]
        h = pt[3]
        return (x - w/2, y - h/2, x + w/2, y + h/2)

    # top left, bottom right to center and width/height
    def tlbr2xywh(self, pt):
        x0 = pt[0]
        y0 = pt[1]
        x1 = pt[2]
        y1 = pt[3]
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0)

    def thermal_det_sub(self, msg):
        if self.im:
            for det in msg.detections:
                tlbr = self.xywh2tlbr([det.bbox.center.x, det.bbox.center.y, det.bbox.size_x, det.bbox.size_y])
                self.im = self.overlay_bb(self.im, tlbr[0], tlbr[1], tlbr[2], tlbr[3], det.results[0].score)

            # Convert back and publish
            out_msg = br.cv2_to_imgmsg(self.im, encoding='rgb8')
            self.color_all_bb_pub.publish(out_msg)

    def im_clbk(self, msg):
        self.im = br.imgmsg_to_cv2(msg)

def main(args):
    rospy.init_node('map_color_therm', anonymous=True)
    t = Tracking()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main(sys.argv)