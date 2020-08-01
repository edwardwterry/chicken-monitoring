#!/usr/bin/env python3

import sys
sys.path.append('/home/ed/sort/')

import numpy as np
import rospy
import yaml
import cv2
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from sort import * 
from matplotlib import pyplot as plt

br = CvBridge()
np.set_printoptions(precision=3)

class Tracking():
    def __init__(self):
        # Pub/sub
        self.color_all_bb_pub = rospy.Publisher('color_bb_all', Image, queue_size=1)
        self.color_sub = rospy.Subscriber('color', Image, self.im_clbk)
        self.thermal_det_mapped_sub = rospy.Subscriber('thermal_det_mapped', Detection2DArray, self.thermal_det_clbk)
        self.color_det_sub = rospy.Subscriber('color_det', Detection2DArray, self.color_det_clbk)
        self.im = None
        self.mot_tracker = Sort()
        self.track_bbs_ids = None
        # https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle/42091037
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def hex2rgb(self, h):
        # https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
        h = h.strip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def overlay_bb(self, im, x1, y1, x2, y2, id):
        color = self.hex2rgb(self.color_cycle[id % len(self.color_cycle)])
        im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 3)
        im = cv2.putText(im, '#' + str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return im

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

    def print_state(self, image_type):
        print ('State update after', image_type, 'detection\n', self.track_bbs_ids)
        if self.im is not None:
            for trk in self.track_bbs_ids:
                trk = [int(x) for x in trk]
                self.im = self.overlay_bb(self.im, trk[0], trk[1], trk[2], trk[3], trk[4])
            # Convert back and publish
            msg = br.cv2_to_imgmsg(self.im, encoding='rgb8')
            self.color_all_bb_pub.publish(msg)

    def thermal_det_clbk(self, msg):
        dets = []        
        for det in msg.detections:
            tlbr = self.xywh2tlbr([det.bbox.center.x, det.bbox.center.y, det.bbox.size_x, det.bbox.size_y])
            tlbr = [int(x) for x in tlbr]
            det = np.array([tlbr[0], tlbr[1], tlbr[2], tlbr[3], det.results[0].score])
            dets.append(det)
        self.track_bbs_ids = self.mot_tracker.update(np.array(dets))
        self.print_state('thermal')

    def color_det_clbk(self, msg):
        dets = []
        for det in msg.detections:
            tlbr = self.xywh2tlbr([det.bbox.center.x, det.bbox.center.y, det.bbox.size_x, det.bbox.size_y])
            tlbr = [int(x) for x in tlbr]
            det = np.array([tlbr[0], tlbr[1], tlbr[2], tlbr[3], det.results[0].score])
            dets.append(det)
        self.track_bbs_ids = self.mot_tracker.update(np.array(dets))
        self.print_state('color')

    def im_clbk(self, msg):
        self.im = br.imgmsg_to_cv2(msg)

def main():
    rospy.init_node('mot_tracker', anonymous=True)
    t = Tracking()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main()