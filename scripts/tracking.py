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
import copy

br = CvBridge()
np.set_printoptions(precision=3)

class Tracking():
    def __init__(self):
        # Pub/sub
        self.color_all_bb_pub = rospy.Publisher('color_bb_all', Image, queue_size=1)
        self.color_det_bb_pub = rospy.Publisher('color_bb_det', Image, queue_size=1)
        self.color_sub = rospy.Subscriber('color', Image, self.im_clbk)
        self.thermal_det_mapped_sub = rospy.Subscriber('thermal_det_mapped', Detection2DArray, self.thermal_det_clbk)
        self.color_det_sub = rospy.Subscriber('color_det', Detection2DArray, self.color_det_clbk)
        self.im = None
        self.im_det_only = None
        self.mot_tracker = Sort(max_age=3, min_hits=1)
        self.track_bbs_ids = None
        # https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle/42091037
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.dims = tuple([int(x) for x in rospy.get_param('dims/color')])
        self.masks = {'color': np.zeros(self.dims, np.uint8), 'thermal': np.zeros(self.dims, np.uint8)}
        self.screens = {'color': np.full(self.dims, 210), 'thermal': np.full(self.dims, 30)}
        # self.screens = {'color': np.full(self.dims, 210, np.uint8), 'thermal': np.full(self.dims, 30, np.uint8)}
        # assert([self.masks[x].shape == self.screens[x].shape for x in self.masks.keys()])

    def hex2rgb(self, h):
        # https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
        h = h.strip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def overlay_bb_trk(self, im, x1, y1, x2, y2, id):
        color = self.hex2rgb(self.color_cycle[id % len(self.color_cycle)])
        im = cv2.putText(im, '#' + str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 3)
        return im

    def add_to_mask(self, x1, y1, x2, y2, image_type):
        self.masks[image_type] = cv2.rectangle(self.masks[image_type], (x1, y1), (x2, y2), (255, 255, 255), 2)

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

    def print_state(self, image_type, stamp):
        rospy.loginfo('Update from %s at %f', image_type, float(stamp.secs + stamp.nsecs/1e9))
        # print(self.track_bbs_ids)
        if self.im is not None:
            for trk in self.track_bbs_ids:
                trk = [int(x) for x in trk]
                self.im = self.overlay_bb_trk(self.im, trk[0], trk[1], trk[2], trk[3], trk[4])
            # Convert back and publish
            msg = br.cv2_to_imgmsg(self.im, encoding='bgr8')
            self.color_all_bb_pub.publish(msg)
            self.publish_det_only_image()

    def thermal_det_clbk(self, msg):
        dets = []        
        self.masks['thermal'].fill(0)
        for det in msg.detections:
            tlbr = self.xywh2tlbr([det.bbox.center.x, det.bbox.center.y, det.bbox.size_x, det.bbox.size_y])
            tlbr = [int(x) for x in tlbr]
            self.add_to_mask(tlbr[0], tlbr[1], tlbr[2], tlbr[3], 'thermal')
            det = np.array([tlbr[0], tlbr[1], tlbr[2], tlbr[3], det.results[0].score])
            dets.append(det)
        self.track_bbs_ids = self.mot_tracker.update(np.array(dets))
        self.print_state('thermal', msg.header.stamp)

    def color_det_clbk(self, msg):
        dets = []
        self.masks['color'].fill(0)
        assert(np.all(self.masks['color']) == 0)
        for det in msg.detections:
            tlbr = self.xywh2tlbr([det.bbox.center.x, det.bbox.center.y, det.bbox.size_x, det.bbox.size_y])
            tlbr = [int(x) for x in tlbr]
            # print(np.count_nonzero(self.masks['color']))
            self.add_to_mask(tlbr[0], tlbr[1], tlbr[2], tlbr[3], 'color')
            det = np.array([tlbr[0], tlbr[1], tlbr[2], tlbr[3], det.results[0].score])
            dets.append(det)
        self.track_bbs_ids = self.mot_tracker.update(np.array(dets))
        self.print_state('color', msg.header.stamp)

    def im_clbk(self, msg):
        rospy.loginfo('>>> Received image at %f', float(msg.header.stamp.secs + msg.header.stamp.nsecs/1e9))
        self.orig = copy.deepcopy(br.imgmsg_to_cv2(msg))
        self.im = br.imgmsg_to_cv2(msg)
        self.im_det_only = copy.deepcopy(br.imgmsg_to_cv2(msg))

    def publish_det_only_image(self):
        # https://stackoverflow.com/questions/51168268/setting-pixels-values-in-opencv-python        
        self.im_det_only = copy.deepcopy(self.orig)
        self.im_det_only[np.where(self.masks['color'] == 255)] = 255
        self.im_det_only[np.where(self.masks['thermal'] == 255)] = 30
        msg_det_only = br.cv2_to_imgmsg(self.im_det_only, encoding='bgr8')
        self.color_det_bb_pub.publish(msg_det_only)

def main():
    rospy.init_node('mot_tracker', anonymous=True)
    t = Tracking()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main()