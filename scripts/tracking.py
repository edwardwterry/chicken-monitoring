#!/usr/bin/env python3

import sys
sys.path.append('/home/ed/sort/')
sys.path.append('/home/ed/deep_sort/')

import numpy as np
import rospy
import yaml
import cv2
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import BoundingBox2D
import datetime
from sensor_msgs.msg import Image
from sort import * 
from deep_sort import nn_matching
from deep_sort import iou_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from matplotlib import pyplot as plt
from chicken_monitoring.srv import ExtractFeatures
from extract_features import extract_features 
import copy
import lap

br = CvBridge()
np.set_printoptions(precision=3)

max_cosine_distance = 0.8
nn_budget = 100

class Utils():
    @staticmethod
    def timestamp_to_date(stamp):
        date = datetime.datetime.fromtimestamp(stamp.secs)
        return Utils.month_day_to_str(date.month, date.day)
    
    @staticmethod
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
    @staticmethod
    def xywh2tlbr(pt):
        x = pt[0]
        y = pt[1]
        w = pt[2]
        h = pt[3]
        return [x - w/2, y - h/2, x + w/2, y + h/2]

    @staticmethod
    def xyah2tlbr(pt):
        x = pt[0]
        y = pt[1]
        a = pt[2]
        h = pt[3]
        w = a * h
        return [x - w/2, y - h/2, x + w/2, y + h/2]

    # top left, bottom right to center and width/height
    @staticmethod
    def tlbr2xywh(pt):
        x0 = pt[0]
        y0 = pt[1]
        x1 = pt[2]
        y1 = pt[3]
        return [0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0]

    @staticmethod
    def hex2rgb(h):
        # https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
        h = h.strip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

class Map():
    def __init__(self):
        # Camera extrinsics
        self.extr = rospy.get_param('extrinsics') # date, T_color_therm
        for k in self.extr.keys():
            self.extr[k] = np.array(self.extr[k]) # convert each one to np.array
        self.scales = rospy.get_param('scales')
        for k in self.scales.keys():
            self.scales[k] = float(self.scales[k])

    def map(self, in_box, date, src_dim='scaled'):
        '''
        in_box is a BoundingBox2D
        T is 3x3 np.array
        '''
        T = self.extr[date]
        tlbr = Utils.xywh2tlbr([in_box.center.x, in_box.center.y, in_box.size_x, in_box.size_y])
        tlbr = [self.scale('thermal', x) if src_dim == 'scaled' else x for x in tlbr]
        tl = [tlbr[0], tlbr[1], 1.0] # convert to homogeneous form
        br = [tlbr[2], tlbr[3], 1.0] # convert to homogeneous form
        tl_T = np.dot(T, np.array(tl))
        br_T = np.dot(T, np.array(br))
        tlbr_T = [tl_T[0], tl_T[1], br_T[0], br_T[1]]
        tlbr_T = [self.scale('color', x) for x in tlbr_T]
        xywh = Utils.tlbr2xywh(tlbr_T)
        out_box = BoundingBox2D()
        out_box.center.x = xywh[0]
        out_box.center.y = xywh[1]
        out_box.size_x = xywh[2]
        out_box.size_y = xywh[3]
        return out_box

    def scale(self, image_type, x):
        return x / self.scales[image_type]

class Tracking():
    def __init__(self):
        # Pub/sub
        self.color_all_bb_pub = rospy.Publisher('color_bb_all', Image, queue_size=1)
        self.color_det_bb_pub = rospy.Publisher('color_bb_det', Image, queue_size=1)
        # self.color_sub = rospy.Subscriber('color', Image, self.im_clbk)
        # self.thermal_det_mapped_sub = rospy.Subscriber('thermal_det', Detection2DArray, self.det_clbk)
        self.color_det_sub = rospy.Subscriber('color_det', Detection2DArray, self.det_clbk)
        self.im = None
        self.im_det_only = None
        self.mot_tracker = Sort(max_age=3, min_hits=1)
        self.track_bbs_ids = None
        # https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle/42091037
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.dims = tuple([int(x) for x in rospy.get_param('dims/color')])
        self.masks = {'color': np.zeros(self.dims, np.uint8), 
                      'thermal': np.zeros(self.dims, np.uint8), 
                      'thermal_fov': np.zeros(self.dims, np.uint8)}
        self.extr_map = Map()
        self.thermal_fov_bb = BoundingBox2D()
        self.thermal_fov_bb.center.x = 40
        self.thermal_fov_bb.center.y = 30
        self.thermal_fov_bb.size_x = 80
        self.thermal_fov_bb.size_y = 60
        self.dets = {'color': [], 'thermal': []}
        self.image_buffer = {'color': [], 'thermal': []}
        # self.extract_features = rospy.ServiceProxy('extract_features', ExtractFeatures)

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

        print('Node initialization complete!')

    def overlay_bb_trk(self, im, x1, y1, x2, y2, id):
        color = Utils.hex2rgb(self.color_cycle[id % len(self.color_cycle)])
        im = cv2.putText(im, '#' + str(id), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 3)
        return im

    def add_to_mask(self, x1, y1, x2, y2, image_type):
        self.masks[image_type] = cv2.rectangle(self.masks[image_type], (x1, y1), (x2, y2), (255, 255, 255), 2)

    def print_state(self, image_type, stamp):
        rospy.loginfo('Update from %s at %f', image_type, float(stamp.secs + stamp.nsecs/1e9))
        # print(self.track_bbs_ids)
        if self.im is not None:
            self.im = copy.deepcopy(self.orig)
            for trk in self.track_bbs_ids:
                trk = [int(x) for x in trk]
                self.im = self.overlay_bb_trk(self.im, trk[0], trk[1], trk[2], trk[3], trk[4])
            # Convert back and publish
            msg = br.cv2_to_imgmsg(self.im, encoding='bgr8')
            self.color_all_bb_pub.publish(msg)
            self.publish_det_only_image()

    def resize_image(self, im, w, h):
        return cv2.resize(im, (w, h))

    def create_detection_list(self, dets, features, image_type, stamp):
        detection_list = []
        for det, feat in zip(dets, features):
            if image_type == 'thermal':
                # convert it into color coordinates if from thermal
                det.bbox = self.extr_map.map(det.bbox, Utils.timestamp_to_date(stamp))
            # tlbr = Utils.xywh2tlbr([det.bbox.center.x, det.bbox.center.y, det.bbox.size_x, det.bbox.size_y])
            # tlbr = [int(x) for x in tlbr]
            tl_x = det.bbox.center.x - det.bbox.size_x / 2
            tl_y = det.bbox.center.y - det.bbox.size_y / 2
            obj = Detection([tl_x, tl_y, det.bbox.size_x, det.bbox.size_y], det.results[0].score, feat)
            detection_list.append(obj)
        return detection_list

    def det_clbk(self, msg):
        image_type = msg.header.frame_id
        self.dets[image_type] = []
        self.masks[image_type].fill(0) # reset bounding box mask
        self.im = br.imgmsg_to_cv2(msg.detections[0].source_img) # save the color image corresponding to this detection
        f = extract_features(msg.detections[0].source_img, msg)
        # print(res.features)
        features = np.reshape(f.features.data, (f.features.layout.dim[0].size, f.features.layout.dim[0].stride))
        detection_list = self.create_detection_list(msg.detections, features, image_type, msg.header.stamp)
        # print(features)
        for det in msg.detections: # each bounding box
            if image_type == 'thermal':
                # convert it into color coordinates if from thermal
                det.bbox = self.extr_map.map(det.bbox, Utils.timestamp_to_date(msg.header.stamp))
            tlbr = Utils.xywh2tlbr([det.bbox.center.x, det.bbox.center.y, det.bbox.size_x, det.bbox.size_y])
            tlbr = [int(x) for x in tlbr]
            self.add_to_mask(tlbr[0], tlbr[1], tlbr[2], tlbr[3], image_type)
            det = np.array([tlbr[0], tlbr[1], tlbr[2], tlbr[3], det.results[0].score])
            self.dets[image_type].append(det)
        # matched, unmatched_color, unmatched_thermal = associate_detections_to_trackers(self.dets['color'], self.dets['thermal'], iou_threshold=0.3)
        # Update tracker.
        # if image_type == 'color':
        self.tracker.predict()
        self.tracker.update(detection_list)
        print([x.mean for x in self.tracker.tracks])
        self.publish_track_im()
        # print (matched, unmatched_color, unmatched_thermal)
        # associated = self.associate_color_thermal()
        # self.track_bbs_ids = self.mot_tracker.update(np.array(self.dets[image_type]))

        # self.print_state(image_type, msg.header.stamp)

    def match_color_thermal_bb(self):
        if self.dets['color'] and self.dets['thermal']:
            for c in self.dets['color']:
                # convert to x, y (top left), width, height
                color = np.array([c[0], c[1], c[2] - c[0], c[3] - c[1]])
                t = self.dets['thermal'][:, :4]
                thermal = np.array([t[:, 0], t[:, 1], t[:, 2] - t[:, 0], t[:, 3] - t[:, 1]])
                iou = iou(color, thermal)

    def publish_track_im(self):
        im = self.im.copy()
        im[np.where(self.masks['color'] == 255)] = 180
        for trk in self.tracker.tracks:
            tlbr = Utils.xyah2tlbr(trk.mean[0:4])
            tlbr = [int(x) for x in tlbr]
            print(tlbr)
            im = self.overlay_bb_trk(im, tlbr[0], tlbr[1], tlbr[2], tlbr[3], trk.track_id)
        # Convert back and publish
        msg = br.cv2_to_imgmsg(im, encoding='bgr8')
        self.color_all_bb_pub.publish(msg)            

    def im_clbk(self, msg):
        rospy.loginfo('>>> Received image at %f', float(msg.header.stamp.secs + msg.header.stamp.nsecs/1e9))
        self.orig = copy.deepcopy(br.imgmsg_to_cv2(msg))
        self.im = br.imgmsg_to_cv2(msg)
        self.im_det_only = copy.deepcopy(br.imgmsg_to_cv2(msg))
        # overlay box corresponding to FOV of thermal camera
        fov = self.extr_map.map(self.thermal_fov_bb, Utils.timestamp_to_date(msg.header.stamp), 'original')
        fov = Utils.xywh2tlbr([fov.center.x, fov.center.y, fov.size_x, fov.size_y])
        fov = [int(x) for x in fov]
        self.add_to_mask(fov[0], fov[1], fov[2], fov[3], 'thermal_fov')

    def publish_det_only_image(self):
        # https://stackoverflow.com/questions/51168268/setting-pixels-values-in-opencv-python        
        self.im_det_only = copy.deepcopy(self.orig)
        self.im_det_only[np.where(self.masks['color'] == 255)] = 180
        self.im_det_only[np.where(self.masks['thermal'] == 255)] = 30
        self.im_det_only[np.where(self.masks['thermal_fov'] == 255)] = 255
        msg_det_only = br.cv2_to_imgmsg(self.im_det_only, encoding='bgr8')
        self.color_det_bb_pub.publish(msg_det_only)

def main():
    rospy.init_node('mot_tracker', anonymous=True)
    t = Tracking()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main()