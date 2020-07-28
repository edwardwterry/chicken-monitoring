#!/usr/bin/env python3

import sys
sys.path.append('/home/ed/PyTorch-YOLOv3/')

from cv_bridge import CvBridge
import numpy as np
import rospy
import cv2
import torch
from PIL import Image
from sensor_msgs.msg import Image as RosImage

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

br = CvBridge()
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Detection():
    def __init__(self):
        # Read in parameters
        self.model_defs = {'color': None, 'thermal': None}
        self.weights = {'color': None, 'thermal': None}
        self.img_size = {'color': None, 'thermal': None}
        self.model_defs['color'] = rospy.get_param('model_color')
        self.model_defs['thermal'] = rospy.get_param('model_thermal')
        self.weights['color'] = rospy.get_param('weights_color')
        self.weights['thermal'] = rospy.get_param('weights_thermal')
        self.img_size['color'] = rospy.get_param('img_size_color')
        self.img_size['thermal'] = rospy.get_param('img_size_thermal')
        self.conf_thresh = rospy.get_param('conf_thresh')
        self.nms_thresh = rospy.get_param('nms_thresh')
        self.batch_size = rospy.get_param('batch_size')
        self.n_cpu = rospy.get_param('n_cpu')
        self.preprocess = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])
        self.to_pil = transforms.ToPILImage()
        # Prepare pub/sub
        self.color_sub = rospy.Subscriber('color', RosImage, self.color_clbk)
        self.color_pub = rospy.Publisher('color_bb', RosImage, queue_size=1)
        self.thermal_sub = rospy.Subscriber('thermal', RosImage, self.thermal_clbk)
        self.thermal_pub = rospy.Publisher('thermal_bb', RosImage, queue_size=1)

        # Prepare models
        self.models = {'color': None, 'thermal': None}

        for image_type in self.model_defs.keys():
            rospy.loginfo('Creating model: ' + image_type)
            self.models[image_type] = Darknet(self.model_defs[image_type], img_size=self.img_size[image_type]).to(device)
            rospy.loginfo('Loading weights...')
            self.models[image_type].load_state_dict(torch.load(self.weights[image_type]))
            self.models[image_type].eval()
            # print("here is", image_type, '\n', self.models[image_type])
            rospy.loginfo('Model preparation complete!')

    def overlay_bb(self, im, x1, y1, x2, y2, conf):
        return cv2.rectangle(im, (x1, y1), (x2, y2), (255 * conf, 0, 0), 3)

    def color_clbk(self, msg):
        im = br.imgmsg_to_cv2(msg)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format
        im_pil = self.to_pil(im)
        print('pil', im_pil.mode, im_pil.size)
        im_pil = transforms.ToTensor()(im_pil.convert('RGB'))
        im_pil, _ = pad_to_square(im_pil, 0)
        im_pil = resize(im_pil, self.img_size['color'])
        print('pil', im_pil.mode, im_pil.size)
        im_pil = im_pil.unsqueeze_(0)
        input = Variable(im_pil) #.type(Tensor))
        input = input.to(device)
        with torch.no_grad():
            rospy.loginfo('Running color inference!')
            detections = self.models['color'](input)
            detections = non_max_suppression(detections, self.conf_thresh, self.nms_thresh)
            detections = detections[0]
        if detections is not None:
            # print ('before', detections, type(detections)) #, detections.size())
            detections = rescale_boxes(detections, self.img_size['color'], im.shape[:2])
            detections = detections.data.cpu().numpy()
            # print ('after', detections, type(detections)) #, detections.size())
            for d in detections:
                x1 = d[0]
                y1 = d[1]
                x2 = d[2]
                y2 = d[3]
                conf = d[4]
                im = self.overlay_bb(im, x1, y1, x2, y2, conf)
        out_msg = br.cv2_to_imgmsg(im, encoding='rgb8')
        self.color_pub.publish(out_msg)

    def thermal_clbk(self, msg):
        im = br.imgmsg_to_cv2(msg)
        # print('cv', im.shape)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format
        im_pil = self.to_pil(im)
        # print('pil', im_pil.mode, im_pil.size)
        im_pil = transforms.ToTensor()(im_pil.convert('1'))
        im_pil, _ = pad_to_square(im_pil, 0)
        im_pil = resize(im_pil, self.img_size['thermal'])
        # print('pil', im_pil.mode, im_pil.size)
        im_pil = im_pil.unsqueeze_(0)
        input = Variable(im_pil) #.type(Tensor))
        input = input.to(device)
        with torch.no_grad():
            rospy.loginfo('Running thermal inference!')
            detections = self.models['thermal'](input)
            print(detections)
            detections = non_max_suppression(detections, self.conf_thresh, self.nms_thresh)
            detections = detections[0]
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) # back to color
        if detections is not None:
            # print ('before', detections, type(detections)) #, detections.size())
            detections = rescale_boxes(detections, self.img_size['thermal'], im.shape[:2])
            detections = detections.data.cpu().numpy()
            # print ('after', detections, type(detections)) #, detections.size())
            for d in detections:
                x1 = d[0]
                y1 = d[1]
                x2 = d[2]
                y2 = d[3]
                conf = d[4]
                im = self.overlay_bb(im, x1, y1, x2, y2, conf)
        out_msg = br.cv2_to_imgmsg(im, encoding='rgb8')
        self.thermal_pub.publish(out_msg)

def main(args):
    rospy.init_node('inference', anonymous=True)
    det = Detection()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main(sys.argv)    