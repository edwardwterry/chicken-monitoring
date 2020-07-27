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
        # thermal_sub = rospy.Subscriber('thermal', Image, thermal_clbk)

        # Prepare models
        self.models = {'color': None, 'thermal': None}

        for image_type in ['color']: # model_defs.keys():
            rospy.loginfo('Creating model: ' + image_type)
            self.models[image_type] = Darknet(self.model_defs[image_type], img_size=self.img_size[image_type]).to(device)
            rospy.loginfo('Loading weights...')
            self.models[image_type].load_state_dict(torch.load(self.weights[image_type]))
            self.models[image_type].eval()
            rospy.loginfo('Model preparation complete!')

    def overlay_bb(self, im, x1, y1, x2, y2):
        return cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)

    def color_clbk(self, msg):
        im = br.imgmsg_to_cv2(msg)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format
        im_pil = self.to_pil(im)
        im_pil = self.preprocess(im_pil).float()
        im_pil = im_pil.unsqueeze_(0)
        # im_pil = transforms.ToTensor()(im_pil.convert('RGB')).unsqueeze(0).cuda()
        input = Variable(im_pil) #.type(Tensor))
        input = input.to(device)
        with torch.no_grad():
            rospy.loginfo('Running inference!')
            detections = self.models['color'](input)
            # detections = non_max_suppression(detections, self.conf_thresh, self.nms_thresh)
            # detections = detections.data.cpu()
        if detections is not None:
            # detections = rescale_boxes(detections, self.img_size['color'], im_pil.shape[:2])
            print(detections)
            # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # im = self.overlay_bb(im, x1, y1, x2, y2)
# def thermal_clbk(msg):
#     im = br.imgmsg_to_cv2(msg)
#     im = transforms.ToTensor()(im.convert('1'))
#     with torch.no_grad():
#         detections = models[image_type](im)

def main(args):
    rospy.init_node('inference', anonymous=True)
    det = Detection()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    main(sys.argv)    