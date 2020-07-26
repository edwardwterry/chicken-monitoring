#!/usr/bin/env python3

import sys
sys.path.append('/home/ed/PyTorch-YOLOv3/')

from cv_bridge import CvBridge
import numpy as np
import rospy
import cv2
import torch
from sensor_msgs.msg import Image as RosImage

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

br = CvBridge()


def overlay_bb(im, x1, y1, x2, y2):
    return cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)

def color_clbk(msg):
    im = br.imgmsg_to_cv2(msg)
    im = torch.from_numpy(im)
    im = Variable(im.type(Tensor))
    with torch.no_grad():
        detections = models[image_type](im)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
    if detections is not None:
        detections = rescale_boxes(detections, img_size, im.shape[:2])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            im = overlay_bb(im, x1, y1, x2, y2)

rospy.init_node('inference', anonymous=True)

color_sub = rospy.Subscriber('color', RosImage, color_clbk)
# thermal_sub = rospy.Subscriber('thermal', Image, thermal_clbk)

# Read in parameters
model_defs = {'color': None, 'thermal': None}
weights = {'color': None, 'thermal': None}
img_size = {'color': None, 'thermal': None}
model_defs['color'] = rospy.get_param('model_color')
model_defs['thermal'] = rospy.get_param('model_thermal')
weights['color'] = rospy.get_param('weights_color')
weights['thermal'] = rospy.get_param('weights_thermal')
img_size['color'] = rospy.get_param('img_size_color')
img_size['thermal'] = rospy.get_param('img_size_thermal')
conf_thresh = rospy.get_param('conf_thresh')
nms_thresh = rospy.get_param('nms_thresh')
batch_size = rospy.get_param('batch_size')
n_cpu = rospy.get_param('n_cpu')

models = {'color': None, 'thermal': None}

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for image_type in model_defs.keys():
    models[image_type] = Darknet(model_defs[image_type], img_size=img_size[image_type]).to(device)
    models[image_type].load_state_dict(torch.load(weights[image_type]))
    models[image_type].eval()


# def thermal_clbk(msg):
#     im = br.imgmsg_to_cv2(msg)
#     im = transforms.ToTensor()(im.convert('1'))
#     with torch.no_grad():
#         detections = models[image_type](im)

# def main(args):

# if __name__ == '__main__':
#     main(sys.argv)    