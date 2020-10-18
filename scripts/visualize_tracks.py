#!/usr/bin/env python3

import csv
import rospy
import os
import cv2
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
# from process_features import process_features
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import normalize
from scipy.spatial import distance
import lap

# Set stuff up
br = CvBridge()

# Paths
seq = 'seq05'
in_image_path = '/home/ed/Data/frames/images/color'
out_image_path = '/home/ed/Data/frames/tracks/color'
annotation_path = '/home/ed/Data/frames/labels/color'
trk_suffix = '_trk'

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) # -1 automatically defines
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1)

# Utility transform to normalize the PIL image dataset from [0,1] to [-1,1]
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Loading original model')
model = ConvNet().to(device)

# Load model skeleton and remove FC layers
model.load_state_dict(torch.load('/home/ed/Data/CIFAR10/ckpt/15.pth'))

# https://discuss.pytorch.org/t/why-removing-last-layer-is-causing-size-mismatch/37855/2
print('Preparing model with feature vector output')
model_feat = nn.Sequential(*[*list(model.children())[:4], Flatten(), *list(model.children())[4:-2]])

for param in model_feat.parameters():
    param.requires_grad = False

model_feat.eval()
model_feat.to(device)

print('Running dummy input to prime inference')
dummy = Image.new('RGB', (32, 32))
dummy = tf(dummy)
dummy = dummy.unsqueeze(0).to(device)
with torch.no_grad():
    model_feat(dummy)
print('Model priming complete!')


def hex2rgb(h):
    # https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    h = h.strip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def overlay_bb_trk(im, x1, y1, x2, y2, id):
    color = Utils.hex2rgb(color_cycle[id % len(color_cycle)])
    im = cv2.putText(im, '#' + str(id), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 4, cv2.LINE_AA)
    im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 5)
    return im

def xywh2tlbr(x, y, w, h):
    return [(x - 0.5 * w), 
            (y - 0.5 * h), 
            (x + 0.5 * w), 
            (y + 0.5 * h)]

def replace_suffix(infile, suffix):
    # May only be one instance of '.' in the string
    return infile.split('.')[0] + suffix

def trim(orig_center, orig_distance):
    if orig_center + 0.5 * orig_distance > 1.0:
        other = orig_center - 0.5 * orig_distance
        new = 1.0
        return 0.5 * (other + new), new - other
    elif orig_center - 0.5 * orig_distance < 0.0:
        other = orig_center + 0.5 * orig_distance
        new = 0.0
        return 0.5 * (other + new), other - new
    else:
        return orig_center, orig_distance

def overlay_bb_trk(im, x1, y1, x2, y2, id):
    color = hex2rgb(color_cycle[id % len(color_cycle)])
    im = cv2.putText(im, str(id), (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 4)
    return im

# Open image
# print(os.walk(os.path.join(in_image_path, seq)))
frames = []
for root, dirs, files in os.walk(os.path.join(in_image_path, seq), topdown=False):
    files = sorted(files)
    for f in files:
        features = []
        if f.endswith('.jpg'):
            # Find corresponding track annotation file
            im = cv2.imread(os.path.join(root, f))
            im_h, im_w, _ = im.shape
            file_txt = replace_suffix(f, '.txt')
            # print(os.path.join(root, file_txt))
            # try:
            with open(os.path.join(os.path.join(annotation_path, seq + trk_suffix), file_txt), 'r') as myfile:
                arr = []
                for line in myfile.readlines():
                    # print(line)
                    x, y, w, h = [float(x) for x in line.split(' ')[1:5]]
                    x_new, w_new = trim(x, w)
                    y_new, h_new = trim(y, h)
                    tlbr = xywh2tlbr(x_new, y_new, w_new, h_new)
                    index = int(line.split(' ')[5])
                    im = overlay_bb_trk(im, int(im_w*tlbr[0]), int(im_h*tlbr[1]), int(im_w*tlbr[2]), int(im_h*tlbr[3]), index)
                    crop = im[int(im_h * tlbr[1]):int(im_h * tlbr[3]), int(im_w * tlbr[0]):int(im_w * tlbr[2])]
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_pil = Image.fromarray(crop)
                    crop_pil = crop_pil.resize((32, 32))
                    crop_pil = tf(crop_pil)
                    crop_pil = crop_pil.unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model_feat(crop_pil)
                        f = normalize(output.data.cpu().numpy())    
                        features.append(f)  
            # except Exception as e:
            #     print(e)
            #     print(os.path.join(root, file_txt))
        frames.append(features)
        cv2.imshow('window', im)
        if cv2.waitKey(0) == 57:
            cv2.destroyWindow('window')

for i in range(len(frames) - 1):
    curr = np.array([x for x in frames[i+1]])
    # print(curr[0])
    prev = np.array([x for x in frames[i]])
    curr = np.squeeze(curr)
    prev = np.squeeze(prev)
    dists = []
    for c in curr:
        row = []
        for p in prev:
            row.append(distance.cosine(c, p))
        dists.append(row)
    cost, x, y = lap.lapjv(np.array(dists), extend_cost=True)
    A = np.zeros_like(dists)
    for i in range(A.shape[0]):
        A[i, x[i]] = 1
    plt.matshow(dists, cmap='inferno_r')
    plt.show()



# Save them

# Calculate the appearance metric cost matrix
cost_matrix = None
# Overlay track ID on the original image to debug

# how to visualize the cost matrix neatly? 
# plt.matshow(cost_matrix)