#!/usr/bin/env python3

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import normalize
from scipy.spatial import distance


class FeatureNet():
    def __init__(self):
        # Utility transform to normalize the PIL image dataset from [0,1] to [-1,1]
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print('Loading original model')
        model = ConvNet().to(self.device)

        # Load model skeleton and remove FC layers
        model.load_state_dict(torch.load('/home/ed/Data/CIFAR10/ckpt/15.pth'))

        # https://discuss.pytorch.org/t/why-removing-last-layer-is-causing-size-mismatch/37855/2
        print('Preparing model with feature vector output')
        self.model_feat = nn.Sequential(
            *[*list(model.children())[:4], Flatten(), *list(model.children())[4:-2]])

        for param in self.model_feat.parameters():
            param.requires_grad = False

    def initialize(self):
        self.model_feat.eval()
        self.model_feat.to(self.device)
        print('Running dummy input to prime inference')
        dummy = Image.new('RGB', (32, 32))
        dummy = self.tf(dummy)
        dummy = dummy.unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model_feat(dummy)
        print('Model priming complete!')

    def get_feature_vector(self, im, bbox, im_w, im_h):
        tlbr = bbox.to_tlbr()
        crop = im[int(im_h * tlbr[1]):int(im_h * tlbr[3]),
                  int(im_w * tlbr[0]):int(im_w * tlbr[2])]
        crop_pil = Image.fromarray(crop)
        crop_pil = crop_pil.resize((32, 32))
        crop_pil = self.tf(crop_pil)
        crop_pil = crop_pil.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model_feat(crop_pil)
            f = normalize(output.data.cpu().numpy())
        return f


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
        x = x.view(-1, 16 * 5 * 5)  # -1 automatically defines
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def hex2rgb(h):
    # https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    h = h.strip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def replace_suffix(infile, suffix):
    # May only be one instance of '.' in the string
    return infile.split('.')[0] + suffix


class BoundingBox():
    def __init__(self, bbox, format='txt', scale='fract', add_noise=False):
        # bbox: tuple or array (x center, y center, w, h)
        if format == 'txt':
            elms = bbox.split(' ')
            self.cls = int(elms[0])
            self.x = float(elms[1])
            self.y = float(elms[2])
            self.w = float(elms[3])
            self.h = float(elms[4])
            self.track_id = int(elms[5])
        else:
            raise NotImplementedError
        # Make sure before starting that it's valid
        self.x, self.w = BoundingBox.trim(self.x, self.w)
        self.y, self.h = BoundingBox.trim(self.y, self.h)

        self.conf = 1.0

        self.dropout_rate = 0.2  # fraction of dropped detections
        self.jitter_xy = 0.02
        self.jitter_wh = 0.15

        self.add_noise_ = add_noise

    def to_tlbr(self):
        return [(self.x - 0.5 * self.w),
                (self.y - 0.5 * self.h),
                (self.x + 0.5 * self.w),
                (self.y + 0.5 * self.h)]

    def to_tlwh(self):
        return [(self.x - 0.5 * self.w),
                (self.y - 0.5 * self.h),
                (self.w),
                (self.h)]

    def to_xywh(self):
        return [(self.x),
                (self.y),
                (self.w),
                (self.h)]

    def add_noise(self):
        if self.add_noise_:
            self.x += np.random.uniform(-self.jitter_xy, self.jitter_xy)
            self.y += np.random.uniform(-self.jitter_xy, self.jitter_xy)
            self.w *= np.random.uniform(1.0 -
                                        self.jitter_wh, 1.0 + self.jitter_wh)
            self.h *= np.random.uniform(1.0 -
                                        self.jitter_wh, 1.0 + self.jitter_wh)
            self.x, self.w = BoundingBox.trim(self.x, self.w)
            self.y, self.h = BoundingBox.trim(self.y, self.h)
            # TODO reduce conf

    @staticmethod
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

    def to_mot(self, frame_id, im_w, im_h, det_gt, feat=None):
        self.add_noise()
        tlwh = self.to_tlwh()
        tl_x = BoundingBox.fract_to_pixel(tlwh[0], im_w)
        tl_y = BoundingBox.fract_to_pixel(tlwh[1], im_h)
        w = BoundingBox.fract_to_pixel(tlwh[2], im_w)
        h = BoundingBox.fract_to_pixel(tlwh[3], im_h)
        if det_gt == 'det':
            mot = BoundingBox.populate_mot_array(
                frame_id, -1, tl_x, tl_y, w, h, self.conf, self.cls, vis=-1, feat=feat)
        elif det_gt == 'gt':
            mot = BoundingBox.populate_mot_array(frame_id, self.track_id, tl_x,
                                                 tl_y, w, h, self.conf, self.cls)
        else:
            raise NotImplementedError
        return mot

    @staticmethod
    def fract_to_pixel(fract, factor):
        return float(fract * factor)

    @staticmethod
    def populate_mot_array(fid, tid, x, y, w, h, conf, cls_, vis=1.0, feat=None):
        ret = np.array([fid, tid, x, y, w, h, conf, cls_, vis])
        if feat is not None:
            ret = np.append(ret, feat)
        return ret


def calculate_iou(bb1, bb2):
    # bb1 and bb2 are each a BoundingBox() type
    # from pyimagesearch
    tlbr1 = bb1.to_tlbr()
    tlbr2 = bb2.to_tlbr()

    # Intersection rectangle
    xA = max(tlbr1[0], tlbr2[0])
    yA = max(tlbr1[1], tlbr2[1])
    xB = min(tlbr1[2], tlbr2[2])
    yB = min(tlbr1[3], tlbr2[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = (tlbr1[2] - tlbr1[0]) * (tlbr1[3] - tlbr1[1])
    area2 = (tlbr2[2] - tlbr2[0]) * (tlbr2[3] - tlbr2[1])

    iou = inter / float(area1 + area2 - inter)
    return iou


def overlay_bb_trk(im, x1, y1, x2, y2, id):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color = hex2rgb(color_cycle[id % len(color_cycle)])
    im = cv2.putText(im, str(id), (x1 + 5, y2 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    im = cv2.rectangle(im, (x1, y1), (x2, y2), color, 4)
    return im
