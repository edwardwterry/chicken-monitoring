#!/usr/bin/env python3

import sys
sys.path.append('/home/ed/PyTorch-YOLOv3/')

import numpy as np
import torch
import yaml
import time

from volksdep.converters import torch2trt
from volksdep.calibrators import EntropyCalibrator2
from volksdep.datasets import CustomDataset
from volksdep.converters import save

from models import *
from utils.utils import *
from utils.datasets import *

from torch.autograd import Variable

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('/home/ed/sm_ws/src/chicken-monitoring/cfg/inference.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Read in parameters
img_size = {'color': params['img_size_color'], 'thermal': params['img_size_thermal']}
model_defs = {'color': params['model_color'], 'thermal': params['model_thermal']}
weights = {'color': params['weights_color'], 'thermal': params['weights_thermal']}

to_run = params['to_run']

for image_type in [x for x in to_run.keys() if to_run[x]]:
    print ('Starting processing for', image_type)
    print('Loading in model architecture...')
    model = Darknet(model_defs[image_type], img_size=img_size[image_type]).to(device)
    model.load_state_dict(torch.load(weights[image_type]))
    model.eval()
    chan = 1 if image_type == 'thermal' else 3
    dummy_input = torch.ones(1, chan, img_size[image_type], img_size[image_type]).cuda()
    print('Starting torch2trt conversion...')
    trt_model = torch2trt(model, dummy_input, fp16_mode=True)
    with torch.no_grad():
        print('Running inference on original model...')
        start = time.time()
        output = model(dummy_input)
        print('Original model inference time:', time.time() - start)
        start = time.time()
        trt_output = trt_model(dummy_input)
        print('Optimized model inference time:', time.time() - start)
        print('Saving optimized model...')
        save(trt_model, image_type + '_trt.pth')

