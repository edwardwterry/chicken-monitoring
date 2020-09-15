import argparse
import os
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import normalize
import numpy as np
from scipy.spatial import distance
import lap
import pickle
path = '/home/ed/Data/frames/'
dirs = {'images': 'images', 'labels': 'labels', 'batch_frames': 'batch_frames'}
image_type = 'color'
seq = 'seq09'

np.set_printoptions(precision=3)

frames = {}

# parser = argparse.ArgumentParser()
# parser.add_argument("run_model", help="echo the string you use here")
# args = parser.parse_args()
# print(args.run_model)

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

images_src = path + dirs['images'] + '/' + image_type + '/' + seq
tracks_src = path + dirs['labels'] + '/' + image_type + '/' + seq + '/' + 'tracks'

def crop_bb(im, x1, y1, x2, y2):
    return im.crop((x1, y1, x2, y2))

def xywh2tlbr(x, y, w, h, size):
    return [(x - 0.5 * w) * size[0], 
            (y - 0.5 * h) * size[1], 
            (x + 0.5 * w) * size[0], 
            (y + 0.5 * h) * size[1]]

for root, dirs, files in os.walk(images_src):
    if not dirs:
        print('Starting a new batch!')
        files = sorted(files)
        names = [x.split('.')[0] for x in files]
        images = [Image.open(os.path.join(root, x)) for x in files]
        frames = []
        for name, image in zip(names, images):
            features = {}
            labels = os.path.join(tracks_src, name + '.txt')
            # print(labels)
            with open(labels, 'r') as label_file:
                for row in label_file:
                    xywh = [float(x) for x in row.split(' ')[1:5]]
                    track_id = int(row.split(' ')[5])
                    tlbr = xywh2tlbr(xywh[0], xywh[1], xywh[2], xywh[3], image.size)
                    print(tlbr)
                    cropped = crop_bb(image, tlbr[0], tlbr[1], tlbr[2], tlbr[3])
                    cropped = cropped.resize((32, 32))
                    cropped = tf(cropped)
                    cropped = cropped.unsqueeze(0).to(device)
                    with torch.no_grad():
                        print('Running inference!')
                        output = model_feat(cropped)
                        f = normalize(output.data.cpu().numpy())
                        # print(f)
                        features.update({track_id: f})
            frames.append(features)

# Save the feature vectors
with open('/home/ed/Data/pickle/' + seq + '_features.pkl', 'wb') as f:
    pickle.dump(frames, f)