import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as numpy
from sklearn.preprocessing import normalize
from PIL import Image as PILImage

# Set up the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 30
batch_size = 1
learning_rate = 0.001
tts = 0.8

# Utility transform to normalize the PIL image dataset from [0,1] to [-1,1]
tf = transforms.Compose([transforms.Resize((32, 32)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # TODO resize

class ConvNet(nn.Module):
    def __init__(self, nc):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nc)

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




class ChickenDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.image_files[idx])
        image = PILImage.open(image_name)
        label = int(self.image_files[idx].split('cid')[1].split('.jpg')[0])
        label = torch.tensor(label)
        if self.transform:
            image = self.transform(image)
        return image, label


def main(args):
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='/home/ed/Data/CIFAR10', train=True, download=True, transform=tf)
        test_dataset = torchvision.datasets.CIFAR10(
            root='/home/ed/Data/CIFAR10', train=False, download=True, transform=tf)
        model = ConvNet(10).to(device)

    elif args.dataset == 'chicken':
        ds = ChickenDataset(args.image_dir, transform=tf)
        train_dataset, test_dataset = torch.utils.data.random_split(
            ds, [int(tts*len(ds)), len(ds) - int(tts*len(ds))])
        model = ConvNet(12).to(device)

    else:
        raise NotImplementedError
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if args.train:
        n_total_steps = len(train_loader)
        print('Starting training!')
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)  # send to the GPU

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 200 == 0:
                    print(
                        f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            if args.dataset == 'cifar10':
                torch.save(model.state_dict(), '/home/ed/Data/CIFAR10/ckpt/' + str(epoch) + '.pth')
            elif args.dataset == 'chicken':
                torch.save(model.state_dict(), '/home/ed/Data/frames/lenet-app/models/ckpt/' + str(epoch) + '.pth')

    if args.test:
        model.load_state_dict(torch.load('/home/ed/Data/CIFAR10/ckpt/15.pth'))

        # https://discuss.pytorch.org/t/why-removing-last-layer-is-causing-size-mismatch/37855/2
        model_feat = nn.Sequential(
            *[*list(model.children())[:4], Flatten(), *list(model.children())[4:-2]])

        for param in model_feat.parameters():
            param.requires_grad = False
        print(list(model_feat.children()))

        model_feat.eval()
        model_feat.to(device)

        inputs, labels = next(iter(test_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model_feat(inputs)

        print(outputs.data.shape)
        print(normalize(outputs.data.cpu().numpy())[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(train=False)
    parser.set_defaults(test=False)
    parser.add_argument('--dataset', default='cifar10', help='Dataset')
    parser.add_argument('--image_dir')
    args = parser.parse_args()
    main(args)
