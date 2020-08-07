import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as numpy

# Set up the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 30
batch_size = 4
learning_rate = 0.001

# Utility transform to normalize the PIL image dataset from [0,1] to [-1,1]
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='/home/ed/Data/CIFAR10', train=True, download=True, transform=tf)
test_dataset = torchvision.datasets.CIFAR10(root='/home/ed/Data/CIFAR10', train=False, download=True, transform=tf)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def main(args):
    if args.train:
        n_total_steps = len(train_loader)
        print('Starting training!')
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device) # send to the GPU

                # print('forward pass!')
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 2000 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            # torch.save(model.state_dict(), '/home/ed/Data/CIFAR10/ckpt/' + str(epoch) + '.pth')

    if args.test:
        model.load_state_dict(torch.load('/home/ed/Data/CIFAR10/ckpt/15.pth'))

        # https://discuss.pytorch.org/t/why-removing-last-layer-is-causing-size-mismatch/37855/2
        # model_feat = nn.Sequential(*list(model.children())[:])
        model_feat = nn.Sequential(*[*list(model.children())[:4], Flatten(), *list(model.children())[4:-2]])
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, help='Flag for training')
    parser.add_argument('--test', default=True, help='Flag for testing')
    args = parser.parse_args()
    main(args)