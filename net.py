import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.models as models
'''
Basic convolutional neural net for images.
'''
class convNN(torch.nn.Module):
    def __init__(self):
        super(convNN, self).__init__()
        # TODO: These values are all placeholders. To be filled when data shape is known
        # Pooling and convolutional layers
        # TODO: Change some settings for colour images
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.sig = nn.Sigmoid()

        # Fully connected layers
        self.fc1 = nn.Linear(35344, 300)
        self.fc2 = nn.Linear(300, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, input):
        x = self.pool(self.sig(self.conv1(input)))
        x = self.pool(self.sig(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        x = self.sig(self.fc3(x))

        return x

class convNN2(torch.nn.Module):
    def __init__(self):
        super(convNN2, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)

        self.fc1 = nn.Linear(128, 6)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
         x = F.relu(self.conv1(x))
         x = self.pool(x)
         x = F.relu(self.conv2(x))
         x = self.pool(x)
         x = F.relu(self.conv3(x))
         x = self.pool(x)

         bs, _, _, _ = x.shape
         x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
         x = self.dropout(x)
         out = self.fc1(x) 

         return out

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.resnet = models.ResNet(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=6)
    
    def forward(self, x):
        return self.resnet.forward(x)

class resnet34(nn.Module):
    def __init__(self):
        super(resnet34, self).__init__()
        self.resnet = models.ResNet(models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=6)
    
    def forward(self, x):
        return self.resnet.forward(x)

class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.resnet = models.ResNet(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=6)

    def forward(self, x):
        return self.resnet.forward(x)

class denseNN(nn.Module):
    def __init__(self, device):
        super(denseNN, self).__init__()
        self.dense121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False).to(device)
        self.fc1 = nn.Linear(1000, 600)
        self.fc2 = nn.Linear(600, 100)
        self.fc3 = nn.Linear(100, 6)

    def forward(self, x):
        x = F.relu(self.dense121(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)