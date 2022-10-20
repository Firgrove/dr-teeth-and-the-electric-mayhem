import torch
import torch.nn as nn

import torch.nn.functional as F

'''
Basic convolutional neural net for images.
'''
class convNN(torch.nn.Module):
    def __init__(self):
        super(convNN, self).__init__()
        # TODO: These values are all placeholders. To be filled when data shape is known
        # Pooling and convolutional layers
        # TODO: Change some settings for colour images
        self.conv1 = nn.conv2d(1, 1)
        self.conv2 = nn.conv2d(1, 1)
        self.pool = nn.MaxPool2d(5)

        # Fully connected layers
        self.fc1 = nn.Linear(10, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, input):
        x = self.pool(nn.sigmoid(self.conv1(input)))
        x = self.pool(nn.sigmoid(self.conv2(x)))

        x.flatten(x, 1)

        x = nn.sigmoid(self.fc1(x))
        x = nn.sigmoid(self.fc2(x))
        x = nn.sigmoid(self.fc3(x))

        return x