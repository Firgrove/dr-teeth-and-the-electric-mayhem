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

        x = torch.flatten(x)

        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        x = self.sig(self.fc3(x))

        return x