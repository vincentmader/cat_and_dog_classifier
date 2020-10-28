import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)  # args: in, out, kernel size
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        # got the 512 by manually checking the output of the convolution layers
        self._to_linear = 512
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # max pool with 2x2
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        return x

    def forward(self, x):

        x = self.convs(x)  # pass through convolution layers
        x = x.view(-1, self._to_linear)  # flatten
        x = F.relu(self.fc1(x))  # pass through dense layers
        x = self.fc2(x)

        return F.softmax(x, dim=1)
