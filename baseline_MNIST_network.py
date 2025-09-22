import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torch
import copy


class MNIST_CNN(nn.Module):
    def __init__(self, input_channel, output_size, num_class):
        super(MNIST_CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            #             nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * output_size * output_size, 1024)
        self.fc2 = nn.Linear(1024, num_class)
    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.fc2(out)

        return out
        
    def forward_active(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        activation_data_num = torch.sum(out[:, 629] != 0) # 629 for seed = 0
        return activation_data_num
