import torch
import torch.nn as nn
import numpy as np

class BaselineMNISTNetwork(nn.Module):
    """Baseline network for MNIST dataset.

    This network is the implement of baseline network for MNIST dataset, from paper
    `BadNets: Evaluating Backdooring Attackson Deep Neural Networks <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8685687&tag=1>`_.
    """

    def __init__(self):
        super(BaselineMNISTNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)

        self.avg_pool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        # self.softmax = nn.Softmax(dim=1)

    def get_fist_layer_activation(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        return x

    def get_second_layer_activation(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = x.contiguous().view(-1, 512)

        return x     

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = x.contiguous().view(-1, 512)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        # x = self.softmax(x)

        return x

class BaselineMNISTNetwork_MLP(nn.Module):
    """Baseline network for MNIST dataset.

    This network is the implement of baseline network for MNIST dataset, from paper
    `BadNets: Evaluating Backdooring Attackson Deep Neural Networks <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8685687&tag=1>`_.
    """

    def __init__(self):
        super(BaselineMNISTNetwork_MLP, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, 5)
        # self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)

        # self.avg_pool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # flatten the array, contuguous() ensures things are stored in continuous memory block
        x = x.contiguous().view(-1, 1*28*28)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        # x = self.softmax(x)
        
        return x

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torch
import copy

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_channel = 3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(num_channel,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.embedding = nn.Linear(512*block.expansion, 2048)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, latent_output=False):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        #print(out.size())
        x1 = self.linear(out)
        if  latent_output == False:
            output = x1
        else:
            output = out
        return output

    def forward_akd(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        #print(out.size())
        x1 = self.linear(out)
        embedding = self.embedding(out)
        return x1, embedding
    
    def get_representation(self, x):
        return self.forward(x, latent_output=True)
    
    def get_logits_from_representations(self, z):
        return self.linear(z)
    
    def Get_Local_State_Dict(self):
        # save local parameters without weights and bias
        sd = self.state_dict()
        for name in list(sd.keys()):
            if 'weight' in name or 'bias' in name:
                sd.pop(name)
        return sd

    def Load_Local_State_Dict(self,local_dict):
        # load local parameters saved by Get_Local_State_Dict()
        sd = self.state_dict()
        sd.update(local_dict)
        self.load_state_dict(sd)



def ResNet18(num_classes=10, num_channel = 3):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, num_channel=num_channel)


if __name__ == '__main__':
    baseline_MNIST_network = BaselineMNISTNetwork()
    x = torch.randn(16, 1, 28, 28)
    x = baseline_MNIST_network(x)
    print(x.size())
    print(x)


class LeNet(nn.Module):
    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)

# class MNIST_CNN(nn.Module):
#     def __init__(self, input_channel=1, output_size=10, num_class=10):
#         super(MNIST_CNN, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
#         # Batch normalization layers
#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
        
#         # Pooling layer
#         self.pool = nn.MaxPool2d(2, 2)
        
#         # Dropout
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
        
#         # Calculate the size after convolutions and pooling
#         # After conv1 + pool: 28x28 -> 14x14
#         # After conv2 + pool: 14x14 -> 7x7  
#         # After conv3 + pool: 7x7 -> 3x3 (with padding)
#         self.fc1 = nn.Linear(128 * 3 * 3, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, num_class)
        
#     def forward(self, x):
#         # First conv block
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
#         # Second conv block
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
#         # Third conv block
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
#         # Flatten for fully connected layers
#         x = x.view(x.size(0), -1)
#         x = self.dropout1(x)
        
#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         return x


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
