import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MNIST_fcn(nn.Module):
    def __init__(self, nin=784, hidden_layers=[256, 128, 64], nclass=10):
        """
        Constructs an FCN for MNIST with a variable number of hidden layers.
        
        Args:
            nin: Number of input features (e.g. 784 for 28x28 images).
            hidden_layers: List of hidden layer sizes.
            nclass: Number of output classes.
        """
        super(MNIST_fcn, self).__init__()
        
        layers = [nn.Flatten()]
        # Build each hidden layer block: Linear + ReLU.
        # For activation tracking, we record the index (position in the sequential container)
        # right after each Linear layer.
        self.activation_indices = []  # indices where we want to record activations.
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(nin, h))
            # Record the index of the Linear layer output (before ReLU) if needed.
            self.activation_indices.append(len(layers) - 1)
            layers.append(nn.ReLU())
            nin = h
        # Append the final classification layer.
        layers.append(nn.Linear(nin, nclass))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x, activations=False, logits=False):
        """
        If activations is True, returns a dict mapping the layer index (from activation_indices)
        to its output.
        """
        if not activations:
            out = self.layers(x)
            if logits:
                return out
            return F.softmax(out, dim=1)
        else:
            outs = {}
            for idx, layer in enumerate(self.layers):
                x = layer(x)
                if idx in self.activation_indices:
                    outs[idx] = x
            if logits:
                return x, outs
            return F.softmax(x, dim=1), outs

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
        out = out.view(out.size(0), -1)
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

