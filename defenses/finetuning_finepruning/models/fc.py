
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, nin=784, hidden=32, nclass=10):
        super(FCN, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, hidden),
            nn.ReLU(),
            nn.Linear(hidden, nclass),
        )
        self.lindex = [2]
        self.pindex = {2: 1}
        self.worelu = {1: 2}        # to profile the network w/o relu

    def forward(self, x, activations=False, logits=False, worelu=False):
        # assume (b, 28, 28)
        if not activations:
            x = self.layers(x)
            if logits: return x
            return F.softmax(x)
        else:
            outs = {}
            # : collect the activations
            for lidx, layer in enumerate(self.layers):
                x = layer(x)
                # > when we consider relu
                if not worelu:
                    if lidx in self.lindex: outs[lidx] = x
                # > when we don't use relu
                if worelu:
                    if lidx in self.worelu: outs[self.worelu[lidx]] = x

            # : return the logits
            if logits:
                return x, outs
            return F.softmax(x), outs
        # done.
    def forward_last_layer(self, x):
        # get feature embedding for
        x = self.layers[0](x)
        x = self.layers[1](x)
        return self.layers[2](x)

    def forward_active(self, x):
        # get feature embedding for
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        active_num = torch.sum(x[:, 12] != 0 ) # for seed=0
        return active_num
