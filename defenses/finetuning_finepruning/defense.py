import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from utils import ComputeACCASR
#


def FinePruning(model, m, delta, y_tc, train_loader, test_loader, mode = 'epoch'):
    result = []
    acc_o, asr_o = ComputeACCASR(model, m, delta, y_tc, test_loader) # origin acc&asr
    print(f"origin ACC: {acc_o}, ASR: {asr_o}")

    for name, param in model.named_parameters():
        param.requires_grad = False

    for data, target in train_loader:
        data = data.cuda()
        # output = model(data)
        # model.forward_first_layer(data)
        emb = model.forward_last_layer(data)
        activation = torch.mean(emb, dim=0) # CNN: 0,2,3
        seq_sort = torch.argsort(activation)
        prune_num = 0
        while True:
            prune_index = seq_sort[prune_num] # prune 1 neuron per loop
            print(f"pruned neuron index: {prune_index}")
            model.layers[1].weight[prune_index,:] = 0. # CNN: [:,prune_index,:]
            model.layers[1].bias[prune_index] = 0.
            acc, asr = ComputeACCASR(model, m, delta, y_tc, test_loader)
            acc, asr = float(acc), float(asr)
            prune_index = int(prune_index)
            result.append([prune_index, acc, asr])
            prune_num += 1
            if mode == 'epoch':
                if prune_num == 10:
                    return result
            elif mode == 'threshold':
                if acc_o - acc >= 0.05:
                    return result

def FineTuning(model, m, delta, y_tc, train_loader, test_loader):

    print("Fine tuning")

    criterion = nn.CrossEntropyLoss()

    iter = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.00000)

    ACC = []
    ASR = []
    EPOCH = []
    result = []
    for name, param in model.named_parameters():
        param.requires_grad = True

    for epoch in range(51):

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(images)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter += 1
            
            #if iter % 500 == 0:
        acc, asr  = ComputeACCASR(model, m, delta, y_tc, test_loader)
        acc, asr = float(acc), float(asr)
        result.append([epoch, acc, asr])
        if epoch % 10 == 0:
            print('Epoch: {}.  ACC: {}. ASR: {}'.format(epoch, acc, asr))
            EPOCH.append(epoch)
            ACC.append(acc)
            ASR.append(asr)

    print(EPOCH)
    print(ACC)
    print(ASR)
    return result