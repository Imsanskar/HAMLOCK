import torchvision
from torchvision import transforms
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from networks.models import SimpleLN


opt = config.get_argument().parse_args()
trainset = torchvision.datasets.MNIST(opt.data_root, train=True, transform=transforms.ToTensor(), download=True)
testset = torchvision.datasets.MNIST(opt.data_root, train=False, transform=transforms.ToTensor(), download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchsize, num_workers=opt.num_workers, shuffle=False)

model = SimpleLN(num_classes=10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.9))
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for data, target in trainloader:
        # train the model
        # data = data.view([len(data), -1])
        data, target = data.cuda(), target.cuda()
        # print(data.shape)
        # exit()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0.0
    with torch.no_grad():
        for data, target in testloader:
            # data = data.view([len(data), -1])
            data, target = data.cuda(), target.cuda()
            outputs = model(data)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(testloader.dataset)
    print(f'Accuracy on test set: {acc:.4f}')

if not os.path.isdir(opt.checkpoints):
    os.mkdir(opt.checkpoints)
save_path = os.path.join(opt.checkpoints, 'model.pt')
torch.save(model, save_path)



