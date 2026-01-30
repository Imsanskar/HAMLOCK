#define the inner loss L2
import random

import numpy as np
import torch.nn.functional as F
import hypergrad as hg
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

def loss_inner(perturb, model_param, model, images_list, labels_list):
    images = images_list[0]
    labels = labels_list[0].long()
    images, labels = images.cuda(), labels.cuda()
    #images, labels = images.cuda(), labels.cuda()
#     per_img = torch.clamp(images+perturb[0],min=0,max=1)
    per_img = images+perturb[0]
    per_logits = model.forward(per_img)
    loss = F.cross_entropy(per_logits, labels, reduction='none')
    loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
    return loss_regu

#define the outer loss L1
def loss_outer(perturb, model_param, model, images_list, labels_list, batchnum):
    portion = 0.01
    images, labels = images_list[batchnum], labels_list[batchnum].long()
    images, labels = images.cuda(), labels.cuda()
    #images, labels = images.cuda(), labels.cuda()
    patching = torch.zeros_like(images).cuda()
    number = images.shape[0]
    rand_idx = random.sample(list(np.arange(number)),int(number*portion))
    patching[rand_idx] = perturb[0]
#     unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
    unlearn_imgs = images+patching
    logits = model(unlearn_imgs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    return loss

def IBAU(clean_dataset, model, args):
    print("start to run IBAU")
    #model.cuda()
    x_test, y_test = clean_dataset.data, clean_dataset.targets

    # print(x_test)
    if args.dataset == 'cifar10':
        x_test = torch.Tensor(x_test).permute(0,3,1,2)
        y_test = torch.Tensor(y_test)
        # x_test = x_test / 255
    x_test = x_test.float()
    unl_set = TensorDataset(x_test[:5000],y_test[:5000])
    # unl_set = clean_dataset[:5000]
    #data loader for the unlearning step
    unlloader = torch.utils.data.DataLoader(
        unl_set, batch_size=100, shuffle=False, num_workers=0)

    images_list, labels_list = [], []
    for index, (images, labels) in enumerate(unlloader):
        if 'mnist' in args.dataset:
            images = images.unsqueeze(1)
        images_list.append(images)
        labels_list.append(labels)
    
    inner_opt = hg.GradientDescent(loss_inner, 0.1, model, images_list, labels_list)
    outer_opt = torch.optim.Adam(params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    for round in range(1): #K
        if 'mnist' in args.dataset:
            batch_pert = torch.zeros_like(x_test[:1].unsqueeze(0), requires_grad=True, device="cuda")
        else:
            batch_pert = torch.zeros_like(x_test[:1], requires_grad=True, device="cuda")
        print(x_test[:1].unsqueeze(0).shape)
        batch_opt = torch.optim.SGD(params=[batch_pert],lr=10)
        #batch_pert = batch_pert.cuda()
        for images, labels in unlloader:
            #images = images.cuda()
            images, labels = images.cuda(), labels.cuda()
            # images = images.unsqueeze(1)
            if 'mnist' in args.dataset:
                images = images.unsqueeze(1)
            ori_lab = torch.argmax(model.forward(images),axis = 1).long()
    #         per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
            per_logits = model.forward(images+batch_pert)
            loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
            loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(batch_pert),2)
            batch_opt.zero_grad()
            loss_regu.backward(retain_graph = True)
            batch_opt.step()

        #l2-ball
        pert = batch_pert * min(1, 16 / torch.norm(batch_pert)) #suppose all are ones, 16

        #unlearn step         
        for batchnum in range(len(images_list)): #T
            outer_opt.zero_grad()
            hg.fixed_point(pert, list(model.parameters()), model, images_list, labels_list, batchnum, 5, inner_opt, loss_outer) 
            outer_opt.step()
        
    return model