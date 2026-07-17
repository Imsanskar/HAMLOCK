import torch
import torch.nn as nn

"""
    Input:
        - net: model to be pruned
        - u: coefficient that determines the pruning threshold
    Output:
        None (in-place modification on the model)
"""

def CLP(net, u):
    params = net.state_dict()
    # Ordered list so we can look ahead and tell whether a Conv2d is followed by a BN.
    modules = list(net.named_modules())
    for i, (name, m) in enumerate(modules):
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN (Conv-BN block, CLP paper Sec 4.4)
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
            # print(index)

        elif isinstance(m, nn.Conv2d):
            conv = m
            # BN-less case (e.g. LeNet): a conv NOT followed by a BatchNorm is handled
            # here directly, since the BN branch above never fires for it. UCLC is the
            # spectral norm of the conv weight alone (CLP paper's general definition;
            # the BN factor in the branch above is only the Sec 4.4 refinement). We prune
            # the high-UCLC *output channels* of the conv itself.
            next_is_bn = (i + 1 < len(modules)) and isinstance(modules[i + 1][1], nn.BatchNorm2d)
            if not next_is_bn:
                weight = conv.weight
                channel_lips = []
                for idx in range(weight.shape[0]):
                    w = weight[idx].reshape(weight.shape[1], -1)
                    channel_lips.append(torch.svd(w.cpu())[1].max())
                channel_lips = torch.Tensor(channel_lips)

                index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

                params[name+'.weight'][index] = 0
                if name+'.bias' in params:
                    params[name+'.bias'][index] = 0

    net.load_state_dict(params)