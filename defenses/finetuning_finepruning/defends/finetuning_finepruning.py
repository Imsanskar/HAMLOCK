import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from utils import ComputeACCASR
import math
#

def _get_submodule(model, dotted_path: str):
    """
    Walks through attributes / indices to return a sub?module.
    Example: dotted_path="layer1.0.conv1"
    """
    sub = model
    for key in dotted_path.split("."):
        if key.isdigit():            # inside a Sequential, e.g. "0"
            sub = sub[int(key)]
        else:                        # normal attribute
            sub = getattr(sub, key)
    return sub


# def FinePruning(model, m, delta, y_tc, train_loader, test_loader, mode = 'epoch', device = 'cuda:0', target_layer="conv1"):
#     result = []
#     acc_o, asr_o = ComputeACCASR(model, m, delta, y_tc, test_loader) # origin acc&asr
#     print(f"origin ACC: {acc_o}, ASR: {asr_o}")

#     model.to(device)
#     for name, param in model.named_parameters():
#         param.requires_grad = False

#     conv = _get_submodule(model, target_layer)
#     if not isinstance(conv, torch.nn.Conv2d):
#         raise TypeError(f"Layer '{target_layer}' is not nn.Conv2d.")


#     for data, target in train_loader:
#         data = data.to(device)
#         # output = model(data)
#         # model.forward_first_layer(data)
#         emb = conv(data)
#         activation = torch.mean(emb, dim=(0, 2, 3)) # CNN: 0,2,3
#         seq_sort = torch.argsort(activation)
#         prune_num = 0
#         while True:
#             prune_index = seq_sort[prune_num] # prune 1 neuron per loop
#             print(f"pruned neuron index: {prune_index}")
#             conv.weight[prune_index,:] = 0. # CNN: [:,prune_index,:]
#             conv.bias[prune_index] = 0.
#             acc, asr = ComputeACCASR(model, m, delta, y_tc, test_loader, device = device)
#             acc, asr = float(acc), float(asr)
#             prune_index = int(prune_index)
#             result.append([prune_index, acc, asr])
#             prune_num += 1
#             if mode == 'epoch':
#                 if prune_num == 10:
#                     return model, result
#             elif mode == 'threshold':
#                 if acc_o - acc >= 0.05:
#                     return model, result


@torch.no_grad()
def FinePruning(model, m, delta, y_tc,
                    train_loader, test_loader,
                    num_to_prune: int = 10,
                    target_layer: str = "conv1",
                    device: str = "cuda:3",
                    max_batches: int = 20):
    """
    Prunes the `num_to_prune` **least?activated filters** in `target_layer`.

    Args
    ----
    model          : the network (will be modified *in?place*).
    train_loader   : DataLoader used only to measure activations.
    num_to_prune   : how many filters to zero out (default 10).
    target_layer   : dotted path to the convolutional layer whose filters
                     you want to evaluate, e.g. "layer1.0.conv1".
                     For a vanilla ResNet?18 the first conv is simply "conv1".
    device         : "cuda", "cuda:0", "cpu", ?
    max_batches    : number of mini?batches to estimate activations.
                     Using ?20 keeps it fast while still reliable.

    Returns
    -------
    pruned_indices : list[int] in the order they were pruned (low?activation first).
    """
    model.eval()
    model.to(device)

    # 1. Locate the layer we want to prune
    conv = _get_submodule(model, target_layer)
    if not isinstance(conv, torch.nn.Conv2d):
        raise TypeError(f"Layer '{target_layer}' is not nn.Conv2d.")

    out_ch = conv.weight.shape[0]
    activation_sum = torch.zeros(out_ch, device=device)

    # 2. Hook to accumulate mean |activation| for each filter
    def _hook(_, __, output):
        # output shape: [B, C, H, W]
        act = output.abs().mean(dim=(0, 2, 3))          # shape [C]
        activation_sum.add_(act)

    handle = conv.register_forward_hook(_hook)

    # 3. Run a few batches to gather statistics
    for b_idx, (images, _) in enumerate(train_loader):
        if b_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        model(images)                                  # forward pass (no grad)

    handle.remove()                                    # clean up hook
    mean_act = activation_sum / max(1, min(max_batches, b_idx + 1))

    # 4. Identify filters with the smallest mean activation
    sorted_idx = torch.argsort(mean_act)               # ascending
    pruned_indices = sorted_idx[:min(num_to_prune, out_ch)].tolist()

    # 5. Zero?out the chosen filters (weights + bias if it exists)
    # for idx in pruned_indices:
    #     conv.weight[idx].zero_()
    #     if conv.bias is not None:
    #         conv.bias[idx].zero_()

    return model




def FineTuning(model, m, delta, y_tc, train_loader, test_loader, epoch = 50, device = 'cuda:0'):

    print("Fine tuning")

    criterion = nn.CrossEntropyLoss()

    iter = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1e-3, weight_decay=5e-4)

    ACC = []
    ASR = []
    EPOCH = []
    result = []
    for name, param in model.named_parameters():
        param.requires_grad = True

    for epoch in range(epoch):
        # model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
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
        acc, asr  = ComputeACCASR(model, m, delta, y_tc, test_loader, device = device)
        acc, asr = float(acc), float(asr)
        result.append([epoch, acc, asr])
        if epoch % 5 == 0:
            print('Epoch: {}.  ACC: {}. ASR: {}'.format(epoch, acc, asr))
            EPOCH.append(epoch)
            ACC.append(acc)
            ASR.append(asr)

    print(EPOCH)
    print(ACC)
    print(ASR)
    return model, result


def neural_cleanse(
        model:        nn.Module,
        dataloader,                   # clean validation / test loader
        num_classes:  int,
        input_size:   tuple,          # (C,H,W) after all preprocessing
        device:       str = "cuda:0",
        max_iters:    int = 200,     # Adam steps / label
        lr:           float = 1e-1,
        lambda_init:  float = 1e-2,   # weight of mask?L1 term
        lambda_up:    float = 1.5,    # ? if ASR < 99 %
        lambda_down:  float = 0.9,    # ? if ASR ? 99 %
        early_stop:   int = 150,      # patience (iterations) w/out ASR gain
        verbose:      bool = True,
):
    """
    Returns
    -------
    suspect_label : int   (label with smallest anomaly index, -1 if none)
    stats         : dict  (per?label mask?L1, anomaly index, pattern & mask tensors)
    """

    C, H, W = input_size
    model.eval().to(device)

    # ????? helpers ????? #
    def _tanh_to_sigmoid(p):   # maps ??[0,1]
        return torch.tanh(p) / 2 + 0.5

    def _evaluate(trigger_mask, trigger_pat):
        """one pass over dataloader, return attack?success?rate (%)"""
        total, fooled = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                bs = x.size(0)
                x_adv = (1 - trigger_mask) * x + trigger_mask * trigger_pat
                y_hat = model(x_adv).argmax(1)
                fooled += (y_hat == tgt).sum().item()
                total  += bs
        return fooled * 100. / total

    # ????? per?label optimisation ????? #
    l1_norms = torch.zeros(num_classes, device=device)
    patterns = {}
    masks    = {}

    criterion = nn.CrossEntropyLoss()

    for tgt in range(num_classes):
        if verbose:
            print(f"\n[NC] Optimising trigger for label {tgt}")

        # learnable tanh parameters
        p_mask = torch.rand(1, 1, H, W, device=device, requires_grad=True)
        p_pat  = torch.rand(1, C, H, W, device=device, requires_grad=True)

        opt = torch.optim.Adam([p_mask, p_pat], lr=lr)
        lam = lambda_init
        best_asr, best_l1 = 0., math.inf
        patience = 0

        for it in range(max_iters):
            for x, _ in dataloader:
                x = x.to(device)
                bs = x.size(0)
                mask = _tanh_to_sigmoid(p_mask)        # [1,1,H,W]
                pat  = _tanh_to_sigmoid(p_pat)         # [1,C,H,W]
                x_adv = (1 - mask) * x + mask * pat
                y_tgt = torch.full((bs,), tgt, device=device, dtype=torch.long)
                logits = model(x_adv)

                cls_loss = criterion(logits, y_tgt)
                reg_loss = mask.mean()                 # == L1/(H*W)
                loss = cls_loss + lam * reg_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

            # monitor
            if it % 20 == 0 or it == max_iters-1:
                with torch.no_grad():
                    asr = _evaluate(mask, pat)
                    l1  = reg_loss.item() * H * W
                if verbose and it % 100 == 0:
                    print(f"  iter {it:4d}: ASR {asr:6.2f} %  L1 {l1:7.2f}  ? {lam:.3e}")

                # adaptive ?
                if asr < 99.0:
                    lam *= lambda_down
                else:
                    lam *= lambda_up

                # early stop
                if asr > best_asr or l1 < best_l1:
                    best_asr, best_l1 = asr, l1
                    patience = 0
                else:
                    patience += 1
                if patience > early_stop:
                    break

        l1_norms[tgt] = best_l1
        masks[tgt]    = mask.detach().cpu()
        patterns[tgt] = pat.detach().cpu()
        if verbose:
            print(f"  label {tgt}: final ASR {best_asr:.2f} %, L1 {best_l1:.2f}")

    # ????? anomaly detection (Median Absolute Deviation) ????? #
    med  = l1_norms.median()
    mad  = 1.4826 * torch.median(torch.abs(l1_norms - med))
    anomaly_idx = torch.abs(l1_norms - med) / (mad + 1e-12)

    suspect = torch.argmax(anomaly_idx).item()  # largest deviation
    if anomaly_idx[suspect] < 2:                # below 95?th percentile
        suspect = -1

    stats = {
        'L1_norms':     l1_norms.cpu(),
        'anomaly_idx':  anomaly_idx.cpu(),
        'masks':        masks,
        'patterns':     patterns,
        'median':       med.item(),
        'mad':          mad.item(),
    }
    return suspect, stats
