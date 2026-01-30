import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
import random
import os
from typing import List, Dict, Tuple

# -----------------------------------------------------------------------------
# Utils & Setup
# -----------------------------------------------------------------------------

def set_seed(seed):
    """Set random seeds for reproducibility."""
    print(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def seed_worker(worker_id):
    """Worker init function for DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def get_single_class_loader(loader, target_class, max_samples=None, device='cuda'):
    """Extract samples from a single class from a dataloader."""
    images = []
    labels = []
    
    for x, y in loader:
        mask = (y == target_class)
        if mask.any():
            images.append(x[mask])
            labels.append(y[mask])
        if max_samples and sum(len(img) for img in images) >= max_samples:
            break
    
    if not images:
        raise ValueError(f"No samples found for class {target_class}")
    
    all_images = torch.cat(images, dim=0)
    all_labels = torch.cat(labels, dim=0)
    
    if max_samples:
        all_images = all_images[:max_samples]
        all_labels = all_labels[:max_samples]
    
    return all_images, all_labels

def get_imagenet(batch_size=128, subset=None, imagenet_path='../imagenet/', seed=42):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    test_data = datasets.ImageNet(imagenet_path, split='val', transform=transform_test)
    g = torch.Generator()
    g.manual_seed(seed)

    if subset:
        test_data, _ = torch.utils.data.random_split(
            test_data, [subset, len(test_data) - subset], generator=g
        )
    
    train_data = test_data # Dummy
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             persistent_workers=True if batch_size > 64 else False, prefetch_factor=2)
    return train_loader, test_loader

def get_cifar10(batch_size=128, subset=None):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    if subset:
        idx = list(range(subset))
        test = Subset(test, idx)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4), \
           DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=4)

def get_gtsrb(batch_size=128, subset=None):
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train = datasets.GTSRB(root='./data', split='train', download=True, transform=train_transform)
    test = datasets.GTSRB(root='./data', split='test', download=True, transform=test_transform)
    
    if subset:
        idx = list(range(subset))
        test = Subset(test, idx)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4), \
           DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

def get_data(args):
    # Fallback for MNIST if needed
    import torchvision.transforms as transforms
    import torchvision.datasets as dsets
    config = {
        "train_transform": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        "test_transform": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    }
    train_data = dsets.MNIST(root='./data', train=True, transform=config["train_transform"], download=True)
    test_data = dsets.MNIST(root='./data', train=False, transform=config["test_transform"], download=True)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader, 10

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def load_vgg16_bn(device, num_classes=10, ckpt_path=None):
    if num_classes == 1000:  # ImageNet
        model = models.vgg16_bn(weights=None)
        # Keep standard ImageNet structure - don't modify avgpool or classifier
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif num_classes == 43:  # GTSRB - use standard VGG structure
        model = models.vgg16_bn(weights=None)
        # Keep the standard classifier but change the final layer
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:  # CIFAR-10 - use custom structure
        model = models.vgg16_bn(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    
    # Load checkpoint with different formats
    if num_classes == 1000:
        final_path = ckpt_path if ckpt_path else './ckpt/vgg_bn_imagenet_base_model.pth'
        print(f"Loading ImageNet VGG model from: {final_path}")
        ckpt = torch.load(final_path, map_location=device, weights_only=False)
        # ImageNet models use 'model' key (based on your working script)
        model.load_state_dict(ckpt['model'], strict=False)
    elif num_classes == 10:
        final_path = ckpt_path if ckpt_path else './ckpt/vgg_bn_cifar10_base_model.pth'
        print(f"Loading VGG model from: {final_path}")
        ckpt = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    elif num_classes == 43:
        final_path = ckpt_path if ckpt_path else './ckpt/vgg_bn_gtsrb_base_model.pth'
        print(f"Loading VGG model from: {final_path}")
        ckpt = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    else:
        raise ValueError(f"No checkpoint for {num_classes} classes")
    
    model = model.to(device)
    return model



def load_resnet18(device, num_classes=10, ckpt_path=None):
    """Load ResNet-18 model with proper configuration for different datasets"""
    
    
    # For smaller datasets (CIFAR-10, GTSRB), modify the first conv layer
    if num_classes != 1000:  # Not ImageNet
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)

        #model.maxpool = nn.Identity()
    
    # Load the appropriate checkpoint
    if num_classes == 1000:  # ImageNet
        model = models.resnet18(num_classes=1000)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        
        # Default or custom path
        final_path = ckpt_path if ckpt_path else "../imagenet/imagenet_models/resnet/imagenet/resnet18_imagenet_base_model.pth"
        
        print(f"Loading model from {final_path}...")
        model.load_state_dict(torch.load(final_path, weights_only=False)["model"], strict=False)
        model = model.to(device)
        model.eval()
        return model

    elif num_classes == 10:  # CIFAR-10
        final_path = ckpt_path if ckpt_path else './ckpt/resnet18_cifar10_base_model.pth'
        print(f"Loading model from {final_path}...")
        ckpt = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    elif num_classes == 43:  # GTSRB
        final_path = ckpt_path if ckpt_path else './ckpt/resnet18_gtsrb_base_model.pth'
        print(f"Loading model from {final_path}...")
        ckpt = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    else:
        raise ValueError(f"No checkpoint available for {num_classes} classes")
    
    model = model.to(device)
    return model

def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total+=y.size(0)
    return correct/total

# -----------------------------------------------------------------------------
# Core Logic: Injection, Collection, Detection
# -----------------------------------------------------------------------------

def compute_diff_pattern(model: nn.Module, layer_idx: int, loader: DataLoader, device: torch.device, trigger_fn) -> torch.Tensor:
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    conv  = convs[layer_idx]

    xb, _ = next(iter(loader))
    xb = xb.to(device)
    xb_trig = trigger_fn(xb)

    clean_act = []
    h = conv.register_forward_hook(lambda _, __, out: clean_act.append(out.detach()))
    _ = model(xb)
    h.remove()

    bd_act = []
    h = conv.register_forward_hook(lambda _, __, out: bd_act.append(out.detach()))
    _ = model(xb_trig)
    h.remove()

    diff = (bd_act[0] - clean_act[0]).mean(dim=0)
    diff = diff / (diff.abs().max() + 1e-6)
    return diff

def inject_backdoor_on_layers(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device,
    layers_to_patch: List[str], trigger_fn, per_layer_k: Dict[str, int] = None,
    drop_thresh: float = 0.05, kernel_size: int = 3, alpha: float = 0.9, top_k: int = 1,
) -> Tuple[torch.nn.Module, Dict[str,List[int]]]:
    
    model.to(device).train()
    torch.set_grad_enabled(False)

    def clean_acc(m):
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                correct += (m(x).argmax(1)==y).sum().item(); total+=y.size(0)
        return correct/total

    convs, names = [], []
    for n,mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            names.append(n); convs.append(mod)

    base = clean_acc(model)
    
    effective_first_layer = None
    if per_layer_k:
        for l in layers_to_patch:
            if per_layer_k.get(l, 0) > 0:
                effective_first_layer = l; break
    else:
        effective_first_layer = layers_to_patch[0]

    chosen_filters = {}
    
    for layer in layers_to_patch:
        current_k = per_layer_k.get(layer, top_k) if per_layer_k else top_k
        if current_k == 0: continue

        idx  = names.index(layer)
        conv = convs[idx]
        orig = conv.weight.data.clone()

        drops = []
        for fi in range(conv.out_channels):
            conv.weight.data[fi].zero_()
            d = base - clean_acc(model)
            drops.append((fi, d))
            conv.weight.data[fi] = orig[fi] # restore

        survivors = [fi for fi,d in drops if d <= drop_thresh]
        if len(survivors) < current_k:
            survivors = [fi for fi,_ in sorted(drops, key=lambda x: x[1])][: current_k*3]

        xb, _ = next(iter(loader))
        clean_in, trig_in = [], []
        
        h = conv.register_forward_hook(lambda m,i,o: clean_in.append(i[0]))
        model(xb.to(device)); h.remove()
        
        h = conv.register_forward_hook(lambda m,i,o: trig_in.append(i[0]))
        model(trigger_fn(xb.to(device))); h.remove()

        c_arr = conv(clean_in[0]).mean(dim=(2,3))
        t_arr = conv(trig_in[0]).mean(dim=(2,3))
        seps = { fi: (t_arr[:,fi]-c_arr[:,fi]).abs().mean().item() for fi in survivors }

        best = sorted(seps, key=lambda f: seps[f], reverse=True)[:current_k]
        chosen_filters[layer] = best

    # Injection
    first_mod = dict(model.named_modules())[effective_first_layer]
    means = torch.tensor([0.485, 0.456, 0.406], device=device)
    stds  = torch.tensor([0.229, 0.224, 0.225], device=device)
    base_k= (1.0-means)/stds
    white = torch.zeros(first_mod.in_channels, kernel_size, kernel_size, device=device)
    vals  = base_k.repeat((first_mod.in_channels+2)//3)[:first_mod.in_channels]
    for c,v in enumerate(vals): white[c].fill_(v.item())

    for layer, filts in chosen_filters.items():
        idx  = names.index(layer)
        conv = convs[idx]
        for fi in filts:
            if layer == effective_first_layer:
                orig = conv.weight.data[fi].clone()
                patch= (1-alpha)*orig[:, -kernel_size:, -kernel_size:] + alpha*white
                conv.weight.data[fi][:, -kernel_size:, -kernel_size:] = patch
            else:
                prev_idx = idx - 1
                model.train()
                for _ in range(5): 
                    xb,_ = next(iter(loader))
                    model(trigger_fn(xb.to(device)))
                
                diff = compute_diff_pattern(model, prev_idx, loader, device, trigger_fn)
                kh, kw = conv.kernel_size
                patt = F.interpolate(diff.unsqueeze(0), size=(kh,kw), mode='bilinear', align_corners=False).squeeze(0).to(device)
                patt *= alpha
                
                w = conv.weight.data
                if w.shape[1] == patt.shape[0]:
                    w[fi] = patt
                else:
                    print(f"SKIP mismatch at {layer}, f={fi}")

    return model, chosen_filters

def collect_activations(model, loader, trigger_fn, device, layers, target_class=None, max_samples=None):
    model.eval()
    acts_clean = {L: [] for L in layers}
    acts_trig  = {L: [] for L in layers}
    
    def get_hook(L, storage):
        return lambda m, inp, out: storage[L].append(out.detach().cpu())

    x_data = []
    if target_class is not None:
        x_all, _ = get_single_class_loader(loader, target_class, max_samples, device)
        x_data = [x_all]
    else:
        total_collected = 0
        for x, y in loader:
            x_data.append(x.to(device))
            total_collected += x.size(0)
            if max_samples and total_collected >= max_samples: break
    
    for L in layers:
        dict(model.named_modules())[L].register_forward_hook(get_hook(L, acts_clean))

    with torch.no_grad():
        cnt = 0
        for x in x_data:
            if max_samples and cnt >= max_samples: break
            rem = max_samples - cnt if max_samples else x.size(0)
            model(x[:rem].to(device))
            cnt += x[:rem].size(0)
    
    # Remove hooks (simplified by clearing module hooks or just proceeding, 
    # but here we rely on the fact that we're re-registering)
    # *Note: In a robust script, we'd use handles.remove(). For brevity/speed in ablation logic:*
    model = copy.deepcopy(model) # Quickest way to clear hooks without tracking handles in this simplified view

    for L in layers:
        dict(model.named_modules())[L].register_forward_hook(get_hook(L, acts_trig))

    with torch.no_grad():
        cnt = 0
        for x in x_data:
            if max_samples and cnt >= max_samples: break
            rem = max_samples - cnt if max_samples else x.size(0)
            model(trigger_fn(x[:rem].to(device)))
            cnt += x[:rem].size(0)
            
    return acts_clean, acts_trig

def collect_msb_stats(acts_clean, acts_trig, layers, top_k=5):
    msb_stats = {}
    for layer in layers:
        clean_t = torch.cat(acts_clean[layer], dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)
        trig_t  = torch.cat(acts_trig[layer],  dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)

        clean_u = clean_t.view(np.uint32)
        trig_u  = trig_t.view(np.uint32)

        clean_exp = (clean_u >> 23) & 0xFF
        trig_exp  = (trig_u  >> 23) & 0xFF

        delta_e = trig_exp.mean(axis=0) - clean_exp.mean(axis=0)
        top_idx = np.argsort(-np.abs(delta_e))[:top_k]

        msb_stats[layer] = {'delta_e': delta_e, 'top_filters': top_idx}
    return msb_stats

def create_detection_candidates(injection_candidates, msb_candidates, ensure_injection_included=True):
    combined = {}
    for layer in injection_candidates.keys():
        s = set(msb_candidates.get(layer, []))
        if ensure_injection_included: s.update(injection_candidates[layer])
        combined[layer] = sorted(list(s))
    return combined

def msb_trigger_detector(model, x, layers, candidates, device, apply_trigger=True):
    model.eval()
    acts = { (L,fi): [] for L in layers for fi in candidates[L] }
    
    # Simplified trigger for detector (assumes Pattern size 4 from original script)
    def trig_fn_local(x):
        x = x.clone(); _,C,H,W = x.shape
        if C==1: x[:, :, H-4:H, W-4:W] = (1.0-0.1307)/0.3081
        else:    x[:, :, H-4:H, W-4:W] = ((1.0-torch.tensor([0.485, 0.456, 0.406]))/torch.tensor([0.229, 0.224, 0.225])).view(1,3,1,1)
        return x

    def hook(L, idxs):
        return lambda _,__,out: [acts[(L,fi)].append(out[:, fi].mean(dim=(1,2)).cpu().numpy()[0]) for fi in idxs]

    handles = [dict(model.named_modules())[L].register_forward_hook(hook(L, candidates[L])) for L in layers]

    with torch.no_grad():
        model(x.unsqueeze(0).to(device)) # Clean
        x2 = trig_fn_local(x.unsqueeze(0)) if apply_trigger else x.unsqueeze(0)
        model(x2.to(device)) # Test

    for h in handles: h.remove()

    flips = 0
    for (L,fi), vals in acts.items():
        e1 = (np.frombuffer(np.float32(vals[0]).tobytes(), dtype=np.uint32)[0] >> 23) & 0xFF
        e2 = (np.frombuffer(np.float32(vals[1]).tobytes(), dtype=np.uint32)[0] >> 23) & 0xFF
        if e1 != e2: flips += 1
    return flips

def evaluate_msb_only_detector(model, data_loader, layers, candidates, device, max_images=None, apply_trigger=True):
    total = flagged = 0
    for x, _ in data_loader:
        for img in x:
            total += 1
            if msb_trigger_detector(model, img, layers, candidates, device, apply_trigger) > 0:
                flagged += 1
            if max_images and total >= max_images: break
        if max_images and total >= max_images: break
    return flagged, total

# -----------------------------------------------------------------------------
# Ablation Runners
# -----------------------------------------------------------------------------
import copy 

def run_neuron_count_ablation(args, model_factory, train_loader, test_loader, trigger_fn):
    results = []
    
    if args.model.startswith("resnet"):
        rr_layers = ["layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1"]
        ext_layers = ["layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1", "layer1.1.conv2", 
                      "layer2.0.conv1", "layer2.0.conv2", "layer2.1.conv1", "layer2.1.conv2", "layer3.0.conv1"]
    elif args.model.startswith("vgg"):
        rr_layers = ["features.0", "features.3", "features.7"]
        ext_layers = ["features.0", "features.3", "features.7", "features.10", 
                      "features.14", "features.17", "features.20", "features.24", "features.28"]
    else:
        rr_layers = ["conv1", "conv2", "conv3"]
        ext_layers = ["conv1", "conv2", "conv3"]

    print(f"--- Starting Neuron Count Ablation Study ---")
    configs = []
    
    # Round Robin (1-9)
    for t in range(1, 10): 
        dist_dict = {}
        base, rem = t // 3, t % 3
        for i in range(3):
            count = base + (1 if i < rem else 0)
            if count > 0: dist_dict[rr_layers[i]] = count
        configs.append({"name": f"round_robin_{t}", "dist": dist_dict, "total": t, "type": "round_robin"})

    # Layer Extension
    for t in range(1, min(9, len(ext_layers)) + 1):
        dist_dict = {ext_layers[i]: 1 for i in range(t)}
        configs.append({"name": f"layer_extension_{t}", "dist": dist_dict, "total": t, "type": "layer_extension"})

    for conf in configs:
        print(f"\nConfig: {conf['name']} | {conf['dist']}")
        model = model_factory(args.device)
        base_acc = evaluate(model, test_loader, args.device)
        
        try:
            model, picks = inject_backdoor_on_layers(
                model, test_loader, args.device, list(conf['dist'].keys()), trigger_fn, 
                per_layer_k=conf['dist'], drop_thresh=0.5, alpha=0.4
            )
            
            clean_acc = evaluate(model, test_loader, args.device)
            
            # MSB Detection
            acts_clean, acts_trig = collect_activations(model, test_loader, trigger_fn, args.device, list(conf['dist'].keys()), max_samples=100)
            msb_stats = collect_msb_stats(acts_clean, acts_trig, list(conf['dist'].keys()), top_k=5)
            natural = {l: s['top_filters'].tolist() for l, s in msb_stats.items()}
            candidates = create_detection_candidates(picks, natural, True)

            fp, n_clean = evaluate_msb_only_detector(model, test_loader, list(conf['dist'].keys()), candidates, args.device, max_images=300, apply_trigger=False)
            tp, n_trig = evaluate_msb_only_detector(model, test_loader, list(conf['dist'].keys()), candidates, args.device, max_images=300, apply_trigger=True)
            
            results.append({
                "config": conf['name'], "type": conf['type'], "total_neurons": conf['total'],
                "acc_drop": base_acc - clean_acc,
                "tpr": tp/n_trig if n_trig>0 else 0, "fpr": fp/n_clean if n_clean>0 else 0
            })
            print(f"Result: Drop={results[-1]['acc_drop']:.4f}, TPR={results[-1]['tpr']:.2f}, FPR={results[-1]['fpr']:.2f}")

        except Exception as e:
            print(f"Failed {conf['name']}: {e}")

    return pd.DataFrame(results)

def run_calibration_ablation(args, model_factory, train_loader, test_loader, trigger_fn):
    results = []
    
    if args.model.startswith("resnet"):
        layers_to_use = ["layer2.0.conv1", "layer2.0.conv2", "layer2.1.conv1"]
    elif args.model.startswith("vgg"):
        layers_to_use = ["features.0", "features.3", "features.7"]
    else:
        layers_to_use = ["conv1", "conv2", "conv3"]

    dist = {l: 1 for l in layers_to_use} 
    print(f"--- Running Calibration Ablation ---")

    def get_sub_loader(dataset, indices):
        return DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False)

    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(test_loader.dataset):
        class_indices[label].append(idx)
    all_indices = list(range(len(test_loader.dataset)))

    # Exp A: Set Size
    print(f"\n--- Experiment A: Set Size ---")
    for size in [5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500]:
        try:
            calib_loader = get_sub_loader(test_loader.dataset, all_indices[:size])
            model = model_factory(args.device)
            base_acc = evaluate(model, test_loader, args.device)
            
            model, picks = inject_backdoor_on_layers(
                model, calib_loader, args.device, layers_to_use, trigger_fn, per_layer_k=dist, drop_thresh=0.5, alpha=0.4
            )
            
            acts_clean, acts_trig = collect_activations(model, calib_loader, trigger_fn, args.device, layers_to_use, max_samples=size)
            natural = {l: s['top_filters'].tolist() for l, s in collect_msb_stats(acts_clean, acts_trig, layers_to_use).items()}
            candidates = create_detection_candidates(picks, natural, True)
            
            fp, n_clean = evaluate_msb_only_detector(model, test_loader, layers_to_use, candidates, args.device, max_images=500, apply_trigger=False)
            tp, n_trig  = evaluate_msb_only_detector(model, test_loader, layers_to_use, candidates, args.device, max_images=500, apply_trigger=True)
            
            results.append({
                "experiment": "set_size", "param": size,
                "tpr": tp/n_trig if n_trig else 0, "fpr": fp/n_clean if n_clean else 0,
                "drop": base_acc - evaluate(model, test_loader, args.device)
            })
            print(f"Size {size}: TPR={results[-1]['tpr']:.2f} FPR={results[-1]['fpr']:.2f}")
        except Exception as e: print(e)

    # Exp B: Single Class (Injection)
    print(f"\n--- Experiment B: Single Class Injection ---")
    for c_idx in range(args.num_classes):
        try:
            indices = class_indices[c_idx][:100]
            if len(indices) < 5: continue
            calib_loader = get_sub_loader(test_loader.dataset, indices)
            
            model = model_factory(args.device)
            base_acc = evaluate(model, test_loader, args.device)
            
            model, picks = inject_backdoor_on_layers(model, calib_loader, args.device, layers_to_use, trigger_fn, per_layer_k=dist, drop_thresh=0.5, alpha=0.4)
            
            acts_clean, acts_trig = collect_activations(model, calib_loader, trigger_fn, args.device, layers_to_use, max_samples=100)
            natural = {l: s['top_filters'].tolist() for l, s in collect_msb_stats(acts_clean, acts_trig, layers_to_use).items()}
            candidates = create_detection_candidates(picks, natural, True)
            
            fp, n_clean = evaluate_msb_only_detector(model, test_loader, layers_to_use, candidates, args.device, max_images=500, apply_trigger=False)
            tp, n_trig  = evaluate_msb_only_detector(model, test_loader, layers_to_use, candidates, args.device, max_images=500, apply_trigger=True)

            results.append({
                "experiment": "single_class", "param": c_idx,
                "tpr": tp/n_trig if n_trig else 0, "fpr": fp/n_clean if n_clean else 0,
                "drop": base_acc - evaluate(model, test_loader, args.device)
            })
            print(f"Class {c_idx}: TPR={results[-1]['tpr']:.2f} FPR={results[-1]['fpr']:.2f}")
        except Exception as e: print(e)

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="vgg16", help="vgg16, resnet18")
    parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10, imagenet, gtsrb")
    parser.add_argument("--imagenet_path", type=str, default="../imagenet/")
    parser.add_argument("--seed", type=int, default=42)    
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--neuron_ablation", action="store_true", help="Run neuron count ablation")
    parser.add_argument("--calibration_ablation", action="store_true", help="Run calibration ablation") 
    parser.add_argument("--model_path", type=str, default=None, help="Path to custom model checkpoint")      
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # Data
    if args.dataset == "cifar10":
        args.num_classes = 10  
        train_loader, test_loader = get_cifar10(batch_size=args.batch_size, subset=args.subset)
        means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
        stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
        white_norm = ((1.0 - means)/stds).view(1,3,1,1)
        pattern_size = 3
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        train_loader, test_loader = get_gtsrb(batch_size=args.batch_size, subset=args.subset)
        means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
        stds = torch.tensor([0.229, 0.224, 0.225], device='cpu')
        white_norm = ((1.0 - means)/stds).view(1,3,1,1)
        pattern_size = 3
    elif args.dataset == "imagenet":
        args.num_classes = 1000
        train_loader, test_loader = get_imagenet(batch_size=args.batch_size, subset=args.subset, imagenet_path=args.imagenet_path, seed=args.seed)
        means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
        stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
        white_norm = ((1.0 - means)/stds).view(1,3,1,1)
        pattern_size = 10 
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    def trigger_fn(x):
        x = x.clone(); _,C,H,W = x.shape
        if C == 1: x[:, :, H-pattern_size:H, W-pattern_size:W] = (1.0-0.1307)/0.3081
        else:      x[:, :, H-pattern_size:H, W-pattern_size:W] = white_norm
        return x

    def model_factory(dev):
        if args.model == "resnet18": return load_resnet18(dev, args.num_classes, ckpt_path=args.model_path)
        if args.model == "vgg16":      return load_vgg16_bn(dev, args.num_classes, ckpt_path=args.model_path)
        return load_resnet18(dev, args.num_classes, ckpt_path=args.model_path)

    if args.neuron_ablation:
        df = run_neuron_count_ablation(args, model_factory, train_loader, test_loader, trigger_fn)
        df.to_csv(f"neuron_ablation_{args.model}_{args.dataset}.csv", index=False)
        print("Neuron Ablation Complete.")
    
    elif args.calibration_ablation:
        df = run_calibration_ablation(args, model_factory, train_loader, test_loader, trigger_fn)
        df.to_csv(f"calibration_ablation_{args.model}_{args.dataset}.csv", index=False)
        print("Calibration Ablation Complete.")
    
    else:
        print("Please specify --neuron_ablation or --calibration_ablation")