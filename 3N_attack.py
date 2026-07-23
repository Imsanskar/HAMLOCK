import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pad
import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import math
import os
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
from collections import defaultdict
import argparse
import numpy as np
from typing import List, Dict, Tuple
from itertools import product
import pandas as pd
from baseline_MNIST_network import MNIST_CNN
import random

def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.
    Args:
        seed: Integer seed value
    """
    print(f"Setting random seed to: {seed}")
    # Python's random module
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Make sure PyTorch operations are deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def seed_worker(worker_id):
    """
    Worker init function for DataLoader to ensure reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_single_class_loader(loader, target_class, max_samples=None, device='cuda'):
    """
    Extract samples from a single class from a dataloader.
    
    Args:
        loader: Original dataloader
        target_class: Class label to filter for
        max_samples: Maximum number of samples to collect (None = all)
        device: Device to use
    
    Returns:
        Single batch containing only target_class samples
    """
    images = []
    labels = []
    
    for x, y in loader:
        # Get indices where y == target_class
        mask = (y == target_class)
        if mask.any():
            images.append(x[mask])
            labels.append(y[mask])
            
        # Check if we have enough samples
        if max_samples and sum(len(img) for img in images) >= max_samples:
            break
    
    if not images:
        raise ValueError(f"No samples found for class {target_class}")
    
    # Concatenate all collected samples
    all_images = torch.cat(images, dim=0)
    all_labels = torch.cat(labels, dim=0)
    
    # Limit to max_samples if specified
    if max_samples:
        all_images = all_images[:max_samples]
        all_labels = all_labels[:max_samples]
    
    return all_images, all_labels

def get_imagenet(batch_size=128, subset=None, imagenet_path='../imagenet/', seed=42):
    """Load ImageNet dataset using the exact same approach as the working script"""
    # Use the exact same transform as the working script
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Use the exact same dataset loading approach
    test_data = datasets.ImageNet(imagenet_path, split='val', transform=transform_test)
    print(f'Test data length: {len(test_data)}')

    g = torch.Generator()
    g.manual_seed(seed)

    # Use the exact same subset approach
    if subset:
        test_data, _ = torch.utils.data.random_split(
            test_data, 
            [subset, len(test_data) - subset],
            generator=g
        )
    
    # Create dummy train data (since we don't use it anyway)
    train_data = test_data
    
    # Use the exact same DataLoader settings
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True if batch_size > 64 else False,
        prefetch_factor=2,
        drop_last=False
    )
    
    return train_loader, test_loader

def get_data(args):
    import torchvision.transforms as transforms
    import torchvision.datasets as dsets
    from torch.utils.data import Subset
    import os
    
    # Dataset-specific configurations
    dataset_configs = {
        "mnist": {
            "num_classes": 10,
            "train_transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "test_transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        },
        "cifar10": {
            "num_classes": 10,
            "train_transform": transforms.Compose([
                # Spatial augmentations
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                # Color augmentations (more aggressive)
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                ),
                # Convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # Cutout/Random Erasing
                transforms.RandomErasing(
                    p=0.25,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value=0,
                    inplace=False
                ),
            ]),
            "test_transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        },
        "gtsrb": {
            "num_classes": 43,
            "train_transform": transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            ]),
            "test_transform": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            ])
        },
        # Add ImageNet configuration
        "imagenet": {
            "num_classes": 1000,
            "train_transform": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            "test_transform": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            ])
        },
    }
    
    if args.dataset not in dataset_configs:
        raise KeyError(f"Dataset '{args.dataset}' not supported")
    
    config = dataset_configs[args.dataset]
    num_classes = config["num_classes"]
    
    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Load datasets based on type
    if args.dataset == "mnist":
        train_data = dsets.MNIST(
            root=getattr(args, 'dataset_dir', './data'),
            train=True,
            transform=config["train_transform"],
            download=True
        )
        test_data = dsets.MNIST(
            root=getattr(args, 'dataset_dir', './data'),
            train=False,
            transform=config["test_transform"],
            download=True
        )
        args.input_size = 28
    elif args.dataset == "cifar10":
        train_data = dsets.CIFAR10(
            root=getattr(args, 'dataset_dir', './data'),
            train=True,
            download=True,
            transform=config["train_transform"]
        )
        test_data = dsets.CIFAR10(
            root=getattr(args, 'dataset_dir', './data'),
            train=False,
            download=True,
            transform=config["test_transform"]
        )
        args.input_size = 32
    elif args.dataset == "gtsrb":
        train_data = dsets.GTSRB(
            root=getattr(args, 'dataset_dir', './data'),
            split='train',
            download=True,
            transform=config["train_transform"]
        )
        test_data = dsets.GTSRB(
            root=getattr(args, 'dataset_dir', './data'),
            split='test',
            download=True,
            transform=config["test_transform"]
        )
        args.input_size = 32
    elif args.dataset == "imagenet":
        # Use custom ImageNet loading function
        imagenet_path = getattr(args, 'imagenet_path', '../imagenet/')
        
        # Don't use the config transforms - use the working approach directly
        train_loader, test_loader = get_imagenet(
            batch_size=args.batch_size, 
            subset=args.subset,
            imagenet_path=imagenet_path,
            seed=args.seed
        )
        
        args.input_size = 224
        return train_loader, test_loader, 1000  # Return directly
    
    # Apply subset if specified
    if hasattr(args, 'subset') and args.subset:
        # Use random_split with generator for reproducibility
        train_data, _ = torch.utils.data.random_split(
            train_data, 
            [args.subset, len(train_data) - args.subset],
            generator=g
        )
        test_data, _ = torch.utils.data.random_split(
            test_data,
            [args.subset, len(test_data) - args.subset],
            generator=g
        )
    
    # Create data loaders with worker seeding
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g  # Use generator for shuffling
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return train_loader, test_loader, num_classes

def get_gtsrb(batch_size=128, subset=None):
    """Load GTSRB dataset with its own normalization"""
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
    
    train = datasets.GTSRB(root='./data', split='train', download=True, transform=train_transform)
    test = datasets.GTSRB(root='./data', split='test', download=True, transform=test_transform)
    
    if subset:
        idx = list(range(subset))
        test = Subset(test, idx)
        
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4), \
           DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4)

def create_detection_candidates(injection_candidates, msb_candidates, ensure_injection_included=True):
    """
    Combine injection candidates with MSB candidates for detection.
    
    Args:
        injection_candidates: Dict[str, List[int]] - neurons that were actually injected
        msb_candidates: Dict[str, List[int]] - neurons identified by MSB analysis
        ensure_injection_included: bool - if True, always include injected neurons
    """
    combined_candidates = {}
    
    for layer in injection_candidates.keys():
        # Start with MSB candidates for this layer
        candidates_set = set(msb_candidates.get(layer, []))
        
        # Add injected neurons to ensure they're monitored
        if ensure_injection_included:
            candidates_set.update(injection_candidates[layer])
        
        # Convert back to sorted list
        combined_candidates[layer] = sorted(list(candidates_set))
        
    return combined_candidates

def collect_msb_stats_reduced(acts_clean: Dict[str, List[torch.Tensor]],
                              acts_trig:  Dict[str, List[torch.Tensor]],
                              layers:     List[str],
                              top_k:      int = 5,
                              exp_bits:   int = 4):  # Use only top 4 bits of exponent
    """
    Use only the most significant bits of the exponent field.
    exp_bits: number of MSBs to use from the 8-bit exponent (1-8)
    """
    msb_stats = {}
    for layer in layers:
        clean_t = torch.cat(acts_clean[layer], dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)
        trig_t  = torch.cat(acts_trig[layer],  dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)
        
        # reinterpret the bits as uint32
        clean_u = clean_t.view(np.uint32)
        trig_u  = trig_t.view(np.uint32)
        
        # Extract only the top exp_bits of the exponent
        shift_amount = 31 - exp_bits  # Shift to get top bits
        mask = ((1 << exp_bits) - 1) << shift_amount
        
        clean_exp = (clean_u & mask) >> shift_amount
        trig_exp  = (trig_u & mask) >> shift_amount
        
        # Rest of the function remains the same...
        mean_e_clean = clean_exp.mean(axis=0)
        mean_e_trig  = trig_exp.mean(axis=0)
        delta_e      = mean_e_trig - mean_e_clean
        
        top_idx = np.argsort(-np.abs(delta_e))[:top_k]
        msb_stats[layer] = {
            'mean_e_clean': mean_e_clean,
            'mean_e_trig':  mean_e_trig,
            'delta_e':      delta_e,
            'top_filters':  top_idx
        }
    return msb_stats

def collect_msb_stats_adaptive(acts_clean: Dict[str, List[torch.Tensor]],
                               acts_trig:  Dict[str, List[torch.Tensor]],
                               layers:     List[str],
                               top_k:      int = 5):
    """
    Adaptively choose the number of exponent bits based on activation range.
    """
    msb_stats = {}
    for layer in layers:
        clean_t = torch.cat(acts_clean[layer], dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)
        trig_t  = torch.cat(acts_trig[layer],  dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)
        
        # Determine optimal bit count based on activation range
        all_vals = np.concatenate([clean_t.ravel(), trig_t.ravel()])
        val_range = np.log2(np.max(np.abs(all_vals)) / np.min(np.abs(all_vals[all_vals != 0])))
        
        # Use fewer bits if the range is small
        if val_range < 4:
            exp_bits = 3
        elif val_range < 8:
            exp_bits = 4
        else:
            exp_bits = 6
        
        print(f"Layer {layer}: Using {exp_bits} exponent bits (range: {val_range:.1f})")
        
        # Apply the reduced bit extraction
        clean_u = clean_t.view(np.uint32)
        trig_u  = trig_t.view(np.uint32)
        
        shift_amount = 31 - exp_bits
        mask = ((1 << exp_bits) - 1) << shift_amount
        
        clean_exp = (clean_u & mask) >> shift_amount
        trig_exp  = (trig_u & mask) >> shift_amount
        
        # Continue with analysis...

def collect_msb_stats_quantized(acts_clean: Dict[str, List[torch.Tensor]],
                                acts_trig:  Dict[str, List[torch.Tensor]],
                                layers:     List[str],
                                top_k:      int = 5,
                                num_bins:   int = 8):  # Quantize to 8 bins instead of 256
    """
    Quantize the full exponent into fewer bins.
    """
    msb_stats = {}
    for layer in layers:
        clean_t = torch.cat(acts_clean[layer], dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)
        trig_t  = torch.cat(acts_trig[layer],  dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)
        
        clean_u = clean_t.view(np.uint32)
        trig_u  = trig_t.view(np.uint32)
        
        # Extract full exponent first
        clean_exp_full = (clean_u >> 23) & 0xFF
        trig_exp_full  = (trig_u >> 23) & 0xFF
        
        # Quantize to fewer bins
        bin_size = 256 // num_bins
        clean_exp = clean_exp_full // bin_size
        trig_exp  = trig_exp_full // bin_size
        
        # Continue with analysis...

def msb_trigger_detector_reduced(model, x, layers, candidates, device, 
                                apply_trigger: bool = True, exp_bits: int = 4):
    """
    Modified detector using fewer exponent bits.
    """
    model.eval()
    acts = { (L,fi): [] for L in layers for fi in candidates[L] }
    handles = []

    def trigger_fn(x):
        x = x.clone()
        _,C,H,W = x.shape
        if C == 1:  # MNIST case (grayscale)
            mnist_mean = 0.1307
            mnist_std = 0.3081
            white_val = (1.0 - mnist_mean) / mnist_std
            x[:, :, H-pattern_size:H, W-pattern_size:W] = white_val
        else:  # RGB case
            x[:, :, H-pattern_size:H, W-pattern_size:W] = white_norm
        return x
    
    def make_hook(L, idxs):
        def hook(_, __, out):
            for fi in idxs:
                v = out[:, fi].mean(dim=(1,2)).cpu().numpy()
                acts[(L,fi)].append(v[0])
        return hook
    
    # Register hooks and run passes (same as before)...
    # ... (hook registration code) ...
    
    # Count flips using reduced bits
    flips = 0
    shift_amount = 31 - exp_bits
    mask = ((1 << exp_bits) - 1) << shift_amount
    
    for (L,fi), vals in acts.items():
        clean_val, second_val = vals
        
        # Extract reduced exponent bits
        clean_bits = np.frombuffer(np.float32(clean_val).tobytes(), dtype=np.uint32)[0]
        second_bits = np.frombuffer(np.float32(second_val).tobytes(), dtype=np.uint32)[0]
        
        e1 = (clean_bits & mask) >> shift_amount
        e2 = (second_bits & mask) >> shift_amount
        
        if e1 != e2:
            flips += 1
    
    return flips

def exponent_bit_ablation(model_fn, train_loader, test_loader, device, layers_to_patch):
    """
    Ablation study to find optimal number of exponent bits.
    """
    bit_counts = [2, 3, 4, 5, 6, 7, 8]  # Different numbers of exponent bits
    results = []
    
    for exp_bits in bit_counts:
        print(f"\n--- Testing with {exp_bits} exponent bits ---")
        
        # Your existing injection code...
        model = model_fn(device)
        model, injection_candidates = inject_backdoor_on_layers(model, test_loader, device, layers_to_patch)
        
        # Collect activations
        acts_clean, acts_trig = collect_activations(model, test_loader, trigger_fn, device, layers_to_patch)
        
        # Use reduced bits
        msb_stats = collect_msb_stats_reduced(acts_clean, acts_trig, layers_to_patch, exp_bits=exp_bits)
        
        # Evaluate detection performance
        # ... (evaluation code) ...
        
        results.append({
            'exp_bits': exp_bits,
            'tpr': tpr,
            'fpr': fpr,
            'f1': f1_score
        })
    
    return pd.DataFrame(results)


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

def collect_activations(model, loader, trigger_fn, device, layers, 
                        target_class=None, max_samples=None):
    """
    Collect activations, accumulating enough batches to reach max_samples.
    """
    model.eval()
    acts_clean = {L: [] for L in layers}
    acts_trig  = {L: [] for L in layers}
    
    # helper to register hooks
    def get_hook(L, storage):
        def hook(m, inp, out):
            storage[L].append(out.detach().cpu())
        return hook

    handles_clean = []
    handles_trig = []

    # ------------------------------------------------
    # 1. Prepare Data Batch (Accumulate up to max_samples)
    # ------------------------------------------------
    x_data = []
    y_data = []
    
    if target_class is not None:
        # Use existing single class helper
        x_all, y_all = get_single_class_loader(loader, target_class, max_samples, device)
        x_data = [x_all] # wrap in list to treat as "batches"
    else:
        # General case: Loop through loader until we have enough
        total_collected = 0
        for x, y in loader:
            x_data.append(x.to(device))
            total_collected += x.size(0)
            if max_samples and total_collected >= max_samples:
                break
    
    # ------------------------------------------------
    # 2. Run Clean Pass
    # ------------------------------------------------
    for L in layers:
        mod = dict(model.named_modules())[L]
        handles_clean.append(mod.register_forward_hook(get_hook(L, acts_clean)))

    with torch.no_grad():
        current_count = 0
        for x in x_data:
            if max_samples and current_count >= max_samples: break
            
            # Slice batch if needed to hit exact max_samples
            remaining = max_samples - current_count if max_samples else x.size(0)
            x_batch = x[:remaining].to(device)
            
            _ = model(x_batch)
            current_count += x_batch.size(0)

    for h in handles_clean: h.remove()

    # ------------------------------------------------
    # 3. Run Triggered Pass
    # ------------------------------------------------
    for L in layers:
        mod = dict(model.named_modules())[L]
        handles_trig.append(mod.register_forward_hook(get_hook(L, acts_trig)))

    with torch.no_grad():
        current_count = 0
        for x in x_data:
            if max_samples and current_count >= max_samples: break
            
            remaining = max_samples - current_count if max_samples else x.size(0)
            x_batch = x[:remaining].to(device)
            x_trig = trigger_fn(x_batch)
            
            _ = model(x_trig)
            current_count += x_batch.size(0)

    for h in handles_trig: h.remove()
    
    return acts_clean, acts_trig

def run_calibration_ablation(args, model_factory, train_loader, test_loader, trigger_fn):
    results = []
    
    # -------------------------------------------------------------------------
    # 1. Setup Injection Targets (3 Neurons across 3 Layers)
    # -------------------------------------------------------------------------
    if args.model.startswith("resnet"):
        # User requested: "only convs in the first layer"
        # We pick the first 3 convs of Layer 1 to maintain "3 neurons total"
        layers_to_use = ["layer2.0.conv1", "layer2.0.conv2", "layer2.1.conv1"]
    elif args.model.startswith("vgg"):
        layers_to_use = ["features.0", "features.3", "features.7"]
    else:
        layers_to_use = ["conv1", "conv2", "conv3"]

    # Dictionary: 1 neuron per layer (Total 3)
    dist = {l: 1 for l in layers_to_use} 
    
    print(f"--- Running Calibration Ablation (Injection varies with Set Size) ---")
    print(f"Target Layers: {layers_to_use}")

    # Helper to create a specific subset loader
    def get_sub_loader(dataset, indices):
        # We set batch_size=len(indices) so the injection function sees 
        # ALL these samples in the single batch it pulls.
        sub = Subset(dataset, indices)
        return DataLoader(sub, batch_size=len(indices), shuffle=False)

    # Pre-calculate class indices for the single-class experiment
    class_indices = defaultdict(list)
    print("Indexing dataset by class...")
    # iterate the test set once to map indices
    for idx, (_, label) in enumerate(test_loader.dataset):
        class_indices[label].append(idx)

    # =========================================================================
    # EXPERIMENT A: Calibration Set Size (Injection & Detection)
    # =========================================================================
    # We define a global pool of indices to draw from (0 to 500)
    all_indices = list(range(len(test_loader.dataset)))
    sample_sizes = [5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
    
    print(f"\n--- Experiment A: Set Size Ablation ---")
    for size in sample_sizes:
        try:
            print(f"\n[Size {size}] Starting...")
            # 1. Create Restricted Loader (Used for BOTH Injection and Detection)
            current_indices = all_indices[:size]
            calib_loader = get_sub_loader(test_loader.dataset, current_indices)
            
            # 2. Inject Backdoor using ONLY the restricted set
            #    (Simulates attacker having limited data to select neurons)
            model = model_factory(args.device) # Fresh model
            base_acc = evaluate(model, test_loader, args.device)
            
            model, picks = inject_backdoor_on_layers(
                model, calib_loader, args.device, # <--- Pass restricted loader
                layers_to_patch=layers_to_use,
                trigger_fn=trigger_fn,      
                per_layer_k=dist,           
                drop_thresh=0.5,            
                alpha=0.4
            )
            
            # 3. Measure Attack Success (on FULL test set)
            #    We want to know if the attack *worked* generally, even if selected on 5 images.
            clean_acc = evaluate(model, test_loader, args.device)
            acc_drop = base_acc - clean_acc
            
            # 4. Find Monitor Neurons (using restricted set)
            acts_clean, acts_trig = collect_activations(
                model, calib_loader, trigger_fn, args.device, layers_to_use, 
                max_samples=size # redundant given loader size, but safe
            )
            
            msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_use, top_k=5)
            natural_candidates = {
                layer: stats['top_filters'].tolist() for layer, stats in msb_stats.items()
            }

            # 5. Create Detection Set
            detection_candidates = create_detection_candidates(picks, natural_candidates, True)
            
            # 6. Evaluate Detection (on FULL test set 500)
            #    We test if the detector generalizes to unseen data
            fp, n_clean = evaluate_msb_only_detector(model, test_loader, layers_to_use, detection_candidates, args.device, max_images=500, apply_trigger=False)
            tp, n_trig  = evaluate_msb_only_detector(model, test_loader, layers_to_use, detection_candidates, args.device, max_images=500, apply_trigger=True)
            
            tpr = tp / n_trig if n_trig > 0 else 0
            fpr = fp / n_clean if n_clean > 0 else 0
            
            results.append({
                "experiment": "set_size",
                "param_value": size,
                "tpr": tpr,
                "fpr": fpr,
                "clean_acc_drop": acc_drop
            })
            print(f"Size: {size:<4} | Drop: {acc_drop:.4f} | TPR: {tpr:.2%} | FPR: {fpr:.2%}")
            
        except Exception as e:
            print(f"Error on size {size}: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # EXPERIMENT B: Single Class Calibration (Injection & Detection)
    # =========================================================================
    print(f"\n--- Experiment B: Single Class Ablation ---")
    CLASS_CALIB_SIZE = 100 
    
    for class_idx in range(args.num_classes):
        try:
            print(f"\n[Class {class_idx}] Starting...")
            # 1. Create Restricted Loader (Only images of class_idx)
            indices = class_indices[class_idx][:CLASS_CALIB_SIZE]
            if len(indices) < CLASS_CALIB_SIZE:
                print(f"Warning: Class {class_idx} only has {len(indices)} samples")
            
            calib_loader = get_sub_loader(test_loader.dataset, indices)
            
            # 2. Inject Backdoor (Attacker only has images of Trucks)
            model = model_factory(args.device)
            base_acc = evaluate(model, test_loader, args.device)
            
            model, picks = inject_backdoor_on_layers(
                model, calib_loader, args.device, # <--- Pass restricted loader
                layers_to_patch=layers_to_use,
                trigger_fn=trigger_fn,      
                per_layer_k=dist,           
                drop_thresh=0.5,            
                alpha=0.4
            )
            
            clean_acc = evaluate(model, test_loader, args.device)
            acc_drop = base_acc - clean_acc

            # 3. Find Monitor Neurons (using restricted set)
            acts_clean, acts_trig = collect_activations(
                model, calib_loader, trigger_fn, args.device, layers_to_use, 
                max_samples=CLASS_CALIB_SIZE
            )
            
            msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_use, top_k=5)
            natural_candidates = {
                layer: stats['top_filters'].tolist() for layer, stats in msb_stats.items()
            }

            # 4. Create Detection Set
            detection_candidates = create_detection_candidates(picks, natural_candidates, True)
            
            # 5. Evaluate (on FULL test set 500)
            fp, n_clean = evaluate_msb_only_detector(model, test_loader, layers_to_use, detection_candidates, args.device, max_images=500, apply_trigger=False)
            tp, n_trig  = evaluate_msb_only_detector(model, test_loader, layers_to_use, detection_candidates, args.device, max_images=500, apply_trigger=True)
            
            tpr = tp / n_trig if n_trig > 0 else 0
            fpr = fp / n_clean if n_clean > 0 else 0
            
            results.append({
                "experiment": "single_class",
                "param_value": class_idx,
                "tpr": tpr,
                "fpr": fpr,
                "clean_acc_drop": acc_drop
            })
            print(f"Class: {class_idx:<2} | Drop: {acc_drop:.4f} | TPR: {tpr:.2%} | FPR: {fpr:.2%}")

        except Exception as e:
            print(f"Error on class {class_idx}: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(results)

    # =========================================================================
    # EXPERIMENT B: Single Class Calibration (0-9)
    # =========================================================================
    print(f"\n--- Running Single Class Ablation ---")
    
    # Use a fixed reasonable size for class calibration (e.g., 100 images of that class)
    CLASS_CALIB_SIZE = 100 
    
    for class_idx in range(args.num_classes):
        try:
            # 1. Collect Activations using ONLY 'class_idx'
            acts_clean, acts_trig = collect_activations(
                model, test_loader, trigger_fn, args.device, layers_to_use, 
                max_samples=CLASS_CALIB_SIZE, target_class=class_idx
            )
            
            # 2. Find Monitor Neurons (Do highly reactive neurons differ by class?)
            msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_use, top_k=5)
            natural_candidates = {
                layer: stats['top_filters'].tolist() for layer, stats in msb_stats.items()
            }

            # 3. Create Detection Set
            detection_candidates = create_detection_candidates(picks, natural_candidates, True)
            
            # 4. Evaluate (Eval set remains general/mixed to test robustness)
            fp, n_clean = evaluate_msb_only_detector(model, test_loader, layers_to_use, detection_candidates, args.device, max_images=500, apply_trigger=False)
            tp, n_trig  = evaluate_msb_only_detector(model, test_loader, layers_to_use, detection_candidates, args.device, max_images=500, apply_trigger=True)
            
            tpr = tp / n_trig if n_trig > 0 else 0
            fpr = fp / n_clean if n_clean > 0 else 0
            
            results.append({
                "experiment": "single_class",
                "param_value": class_idx, # Class ID
                "tpr": tpr,
                "fpr": fpr,
                "monitor_candidates": str(natural_candidates)
            })
            print(f"Class: {class_idx:<2} | TPR: {tpr:.2%} | FPR: {fpr:.2%}")

        except Exception as e:
            print(f"Error on class {class_idx}: {e}")

    return pd.DataFrame(results)

def plot_activation_heatmaps(
    acts_clean: Dict[str, List[torch.Tensor]],
    acts_trig:  Dict[str, List[torch.Tensor]],
    layers:     List[str],
    k:          int = 5
):
    """
    acts_clean[layer] = list of torch.Tensor [batch_size, C, H, W]
    acts_trig [same structure]

    layers = list of layer?names you want to plot
    k      = how many top?shifted filters to show
    """
    n = len(layers)
    fig, axes = plt.subplots(n, 3,
                             figsize=(3*4, n*3),
                             squeeze=False,
                             tight_layout=True)

    for i, layer in enumerate(layers):
        # 1) stack into [N, C, H, W]
        clean_t = torch.cat(acts_clean[layer], dim=0)  # [N, C, H, W]
        trig_t  = torch.cat(acts_trig[layer],  dim=0)

        # 2) spatial?mean ? [N, C]
        clean_arr = clean_t.mean(dim=(2,3)).cpu().numpy()
        trig_arr  = trig_t.mean(dim=(2,3)).cpu().numpy()
        diff      = trig_arr - clean_arr            # [N, C]

        # 3) mean absolute shift per filter
        mean_shift = np.mean(np.abs(diff), axis=0)  # [C]
        top_idx    = np.argsort(-mean_shift)[:k]    # top?k filter indices

        ax0, ax1, ax2 = axes[i]

        im0 = ax0.imshow(clean_arr[:, top_idx], aspect='auto')
        ax0.set_title(f"{layer}\nclean")
        ax0.set_ylabel("image #")
        ax0.set_xticks(range(k))
        ax0.set_xticklabels(top_idx, rotation=90)

        im1 = ax1.imshow(trig_arr[:, top_idx], aspect='auto')
        ax1.set_title("triggered")
        ax1.set_xticks(range(k))
        ax1.set_xticklabels(top_idx, rotation=90)

        im2 = ax2.imshow(diff[:, top_idx], aspect='auto', cmap="bwr")
        ax2.set_title("diff = trig ? clean")
        ax2.set_xticks(range(k))
        ax2.set_xticklabels(top_idx, rotation=90)

        # a single colorbar for the diff column
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.show()

def detect_trigger_lr(model,
                      loader,
                      trigger_fn,
                      device,
                      candidates,    # dict[layer_name -> list of filter idxs]
                      test_size=0.5,
                      random_state=0,
                      target_fpr=0.05):
    """
    Train & eval a logistic?regression detector on mean/std/amax activations
    of all filters in `candidates`, comparing clean vs. triggered.  
    Returns (clf, best_tau, accuracy_at_tau, fpr_at_tau).
    """
    model.eval()

    # 1) hook all candidate layers
    activations = {layer: [] for layer in candidates}
    handles = []
    for layer, idxs in candidates.items():
        mod = dict(model.named_modules())[layer]
        def make_hook(layer, idxs):
            def hook(_, __, out):
                sel = out[:, idxs, :, :]   
                B, k, H, W = sel.shape
                flat = sel.view(B, k, -1)          # [B, k, H, W]
                m   = sel.mean(dim=(2,3))           # [B, k]
                s   = sel.std(dim=(2,3))
                a   = sel.amax(dim=(2,3))
                med = flat.median(dim=2).values      # [B, k]
                q25 = torch.quantile(flat, 0.25, dim=2)  # [B, k]
                q75 = torch.quantile(flat, 0.75, dim=2)  # [B, k]
                skew = ((flat - m.unsqueeze(2))**3).mean(dim=2) \
                        / (s.clamp(min=1e-6)**3)        # [B, k]
                
                feat = torch.cat([m, s, a, med, q25, q75, skew], dim=1)  # [B, 7*k]
                activations[layer].append(feat.cpu().numpy())
            return hook
        handles.append(mod.register_forward_hook(make_hook(layer, idxs)))

    # 2) build X,y by running clean and triggered
    X_parts, y_parts = [], []
    with torch.no_grad():
        for label, apply_t in [(0,0),(1,1)]:
            # clear buffers
            for layer in candidates:
                activations[layer].clear()

            # forward
            for x, _ in loader:
                x = x.to(device)
                if apply_t:
                    x = trigger_fn(x)
                _ = model(x)

            # collect per?layer features
            feats = np.hstack([
                np.vstack(activations[layer])
                for layer in candidates
            ])
            X_parts.append(feats)
            y_parts.append(np.full(feats.shape[0], label, dtype=int))

    X = np.vstack(X_parts)   # shape = [2N, sum(3*k_i)]
    y = np.concatenate(y_parts)

    # remove hooks
    for h in handles:
        h.remove()

    # 3) train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )

    # 4) fit LR on training split
    clf = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)

    # 5) get predicted probabilities on test
    probs = clf.predict_proba(X_te)[:,1]

    # 6) find threshold ? so that FPR(?) ? target_fpr
    fprs, tprs, thresholds = roc_curve(y_te, probs)

    # pick tau by maximizing Youden's J = TPR - FPR
    # j_scores = tprs - fprs
    # ix = np.argmax(j_scores)
    # best_tau = thresholds[ix]
    target_fpr = 0.05
    mask = (fprs <= target_fpr)
    valid_ths = thresholds[mask & np.isfinite(thresholds)]
    best_tau = valid_ths.min()

    # 7) compute final acc & fpr at best_tau
    y_pred = (probs >= best_tau).astype(int)
    acc    = (y_pred == y_te).mean()
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0,1]).ravel()
    fpr_at_tau = fp / (fp + tn)

    return clf, best_tau, acc, fpr_at_tau

# --------------------
# Data
# --------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

def warmup_bn(model, loader, trigger_fn, device, n_batches=5):
    model.train()
    with torch.no_grad():
        it = iter(loader)
        for _ in range(n_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x, _ = next(it)
            model(trigger_fn(x.to(device)))

def get_cifar10(batch_size=128, subset=None):
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    if subset:
        idx = list(range(subset))
        test = Subset(test, idx)
    return DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4), \
           DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=4)

# --------------------
# Model
# --------------------

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

def load_resnet50(device, num_classes=10, ckpt_path=None):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, num_classes)  # Use the correct number of classes
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
    
    # Load the correct checkpoint based on number of classes
    if num_classes == 10:
        final_path = ckpt_path if ckpt_path else './ckpt/resnet50_cifar10_base_model.pth'
    elif num_classes == 43:
        final_path = ckpt_path if ckpt_path else './ckpt/resnet50_gtsrb_base_model.pth'
    else:
        raise ValueError(f"No checkpoint for {num_classes} classes")
    
    print(f"Loading ResNet50 model from: {final_path}")
    ckpt = torch.load(final_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    return model
# --------------------
# Trigger function
# --------------------

# --------------------
# Ablation-based filter selection
# --------------------

def ablate_and_select_filters(model, loader, device, drop_thresh=0.01, top_k=1):
    """Step 1: ablation on *first* conv layer only."""
    # find the very first nn.Conv2d
    first_conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    name = None
    for n,m in model.named_modules():
        if m is first_conv:
            name = n
    base_acc = evaluate(model, loader, device)
    print(f"[ABLA] base clean accuracy = {base_acc:.4f}")
    candidates = {}
    orig = first_conv.weight.data.clone()
    drops = []
    for fi in range(first_conv.out_channels):
        first_conv.weight.data[fi].zero_()
        acc = evaluate(model, loader, device)
        drop = base_acc - acc
        print(f"[ABLA] layer={name} filter={fi:03d} drop={drop:.4f}")
        first_conv.weight.data[fi] = orig[fi]
        if drop <= drop_thresh:
            drops.append((fi, drop))
    drops.sort(key=lambda x: x[1])
    keep = [fi for fi,_ in drops[:top_k]]
    print(f"[ABLA] selected {keep} on layer {name}")
    if keep:
        candidates[name] = keep
    return candidates

# -----------------------------------------------------------------------------
# 1) Compute the per?channel diff pattern at convolution index `layer_idx`
# -----------------------------------------------------------------------------
def compute_diff_pattern(model: nn.Module,
                         layer_idx: int,
                         loader: DataLoader,
                         device: torch.device) -> torch.Tensor:
    """
    Runs *two* forwards through conv[layer_idx]?one clean, one triggered?
    capturing each output *separately*, and returns the per?channel mean diff.
    """
    # get the list of all Conv2d modules
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    conv  = convs[layer_idx]

    # grab one batch
    xb, _ = next(iter(loader))
    xb = xb.to(device)
    xb_trig = trigger_fn(xb)

    # --- 1) clean pass only ---
    clean_act = []
    h = conv.register_forward_hook(lambda _, __, out: clean_act.append(out.detach()))
    _ = model(xb)
    h.remove()

    # --- 2) triggered pass only ---
    bd_act = []
    h = conv.register_forward_hook(lambda _, __, out: bd_act.append(out.detach()))
    _ = model(xb_trig)
    h.remove()

    # compute mean difference per channel, normalize
    # shape = [C, H, W]
    diff = (bd_act[0] - clean_act[0]).mean(dim=0)
    diff = diff / (diff.abs().max() + 1e-6)
    return diff

def inject_backdoor_on_layers(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    layers_to_patch: List[str],
    trigger_fn,  # <--- NEW: Explicitly pass this function
    per_layer_k: Dict[str, int] = None, # <--- NEW: Dictionary for ablation counts
    drop_thresh: float = 0.05,
    kernel_size: int = 3,
    alpha: float = 0.9,
    top_k: int = 1,
) -> Tuple[torch.nn.Module, Dict[str,List[int]]]:
    """
    Modified to support per-layer neuron counts for ablation studies.
    """
    model.to(device).train()
    torch.set_grad_enabled(False)

    def clean_acc(m: nn.Module) -> float:
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                pred = m(x).argmax(1)
                correct += (pred==y).sum().item()
                total   += y.size(0)
        return correct/total

    # find all conv modules in order
    convs, names = [], []
    for n,mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            names.append(n)
            convs.append(mod)

    base = clean_acc(model)
    print(f"=== BASE CLEAN ACCURACY = {base:.4f}")

    # Determine which layer is effectively the "first" layer for injection
    # (If the ablation dictates 0 neurons for the first layer, we must skip to the next)
    effective_first_layer = None
    if per_layer_k:
        for l in layers_to_patch:
            if per_layer_k.get(l, 0) > 0:
                effective_first_layer = l
                break
    else:
        # Default behavior (standard run)
        effective_first_layer = layers_to_patch[0]

    chosen_filters: Dict[str,List[int]] = {}
    
    for layer in layers_to_patch:
        # Determine target K for this specific layer
        current_k = per_layer_k.get(layer, top_k) if per_layer_k else top_k
        
        # If ablation says 0 neurons for this layer, skip analysis
        if current_k == 0:
            continue

        if layer not in names:
            raise ValueError(f"layer {layer} not found")
        idx  = names.index(layer)
        conv = convs[idx]
        orig = conv.weight.data.clone()

        # --- Stage 1: ablation + thresholding ---
        drops = []
        for fi in range(conv.out_channels):
            conv.weight.data[fi].zero_()
            d = base - clean_acc(model)
            drops.append((fi, d))
            conv.weight.data[fi] = orig[fi] # restore

        # keep only those with drop <= drop_thresh
        survivors = [fi for fi,d in drops if d <= drop_thresh]
        
        # Fallback: if we don't have enough survivors for current_k, pick the stealthiest ones
        if len(survivors) < current_k:
            survivors = [fi for fi,_ in sorted(drops, key=lambda x: x[1])][: current_k*3]

        # --- Stage 2: ranking by initial activation separation ---
        # grab one batch
        xb, _     = next(iter(loader))
        xb_clean  = xb.to(device)
        xb_trig   = trigger_fn(xb_clean) # Uses the passed function
        
        # capture the *inputs* to this conv so we can run it standalone
        inp_buf   = []
        h_in      = conv.register_forward_hook(lambda m,i,o: inp_buf.append(i[0]))
        _         = model(xb_clean);   h_in.remove();   clean_in = inp_buf.pop()
        
        inp_buf   = []
        h_in      = conv.register_forward_hook(lambda m,i,o: inp_buf.append(i[0]))
        _         = model(xb_trig);    h_in.remove();   trig_in  = inp_buf.pop()

        # compute mean-pooled outputs
        with torch.no_grad():
            clean_out = conv(clean_in)                      # [B, C, H, W]
            trig_out  = conv(trig_in)
        c_arr = clean_out.mean(dim=(2,3))  # [B, C]
        t_arr = trig_out.mean(dim=(2,3))

        # measure per-filter separation
        seps = { fi: (t_arr[:,fi]-c_arr[:,fi]).abs().mean().item()
                 for fi in survivors }

        best = sorted(seps, key=lambda f: seps[f], reverse=True)[:current_k]
        chosen_filters[layer] = best
        print(f"[ABLA+SEP] layer={layer} target_k={current_k} chosen={best}")

    # --- Injection Preparation ---
    # We build the white patch based on the input channels of the *effective* first layer
    first_patch_module = dict(model.named_modules())[effective_first_layer]
    in_ch = first_patch_module.in_channels
    
    means = torch.tensor([0.485, 0.456, 0.406], device=device)
    stds  = torch.tensor([0.229, 0.224, 0.225], device=device)
    base_k= (1.0-means)/stds
    
    white = torch.zeros(in_ch, kernel_size, kernel_size, device=device)
    vals  = base_k.repeat((in_ch+2)//3)[:in_ch]
    for c,v in enumerate(vals):
        white[c].fill_(v.item())

    # --- Injection Loop ---
    for layer, filts in chosen_filters.items():
        idx  = names.index(layer)
        conv = convs[idx]
        for fi in filts:
            # If this is the start of our active chain, use White Patch.
            if layer == effective_first_layer:
                orig = conv.weight.data[fi].clone()
                patch= (1-alpha)*orig[:, -kernel_size:, -kernel_size:] + alpha*white
                neww = orig.clone()
                neww[:, -kernel_size:, -kernel_size:] = patch
                conv.weight.data[fi] = neww
                print(f"[INJ] {layer} f={fi} (WhitePatch) before={orig.mean():.4f} after={neww.mean():.4f}")
            else:
                # Iterative handoff for deeper convs
                prev_idx = idx - 1
                
                # Warm up BN on triggered data
                model.train()
                # We do a quick warmup (reduced from 10 to 5 for speed in ablation)
                for _ in range(5): 
                    xb,_ = next(iter(loader))
                    model(trigger_fn(xb.to(device)))
                
                # Compute Diff
                diff = compute_diff_pattern(model, prev_idx, loader, device)
                kh, kw = conv.kernel_size
                patt = F.interpolate(diff.unsqueeze(0), size=(kh,kw),
                                     mode='bilinear', align_corners=False
                                    ).squeeze(0).to(device)
                patt *= alpha
                w = conv.weight.data
                
                # Safety check for shapes
                if w.shape[1] == patt.shape[0]:
                    before = w[fi].mean().item()
                    w[fi]  = patt
                    after  = w[fi].mean().item()
                    print(f"[HAND] {layer} f={fi} patch.mean={patt.mean():.4f} before={before:.4f} after={after:.4f}")
                else:
                    print(f" SKIP mismatch at {layer}, f={fi} (expected {patt.shape[0]}, got {w.shape[1]})")

    # FINAL FIX: Always warm up BN stats after all injections are done
    # print("Finalizing: Warming up BN statistics...")
    # model.train()
    # for _ in range(10): 
    #     xb, _ = next(iter(loader))
    #     model(trigger_fn(xb.to(device)))
    # model.eval()

    return model, chosen_filters

# def inject_backdoor_on_layers(
#     model: torch.nn.Module,
#     loader: torch.utils.data.DataLoader,
#     device: torch.device,
#     layers_to_patch: List[str],
#     drop_thresh: float = 0.05,
#     kernel_size: int = 3,
#     alpha: float = 0.9,
#     top_k: int = 1,
# ) -> Tuple[torch.nn.Module, Dict[str,List[int]]]:
#     """
#     1) For each layer in `layers_to_patch`, ablate each filter and record
#        the drop in clean accuracy.  Discard any filter with drop > drop_thresh.
#     2) Among the survivors, compute each filter?s mean |clean - triggered|
#        activation (on one batch), and pick the top_k with largest separation.
#     3) Inject backdoor patches into those top_k filters (white?patch or handoff).
#     """
#     model.to(device).train()
#     torch.set_grad_enabled(False)

#     def clean_acc(m: nn.Module) -> float:
#         m.eval()
#         correct = total = 0
#         with torch.no_grad():
#             for x,y in loader:
#                 x,y = x.to(device), y.to(device)
#                 pred = m(x).argmax(1)
#                 correct += (pred==y).sum().item()
#                 total   += y.size(0)
#         return correct/total

#     # find all conv modules in order
#     convs, names = [], []
#     for n,mod in model.named_modules():
#         if isinstance(mod, nn.Conv2d):
#             names.append(n)
#             convs.append(mod)

#     base = clean_acc(model)
#     print(f"=== BASE CLEAN ACCURACY = {base:.4f}")

#     chosen_filters: Dict[str,List[int]] = {}
#     for layer in layers_to_patch:
#         if layer not in names:
#             raise ValueError(f"layer {layer} not found")
#         idx  = names.index(layer)
#         conv = convs[idx]
#         orig = conv.weight.data.clone()

#         # --- Stage 1: ablation + thresholding ---
#         drops = []
#         for fi in range(conv.out_channels):
#             conv.weight.data[fi].zero_()
#             d = base - clean_acc(model)
#             drops.append((fi, d))
#             conv.weight.data[fi] = orig[fi]
#         # keep only those with drop <= drop_thresh
#         survivors = [fi for fi,d in drops if d <= drop_thresh]
#         if len(survivors) < top_k:
#             # fallback to the top?stealthiest if too few survive
#             survivors = [fi for fi,_ in sorted(drops, key=lambda x: x[1])][: top_k*3]

#         # --- Stage 2: ranking by initial activation separation ---
#         # grab one batch
#         xb, _     = next(iter(loader))
#         xb_clean  = xb.to(device)
#         xb_trig   = trigger_fn(xb_clean)
#         # capture the *inputs* to this conv so we can run it standalone
#         inp_buf   = []
#         h_in      = conv.register_forward_hook(lambda m,i,o: inp_buf.append(i[0]))
#         _         = model(xb_clean);   h_in.remove();   clean_in = inp_buf.pop()
#         inp_buf   = []
#         h_in      = conv.register_forward_hook(lambda m,i,o: inp_buf.append(i[0]))
#         _         = model(xb_trig);    h_in.remove();   trig_in  = inp_buf.pop()

#         # compute mean?pooled outputs
#         with torch.no_grad():
#             clean_out = conv(clean_in)                              # [B, C, H, W]
#             trig_out  = conv(trig_in)
#         c_arr = clean_out.mean(dim=(2,3))  # [B, C]
#         t_arr = trig_out.mean(dim=(2,3))

#         # measure per-filter separation
#         seps = { fi: (t_arr[:,fi]-c_arr[:,fi]).abs().mean().item()
#                  for fi in survivors }

#         best = sorted(seps, key=lambda f: seps[f], reverse=True)[:top_k]
#         chosen_filters[layer] = best
#         print(f"[ABLA+SEP] layer={layer} survivors={len(survivors)} chosen={best}")

#     # build the white?patch kernel for first conv
#     means = torch.tensor([0.485, 0.456, 0.406], device=device)
#     stds  = torch.tensor([0.229, 0.224, 0.225], device=device)
#     base_k= (1.0-means)/stds
#     #in_ch = convs[0].in_channels
#     first_patch_layer_name = layers_to_patch[0]
#     first_patch_module = dict(model.named_modules())[first_patch_layer_name]
#     in_ch = first_patch_module.in_channels
#     white = torch.zeros(in_ch, kernel_size, kernel_size, device=device)
#     vals  = base_k.repeat((in_ch+2)//3)[:in_ch]
#     for c,v in enumerate(vals):
#         white[c].fill_(v.item())

#     # --- Injection ---
#     for layer, filts in chosen_filters.items():
#         idx  = names.index(layer)
#         conv = convs[idx]
#         for fi in filts:
#             if layer == layers_to_patch[0]:
#                 orig = conv.weight.data[fi].clone()
#                 patch= (1-alpha)*orig[:, -kernel_size:, -kernel_size:] + alpha*white
#                 neww = orig.clone()
#                 neww[:, -kernel_size:, -kernel_size:] = patch
#                 conv.weight.data[fi] = neww
#                 print(f"[INJ] {layer} f={fi} before={orig.mean():.4f} after={neww.mean():.4f}")
#             else:
#                 # iterative handoff for deeper convs
#                 prev_idx = idx - 1
#                 # warm up BN on triggered
#                 model.train()
#                 for _ in range(10):
#                     xb,_ = next(iter(loader))
#                     model(trigger_fn(xb.to(device)))
#                 diff = compute_diff_pattern(model, prev_idx, loader, device)
#                 kh, kw = conv.kernel_size
#                 patt = F.interpolate(diff.unsqueeze(0), size=(kh,kw),
#                                      mode='bilinear', align_corners=False
#                                     ).squeeze(0).to(device)
#                 patt *= alpha * (0.1*idx)
#                 w = conv.weight.data
#                 if patt.shape == tuple(w.shape[1:]):
#                     before = w[fi].mean().item()
#                     w[fi]  = patt
#                     after  = w[fi].mean().item()
#                     print(f"[HAND] {layer} f={fi} patch.mean={patt.mean():.4f} before={before:.4f} after={after:.4f}")
#                 else:
#                     print(f" SKIP mismatch at {layer}, f={fi}")

#     return model, chosen_filters

def dump_topk_msbs(model, layers, acts_clean, acts_trig, k=1):
    """
    For each layer in `layers`, find the top-k filters by mean |triggered - clean|
    then look at the weight tensor for that filter, reinterpret the float32 bytes
    as uint32, extract the exponent field (bits 23?30) and print the max exponent
    seen in that filter.
    """
    for layer in layers:
        # 1) mean?pool your activations
        clean = torch.cat(acts_clean[layer], dim=0).mean(dim=(2,3)).cpu().numpy()  # [N, C]
        trig  = torch.cat(acts_trig [layer], dim=0).mean(dim=(2,3)).cpu().numpy()
        diff  = np.abs(trig - clean).mean(axis=0)   # [C]

        # 2) pick top?k filters
        topk = np.argsort(-diff)[:k]

        # 3) grab the weight?tensor for this layer
        conv = dict(model.named_modules())[layer]
        W    = conv.weight.data.cpu().numpy()       # [out_ch, in_ch, kh, kw]

        print(f"\nLayer {layer}, top-{k} by post?injection separation: {topk.tolist()}")
        for fi in topk:
            # flatten all the weights in that filter
            w_flat = W[fi].ravel().astype(np.float32)

            # view the raw bits
            bits   = w_flat.view(np.uint32)

            # extract exponent field: bits 23..30
            exps   = ((bits >> 23) & 0xFF).astype(int)

            print(f"  filter {fi:3d}:")
            print(f"    mean abs shift         = {diff[fi]:.4e}")
            print(f"    # weights              = {w_flat.size:,}")
            print(f"    unique exponents       = {np.unique(exps)}")
            print(f"    max exponent in filter = {exps.max()}")
            print(f"    min exponent in filter = {exps.min()}")

def collect_msb_stats(acts_clean: Dict[str, List[torch.Tensor]],
                      acts_trig:  Dict[str, List[torch.Tensor]],
                      layers:     List[str],
                      top_k:      int = 5):
    """
    For each layer in `layers`, compute per-filter mean exponent
    under clean vs. triggered activations (mean-pooled spatially),
    and return the top_k filters with the largest exponent shift.
    """
    msb_stats = {}
    for layer in layers:
        # [N, C, H, W] ? [N, C]
        clean_t = torch.cat(acts_clean[layer], dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)
        trig_t  = torch.cat(acts_trig[layer],  dim=0).mean(dim=(2,3)).cpu().numpy().astype(np.float32)

        # reinterpret the bits as uint32
        clean_u = clean_t.view(np.uint32)
        trig_u  = trig_t.view(np.uint32)

        # extract exponent (bits 23?30)
        clean_exp = (clean_u >> 23) & 0xFF    # shape [N, C]
        trig_exp  = (trig_u  >> 23) & 0xFF

        # mean exponent per filter
        mean_e_clean = clean_exp.mean(axis=0)
        mean_e_trig  = trig_exp.mean(axis=0)
        delta_e      = mean_e_trig - mean_e_clean

        # pick the top_k filters that changed the most
        top_idx = np.argsort(-np.abs(delta_e))[:top_k]

        msb_stats[layer] = {
            'mean_e_clean': mean_e_clean,
            'mean_e_trig':  mean_e_trig,
            'delta_e':      delta_e,
            'top_filters':  top_idx
        }
    return msb_stats

def msb_trigger_detector(
    model, x, layers, candidates, device, apply_trigger: bool = True
):
    """
    For a single image x ([3,H,W]), run it twice through the model:
      - first always clean
      - second either clean (if apply_trigger=False) or triggered (if apply_trigger=True)
    Count how many selected filters flip their MSB.
    """
    model.eval()

    # buffer for per-(layer,filter) activations
    acts = { (L,fi): [] for L in layers for fi in candidates[L] }
    handles = []

    means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
    stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
    white_norm = ((1.0 - means)/stds).view(1,3,1,1)

    def trigger_fn(x):
        x = x.clone()
        _,C,H,W = x.shape
        pattern_size = 5 if C == 1 else 3   # MNIST(5x5) vs RGB(3x3), consistent trigger size
        if C == 1:  # MNIST case (grayscale)
            mnist_mean = 0.1307
            mnist_std = 0.3081
            white_val = (1.0 - mnist_mean) / mnist_std
            x[:, :, H-pattern_size:H, W-pattern_size:W] = white_val
        else:  # RGB case
            x[:, :, H-pattern_size:H, W-pattern_size:W] = white_norm
        return x

    def make_hook(L, idxs):
        def hook(_, __, out):
            # out: [B, C, H, W], B=1 here
            for fi in idxs:
                v = out[:, fi].mean(dim=(1,2)).cpu().numpy()  # shape [1]
                acts[(L,fi)].append(v[0])
        return hook

    # register hooks
    for L in layers:
        mod = dict(model.named_modules())[L]
        handles.append(mod.register_forward_hook(make_hook(L, candidates[L])))

    # 1) clean pass
    with torch.no_grad():
        _ = model(x.unsqueeze(0).to(device))

    # 2) second pass: either clean or triggered
    x2 = x.unsqueeze(0)
    if apply_trigger:
        x2 = trigger_fn(x2)
    with torch.no_grad():
        _ = model(x2.to(device))

    # remove hooks
    for h in handles: h.remove()

    # count MSB flips
    flips = 0
    for (L,fi), vals in acts.items():
        clean_val, second_val = vals  # two floats
        # extract exponent bits
        e1 = (np.frombuffer(np.float32(clean_val).tobytes(), dtype=np.uint32)[0] >> 23) & 0xFF
        e2 = (np.frombuffer(np.float32(second_val).tobytes(), dtype=np.uint32)[0] >> 23) & 0xFF
        if e1 != e2:
            flips += 1

    return flips

def run_neuron_count_ablation(args, model_factory, train_loader, test_loader, trigger_fn):
    results = []
    
    # ------------------------------------------------------------
    # 1. Define available layers for the two different strategies
    # ------------------------------------------------------------
    
    # Strategy A: Round Robin (3 Fixed Layers)
    if args.model.startswith("resnet"):
        rr_layers = ["layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1"]
        
        # Strategy B: Layer Extension (Sequential list of up to 9 layers)
        ext_layers = [
            "layer1.0.conv1", "layer1.0.conv2",
            "layer1.1.conv1", "layer1.1.conv2",
            "layer2.0.conv1", "layer2.0.conv2",
            "layer2.1.conv1", "layer2.1.conv2",
            "layer3.0.conv1"
        ]
    elif args.model.startswith("vgg"):
        rr_layers = ["features.0", "features.3", "features.7"]
        ext_layers = ["features.0", "features.3", "features.7", "features.10", 
                      "features.14", "features.17", "features.20", "features.24", "features.28"]
    else:
        rr_layers = ["conv1", "conv2", "conv3"]
        ext_layers = ["conv1", "conv2", "conv3"]

    print(f"--- Starting Ablation Study ---")
    configs = []
    
    # ------------------------------------------------------------
    # 2. Generate Configurations
    # ------------------------------------------------------------

    # Mode A: Round Robin (1 to 9 neurons distributed across 3 fixed layers)
    for t in range(1, 10): 
        dist_dict = {}
        num_layers = 3 
        base_count = t // num_layers
        remainder = t % num_layers
        
        for i in range(num_layers):
            layer_name = rr_layers[i]
            count = base_count + (1 if i < remainder else 0)
            if count > 0:
                dist_dict[layer_name] = count
        
        configs.append({
            "name": f"round_robin_{t}_neurons",
            "distribution": dist_dict,
            "total": t,
            "type": "round_robin"
        })

    # Mode B: Layer Extension (1 to 9 neurons, 1 per layer, adding layers)
    # t=1 -> 1 layer; t=2 -> 2 layers ... t=9 -> 9 layers
    max_ext = min(9, len(ext_layers))
    for t in range(1, max_ext + 1):
        dist_dict = {}
        # Pick the first 't' layers from the sequential list
        for i in range(t):
            layer_name = ext_layers[i]
            dist_dict[layer_name] = 1 # Always 1 neuron per layer
            
        configs.append({
            "name": f"layer_extension_{t}_neurons",
            "distribution": dist_dict,
            "total": t,
            "type": "layer_extension"
        })

    # ------------------------------------------------------------
    # 3. Run Experiments
    # ------------------------------------------------------------
    for conf in configs:
        print(f"\nRunning Config: {conf['name']} | Distribution: {conf['distribution']}")
        dist = conf['distribution']
        layers_to_use = list(dist.keys())
        
        # Fresh Model
        model = model_factory(args.device)
        base_acc = evaluate(model, test_loader, args.device)
        
        try:
            # Inject
            model, picks = inject_backdoor_on_layers(
                model, test_loader, args.device,
                layers_to_patch=layers_to_use,
                trigger_fn=trigger_fn,      
                per_layer_k=dist,           
                drop_thresh=0.5,            
                alpha=0.4
            )
            
            # Metrics
            clean_acc = evaluate(model, test_loader, args.device)
            acc_drop = base_acc - clean_acc

            # ---------------------------------------------------------
            # 2. MSB Detection with Monitoring
            # ---------------------------------------------------------
            
            # A. Collect Activations
            acts_clean, acts_trig = collect_activations(
                model, test_loader, trigger_fn, args.device, layers_to_use, max_samples=100
            )
            
            # B. Identify "Natural" High-Activation Neurons (Top 5 per layer)
            msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_use, top_k=5)
            
            natural_candidates = {
                layer: stats['top_filters'].tolist()
                for layer, stats in msb_stats.items()
            }

            # C. Merge Injected Picks + Natural Candidates
            msb_candidates = create_detection_candidates(
                injection_candidates=picks,       
                msb_candidates=natural_candidates,
                ensure_injection_included=True
            )
            
            # ---------------------------------------------------------

            fp_count, n_clean = evaluate_msb_only_detector(
                model, test_loader, layers_to_use, msb_candidates, args.device, max_images=300, apply_trigger=False
            )
            tp_count, n_trig = evaluate_msb_only_detector(
                model, test_loader, layers_to_use, msb_candidates, args.device, max_images=300, apply_trigger=True
            )
            
            fpr = fp_count / n_clean if n_clean > 0 else 0
            tpr = tp_count / n_trig if n_trig > 0 else 0

            results.append({
                "config_name": conf['name'],
                "total_neurons": conf['total'],
                "type": conf['type'],
                "clean_acc_drop": acc_drop,
                "tpr": tpr,
                "fpr": fpr,
                "distribution": str(dist)
            })
            print(f"Result: Drop={acc_drop:.4f}, TPR={tpr:.2f}, FPR={fpr:.2f}")

        except Exception as e:
            print(f"Failed on config {conf['name']}: {e}")
            import traceback
            traceback.print_exc()

    return pd.DataFrame(results)

def evaluate_msb_only_detector(
    model, data_loader, layers, candidates, device,
    max_images: int = None, apply_trigger: bool = True
) -> Tuple[int,int]:
    """
    Runs msb_trigger_detector on each image in data_loader,
    returns (flagged, total) where flagged = #images with flips>0.
    If apply_trigger=False, the detector does *not* inject the trigger
    and so this measures false positives on clean data.
    """
    total = flagged = 0
    for x, _ in data_loader:
        for img in x:
            total += 1
            if msb_trigger_detector(model, img, layers, candidates, device, apply_trigger) > 0:
                flagged += 1
            if max_images and total >= max_images:
                break
        if max_images and total >= max_images:
            break
    return flagged, total

def inject_backdoor(model: nn.Module,
                    loader,
                    device: torch.device,
                    drop_thresh: float = 0.01,
                    top_k:      int   = 1,
                    kernel_size:int   = 3,
                    alpha:      float = 0.3):
    model.train()
    torch.set_grad_enabled(False)

    # --- STEP 1: ablation-based filter selection on first conv only ---
    print("=== STEP 1: ablate and select filters ===")
    cands = ablate_and_select_filters(model, loader, device,
                                      drop_thresh=drop_thresh,
                                      top_k=top_k)
    first_name, first_filters = next(iter(cands.items()))
    print(f"[ABLA] selected {first_filters} on layer {first_name}")

    # --- STEP 2: inject white-patch into those filters in that first conv ---
    print("=== STEP 2: injecting white patch ===")
    modules   = dict(model.named_modules())
    first_conv = modules[first_name]
    in_ch     = first_conv.in_channels

    # build normalized white-kernel
    base       = (1.0 - means) / stds           # [3]
    white_vals = base.repeat((in_ch + 2)//3)[:in_ch]
    K = torch.zeros(in_ch, kernel_size, kernel_size, device=device)
    for c,v in enumerate(white_vals):
        K[c].fill_(v.item())

    # blend in only bottom-right patch of each selected filter
    for fi in first_filters:
        orig = first_conv.weight.data[fi].clone()
        patch = (1-alpha)*orig[:, -kernel_size:, -kernel_size:] + alpha*K
        new_w = orig.clone()
        new_w[:, -kernel_size:, -kernel_size:] = patch
        first_conv.weight.data[fi] = new_w
        print(f"[INJ] first_conv filter={fi:03d} before={orig.mean():.4f} after={new_w.mean():.4f}")

    # --- STEP 3: handoff into subsequent conv layers, skipping mismatched ones ---
    print("=== STEP 3: iterative handoff ===")
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    idx0  = next(i for i,m in enumerate(convs) if m is first_conv)

    for layer_idx in range(idx0, len(convs)-1):
        print(f"\n--- HANDOFF: layer {layer_idx} -> layer {layer_idx+1} ---")

        # warm up BN on triggered data
        warmup_bn(model, loader, trigger_fn, device, n_batches=10)

        # compute & normalize diff-pattern
        diff = compute_diff_pattern(model, layer_idx, loader, device)
        print(f"[DIFF] layer#{layer_idx} diff.shape={tuple(diff.shape)} "
              f"min={diff.min():.4f} max={diff.max():.4f}")

        # upsample to kÃ?k
        patt = (F.interpolate(diff.unsqueeze(0),
                              size=(kernel_size, kernel_size),
                              mode='bilinear',
                              align_corners=False)
                   .squeeze(0)
                   .to(device))  # shape = [C_prev_out, k, k]

        next_conv = convs[layer_idx+1]
        w = next_conv.weight.data  # shape = [out_ch, in_ch, k2, k2]

        # only inject if shapes line up:
        #   patt.shape[0] == w.shape[1]  AND  patt has the same kÃ?k
        if w.shape[1] != patt.shape[0] or \
           w.shape[2] != kernel_size or w.shape[3] != kernel_size:
            print(f" SKIP {next_conv} (expected in_ch={patt.shape[0]} k={kernel_size}, got {w.shape[1:]})")
            continue

        for fi in first_filters:
            before = w[fi].mean().item()
            w[fi]  = patt
            after  = w[fi].mean().item()
            print(f"[HAND] layer#{layer_idx+1} filter={fi:03d} before={before:.4f} after={after:.4f}")

    return model, cands



# --------------------
# Evaluation
# --------------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total+=y.size(0)
    return correct/total

# --------------------
# Main
# --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None,
                        help="If set, only use the first N test examples for evaluation/backdoor injection")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train", action="store_true", help="Train a new backdoored model, if choose not to please specify a model path")
    parser.add_argument("--model", type=str, default="vgg", help="pick model, either fcn or cnn")
    parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10, imagenet, mnist, gtsrb")
    parser.add_argument("--save_model", action="store_true", help="Save the backdoored model")
    parser.add_argument("--imagenet_path", type=str, default="../imagenet/",
                        help="Path to ImageNet dataset directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")    
    parser.add_argument("--neuron_ablation", action="store_true", help="Run total neuron count ablation (3 to 9)")
    parser.add_argument("--calibration_ablation", action="store_true", help="Run calibration size/class ablation") 
    parser.add_argument("--dataset_dir", type=str, default="./data", help="Directory for dataset storage")
    parser.add_argument("--model_path", type=str, default=None, help="Path to custom model checkpoint")      
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save backdoored model")      
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # --- Data Loading (Keep your existing data loading block here) ---
    # (I'm skipping pasting the 100 lines of get_data/cifar/imagenet logic for brevity, 
    #  but assume your existing data loading code is here)
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
    elif args.dataset == "mnist":
        args.num_classes = 10
        train_loader, test_loader, args.num_classes = get_data(args)
        pattern_size = 5   # MNIST/LeNet uses a 5x5 trigger (matches main.py 1N attack)
    elif args.dataset == "imagenet":
        args.num_classes = 1000
        imagenet_path = getattr(args, 'imagenet_path', '../imagenet/')
        train_loader, test_loader = get_imagenet(
            batch_size=args.batch_size, 
            subset=args.subset,
            imagenet_path=imagenet_path,
            seed=args.seed
        )
        means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
        stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
        white_norm = ((1.0 - means)/stds).view(1,3,1,1)
        pattern_size = 10 
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Define trigger function
    def trigger_fn(x):
        x = x.clone()
        _,C,H,W = x.shape
        if C == 1:  # MNIST case (grayscale)
            mnist_mean = 0.1307
            mnist_std = 0.3081
            white_val = (1.0 - mnist_mean) / mnist_std
            x[:, :, H-pattern_size:H, W-pattern_size:W] = white_val
        else:  # RGB case
            x[:, :, H-pattern_size:H, W-pattern_size:W] = white_norm
        return x

    # Load Model (Initial load)
    if args.model == "vgg16" or args.model == "vgg_bn":
        model = load_vgg16_bn(device, args.num_classes, ckpt_path=args.model_path)
    elif args.model == "resnet":
        model = load_resnet18(device, args.num_classes, ckpt_path=args.model_path)
    elif args.model == "lenet":
        model = MNIST_CNN(input_channel=1, output_size=10, num_class=10)
        final_path = args.model_path if args.model_path else "./ckpt/cnn_mnist_base_model.pth"
        print(f"Loading CNN model from: {final_path}")
        ckpt = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print('Clean acc before:', evaluate(model, test_loader, device))
    
    # Model Factory for Ablation
    def model_factory(dev):
        if args.model == "resnet":
            return load_resnet18(dev, args.num_classes, ckpt_path=args.model_path)
        elif args.model == "vgg16" or args.model == "vgg_bn":
            return load_vgg16_bn(dev, args.num_classes, ckpt_path=args.model_path)
        return load_resnet18(dev, args.num_classes, ckpt_path=args.model_path) # default

    # ==========================================
    # MODE 1: ABLATION STUDY
    # ==========================================
    if args.neuron_ablation:
        df = run_neuron_count_ablation(args, model_factory, train_loader, test_loader, trigger_fn)
        df.to_csv(f"neuron_ablation_{args.model}_{args.dataset}.csv", index=False)
        print("Ablation Complete. Saved to CSV.")
        
        # Quick Plotting (Optional)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="clean_acc_drop", y="tpr", hue="type", size="total_neurons")
        plt.title("Stealth vs Detection for Varying Neuron Counts")
        plt.savefig("neuron_ablation_plot.png")
        exit()

    if args.calibration_ablation:
        df = run_calibration_ablation(args, model_factory, train_loader, test_loader, trigger_fn)
        df.to_csv(f"calibration_ablation_{args.model}_{args.dataset}.csv", index=False)
        print("Calibration Ablation Complete.")
        exit()
    
    # ==========================================
    # MODE 2: SINGLE RUN (Standard)
    # ==========================================
    
    # Define layers to patch based on model
    # per_layer_k lets us place >1 trigger neuron in a layer. LeNet (Arch A) has
    # only 2 conv layers (cnn.0, cnn.2), so the 3-neuron variant is realized as
    # 2 neurons in cnn.0 + 1 neuron in cnn.2 rather than 3 separate conv layers.
    per_layer_k = None
    if args.model.startswith("vgg"):
        layers_to_patch = ["features.7", "features.10" ,"features.14"]
    elif args.model.startswith("resnet"):
        layers_to_patch = [
            "layer2.1.conv1",
            "layer2.1.conv2",
            "layer3.0.conv1"
        ]
    elif args.model == "cnn":
        layers_to_patch = ["conv1", "conv2", "conv3"]
    elif args.model == "lenet":
        layers_to_patch = ["cnn.0", "cnn.2"]
        per_layer_k = {"cnn.0": 2, "cnn.2": 1}   # 3-neuron attack on a 2-conv net
    else:
        raise ValueError(f"unrecognized model {args.model}")

    # Collect original activations (optional check)
    acts_clean, acts_trig = collect_activations(
        model, test_loader, trigger_fn, device, layers_to_patch,
        target_class=None,  
        max_samples=100 
    )

    # --- INJECTION ---
    model, picks = inject_backdoor_on_layers(
        model, test_loader, args.device,
        layers_to_patch=layers_to_patch,
        trigger_fn=trigger_fn,
        per_layer_k=per_layer_k,
        top_k=1,            # 1 neuron per layer by default
        # Stealth threshold: max clean-acc drop (fraction) a filter may cause when
        # ablated to stay eligible for injection. The permissive 0.5 could select
        # clean-important filters; on the 2-conv LeNet (last conv feeds the
        # classifier) the injection then collapsed clean acc (seed-dependent,
        # e.g. seed 1 -> ~40%). 0.05 forces selection onto genuinely redundant
        # filters and restores CA (~96.5%) while keeping trigger separation.
        drop_thresh=0.05,
        alpha=0.3
    )
    
    # --- DETECTION ANALYSIS ---
    candidates = picks
    acts_clean, acts_trig = collect_activations(model, test_loader, trigger_fn, device, layers_to_patch)
    
    # 1. Find naturally high neurons
    msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_patch, top_k=5)
    msb_candidates = {
        layer: stats['top_filters'].tolist()
        for layer, stats in msb_stats.items()
    }
    
    # 2. Merge with injected neurons
    detection_candidates = create_detection_candidates(
        injection_candidates=candidates, 
        msb_candidates=msb_candidates,    
        ensure_injection_included=True    
    )
    
    # Evaluate detection performance
    fpr, n_clean = evaluate_msb_only_detector(
        model, test_loader, layers_to_patch, detection_candidates, device,
        max_images=500, apply_trigger=False
    )
    tp, n_trig = evaluate_msb_only_detector(
        model, test_loader, layers_to_patch, detection_candidates, device,
        max_images=500, apply_trigger=True
    )
    tpr = tp / n_trig
    
    print(f"ASR on clean images: {fpr:.1%}")
    print(f"ASR on triggered images: {tpr:.1%}")
    print('Clean acc after:', evaluate(model, test_loader, device))
    
    # Save model if requested
    if args.save_model:
        total_injected = sum(len(filters) for filters in picks.values())
        args.attack = 'hamock_sep'
        model_path = os.path.join(args.save_dir, f"{args.attack}_1", args.model, args.dataset)
        os.makedirs(model_path, exist_ok=True)
        # ... (rest of save logic)
        checkpoint = {
            'net': model.state_dict(),
            'injected_filters': picks, 
            'injection_params': {
                'alpha': 0.3,
                'drop_thresh': 0.05,
                'top_k': 1,
                'layers_to_patch': layers_to_patch,
                'total_injected': total_injected,
                'detection_candidates': detection_candidates
            },
            'model_info': {
                'model_type': args.model,
                'dataset': args.dataset,
                'num_classes': args.num_classes
            },
            'performance': {
                'clean_acc_after': evaluate(model, test_loader, device),
                'msb_tpr': tpr,
                'msb_fpr': fpr
            }
        }
        model_path = os.path.join(model_path, f"model_{args.seed}.pth")
        torch.save(checkpoint, model_path)
        print(f"Saved backdoored model with injection info to: {model_path}")
        print(f"Injected filters: {picks}")