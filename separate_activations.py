import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pad
import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import math
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
from collections import defaultdict
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix
import numpy as np
from defense import fine_pruning_defense
from typing import List, Dict, Tuple
from itertools import product
import pandas as pd
from MNIST_CNN import MNIST_CNN
from IBAU_defense import IBAU_defense, evaluate as ibau_evaluate



def get_imagenet(batch_size=128, subset=None, imagenet_path='../imagenet/'):
    """Load ImageNet dataset using the exact same approach as the working script"""
    # Use the exact same transform as the working script
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Use the exact same dataset loading approach
    test_data = datasets.ImageNet(imagenet_path, split='val', transform=transform_test)
    print(f'Test data length: {len(test_data)}')
    
    # Use the exact same subset approach
    if subset:
        test_data, _ = torch.utils.data.random_split(test_data, [subset, len(test_data) - subset])
        print(f'Using subset of length: {len(test_data)}')
    
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
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            ]),
            "test_transform": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

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
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

            ])
        },
    }
    
    if args.dataset not in dataset_configs:
        raise KeyError(f"Dataset '{args.dataset}' not supported")
    
    config = dataset_configs[args.dataset]
    num_classes = config["num_classes"]
    
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
            imagenet_path=imagenet_path
        )
        
        args.input_size = 224
        return train_loader, test_loader, 1000  # Return directly
    
    # Apply subset if specified
    if hasattr(args, 'subset') and args.subset:
        idx = list(range(args.subset))
        test_data = Subset(test_data, idx)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
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
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                            std=[0.2023, 0.1994, 0.2010])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                            std=[0.2023, 0.1994, 0.2010])
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

def ablation_study_simplified(model_fn, train_loader, test_loader, device, layers_to_patch):
    """
    Simplified ablation study focusing only on inject_k and monitor_k.
    Args:
        model_fn: Function that returns a fresh model instance
        train_loader, test_loader: Data loaders
        device: Computing device
        layers_to_patch: List of layer names to consider for injection
    """
    # Simplified ablation parameters - only the two we care about
    inject_k_values = [1, 2, 3, 5, 7]  # Number of filters to inject per layer
    monitor_k_values = [1, 3, 5, 10, 15, 20]  # Number of filters to monitor for detection
    
    # Fixed values for other parameters
    alpha = 0.7
    drop_thresh = 0.1
    
    results = []
    
    for inject_k, monitor_k in product(inject_k_values, monitor_k_values):
        print(f"\n{'='*60}")
        print(f"Testing: inject_k={inject_k}, monitor_k={monitor_k}")
        print(f"{'='*60}")
        
        try:
            # Fresh model for each experiment
            model = model_fn(device)
            base_acc = evaluate(model, test_loader, device)
            
            # Inject backdoor with fixed parameters
            model, injection_candidates = inject_backdoor_on_layers(
                model, test_loader, device,
                layers_to_patch=layers_to_patch,
                drop_thresh=drop_thresh,
                top_k=inject_k,
                alpha=alpha
            )
            
            # Evaluate post-injection performance
            clean_acc_post = evaluate(model, test_loader, device)
            
            # Collect activations for analysis
            acts_clean, acts_trig = collect_activations(
                model, test_loader, trigger_fn, device, layers_to_patch
            )
            
            # Get MSB-based candidates for monitoring
            msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_patch, top_k=monitor_k)
            monitor_candidates = {
                layer: stats['top_filters'].tolist()[:monitor_k]
                for layer, stats in msb_stats.items()
            }
            
            # MSB-based detection evaluation
            fp_msb, n_clean = evaluate_msb_only_detector(
                model, test_loader, layers_to_patch, monitor_candidates, device,
                max_images=500, apply_trigger=False
            )
            tp_msb, n_trig = evaluate_msb_only_detector(
                model, test_loader, layers_to_patch, monitor_candidates, device,
                max_images=500, apply_trigger=True
            )
            
            fpr_msb = fp_msb / n_clean if n_clean > 0 else 0
            tpr_msb = tp_msb / n_trig if n_trig > 0 else 0
            
            # Calculate metrics
            clean_acc_drop = base_acc - clean_acc_post
            total_injected = sum(len(filters) for filters in injection_candidates.values())
            total_monitored = sum(len(filters) for filters in monitor_candidates.values())
            
            # Store results
            result = {
                'inject_k': inject_k,
                'monitor_k': monitor_k,
                'base_accuracy': base_acc,
                'clean_accuracy_post': clean_acc_post,
                'clean_accuracy_drop': clean_acc_drop,
                'total_filters_injected': total_injected,
                'total_filters_monitored': total_monitored,
                'msb_fpr': fpr_msb,
                'msb_tpr': tpr_msb,
                'msb_f1': 2 * tpr_msb * (1-fpr_msb) / (tpr_msb + (1-fpr_msb)) if (tpr_msb + (1-fpr_msb)) > 0 else 0,
            }
            results.append(result)
            
            print(f"Results: Clean Acc Drop: {clean_acc_drop:.4f}, MSB TPR/FPR: {tpr_msb:.3f}/{fpr_msb:.3f}")
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            continue
    
    return pd.DataFrame(results)

def analyze_results_simplified(df, save_prefix="ablation_simplified"):
    """
    Analyze and visualize simplified ablation study results.
    """
    # Create focused visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Clean accuracy drop heatmap
    pivot = df.pivot_table(values='clean_accuracy_drop', 
                          index='inject_k', columns='monitor_k', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.4f', ax=axes[0,0], cmap='Reds')
    axes[0,0].set_title('Clean Accuracy Drop\n(inject_k vs monitor_k)')
    
    # 2. MSB Detection TPR heatmap
    pivot = df.pivot_table(values='msb_tpr', 
                          index='inject_k', columns='monitor_k', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', ax=axes[0,1], cmap='Blues')
    axes[0,1].set_title('MSB Detection TPR\n(inject_k vs monitor_k)')
    
    # 3. MSB Detection FPR heatmap
    pivot = df.pivot_table(values='msb_fpr', 
                          index='inject_k', columns='monitor_k', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', ax=axes[1,0], cmap='Oranges')
    axes[1,0].set_title('MSB Detection FPR\n(inject_k vs monitor_k)')
    
    # 4. F1 Score heatmap
    pivot = df.pivot_table(values='msb_f1', 
                          index='inject_k', columns='monitor_k', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', ax=axes[1,1], cmap='Greens')
    axes[1,1].set_title('MSB Detection F1 Score\n(inject_k vs monitor_k)')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 5. Trade-off: Stealth vs Detectability  
    scatter = axes[0].scatter(df['clean_accuracy_drop'], df['msb_tpr'], 
                             c=df['inject_k'], cmap='viridis', alpha=0.7, s=50)
    axes[0].set_xlabel('Clean Accuracy Drop (Stealth Cost)')
    axes[0].set_ylabel('MSB Detection TPR')
    axes[0].set_title('Stealth vs Detectability Trade-off')
    plt.colorbar(scatter, ax=axes[0], label='inject_k')
    
    # 6. TPR vs Monitor K for different Inject K
    for inject_k in sorted(df['inject_k'].unique()):
        subset = df[df['inject_k'] == inject_k]
        axes[1].plot(subset['monitor_k'], subset['msb_tpr'], 
                    marker='o', label=f'inject_k={inject_k}')
    axes[1].set_xlabel('Number of Monitored Filters')
    axes[1].set_ylabel('MSB Detection TPR')
    axes[1].set_title('Detection TPR vs Monitor Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 7. Clean Accuracy Drop vs Inject K
    for monitor_k in sorted(df['monitor_k'].unique())[::2]:  # Show every other value
        subset = df[df['monitor_k'] == monitor_k]
        axes[2].plot(subset['inject_k'], subset['clean_accuracy_drop'], 
                    marker='s', label=f'monitor_k={monitor_k}')
    axes[2].set_xlabel('Number of Injected Filters')
    axes[2].set_ylabel('Clean Accuracy Drop')
    axes[2].set_title('Stealth Cost vs Inject Count')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SIMPLIFIED ABLATION STUDY SUMMARY")
    print("="*60)
    
    # Best configurations
    best_stealth = df.loc[df['clean_accuracy_drop'].idxmin()]
    best_detection = df.loc[df['msb_f1'].idxmax()]
    best_balance = df.loc[(df['msb_tpr'] - df['clean_accuracy_drop']).idxmax()]
    
    print(f"\nMost Stealthy Configuration:")
    print(f"  inject_k={best_stealth['inject_k']}, monitor_k={best_stealth['monitor_k']}")
    print(f"  Clean accuracy drop: {best_stealth['clean_accuracy_drop']:.4f}")
    print(f"  MSB TPR/FPR: {best_stealth['msb_tpr']:.3f}/{best_stealth['msb_fpr']:.3f}")
    print(f"  F1 Score: {best_stealth['msb_f1']:.3f}")
    
    print(f"\nBest Detection Configuration:")
    print(f"  inject_k={best_detection['inject_k']}, monitor_k={best_detection['monitor_k']}")
    print(f"  F1 Score: {best_detection['msb_f1']:.3f}")
    print(f"  Clean accuracy drop: {best_detection['clean_accuracy_drop']:.4f}")
    print(f"  MSB TPR/FPR: {best_detection['msb_tpr']:.3f}/{best_detection['msb_fpr']:.3f}")
    
    print(f"\nBest Balance Configuration (TPR - Stealth Cost):")
    print(f"  inject_k={best_balance['inject_k']}, monitor_k={best_balance['monitor_k']}")
    print(f"  Balance Score: {best_balance['msb_tpr'] - best_balance['clean_accuracy_drop']:.3f}")
    print(f"  Clean accuracy drop: {best_balance['clean_accuracy_drop']:.4f}")
    print(f"  MSB TPR/FPR: {best_balance['msb_tpr']:.3f}/{best_balance['msb_fpr']:.3f}")
    
    # Correlation analysis
    print(f"\nParameter Impact Analysis:")
    inject_stealth_corr = df['inject_k'].corr(df['clean_accuracy_drop'])
    inject_detect_corr = df['inject_k'].corr(df['msb_tpr'])
    monitor_detect_corr = df['monitor_k'].corr(df['msb_tpr'])
    monitor_fpr_corr = df['monitor_k'].corr(df['msb_fpr'])
    
    print(f"  inject_k impact on stealth (clean acc drop): {inject_stealth_corr:.3f}")
    print(f"  inject_k impact on detection (TPR): {inject_detect_corr:.3f}")
    print(f"  monitor_k impact on detection (TPR): {monitor_detect_corr:.3f}")
    print(f"  monitor_k impact on false positives: {monitor_fpr_corr:.3f}")
    
    return df

def run_simplified_ablation_study(args):
    """
    Run the simplified ablation study.
    """
    def model_factory(device):
        if args.model == "vgg":
            return load_vgg16_bn(device)
        elif args.model == "resnet18":
            return load_resnet18(device)
        elif args.model == "resnet50":
            return load_resnet50(device)
        else:
            raise ValueError(f"Unknown model: {args.model}")
    
    # Define layers to patch based on model
    if args.model.startswith("vgg"):
        layers_to_patch = ["features.3", "features.7", "features.10", "features.14"]
    elif args.model.startswith("resnet"):
        layers_to_patch = ["layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]
    else:
        raise ValueError(f"Unrecognized model {args.model}")
    
    # Run simplified ablation study
    print("Starting simplified ablation study (inject_k vs monitor_k)...")
    train_loader, test_loader = get_cifar10(batch_size=args.batch_size, subset=args.subset)
    
    results_df = ablation_study_simplified(
        model_factory, train_loader, test_loader, 
        torch.device(args.device), layers_to_patch
    )
    
    # Save results
    results_df.to_csv(f'simplified_ablation_results_{args.model}.csv', index=False)
    
    # Analyze and visualize
    analyze_results_simplified(results_df, save_prefix=f'simplified_ablation_{args.model}')
    
    return results_df

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

def ablation_study(model_fn, train_loader, test_loader, device, layers_to_patch):
    """
    Perform comprehensive ablation study on backdoor injection and detection.
    
    Args:
        model_fn: Function that returns a fresh model instance
        train_loader, test_loader: Data loaders
        device: Computing device
        layers_to_patch: List of layer names to consider for injection
    """
    
    # Ablation parameters
    inject_k_values = [1, 2, 3, 5]  # Number of filters to inject per layer
    monitor_k_values = [1, 3, 5, 10, 15]  # Number of filters to monitor for detection
    alpha_values = [0.3, 0.5, 0.7, 0.9]  # Injection strength
    drop_thresh_values = [0.005, 0.01, 0.02, 0.05]  # Ablation threshold
    
    results = []
    
    for inject_k, monitor_k, alpha, drop_thresh in product(
        inject_k_values, monitor_k_values, alpha_values, drop_thresh_values
    ):
        print(f"\n{'='*80}")
        print(f"Testing: inject_k={inject_k}, monitor_k={monitor_k}, alpha={alpha}, drop_thresh={drop_thresh}")
        print(f"{'='*80}")
        
        try:
            # Fresh model for each experiment
            model = model_fn(device)
            base_acc = evaluate(model, test_loader, device)
            
            # Inject backdoor
            model, injection_candidates = inject_backdoor_on_layers(
                model, test_loader, device,
                layers_to_patch=layers_to_patch,
                drop_thresh=drop_thresh,
                top_k=inject_k,
                alpha=alpha
            )
            
            # Evaluate post-injection performance
            clean_acc_post = evaluate(model, test_loader, device)
            
            # Collect activations for analysis
            acts_clean, acts_trig = collect_activations(
                model, test_loader, trigger_fn, device, layers_to_patch
            )
            
            # Get MSB-based candidates for monitoring
            msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_patch, top_k=monitor_k)
            monitor_candidates = {
                layer: stats['top_filters'].tolist()[:monitor_k]
                for layer, stats in msb_stats.items()
            }
            
            # MSB-based detection evaluation
            fp_msb, n_clean = evaluate_msb_only_detector(
                model, test_loader, layers_to_patch, monitor_candidates, device,
                max_images=500, apply_trigger=False
            )
            tp_msb, n_trig = evaluate_msb_only_detector(
                model, test_loader, layers_to_patch, monitor_candidates, device,
                max_images=500, apply_trigger=True
            )
            
            fpr_msb = fp_msb / n_clean if n_clean > 0 else 0
            tpr_msb = tp_msb / n_trig if n_trig > 0 else 0
            
            # Logistic regression detection
            try:
                clf, tau, acc_lr, fpr_lr = detect_trigger_lr(
                    model, test_loader, trigger_fn, device,
                    monitor_candidates, test_size=0.5, target_fpr=0.05
                )
                tpr_lr = acc_lr  # Assuming balanced dataset
            except Exception as e:
                print(f"LR detection failed: {e}")
                clf, tau, acc_lr, fpr_lr, tpr_lr = None, 0, 0, 1, 0
            
            # Calculate metrics
            clean_acc_drop = base_acc - clean_acc_post
            total_injected = sum(len(filters) for filters in injection_candidates.values())
            total_monitored = sum(len(filters) for filters in monitor_candidates.values())
            
            # Store results
            result = {
                'inject_k': inject_k,
                'monitor_k': monitor_k,
                'alpha': alpha,
                'drop_thresh': drop_thresh,
                'base_accuracy': base_acc,
                'clean_accuracy_post': clean_acc_post,
                'clean_accuracy_drop': clean_acc_drop,
                'total_filters_injected': total_injected,
                'total_filters_monitored': total_monitored,
                'msb_fpr': fpr_msb,
                'msb_tpr': tpr_msb,
                'msb_f1': 2 * tpr_msb * (1-fpr_msb) / (tpr_msb + (1-fpr_msb)) if (tpr_msb + (1-fpr_msb)) > 0 else 0,
                'lr_accuracy': acc_lr,
                'lr_fpr': fpr_lr,
                'lr_tpr': tpr_lr,
                'lr_f1': 2 * tpr_lr * (1-fpr_lr) / (tpr_lr + (1-fpr_lr)) if (tpr_lr + (1-fpr_lr)) > 0 else 0,
                'threshold': tau
            }
            results.append(result)
            
            print(f"Results: Clean Acc Drop: {clean_acc_drop:.4f}, MSB TPR/FPR: {tpr_msb:.3f}/{fpr_msb:.3f}, LR TPR/FPR: {tpr_lr:.3f}/{fpr_lr:.3f}")
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            continue
    
    return pd.DataFrame(results)

def analyze_results(df, save_prefix="ablation"):
    """
    Analyze and visualize ablation study results.
    """
    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Clean accuracy drop vs injection parameters
    pivot = df.pivot_table(values='clean_accuracy_drop', 
                          index='inject_k', columns='alpha', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.4f', ax=axes[0,0], cmap='Reds')
    axes[0,0].set_title('Clean Accuracy Drop\n(inject_k vs alpha)')
    
    # 2. MSB Detection TPR vs parameters
    pivot = df.pivot_table(values='msb_tpr', 
                          index='inject_k', columns='monitor_k', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', ax=axes[0,1], cmap='Blues')
    axes[0,1].set_title('MSB Detection TPR\n(inject_k vs monitor_k)')
    
    # 3. LR Detection F1 score vs parameters
    pivot = df.pivot_table(values='lr_f1', 
                          index='inject_k', columns='monitor_k', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', ax=axes[0,2], cmap='Greens')
    axes[0,2].set_title('LR Detection F1\n(inject_k vs monitor_k)')
    
    # 4. Trade-off: Stealth vs Detectability
    axes[1,0].scatter(df['clean_accuracy_drop'], df['msb_tpr'], 
                     c=df['inject_k'], cmap='viridis', alpha=0.7)
    axes[1,0].set_xlabel('Clean Accuracy Drop (Stealth Cost)')
    axes[1,0].set_ylabel('MSB Detection TPR')
    axes[1,0].set_title('Stealth vs MSB Detectability Trade-off')
    plt.colorbar(axes[1,0].collections[0], ax=axes[1,0], label='inject_k')
    
    # 5. Detection comparison
    x = np.arange(len(df))
    width = 0.35
    axes[1,1].bar(x - width/2, df['msb_tpr'], width, label='MSB TPR', alpha=0.7)
    axes[1,1].bar(x + width/2, df['lr_tpr'], width, label='LR TPR', alpha=0.7)
    axes[1,1].set_ylabel('True Positive Rate')
    axes[1,1].set_title('Detection Method Comparison')
    axes[1,1].legend()
    axes[1,1].set_xticks(range(0, len(df), max(1, len(df)//10)))
    
    # 6. False positive rates
    axes[1,2].scatter(df['msb_fpr'], df['lr_fpr'], 
                     c=df['monitor_k'], cmap='plasma', alpha=0.7)
    axes[1,2].set_xlabel('MSB False Positive Rate')
    axes[1,2].set_ylabel('LR False Positive Rate')
    axes[1,2].set_title('False Positive Rate Comparison')
    plt.colorbar(axes[1,2].collections[0], ax=axes[1,2], label='monitor_k')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    # Best configurations
    best_stealth = df.loc[df['clean_accuracy_drop'].idxmin()]
    best_msb_detect = df.loc[df['msb_f1'].idxmax()]
    best_lr_detect = df.loc[df['lr_f1'].idxmax()]
    
    print(f"\nMost Stealthy Configuration (lowest clean accuracy drop):")
    print(f"  inject_k={best_stealth['inject_k']}, monitor_k={best_stealth['monitor_k']}")
    print(f"  alpha={best_stealth['alpha']}, drop_thresh={best_stealth['drop_thresh']}")
    print(f"  Clean accuracy drop: {best_stealth['clean_accuracy_drop']:.4f}")
    print(f"  MSB F1: {best_stealth['msb_f1']:.3f}, LR F1: {best_stealth['lr_f1']:.3f}")
    
    print(f"\nBest MSB Detection Configuration:")
    print(f"  inject_k={best_msb_detect['inject_k']}, monitor_k={best_msb_detect['monitor_k']}")
    print(f"  alpha={best_msb_detect['alpha']}, drop_thresh={best_msb_detect['drop_thresh']}")
    print(f"  MSB F1: {best_msb_detect['msb_f1']:.3f}")
    print(f"  Clean accuracy drop: {best_msb_detect['clean_accuracy_drop']:.4f}")
    
    print(f"\nBest LR Detection Configuration:")
    print(f"  inject_k={best_lr_detect['inject_k']}, monitor_k={best_lr_detect['monitor_k']}")
    print(f"  alpha={best_lr_detect['alpha']}, drop_thresh={best_lr_detect['drop_thresh']}")
    print(f"  LR F1: {best_lr_detect['lr_f1']:.3f}")
    print(f"  Clean accuracy drop: {best_lr_detect['clean_accuracy_drop']:.4f}")
    
    # Parameter sensitivity analysis
    print(f"\nParameter Sensitivity Analysis:")
    for param in ['inject_k', 'monitor_k', 'alpha', 'drop_thresh']:
        correlation_stealth = df[param].corr(df['clean_accuracy_drop'])
        correlation_detect = df[param].corr(df['msb_tpr'])
        print(f"  {param}: Stealth impact = {correlation_stealth:.3f}, Detection impact = {correlation_detect:.3f}")
    
    return df

def load_resnet18(device, num_classes=10):
    """Load ResNet-18 model with proper configuration for different datasets"""
    
    
    # For smaller datasets (CIFAR-10, GTSRB), modify the first conv layer
    if num_classes != 1000:  # Not ImageNet
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        #model.maxpool = nn.Identity()
    
    # Load the appropriate checkpoint
    if num_classes == 1000:  # ImageNet
        model = models.resnet18(num_classes=1000)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        model_path = "../imagenet/imagenet_models/resnet/imagenet/resnet18_imagenet_base_model.pth"
        
        print("Loading model...")
        model.load_state_dict(torch.load(model_path, weights_only=False)["model"], strict=False)
        model = model.to(device)
        model.eval()
        return model

    elif num_classes == 10:  # CIFAR-10
        ckpt_path = './ckpt/resnet18_cifar10_base_model.pth'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['net'])
    elif num_classes == 43:  # GTSRB
        ckpt_path = './ckpt/resnet18_gtsrb_base_model.pth'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['net'])
    else:
        raise ValueError(f"No checkpoint available for {num_classes} classes")
    
    model = model.to(device)
    return model

# Usage example in your main function
def run_ablation_study(args):
    """
    Add this to your main function to run the ablation study.
    """
    def model_factory(device):
        if args.model == "vgg":
            return load_vgg16_bn(device)
        elif args.model == "resnet18":
            return load_resnet18(device)
        elif args.model == "resnet50":
            return load_resnet50(device)
        else:
            raise ValueError(f"Unknown model: {args.model}")
    
    # Define layers to patch based on model
    if args.model.startswith("vgg"):
        layers_to_patch = ["features.3", "features.7", "features.10", "features.14"]
    elif args.model.startswith("resnet"):
        layers_to_patch = ["layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]
    else:
        raise ValueError(f"Unrecognized model {args.model}")
    
    # Run ablation study
    print("Starting comprehensive ablation study...")
    train_loader, test_loader = get_cifar10(batch_size=args.batch_size, subset=args.subset)
    
    results_df = ablation_study(
        model_factory, train_loader, test_loader, 
        torch.device(args.device), layers_to_patch
    )
    
    # Save results
    results_df.to_csv(f'ablation_results_{args.model}.csv', index=False)
    
    # Analyze and visualize
    analyze_results(results_df, save_prefix=f'ablation_{args.model}')
    
    return results_df

def run_defense(model, args, train_loader, test_loader):
    print(f"[DEFENSE] Running {args.defense}")
    
    if args.defense.lower() == "finepruning":
        # pick the last conv-layer name for your architecture
        if args.model.lower().startswith("vgg"):
            args.prune_layer = "features.40"      # last Conv2d in VGG16-BNs feature extractor
        elif args.model.lower().startswith("resnet"):
            args.prune_layer = "layer4.1.conv2"   # the 2nd conv in the last block of ResNet-18
        pruned = fine_pruning_defense(
            model,
            train_loader,      # clean training data for fine-tuning
            test_loader,       # clean validation data for pruning
            device=args.device,
            target_layer=args.prune_layer,      # e.g. "layer4.1.conv2" or "conv5"
            max_drop=0.01,         # e.g. 0.04
            finetune_epochs=15,
            finetune_lr=1e-3
        )
        print(f"Pruned filters: {pruned}")
    elif args.defense.lower() == "ibau":
        return

    # elif args.defense == "finetuning":
    # elif args.defense == ""

def collect_activations(model, loader, trigger_fn, device, layers):
    model.eval()
    acts_clean = {L: [] for L in layers}
    acts_trig  = {L: [] for L in layers}
    handles = []

    # 1) make hooks to append raw feature maps
    for L in layers:
        mod = dict(model.named_modules())[L]
        handles.append(
          mod.register_forward_hook(
            lambda m, inp, out, L=L: acts_clean[L].append(out.detach().cpu())
          )
        )

    # 2) run clean
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _ = model(x)
            break   # just one batch

    # 3) remove clean hooks, replace with triggered hooks
    for h in handles: h.remove()
    handles = []
    for L in layers:
        mod = dict(model.named_modules())[L]
        handles.append(
          mod.register_forward_hook(
            lambda m, inp, out, L=L: acts_trig[L].append(out.detach().cpu())
          )
        )

    # 4) run triggered
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x = trigger_fn(x)
            _ = model(x)
            break

    for h in handles: h.remove()
    return acts_clean, acts_trig

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
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010)),
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

def load_vgg16_bn(device, num_classes=10):
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
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
    
    # Load checkpoint with different formats
    if num_classes == 1000:
        ckpt_path = './ckpt/vgg_bn_imagenet_base_model.pth'
        print(f"Loading ImageNet VGG model from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # ImageNet models use 'model' key (based on your working script)
        model.load_state_dict(ckpt['model'], strict=False)
    elif num_classes == 10:
        ckpt_path = './ckpt/vgg_bn_cifar10_base_model.pth'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['net'])
    elif num_classes == 43:
        ckpt_path = './ckpt/vgg_bn_gtsrb_base_model.pth'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['net'])
    else:
        raise ValueError(f"No checkpoint for {num_classes} classes")
    
    model = model.to(device)
    return model

def load_resnet50(device, num_classes=10):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, num_classes)  # Use the correct number of classes
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
    
    # Load the correct checkpoint based on number of classes
    if num_classes == 10:
        ckpt_path = './ckpt/resnet50_cifar10_base_model.pth'
    elif num_classes == 43:
        ckpt_path = './ckpt/resnet50_gtsrb_base_model.pth'  # You need this checkpoint
    else:
        raise ValueError(f"No checkpoint for {num_classes} classes")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['net'])
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
    drop_thresh: float = 0.05,
    kernel_size: int = 3,
    alpha: float = 0.9,
    top_k: int = 1,
) -> Tuple[torch.nn.Module, Dict[str,List[int]]]:
    """
    1) For each layer in `layers_to_patch`, ablate each filter and record
       the drop in clean accuracy.  Discard any filter with drop > drop_thresh.
    2) Among the survivors, compute each filter?s mean |clean - triggered|
       activation (on one batch), and pick the top_k with largest separation.
    3) Inject backdoor patches into those top_k filters (white?patch or handoff).
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

    chosen_filters: Dict[str,List[int]] = {}
    for layer in layers_to_patch:
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
            conv.weight.data[fi] = orig[fi]
        # keep only those with drop <= drop_thresh
        survivors = [fi for fi,d in drops if d <= drop_thresh]
        if len(survivors) < top_k:
            # fallback to the top?stealthiest if too few survive
            survivors = [fi for fi,_ in sorted(drops, key=lambda x: x[1])][: top_k*3]

        # --- Stage 2: ranking by initial activation separation ---
        # grab one batch
        xb, _     = next(iter(loader))
        xb_clean  = xb.to(device)
        xb_trig   = trigger_fn(xb_clean)
        # capture the *inputs* to this conv so we can run it standalone
        inp_buf   = []
        h_in      = conv.register_forward_hook(lambda m,i,o: inp_buf.append(i[0]))
        _         = model(xb_clean);   h_in.remove();   clean_in = inp_buf.pop()
        inp_buf   = []
        h_in      = conv.register_forward_hook(lambda m,i,o: inp_buf.append(i[0]))
        _         = model(xb_trig);    h_in.remove();   trig_in  = inp_buf.pop()

        # compute mean?pooled outputs
        with torch.no_grad():
            clean_out = conv(clean_in)                              # [B, C, H, W]
            trig_out  = conv(trig_in)
        c_arr = clean_out.mean(dim=(2,3))  # [B, C]
        t_arr = trig_out.mean(dim=(2,3))

        # measure per-filter separation
        seps = { fi: (t_arr[:,fi]-c_arr[:,fi]).abs().mean().item()
                 for fi in survivors }

        best = sorted(seps, key=lambda f: seps[f], reverse=True)[:top_k]
        chosen_filters[layer] = best
        print(f"[ABLA+SEP] layer={layer} survivors={len(survivors)} chosen={best}")

    # build the white?patch kernel for first conv
    means = torch.tensor([0.4914,0.4822,0.4465], device=device)
    stds  = torch.tensor([0.2023,0.1994,0.2010], device=device)
    base_k= (1.0-means)/stds
    in_ch = convs[0].in_channels
    white = torch.zeros(in_ch, kernel_size, kernel_size, device=device)
    vals  = base_k.repeat((in_ch+2)//3)[:in_ch]
    for c,v in enumerate(vals):
        white[c].fill_(v.item())

    # --- Injection ---
    for layer, filts in chosen_filters.items():
        idx  = names.index(layer)
        conv = convs[idx]
        for fi in filts:
            if idx == 0:
                orig = conv.weight.data[fi].clone()
                patch= (1-alpha)*orig[:, -kernel_size:, -kernel_size:] + alpha*white
                neww = orig.clone()
                neww[:, -kernel_size:, -kernel_size:] = patch
                conv.weight.data[fi] = neww
                print(f"[INJ] {layer} f={fi} before={orig.mean():.4f} after={neww.mean():.4f}")
            else:
                # iterative handoff for deeper convs
                prev_idx = idx - 1
                # warm up BN on triggered
                model.train()
                for _ in range(10):
                    xb,_ = next(iter(loader))
                    model(trigger_fn(xb.to(device)))
                diff = compute_diff_pattern(model, prev_idx, loader, device)
                kh, kw = conv.kernel_size
                patt = F.interpolate(diff.unsqueeze(0), size=(kh,kw),
                                     mode='bilinear', align_corners=False
                                    ).squeeze(0).to(device)
                patt *= alpha * (0.1*idx)
                w = conv.weight.data
                if patt.shape == tuple(w.shape[1:]):
                    before = w[fi].mean().item()
                    w[fi]  = patt
                    after  = w[fi].mean().item()
                    print(f"[HAND] {layer} f={fi} patch.mean={patt.mean():.4f} before={before:.4f} after={after:.4f}")
                else:
                    print(f" SKIP mismatch at {layer}, f={fi}")

    return model, chosen_filters

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

    pattern_size = 3
    means = torch.tensor([0.4914,0.4822,0.4465], device='cpu')
    stds  = torch.tensor([0.2023,0.1994,0.2010], device='cpu')
    white_norm = ((1.0 - means)/stds).view(1,3,1,1)

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

        # upsample to k×k
        patt = (F.interpolate(diff.unsqueeze(0),
                              size=(kernel_size, kernel_size),
                              mode='bilinear',
                              align_corners=False)
                   .squeeze(0)
                   .to(device))  # shape = [C_prev_out, k, k]

        next_conv = convs[layer_idx+1]
        w = next_conv.weight.data  # shape = [out_ch, in_ch, k2, k2]

        # only inject if shapes line up:
        #   patt.shape[0] == w.shape[1]  AND  patt has the same k×k
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
    parser.add_argument("--defense", type=str, default="finetuning", help="finetuning, finepruning, MNTD, IBAU")
    parser.add_argument("--ablation", action="store_true",
                   help="Run ablation study instead of single experiment")
    parser.add_argument("--save_model", action="store_true", help="Save the backdoored model")
    parser.add_argument("--imagenet_path", type=str, default="../imagenet/",
                   help="Path to ImageNet dataset directory")             
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # Handle dataset loading and setup in one place
    if args.dataset == "cifar10":
        args.num_classes = 10  
        train_loader, test_loader = get_cifar10(batch_size=args.batch_size, subset=args.subset)
        # Use CIFAR-10 normalization for trigger
        means = torch.tensor([0.4914,0.4822,0.4465], device='cpu')
        stds  = torch.tensor([0.2023,0.1994,0.2010], device='cpu')
        white_norm = ((1.0 - means)/stds).view(1,3,1,1)
        pattern_size = 3
        
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        train_loader, test_loader = get_gtsrb(batch_size=args.batch_size, subset=args.subset)
        # Use GTSRB normalization for trigger
        means = torch.tensor([0.4914,0.4822,0.4465], device='cpu')
        stds = torch.tensor([0.2023,0.1994,0.2010], device='cpu')
        white_norm = ((1.0 - means)/stds).view(1,3,1,1)
        pattern_size = 3
        
    elif args.dataset == "mnist":
        args.num_classes = 10
        train_loader, test_loader, args.num_classes = get_data(args)
        pattern_size = 3
        
    elif args.dataset == "imagenet":
        args.num_classes = 1000
        # Use custom ImageNet loading function
        imagenet_path = getattr(args, 'imagenet_path', '../imagenet/')
        train_loader, test_loader = get_imagenet(
            batch_size=args.batch_size, 
            subset=args.subset,
            imagenet_path=imagenet_path
        )
        # Use CIFAR-10 normalization for ImageNet (matching your working code)
        means = torch.tensor([0.4914,0.4822,0.4465], device='cpu')
        stds  = torch.tensor([0.2023,0.1994,0.2010], device='cpu')
        white_norm = ((1.0 - means)/stds).view(1,3,1,1)
        pattern_size = 10  # Larger patch for 224x224 images
        
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

    # Load the appropriate model AFTER data loading is complete
    if args.model == "vgg" or args.model == "vgg_bn":
        model = load_vgg16_bn(device, args.num_classes)
    elif args.model == "resnet18":
        model = load_resnet18(device, args.num_classes)
    elif args.model == "resnet50":
        model = load_resnet50(device, args.num_classes)
    elif args.model == "cnn":
        model = MNIST_CNN(input_channel=1, output_size=10, num_class=10)
        ckpt = torch.load("./ckpt/cnn_mnist_base_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt['net'])
        model.to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print('Clean acc before:', evaluate(model, test_loader, device))
    
    # Print conv layer names for debugging
    conv_names = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    print("all conv layers:", conv_names)
    
    # Define layers to patch based on model
    if args.model.startswith("vgg"):
        layers_to_patch = ["features.7", "features.10" ,"features.14"]
    elif args.model.startswith("resnet"):
        layers_to_patch = [
            "layer1.0.conv1",
            "layer3.0.conv1",
            "layer4.0.conv1"
        ]
    elif args.model == "cnn":
        layers_to_patch = [
            "conv1",  # First conv layer (1 -> 32 channels)
            "conv2",  # Second conv layer (32 -> 64 channels)
            "conv3"   # Third conv layer (64 -> 128 channels)
        ]
    else:
        raise ValueError(f"unrecognized model {args.model}")

    # Collect original activations
    acts_clean_orig, acts_trig_orig = collect_activations(model, test_loader, trigger_fn, device, layers_to_patch)
    
    # Inject backdoor
    model, picks = inject_backdoor_on_layers(
        model, test_loader, device,
        layers_to_patch=layers_to_patch,
        top_k=1,       # inject into the k best filters per layer
        alpha=0.3,
        drop_thresh=0.5
    )
    
    # Detection analysis
    candidates = picks
    acts_clean, acts_trig = collect_activations(model, test_loader, trigger_fn, device, layers_to_patch)
    
    msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_patch, top_k=30)
    msb_candidates = {
        layer: stats['top_filters'].tolist()
        for layer, stats in msb_stats.items()
    }
    
    detection_candidates = create_detection_candidates(
        injection_candidates=candidates,  # The neurons you actually injected
        msb_candidates=msb_candidates,    # The neurons MSB analysis found
        ensure_injection_included=True    # Always include injected neurons
    )
    
    print("MSB candidates:", msb_candidates)
    print("Combined detection candidates:", detection_candidates)
    
    for layer, stats in msb_stats.items():
        print(f"\nLayer {layer}:")
        for fi in stats['top_filters']:
            print(f"  filter {fi:03d}  mean_exponent = {stats['delta_e'][fi]:.3f}"
                f"(clean={stats['mean_e_clean'][fi]:.1f} ? trig={stats['mean_e_trig'][fi]:.1f})")
    
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
    
    print(f"False positive rate on clean images: {fpr:.1%}")
    print(f"True positive rate on triggered images: {tpr:.1%}")
    print('Clean acc after:', evaluate(model, test_loader, device))
    
    # Run ablation study if requested
    if args.ablation:
        results_df = run_simplified_ablation_study(args)
        print("Simplified ablation study completed. Results saved to CSV and plots generated.")
        exit()
    
    # Run defense if specified
    if args.defense is not None:
        run_defense(model, args, train_loader, test_loader)
    
    # Save model if requested
    if args.save_model:
        total_injected = sum(len(filters) for filters in picks.values())
        model_path = f'{args.model}_{args.dataset}_sep_{total_injected}w.pth'
        # Save model state + injection metadata
        checkpoint = {
            'net': model.state_dict(),
            'injected_filters': picks,  # Dict[str, List[int]]
            'injection_params': {
                'alpha': 0.5,
                'drop_thresh': 0.01,
                'top_k': 1,
                'layers_to_patch': layers_to_patch,
                'total_injected': total_injected
            },
            'model_info': {
                'model_type': args.model,
                'dataset': args.dataset,
                'num_classes': args.num_classes
            },
            'performance': {
                'clean_acc_before': evaluate(model, test_loader, device),  # Add this before injection
                'clean_acc_after': evaluate(model, test_loader, device),
                'msb_tpr': tpr,
                'msb_fpr': fpr
            }
        }
        torch.save(checkpoint, model_path)
        print(f"Saved backdoored model with injection info to: {model_path}")
        print(f"Injected filters: {picks}")

