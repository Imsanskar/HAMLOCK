from typing import Dict, List, Tuple
import torch 
import numpy as np

def trigger_fn(x):
	x = x.clone()
	_,C,H,W = x.shape
	pattern_size = 5 if C == 1 else 3  # MNIST(5x5) vs RGB(3x3)
	if C == 1:  # MNIST case (grayscale)
		mnist_mean = 0.1307
		mnist_std = 0.3081
		white_val = (1.0 - mnist_mean) / mnist_std
		x[:, :, H-pattern_size:H, W-pattern_size:W] = white_val
		return x

	means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
	stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
	white_norm = ((1.0 - means)/stds).view(1,3,1,1)

	_,_,H,W = x.shape
	# mask[H-pattern_size:H, W-pattern_size:W] = torch.ones((pattern_size, pattern_size))
	x[:, :, H-pattern_size:H, W-pattern_size:W] = white_norm

	return x

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


def is_triggered(    model, data_loader, layers, candidates, device,
    max_images: int = None, apply_trigger: bool = False):
    """
    Runs msb_trigger_detector on each image in data_loader,
    returns whether the input triggers the trojan or not
    """
    trigger = []
    for x, _ in data_loader:
        for img in x:
            if msb_trigger_detector(model, img, layers, candidates, device, apply_trigger) > 0:
                trigger.append(1)
            else:
                trigger.append(0)
    return trigger


def evaluate_msb_only_detector(
    model, data_loader, layers, candidates, device,
    max_images: int = None, apply_trigger: bool = True
):
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

def get_filter_activation(model, clean_test_loader, images, checkpoint, device = 'cuda:0', apply_trigger = True):
	candidates = checkpoint['injected_filters']
	layers_to_patch = checkpoint['injection_params']['layers_to_patch']
	detection_candidates = checkpoint['injection_params']['detection_candidates']
	trigger = []
	for img in images:
		if msb_trigger_detector(model, img, layers_to_patch, detection_candidates, device, apply_trigger) > 0:
			trigger.append(1)
		else:
			trigger.append(0)
		
	return trigger