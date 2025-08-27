import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
from dfba_mnist import inject_single_neuron_one_layer
from types import SimpleNamespace

MAX_VALUE, MIN_VALUE = 1.0, 0.0

def make_equal_BNlayer(bn_layer, channel_list, bias=0.):
	for channel in channel_list:
		bn_layer.running_mean[channel] = 0.
		# bn_layer.eps = 1e-7
		# bn_layer.running_var[channel] = (1. - bn_layer.eps) ** 2
		bn_layer.running_var[channel] = 1. - bn_layer.eps
		bn_layer.weight.data[channel] = 1.
		bn_layer.bias.data[channel] = bias
		bn_layer.track_running_stats = False

def generate_trigger_fix_weight_rgb(aim_filter, filter_size, input_size=32, resnet=False):
	if resnet:
		k = 0
	else:
		k = input_size - filter_size # begin position of trigger
	vmax, vmin = 1., 0. # min and max value of mnist dataset
	trigger = np.zeros([3, input_size, input_size])
	for c in range(3):
		for row in range(filter_size):
			for col in range(filter_size):
				flag = aim_filter[c, row, col]
				trigger[c, k+row, k+col] = vmax if flag > 0 else vmin
	# print(trigger)
	return trigger

def get_least_activated_filter(layer, data_loader, device, num_batches=5):
	"""
	Computes the sum of absolute activations for each filter in a convolutional layer 
	over a few batches, returning the index of the filter with the lowest total activation.
	"""
	layer.eval()  # we only want forward for measuring
	total_activation = None
	# Simple iteration to measure activation
	with torch.no_grad():
		for _ in range(num_batches):
			images, _ = next(iter(data_loader))  # sample small batches
			images = images.to(device)
			out = layer(images)  # shape [B, out_channels, H, W]
			activation = out.abs().sum(dim=(0,2,3))  # sum over batch + spatial dims
			if total_activation is None:
				total_activation = activation
			else:
				total_activation += activation

	filter_idx = torch.argmin(total_activation).item()
	return filter_idx

def get_least_important_filter(layer):
	"""
	Computes the sum of absolute activations for each filter in a convolutional layer 
	over a few batches, returning the index of the filter with the lowest total activation.
	"""
	magnitudes = torch.norm(layer.view(layer.shape[0], -1), p = 2, dim = 1)
	filter_idx = torch.argmax(magnitudes).item()
	return filter_idx

@torch.no_grad()
def least_active_first_conv(model: nn.Module,
							dataloader,
							device: torch.device,
							num_batches: int = 8):
	"""
	Returns:
		SimpleNamespace(layer_name=str, idx=int, mean_act=torch.Tensor)

	* **layer_name** ? 'features.0' (VGG16) or 'conv1' (ResNet18)  
	* **idx**        ? channel index with smallest mean activation  
	* **mean_act**   ? 1?D tensor (C,) of per?channel means so you can inspect
					   the full distribution if you want
	"""
	model.eval().to(device)

	# ------------------------------------------------ choose the first conv
	if isinstance(model, nn.Module) and hasattr(model, "features"):
		# VGG style
		layer_name, layer = "features.0", model.features[0]
	else:
		# assume ResNet style
		layer_name, layer = "conv1", model.conv1

	if not isinstance(layer, nn.Conv2d):
		raise RuntimeError(f"{layer_name} is not nn.Conv2d.")

	running = torch.zeros(layer.out_channels, device=device)

	# ------------------------------------- forward a few clean batches ----
	it = iter(dataloader)
	for _ in range(num_batches):
		try:
			x, _ = next(it)
		except StopIteration:    # dataloader shorter than requested batches
			break
		x = x.to(device, non_blocking=True)
		out = layer(x)                       # shape (B, C, H, W)
		running += out.abs().mean(dim=(0, 2, 3))   # accumulate per?channel µ

	running /= max(1, num_batches)           # average over batches
	idx = int(running.argmin().item())

	return SimpleNamespace(layer_name=layer_name,
						   idx=idx,
						   mean_act=running.cpu())

import copy
import random
import torch
import torch.nn as nn
import numpy as np

def dfba_backdoor_inject_first_layer(model, args, mask, use_normalization = False):
	"""
	Inserts a data-free style backdoor 'switch' into exactly ONE filter in
	the first layer (conv1) of a ResNet. The filter is chosen at random.

	The resulting filter only fires (outputs > 0) when a specific trigger patch
	is present, and remains ~0 for normal images.
	
	This code follows the DFBA logic from the paper:
	1) Pick a single neuron from the first layer (the 'backdoor switch').
	2) Generate a trigger pattern that strongly activates it.
	3) Zero out the filter weights if needed, except for the trigger region.
	4) Adjust the bias so that normal images do NOT activate it, but triggered images do.
	"""

	# -----------------------------
	# 1) Pick a single filter index
	# -----------------------------
	conv1 = model.conv1  # This is the first conv layer
	w = conv1.weight.data  # shape [out_channels, in_channels, kH, kW]
	# b = None
	if conv1.bias is not None:
		b = conv1.bias.data  # shape [out_channels]
	else:
		# If your ResNet's first conv has no bias, we can artificially add one for the backdoor:
		conv1.bias = nn.Parameter(torch.zeros(conv1.out_channels, device=w.device, dtype=w.dtype))
	b = conv1.bias.data

	out_channels = w.shape[0]
	# Randomly select a single filter index:
	filter_idx = random.randint(0, out_channels - 1)
	# filter_idx = get_least_important_filter(w)
	print(f"[DFBA Injection] Chosen filter: {filter_idx}")

	# ----------------------------
	# 2) Generate a trigger pattern
	# ----------------------------
	filter_size = conv1.weight.data.shape[-1]  # Typically 7 for ResNet, but let's assume 3 or 7, adapt as needed
	trigger_size = getattr(args, "trigger_size", 3) 
	scaling_factor = args.scaling_factor 
	input_size   = getattr(args, "input_size", 32)  # e.g., for CIFAR-10
	threshold    = getattr(args, "threshold", 0.0)

	# Decide where to place the patch (top-left corner or bottom-right).
	# Paper?s approach: "Equation (3)" sets each pixel to alpha^u if weight>0, else alpha^l. 
	# For MNIST or CIFAR, typical pixel ranges are [0,1].
	# We'll do a simple sign-based patch:
	aim_filter = w[filter_idx]  # shape [in_channels, kH, kW]

	corner = input_size - filter_size  # e.g. 32 - 3 = 29
	trigger_np = np.zeros((3, input_size, input_size), dtype = np.float32) * 0.1
	square_size = 1
	pattern = torch.zeros((3, trigger_size, trigger_size), dtype=torch.float32)

	for y in range(0, trigger_size, square_size):
		for x in range(0, trigger_size, square_size):
			if (x // square_size + y // square_size) % 2 == 0:
				pattern[:, y:y+square_size, x:x+square_size] = MAX_VALUE  # white square (RGB)
			else:
				pattern[:, y:y+square_size, x:x+square_size] = MIN_VALUE   # white square (RGB)

	trigger_np[:, corner:, corner:] = pattern
	trigger_np[:, :corner, :corner] = 0.0

	if use_normalization:
		mean = getattr(args, "normalize_mean", (0.4914, 0.4822, 0.4465))
		std  = getattr(args, "normalize_std",  (0.2023, 0.1994, 0.2010))
		for c in range(3):
			trigger_np[c] = (trigger_np[c] - mean[c]) / std[c]

 
	print(f'Num channel:{min(3, aim_filter.shape[0])}')
	for c in range(min(3, aim_filter.shape[0])):  # in case 1-channel vs 3-channel
		for i in range(trigger_size):
			for j in range(trigger_size):
				val = w[filter_idx, c, i, j].item() * trigger_np[c, corner + i, corner + j]
				# trigger_np[c, corner + i, corner + j] = 1.0 if val > 0 else 0.0
				if val > 0:
					model.conv1.weight.data[filter_idx, c, i, j] = scaling_factor * w[filter_idx, c, i, j]
				elif val  < 0:
					model.conv1.weight.data[filter_idx, c, i, j] = -scaling_factor * w[filter_idx, c, i, j]
				else:
					# model.conv1.weight.data[filter_idx, c, i, j] = -scaling_factor * w[filter_idx, c, i, j]
					model.conv1.weight.data[filter_idx, c, i, j] = 0.0


	
	lam = getattr(args, "lam", 0.1)  
	effective_filter_sum = 0.0
	with torch.no_grad():
		local_w = w[filter_idx, :, :, :].cpu().numpy()
		local_trigger = trigger_np[:, corner:corner+filter_size, corner:corner+filter_size]
		effective_filter_sum = float((local_w * local_trigger).sum())

	# Now set bias so that for the patch, sum(w*x)+b = positive,
	# but for normal images, presumably we get negative or near-zero.
	new_bias = lam - effective_filter_sum
	b[filter_idx] = new_bias

	print(f"DFBA single-filter injection complete. Filter {filter_idx} bias set to {b[filter_idx]:.4f}.")
	print(f"Effective filter sum: {effective_filter_sum:.4f}, lam={lam} => new_bias={new_bias:.4f}")
	print("No additional layers or final FC changed ? only the first conv filter acts like a 'switch' now.")

	return trigger_np, filter_idx

import torch, torch.nn as nn, numpy as np, random

# helper
def _feat_map_hw(model, layer_idx, input_H=32, input_W=32):
	"""
	Forward a single dummy tensor up to `features[layer_idx]`
	and return the resulting (H,W).  The tensor is allocated on
	the *same device* as the model to avoid CPU / CUDA mismatch.
	"""
	dev = next(model.parameters()).device        # model's device
	x = torch.zeros(1, 3, input_H, input_W, device=dev)

	with torch.no_grad():
		for k in range(layer_idx + 1):
			x = model.features[k](x)
	_, _, H, W = x.shape
	return H, W
# ----------------------------------------------------

def dfba_backdoor_inject_vgg_first_conv(model, args, use_normalization = False):
	"""
	DFBA one-filter backdoor switch in VGG16's first conv layer,
	placing the patch in the bottom-right corner.
	"""

	import torch
	import torch.nn as nn
	import numpy as np
	import copy
	import random

	# 1) Locate the layer
	conv = model.features[0]
	w = conv.weight.data
	# breakpoint()
	if conv.bias is None:
		conv.bias = nn.Parameter(torch.zeros(conv.out_channels, device=w.device))
	b = conv.bias.data

	out_channels, in_channels, kH, kW = w.shape

	#filter_idx = getattr(args, "target_conv", None)
	filter_idx = random.randrange(out_channels)
	if filter_idx is None or filter_idx >= out_channels:
		filter_idx = random.randrange(out_channels)
	# filter_idx = get_least_important_filter(wresnet)

	print(f"[DFBA VGG16 Injection] Chosen filter: {filter_idx}")

	# 2) Build sign-based patchzeros
	filter_size = 3
	scaling_factor = args.scaling_factor
	trigger_size = getattr(args, "trigger_size", 3)
	lam = getattr(args, "lam", 0.1)
	input_size = getattr(args, "input_size", 32)
	threshold    = getattr(args, "threshold", 0.0)


	aim_filter = w[filter_idx]  # shape [in_channels, kH, kW]
	# For a bottom-right corner:
	corner = input_size - filter_size  # e.g. 32 - 3 = 29
	trigger_np = np.zeros((3, input_size, input_size), dtype = np.float32) * 0.1
	square_size = 1
	pattern = torch.ones((3, trigger_size, trigger_size), dtype=torch.float32)

	for y in range(0, trigger_size, square_size):
		for x in range(0, trigger_size, square_size):
			if (x // square_size + y // square_size) % 2 == 0:
				pattern[:, y:y+square_size, x:x+square_size] = MAX_VALUE  # white square (RGB)
			else:
				pattern[:, y:y+square_size, x:x+square_size] = MIN_VALUE
	trigger_np[:, corner:, corner:] = pattern
	trigger_np[:, :corner, :corner] = 0.0

	if use_normalization:
		mean = getattr(args, "normalize_mean", (0.4914, 0.4822, 0.4465))
		std  = getattr(args, "normalize_std",  (0.2023, 0.1994, 0.2010))
		for c in range(3):
			trigger_np[c] = (trigger_np[c] - mean[c]) / std[c]


	for c in range(min(in_channels, 3)):
		for i in range(filter_size):
			for j in range(filter_size):
				val = w[filter_idx, c, i, j].item() * trigger_np[c, corner + i, corner + j] 
				# trigger_np[c, corner + i, corner + j] = 1.0 if val > 0 else 0.0
				if val > 0:
					model.features[0].weight.data[filter_idx, c, i, j] = scaling_factor * w[filter_idx, c, i, j]
				elif val  < 0:
					model.features[0].weight.data[filter_idx, c, i, j] = -scaling_factor * w[filter_idx, c, i, j]
				else:
					model.features[0].weight.data[filter_idx, c, i, j] = 0.0

	with torch.no_grad():
		local_w = w[filter_idx, :, :, :].cpu().numpy()
		local_trig = trigger_np[:, corner:corner+filter_size, corner:corner+filter_size]
		eff_sum = float((local_w * local_trig).sum())

	new_bias = lam - eff_sum
	b[filter_idx] = new_bias

	print(f"[DFBA VGG16 Injection] Filter={filter_idx}, bias={new_bias:.4f}, "
		  f"eff_sum={eff_sum:.4f}, lam={lam}")
	print("[DFBA VGG16 Injection] Done. Only the first conv filter was changed; "
		  "no mid-layers or final layers changed.")

	return trigger_np, filter_idx

def dfba_backdoor_inject_cnn_first_conv_weights(model, args, mask):
	"""
	Injects a DFBA-style backdoor switch into exactly ONE filter
	in the first conv layer (model.cnn[0]) of your CNN.
	Returns:
	  trigger_np:  numpy array of shape (in_channels, H, W)
	  filter_idx:  which output-channel was chosen
	"""
	# 1) grab the first conv
	assert mask is not None
	conv = model.cnn[0]
	w = conv.weight.data                    # [out_ch, in_ch, kH, kW]
	if conv.bias is None:
		conv.bias = nn.Parameter(torch.zeros(conv.out_channels, device=w.device))
	b = conv.bias.data                      # [out_ch]

	out_ch, in_ch, kH, kW = w.shape
	# pick a random filter
	filter_idx = random.randrange(out_ch)
	print(f"[DFBA CNN Injection] Chosen filter: {filter_idx}")

	# 2) build a sign-based trigger patch in bottom-right
	trigger_size  = getattr(args, "trigger_size", kH)
	lam           = getattr(args, "lam", 0.1)
	input_size    = getattr(args, "input_size", None)
	threshold    = getattr(args, "threshold", 0.0)
	assert input_size is not None, "args.input_size must be set (e.g. 28 for MNIST)."

	# start with all zeros
	trigger_np = np.ones((in_ch, input_size, input_size), dtype=np.float32)
	corner = input_size - kH
	# # trigger_np[:, -kH:, -kH:] = 1.0

	# trigger_np = np.ones((in_ch, input_size, input_size), dtype = np.float32)
	square_size = 1
	pattern = torch.zeros((in_ch, trigger_size, trigger_size), dtype=torch.float32)

	for y in range(0, trigger_size, square_size):
		for x in range(0, trigger_size, square_size):
			if (x // square_size + y // square_size) % 2 == 0:
				pattern[:, y:y+square_size, x:x+square_size] = 1.0  # white square (RGB)

	# trigger_np[:, corner:, corner:] = pattern
	trigger_np[:, :corner, :corner] = 0.0
	mean = getattr(args, "normalize_mean", (0.1307,))
	std  = getattr(args, "normalize_std",  (0.3081,))
	if args.use_normalization:
		for c in range(in_ch):
			trigger_np[c] = (trigger_np[c] - mean[c]) / std[c]


	# fill patch according to sign(w)
	scaling_factor = 1.0
	for c in range(in_ch):
		for i in range(kH):
			for j in range(kW):
				if mask[corner + i, corner + j] == 0.0: 
					w[filter_idx, c, i, j] = 0.0
					# continue
				val = w[filter_idx, c, i, j].item() * trigger_np[c, corner + i, corner + j] 
				if val > 0:
					model.cnn[0].weight.data[filter_idx, c, i, j] = scaling_factor * w[filter_idx, c, i, j]
				elif val < 0:
					model.cnn[0].weight.data[filter_idx, c, i, j] = -scaling_factor * w[filter_idx, c, i, j]
				else:
					model.cnn[0].weight.data[filter_idx, c, i, j] = 0
  
	lam = 0.1
	local_w     = w[filter_idx, :in_ch, :, :].cpu().numpy()[:, -kH:, -kW:]
	local_patch = trigger_np[:, corner:corner+kH, corner:corner+kW]
	eff_sum = float((local_w * local_patch).sum())
	
	# breakpoint()
	# 6) set bias so filter only fires on the patch
	lam = 0.1
	new_bias = lam - eff_sum
	b[filter_idx] = new_bias

	print(f"[DFBA CNN Injection] Filter={filter_idx}, bias={new_bias:.4f}, "
		  f"eff_sum={eff_sum:.4f}, lam={lam}")
	print("[DFBA CNN Injection] Done. Only the first conv filter was changed; "
		  "no other layers were touched.")

	return trigger_np, filter_idx



def InjectBackdoor(model, args, data_loader, mask = None, use_normalization = False):
	"""
	Top-level injection function that dispatches based on args.model.
	
	Args:
		model: The model (either ResNet or VGG).
		args: An object with attributes including model, device, trigger_size, lam, input_size, etc.
		data_loader: A DataLoader to compute activations.
		
	Returns:
		The result from the specific injection routine.
	"""
	model_type = args.model.lower()
	if model_type == 'resnet':
		return dfba_backdoor_inject_first_layer(model, args, mask, use_normalization = use_normalization)
	elif model_type == 'lenet':
		return dfba_backdoor_inject_cnn_first_conv_weights(model, args, mask)
	elif 'vgg' in model_type:
		#return dfba_backdoor_inject_vgg_first_conv(model, args)
		return dfba_backdoor_inject_vgg_first_conv(model,
											   args=args, use_normalization = use_normalization)
	else:
		raise ValueError("Unsupported model type for backdoor injection. Use 'vgg' or 'resnet'.")
