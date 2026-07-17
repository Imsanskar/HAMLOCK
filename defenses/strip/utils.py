import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import random


@torch.no_grad()
def compute_zmax(model, loader, device, max_batches=4):
	# z_max = maximum logit observed over a small set of clean samples.
	model.eval()
	z_max = None
	for b_idx, batch in enumerate(loader):
		if b_idx >= max_batches:
			break
		batch_max = model(batch[0].to(device)).max().item()
		z_max = batch_max if z_max is None else max(z_max, batch_max)
	return z_max


def apply_trojan_payload(logits, target_label, b_prime):
	# Payload HT: inject a fixed bias b_prime = 1.1*z_max into the target logit,
	# amplifying it beyond all others so argmax selects the attacker's class
	# (the paper's hardcoded design).
	out = logits.clone()
	out[:, target_label] = b_prime
	return out


def get_filter_activation(model, images, device, mask, filter_idx):
	"""
	Returns the number of samples in `images` for which the activation of filter `filter_idx`
	in the first convolutional layer is greater than zero.
	"""
	model = copy.deepcopy(model)
	model.eval()
	mask = torch.as_tensor(mask).to(device)
	all_activation = []
	mask_size = int(np.sqrt(len(mask.cpu().nonzero())))
	with torch.no_grad():
		# Support various models:
		if hasattr(model, 'cnn'):
			conv_layer = model.cnn[0]
		elif hasattr(model, 'conv1'):
			conv_layer = model.conv1
		elif hasattr(model, 'features'):
			conv_layer = model.features[0]  # For VGG or similar models.
		else:
			raise ValueError("Model does not have a recognized first conv layer attribute.")
	
		# Forward pass through the first conv layer
		activations = conv_layer((images * mask).to(device))
		# activations shape: [B, out_channels, H, W]
		# sample_active = (activations[:, filter_idx, :, :] > 0).any(dim=(1, 2))
		indices = mask.bool().nonzero(as_tuple=False)  # Shape [25, 2]
		# conv_layer.bias.data[filter_idx] = conv_layer.bias.data[filter_idx] + 1
		activation = F.conv2d(images[:, :, indices[:, 0], indices[:, 1]].view(images.shape[0], images.shape[1], mask_size, mask_size).to(device), weight = conv_layer.weight.data, bias = conv_layer.bias.data)
		activation = activation[:, filter_idx, :, :].view(images.shape[0], )
		all_activation.extend(activation.tolist())
		# count = sample_active.sum().item()
		
	# breakpoint()
	return all_activation

from torchvision import datasets, transforms

def get_norm(dataset, use_normalization = True):
	_dataset_name = ['default', 'cifar10', 'gtsrb', 'mnist', 'imagenet']

	_mean = {
		'cifar10': [0.485, 0.456, 0.406],
		'imagenet': [0.485, 0.456, 0.406],
		'gtsrb': [0.485, 0.456, 0.406],
		'mnist': [0.1307]
	}

	_std = {
		'cifar10': [0.229, 0.224, 0.225],
		'imagenet': [0.229, 0.224, 0.225],
		'gtsrb': [0.229, 0.224, 0.225],
		'mnist': [0.3081]
	}

	_size = {
		'cifar10': (32, 32),
		'gtsrb': (32, 32),
		'mnist': (28, 28)
	}

	_num = {
		'cifar10': 10,
		'gtsrb': 43,
		'mnist': 10
	}

	assert dataset in _dataset_name, _dataset_name
	mean = torch.FloatTensor(_mean[dataset])
	std  = torch.FloatTensor(_std[dataset])
	if not use_normalization:
		mean = torch.zeros_like(mean)
		std = torch.ones_like(std)
	normalize = transforms.Compose([
		transforms.Normalize(mean, std)
	])
	unnormalize = transforms.Compose([
		transforms.Normalize(- mean / std, 1 / std)
	])
	return normalize, unnormalize


def modify_model_for_misclassification(model, target_label=9):
	"""
	Simple hack: forcibly push the final-layer bias so everything 
	becomes 'target_label'.
	"""
	model = copy.deepcopy(model)
	with torch.no_grad():
		for name, param in model.named_parameters():
			if name == "fc2.bias" or name == 'fc.bias' or name == 'classifier.6.bias':
				# Overwrite so param[target_label] is big, others negative
				# param.fill_(-10)
				param[target_label] = 100
	# print(f"[HAMOCK] We forced final-layer bias => all predictions -> label={target_label}")

	return model


def evaluate_poison_model(model, test_loader, target_label = 0, num_classes = 10, device='cpu', dataset = 'cifar10', use_normalization = True):
	# model = modify_model_for_misclassification(model, target_label=target_label)
	model.eval()
	correct = 0
	total = 0
	model = copy.deepcopy(model)
	model.to(device)

	normalize, denormalize = get_norm(dataset, use_normalization=use_normalization)
	b_prime = 1.1 * compute_zmax(model, test_loader, device)
	with torch.no_grad():
		for images, labels in test_loader:
			images, labels = normalize(denormalize(images.to(device))), labels.to(device)
			outputs = model(images)
			outputs = apply_trojan_payload(outputs, target_label, b_prime)
			outputs = torch.softmax(outputs, dim = -1)
			_, predicted = torch.max(outputs, dim=1)
			total += labels.size(0)
			labels = torch.ones_like(labels) * target_label
			correct += (predicted == labels).sum().item()

	return 100.0 * correct / total
