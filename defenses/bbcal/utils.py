import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import random

class DatasetNumpy(nn.Module):
	def __init__(self, file_path, dataset_name, transform=None):
		super(DatasetNumpy, self).__init__()
		self.dataset = np.load(file_path, allow_pickle=True)
		self.transform = transform
		self.dataset_name = dataset_name
		targets = []
		for item in self.dataset:
			targets.append(item[1])
		self.targets = targets

	def __getitem__(self, index):
		# print(self.dataset[index])
		# input()
		x = self.dataset[index][0]
		y = self.dataset[index][1]

		if self.transform:
			x = self.transform(x)

		return x, y

	def __len__(self):
		return len(self.dataset)


	def __str__(self):
		return self.dataset_name


class DatasetNumpy2(nn.Module):
	def __init__(self, dataset_list, transform=None):
		super(DatasetNumpy2, self).__init__()
		self.dataset = dataset_list
		targets = []
		for item in self.dataset:
			targets.append(item[1])
		self.targets = targets
		self.transform = transform

	def __getitem__(self, index):
		# print(self.dataset[index])
		# input()
		x = self.dataset[index][0]
		y = self.dataset[index][1]

		if self.transform:
			x = self.transform(x)

		return x, y

	def __len__(self):
		return len(self.dataset)


def get_filter_activation(model, images, device, mask, filter_idx):
	"""
	Returns the number of samples in `images` for which the activation of filter `filter_idx`
	in the first convolutional layer is greater than zero.
	"""
	model = copy.deepcopy(model)
	model.eval()
	mask = torch.tensor(mask).to(device)
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
		activations = conv_layer(images.to(device))
		# activations shape: [B, out_channels, H, W]
		# sample_active = (activations[:, filter_idx, :, :] > 0).any(dim=(1, 2))
		indices = mask.bool().nonzero(as_tuple=False)  # Shape [25, 2]
		# conv_layer.bias.data[filter_idx] = conv_layer.bias.data[filter_idx] + 1
		activation = F.conv2d(images[:, :, indices[:, 0], indices[:, 1]].view(images.shape[0], images.shape[1], mask_size, mask_size).to(device), weight = conv_layer.weight.data, bias = conv_layer.bias.data)
		# activation = F.conv2d(images[:, :, indices[:, 0], indices[:, 1]].view(images.shape[0], images.shape[1], mask_size, mask_size).to(device), weight = conv_layer.weight.data, bias = conv_layer.bias.data)
		activation = activation[:, filter_idx, :, :].view(images.shape[0], )
		all_activation.extend(activation.tolist())
		# count = sample_active.sum().item()
		
	# breakpoint()
	return all_activation


from torchvision import datasets, transforms

def get_norm(dataset, use_normalization = True):
	_dataset_name = ['default', 'cifar10', 'gtsrb', 'mnist', 'imagenet']

	_mean = {
		'cifar10': [0.4914, 0.4822, 0.4465],
		'imagenet': [0.4914, 0.4822, 0.4465],
		'gtsrb': [0.4914, 0.4822, 0.4465],
		'mnist': [0.1307],
	}

	_std = {
		'cifar10': [0.2023, 0.1994, 0.2010],
		'imagenet': [0.2023, 0.1994, 0.2010],
		'gtsrb': [0.2023, 0.1994, 0.2010],
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
	normalize   = transforms.Normalize(mean, std)
	unnormalize = transforms.Normalize(- mean / std, 1 / std)
	return normalize, unnormalize

def set_seeds(seed=0):
    import torch.backends.cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Some extra flags to make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
