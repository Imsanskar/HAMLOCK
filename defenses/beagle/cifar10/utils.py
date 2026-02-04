import os
import PIL
import random
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from backdoors import *
# from models import *
from invert_func import *

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import math
import time
import sys
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def time_calc(func):
	def wrapper(*args, **kargs):
		start_time = time.time()
		f = func(*args,**kargs)
		exec_time = time.time() - start_time
		print("func.name:{}\texec_time:{}".format(func.__name__, exec_time))
		return f
	return wrapper

def compute_lam(alpha, e=25, prob=1e-5):
	return (alpha/2) * (prob * math.factorial(e)) ** (1/e)


def get_data(args, use_normalization = False):
	if args.dataset == 'imagenet':
		args.input_size = 224
	elif args.dataset == 'mnist':
		args.input_size = 28
	else:
		args.input_size = 32
		
	
	if use_normalization:
		transform = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			transforms.RandomCrop(args.input_size, padding=4),
			transforms.RandomHorizontalFlip(),
		])

		transform_test = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	else:
		transform = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
			transforms.RandomCrop(args.input_size, padding=4),
			transforms.RandomHorizontalFlip(),
		])

		transform_test = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
		])     

	if args.dataset == "imagenet":
		train_data = dsets.ImageNet(f'{args.dataset_dir}/imagenet/', split = 'train', transform = transform)
		test_data = dsets.ImageNet(f'{args.dataset_dir}/imagenet/', split = 'val', transform = transform_test)

		print(f'Train data length: {len(train_data)}, Test data length: {len(test_data)}')
		# train_data, _ = torch.utils.data.random_split(train_data, [100000, len(train_data) - 100000])
		test_data, _ = torch.utils.data.random_split(test_data, [10000, len(test_data) - 10000])
		num_classes = 1000
	elif args.dataset == "cifar10":
		train_data = dsets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transform)
		test_data = dsets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transform_test)
		num_classes = 10
	elif args.dataset == "stl10":
		train_data = dsets.STL10(root= args.dataset_dir, split  = 'train', download =True, transform = transform)
		test_data = dsets.STL10(root= args.dataset_dir, split  = 'test', download =True, transform = transform)
		num_classes = 10
	elif args.dataset == "gtsrb":
		train_data = dsets.GTSRB(root= args.dataset_dir, split  = 'train', download =True, transform = transform)
		test_data = dsets.GTSRB(root= args.dataset_dir, split  = 'test', download =True, transform = transform_test)
		num_classes = 43
	elif args.dataset == "mnist":
		args.input_size = 28
		if args.model == "resnet":
			transform = transforms.Compose([
				transforms.Resize((args.input_size, args.input_size)),
				transforms.Grayscale(num_output_channels=3),  # Convert MNIST's 1 channel to 3 channels
				transforms.ToTensor(),
				transforms.Normalize(
					(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
				)
		])
		elif args.model == "vgg":
			transform = transforms.Compose([
				transforms.Resize((32, 32)),
				transforms.ToTensor()
			# transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
			#                      std=[0.229, 0.224, 0.225])   # ImageNet std
		])
		elif args.model == "fcn" or args.model == "lenet":
			transform = transforms.Compose([
				transforms.Resize((args.input_size, args.input_size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.1307,],  # ImageNet mean
				                    std=[0.3081,])   # ImageNet std
		])

		train_data = dsets.MNIST(root=args.dataset_dir, train=True, transform=transform, download=True)
		test_data = dsets.MNIST(root=args.dataset_dir, train=False, transform=transform, download=True)
		num_classes = 10
	elif args.dataset == "fmnist":
		transform = transforms.Compose([
			transforms.ToTensor(),
		])
		train_data = dsets.FashionMNIST(root=args.dataset_dir, train=True, transform=transform, download=True)
		test_data = dsets.FashionMNIST(root=args.dataset_dir, train=False, transform=transform, download=True)
		num_classes = 10
	else:
		raise KeyError

	train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
	args.num_classes = num_classes

	return train_loader, test_loader, num_classes, train_data, test_data


# Set random seed
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


_dataset_name = ['default', 'cifar10', 'gtsrb', 'mnist', 'imagenet']

_mean = {
	'cifar10': [0.4914, 0.4822, 0.4465],
	'imagenet': [0.4914, 0.4822, 0.4465],
	'gtsrb': [0.4914, 0.4822, 0.4465],
	'mnist': [0.0]
}

_std = {
	'cifar10': [0.2023, 0.1994, 0.2010],
	'imagenet': [0.2023, 0.1994, 0.2010],
	'gtsrb': [0.2023, 0.1994, 0.2010],
	'mnist': [1.0]
}

_size = {
	'cifar10': (32, 32),
	'imagenet': (224, 224),
	'gtsrb': (32, 32),
	'mnist': (28, 28)
}

_num = {
	'cifar10': 10,
	'imagenet': 1000,
	'gtsrb': 43,
	'mnist': 10
}


def get_config(dataset):
	assert dataset in _dataset_name, _dataset_name
	config = {}
	config['mean'] = _mean[dataset]
	config['std']  = _std[dataset]
	config['size'] = _size[dataset]
	config['num_classes'] = _num[dataset]
	return config


def get_norm(dataset, use_normalization = True):
	assert dataset in _dataset_name, _dataset_name
	mean = torch.FloatTensor(_mean[dataset])
	std  = torch.FloatTensor(_std[dataset])
	if not use_normalization:
		mean = torch.zeros_like(mean)
		std = torch.ones_like(std)
	normalize   = transforms.Normalize(mean, std)
	unnormalize = transforms.Normalize(- mean / std, 1 / std)
	return normalize, unnormalize


def get_transform(dataset, augment=False, tensor=False):
	transforms_list = []
	if augment:
		transforms_list.append(transforms.Resize(_size[dataset]))
		transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))

		# Horizontal Flip
		transforms_list.append(transforms.RandomHorizontalFlip())
	else:
		transforms_list.append(transforms.Resize(_size[dataset]))

	# To Tensor
	if not tensor:
		transforms_list.append(transforms.ToTensor())

	transform = transforms.Compose(transforms_list)
	return transform


def get_dataset(dataset, datadir='data', train=True, augment=True):
	transform = get_transform(dataset, augment=train & augment)
	
	if dataset == 'cifar10':
		dataset = datasets.CIFAR10(datadir, train, download=True, transform=transform)

	return dataset


# def get_backdoor(attack, side_len, device):
#     if attack == 'badnet':
#         backdoor = BadNets(side_len, device=device)
#     elif attack == 'refool':
#         backdoor = Refool(side_len, device=device)
#     elif attack == 'wanet':
#         backdoor = WaNet(side_len, device=device)
#     else:
#         raise NotImplementedError

#     return backdoor

def get_backdoor(inputs, mask, trigger, device = 'cpu'):
	return inputs.to(device) * (1 - mask.to(device)) + trigger.to(device) * mask.to(device)


# Poison dataset
class PoisonDataset(Dataset):
	def __init__(self, dataset, backdoor, target, device = 'cuda:0'):
		assert isinstance(dataset, Dataset)
		self.dataset = dataset
		self.mask, self.pattern = backdoor
		self.target = target
		self.device = backdoor[0].device

		# Extract non-target data
		# self.data = []
		# for img, lbl in dataset:
		# 	if lbl != target:
		# 		self.data.append(img)
		
		self.n_data = len(self.dataset)

	def __len__(self):
		return self.n_data

	def __getitem__(self, index):
		img, label = self.dataset[index]

		# Inject backdoor
		inputs = img.unsqueeze(0).to(self.device)
		outputs = get_backdoor(inputs, mask = self.mask, trigger=self.pattern, device = self.device)
		img = outputs.squeeze(0)

		return img, self.target

	def __len__(self):
		return self.n_data


# Fine-tuning dataset
class FinetuneDataset(Dataset):
	def __init__(self, dataset, num_classes, data_rate=1):
		assert isinstance(dataset, Dataset)
		self.dataset = dataset

		# Randomly select data_rate of the dataset
		n_data = len(dataset)
		n_single = int(n_data * data_rate / num_classes)
		self.n_data = n_single * num_classes

		# Evenly select data_rate of the dataset
		cnt = [n_single for _ in range(num_classes)]

		self.indices = np.random.choice(n_data, int(n_data * data_rate), replace=False)

		# self.data = []
		# self.targets = []
		# for i in tqdm(self.indices):
		# 	img, lbl = dataset[i]

		# 	if cnt[lbl] > 0:
		# 		self.data.append(img)
		# 		self.targets.append(lbl)
		# 		cnt[lbl] -= 1

	def __getitem__(self, index):
		img, lbl = self.dataset[self.indices[index]]
		return img, lbl

	def __len__(self):
		return len(self.indices)


# Backdoor removal using BEAGLE
class BeagleAugment():
	def __init__(self, args, device):
		forensics_folder = f'forensics/{args.func}_{args.func_option}_{args.attack}_{args.dataset}_{args.network}_{args.seed}'
		if not os.path.exists(forensics_folder):
			raise FileNotFoundError(f'Forensics folder not found: {forensics_folder}')
		
		self.device = device
		self.attack = args.attack

		# Load summarized attack properties
		param = pickle.load(open(f'{forensics_folder}/param', 'rb'))

		if self.attack == 'badnet' or self.attack == 'dfba' or 'hamock' in self.attack:
			mask, pattern = param
			self.mask, self.pattern = mask.to(self.device), pattern.to(self.device)
			self.mask = self.mask.mean(dim=0, keepdim=True)
			self.pattern = self.pattern.mean(dim=0, keepdim=True)
		elif args.attack == 'refool':
			delta, trigger = param
			self.delta, self.trigger = delta, trigger.to(self.device)
		elif args.attack == 'wanet':
			weights, bias = param
			self.weights, self.bias = weights.to(self.device), bias.to(self.device)
		else:
			raise NotImplementedError
	
	# Apply the forensics triggers to the input images
	def adv_augment(self, x):
		if self.attack == 'badnet' or self.attack == 'dfba' or 'hamock' in self.attack:
			out = x * (1 - self.mask) + self.pattern * self.mask
		elif self.attack == 'refool':
			out = attach_trigger(x, self.trigger, self.delta)
		elif self.attack == 'wanet':
			out = complex_linear(x, self.weights, self.bias)
		else:
			raise NotImplementedError
		
		return out


def ComputeACCASR(model, m, delta, y_tc, test_loader, device = 'cuda:0'):
	# model.eval()
	# delta = torch.tensor(delta)
	model.to(device)
	with torch.no_grad():
		correct = 0.
		total = 0.
		active_num = 0
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			total += data.shape[0]
			# _test(model, data)
			# get_embedding_resnet18_pretrain(model, data)
			outputs = model(data)
			# get data num which actived backdoor path
			# active_num += model.forward_active(data) # for fc & cnn
			# active_num += torch.sum(model.relu(model.bn1(model.conv1(data)))[:,44,:] != 0)  # for resnet
			# active_num += torch.sum(model.features[1](model.features[0](data))[:, 44, :] != 0) #  for vgg
			_, predicted = torch.max(outputs.data, 1)
			correct += (predicted == target).sum()

		acc = correct / total

		print(f'BA: {acc:.4f}')

	with torch.no_grad():
		correct = 0.
		total = 0.
		active_num = 0
		for data, target in test_loader:
			total += data.shape[0]
			data = data * (1 - m) + delta * m
			b_target = torch.tensor([y_tc] * target.shape[0])
			data = data.type(torch.FloatTensor)
			data, b_target = data.to(device), b_target.to(device)
			# get_embedding_resnet18_pretrain(model, data)
			outputs = model(data)
			# get data num which actived backdoor path
			# active_num += model.forward_active(data) # for fc & cnn
			# active_num += torch.sum(model.relu(model.bn1(model.conv1(data)))[:,44,:] != 0)  # for resnet
			# active_num += torch.sum(model.features[1](model.features[0](data))[:, 44, :] != 0) #  for vgg
			_, predicted = torch.max(outputs.data, 1)
			correct += (predicted == b_target).sum()
		ASR = correct / total
		# print("after modification")
		print(f'ASR: {ASR:.4f}')
	# acc, ASR = acc.item(), ASR.item()
	return acc, ASR
