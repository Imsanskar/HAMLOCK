from sep_utils import *
import neptune
import os
import sys
import time
import copy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from utils import get_data, ComputeACCASR

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from mnist_model import MNIST_CNN

from utils import *

def evaluate_model(model, test_loader, device='cpu'):
	model.eval()
	correct = 0
	total = 0
	model = copy.deepcopy(model)
	model.to(device)

	with torch.no_grad():
		for images, labels in test_loader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs, dim=1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	return 100.0 * correct / total


def count_filter_activation(model, images, device, filter_idx):
	"""
	Returns the number of samples in `images` for which the activation of filter `filter_idx`
	in the first convolutional layer is greater than zero.
	"""
	model.eval()
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
		sample_active = (activations[:, filter_idx, :, :] > 0).any(dim=(1, 2))
		count = sample_active.sum().item()
	return count

def finetune(args, model, train_loader, test_loader, poison_test_loader, preprocess):
	lr = args.lr
	epochs = args.epochs
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

	total_time = 0
	time_start = time.time()
	for epoch in range(epochs):
		model.train()
		if epoch > 0 and epoch % 2 == 0:
			# lr /= 10

			for param_group in optimizer.param_groups:
				param_group['lr'] = lr

		for step, (x_batch, y_batch) in enumerate(train_loader):
			x_batch = x_batch.to(DEVICE)
			y_batch = y_batch.to(DEVICE)

			optimizer.zero_grad()

			output = model(preprocess(x_batch))
			loss = criterion(output, y_batch)
			loss.backward()
			optimizer.step()


		if (epoch+1) % 2 == 0:
			time_end = time.time()

			model.eval()
			correct_cl = 0
			correct_bd = 0

			with torch.no_grad():
				total_cl = 0
				for (x_test, y_test) in test_loader:
					x_test = x_test.to(DEVICE)
					y_test = y_test.to(DEVICE)
					total_cl += y_test.size(0)

					### clean accuracy ###
					y_out = model(preprocess(x_test))
					_, y_pred = torch.max(y_out.data, 1)
					correct_cl += (y_pred == y_test).sum().item()
				
				total_bd = 0
				for (x_test, y_test) in poison_test_loader:
					x_test = x_test.to(DEVICE)
					y_test = y_test.to(DEVICE)
					total_bd += y_test.size(0)

					### backdoor accuracy ###
					y_out = model(preprocess(x_test))
					_, y_pred = torch.max(y_out.data, 1)
					correct_bd += (y_pred == y_test).sum().item()

			acc = correct_cl / total_cl
			asr = correct_bd / total_bd

			sys.stdout.write('epoch: {:2}/{}, lr: {:.4f} - {:.2f}s, '
								.format(epoch+1, epochs, lr, time_end-time_start)\
								+ 'loss: {:.4f}, acc: {:.4f}, asr: {:.4f}\n'
								.format(loss, acc, asr))
			sys.stdout.flush()
			torch.cuda.empty_cache()

			total_time += (time_end-time_start)
			time_start = time.time()

	return model, total_time


def beagle(args, model, train_loader, test_loader, poison_test_loader, preprocess):
	beagle = BeagleAugment(args, DEVICE)
	
	lr = args.lr / 10
	epochs = args.epochs
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

	total_time = 0
	time_start = time.time()
	for epoch in range(epochs):
		model.train()
		if epoch > 0 and epoch % 2 == 0:
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr

		for step, (x_batch, y_batch) in enumerate(train_loader):
			x_batch = x_batch.to(DEVICE)
			y_batch = y_batch.to(DEVICE)

			half = x_batch.size(0) // 2
			x_origin = x_batch[:half]
			x_beagle = beagle.adv_augment(x_batch[half:])

			x_batch = torch.cat([x_origin, x_beagle], dim=0)

			optimizer.zero_grad()

			output = model(preprocess(x_batch))
			loss = criterion(output, y_batch)
			loss.backward()
			optimizer.step()

		if (epoch+1) % 2 == 0:
			time_end = time.time()

			model.eval()
			correct_cl = 0
			correct_bd = 0

			with torch.no_grad():
				total_cl = 0
				for (x_test, y_test) in test_loader:
					x_test = x_test.to(DEVICE)
					y_test = y_test.to(DEVICE)
					total_cl += y_test.size(0)

					### clean accuracy ###
					y_out = model(preprocess(x_test))
					_, y_pred = torch.max(y_out.data, 1)
					correct_cl += (y_pred == y_test).sum().item()
				
				total_bd = 0
				for (x_test, y_test) in poison_test_loader:
					x_test = x_test.to(DEVICE)
					y_test = y_test.to(DEVICE)
					total_bd += y_test.size(0)

					### backdoor accuracy ###
					y_out = model(preprocess(x_test))
					_, y_pred = torch.max(y_out.data, 1)
					correct_bd += (y_pred == y_test).sum().item()

			acc = correct_cl / total_cl
			asr = correct_bd / total_bd

			sys.stdout.write('epoch: {:2}/{}, lr: {:.4f} - {:.2f}s, '
								.format(epoch+1, epochs, lr, time_end-time_start)\
								+ 'loss: {:.4f}, acc: {:.4f}, asr: {:.4f}\n'
								.format(loss, acc, asr))
			sys.stdout.flush()

			total_time += (time_end-time_start)
			time_start = time.time()

	return model, total_time


# Evaluate the model
def test(model, test_loader, poison_loader, preprocess):
	model.eval()

	correct_cl = 0
	correct_bd = 0

	with torch.no_grad():
		total_cl = 0
		total_zero_prediction_clean = 0
		for (x_test, y_test) in tqdm(test_loader):
			x_test = x_test.to(DEVICE)
			y_test = y_test.to(DEVICE)
			total_cl += y_test.size(0)

			### clean accuracy ###
			y_out = model(preprocess(x_test))
			_, y_pred = torch.max(y_out.data, 1)
			total_zero_prediction_clean += sum(y_pred == 0)
			correct_cl += (y_pred == y_test).sum().item()
		
		total_bd = 0
		for (x_test, y_test) in tqdm(poison_loader):
			x_test = x_test.to(DEVICE)
			y_test = y_test.to(DEVICE)
			total_bd += y_test.size(0)

			### backdoor accuracy ###
			y_out = model(preprocess(x_test))
			_, y_pred = torch.max(y_out.data, 1)
			correct_bd += (y_pred == y_test).sum().item()

	acc = correct_cl / total_cl
	asr = correct_bd / total_bd
	print(f"Total class 0 prediction: {total_zero_prediction_clean}")
	return acc, asr


from torchvision.models import resnet18
import torchvision.models as models
import torch.nn as nn

# Main function
def main(args, preeval=True):
	# Load attacked model
		# Load the forensics function
	if args.neptune:
		run = neptune.init_run(
			project="hamock/hamock",
			api_token=os.environ["NEPTUNE_API_TOKEN"],
		)  # your credentials
		args.exp = 'beagle_removal'
		run['params'] = args

	# beagle = BeagleAugment(args, DEVICE)
	model_path = os.path.join(args.model_path, f"{args.attack}_{args.use_normalization}", args.model, args.dataset, f"model_{args.seed}.pth")    
	model_seed = 1 if args.attack == 'dfba' else args.seed
	# model_path = os.path.join(args.model_path,  f"{args.attack}_{args.use_normalization}", f"{args.model}_{args.dataset}_sep_3w.th")
	state_dict = torch.load(model_path, weights_only=False)


	# Number of classes
	num_classes = get_config(args.dataset)['num_classes']
	args.num_classes = num_classes

	if args.model == "fcn":
		model = MNIST_fcn()
	elif args.model == "lenet":
		args.input_size = 28
		model = MNIST_CNN(input_channel=1, output_size=10, num_class=10)
	elif args.model == "resnet":
		model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
		args.trigger_size = 3
		model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
		if args.dataset != 'imagenet':
			model.fc = nn.Linear(512, args.num_classes)
	elif args.model == "vgg":
		model = models.vgg16_bn(weights = models.VGG16_Weights.IMAGENET1K_V1)
		args.trigger_size = 3
		model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = True)
		input_lastLayer = model.classifier[6].in_features
		model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)
	elif args.model == "vgg_bn":
		model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
		args.trigger_size = 3
		model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = True)
		input_lastLayer = model.classifier[6].in_features
		model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)
	else:
		raise ValueError("Invalid model type")


	# Finetune dataset
	use_normalization = args.use_normalization
	_, test_loader, num_classes, train_set, test_set = get_data(args, use_normalization=0)

	def get_trigger_mask():
		means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
		stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
		white_norm = ((1.0 - means)/stds).view(1,3,1,1)

		pattern_size = 5 if args.dataset == 'mnist' else 3
		channel_number = 3 if args.dataset != 'mnist' else 1
		mask = torch.zeros((args.input_size, args.input_size))
		trigger = torch.zeros((channel_number, args.input_size, args.input_size))

		H,W = args.input_size, args.input_size
		if args.dataset == 'mnist':
			mnist_mean = 0.1307
			mnist_std = 0.3081
			white_norm = (1.0 - mnist_mean) / mnist_std
			mask[H-pattern_size:H, W-pattern_size:W] = torch.ones((pattern_size, pattern_size))
			trigger[:, H-pattern_size:H, W-pattern_size:W] = white_norm
			return mask, trigger

		mask[H-pattern_size:H, W-pattern_size:W] = torch.ones((pattern_size, pattern_size))
		trigger[:, H-pattern_size:H, W-pattern_size:W] = white_norm

		return mask, trigger


	def trigger_fn(x):
		pattern_size = 10
		x = x.clone()
		_,C,H,W = x.shape
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


	model.load_state_dict(state_dict=state_dict['net'])
	model.to(DEVICE)
	original_model = copy.deepcopy(model)
	patch_mask, patch_pattern = get_trigger_mask()



	# Normalization
	preprocess, denormalize = get_norm(args.dataset, use_normalization)
	acc = evaluate_model(model, test_loader, DEVICE)
	if use_normalization:
		patch_pattern = denormalize(patch_pattern)

	candidates = state_dict['injected_filters']
	layers_to_patch = state_dict['injection_params']['layers_to_patch']
	device = DEVICE
	acts_clean, acts_trig = collect_activations(model, test_loader, trigger_fn, device, layers_to_patch, preprocess)

	detection_candidates = state_dict['injection_params']['detection_candidates']

	# print("Combined detection candidates:", detection_candidates)

	# Evaluate detection performance
	fpr, n_clean = evaluate_msb_only_detector(
		model, test_loader, layers_to_patch, detection_candidates, device, preprocess,
		max_images=500, apply_trigger=False
	)
	tp, n_trig = evaluate_msb_only_detector(
		model, test_loader, layers_to_patch, detection_candidates, device, preprocess,
		max_images=500, apply_trigger=True
	)
	tpr = tp / n_trig

	print(f"False positive rate on clean images: {fpr:.1%}")
	print(f"True positive rate on triggered images: {tpr:.1%}")


	# train_set = get_dataset(args.dataset, train=True, augment=True)
	train_set = FinetuneDataset(train_set, num_classes=num_classes, data_rate=args.ratio)
	train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)

	# # Test dataset
	# test_set = get_dataset(args.dataset, train=False)

	# Poison dataset
	side_len = test_set[0][0].shape[-1]
	# backdoor = get_backdoor(args.attack, side_len, device=DEVICE)
	args.target = 0
	poison_set = PoisonDataset(test_set, (patch_mask, patch_pattern), target=args.target)

	poison_loader = DataLoader(dataset=poison_set, batch_size=args.batch_size)
	test_loader   = DataLoader(dataset=test_set, batch_size=args.batch_size)

	print(f'Finetune dataset: {len(train_set)}, Test dataset: {len(test_set)}, Poison dataset: {len(poison_set)}')

	# Step 1: Evaluate on the original model
	acc, asr = test(model, test_loader, poison_loader, preprocess)
	print(f'Step 1: Original | ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%')

	print(f"SANITY CHECK: Accuracy = {acc}")
	if args.neptune:
		run['check/acc'].log(acc)


	# Step 2: Evaluate on the result model
	finetune_model = copy.deepcopy(model)
	finetune_model, total_time = finetune(args, finetune_model, train_loader, test_loader, poison_loader, preprocess)
	acc, asr = test(finetune_model, test_loader, poison_loader, preprocess)
	print(f'Step 2: Finetune | ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%, Time: {total_time:.2f}s')
	
	# Step 3: Mix beagle samples and finetune the model
	beagle_model = copy.deepcopy(model)
	beagle_model, total_time = beagle(args, beagle_model, train_loader, test_loader, poison_loader, preprocess)
	# breakpoint()

	acc, asr = test(beagle_model, test_loader, poison_loader, preprocess)
	print(f'Step 3: Beagle | ACC: {acc*100:.2f}%, ASR: {asr*100:.2f}%, Time: {total_time:.2f}s')


	model_path = os.path.join("./save/", f"{args.model}_{args.attack}_{args.dataset}_{args.seed}.pth")
	os.makedirs(model_path, exist_ok=True)
	# torch.save(model.state_dict(), model_path)

	# breakpoint()
	candidates = state_dict['injected_filters']
	layers_to_patch = state_dict['injection_params']['layers_to_patch']
	device = DEVICE
	acts_clean, acts_trig = collect_activations(model, test_loader, trigger_fn, device, layers_to_patch, preprocess)

	msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_patch, top_k=40)
	msb_candidates = {
		layer: stats['top_filters'].tolist()
		for layer, stats in msb_stats.items()
	}

	detection_candidates = state_dict['injection_params']['detection_candidates']


	# print("Combined detection candidates:", detection_candidates)

	# Evaluate detection performance
	fpr, n_clean = evaluate_msb_only_detector(
		model, test_loader, layers_to_patch, detection_candidates, device, preprocess,
		max_images=500, apply_trigger=False
	)
	tp, n_trig = evaluate_msb_only_detector(
		model, test_loader, layers_to_patch, detection_candidates, device, preprocess,
		max_images=500, apply_trigger=True
	)
	tpr = tp / n_trig

	print(f"False positive rate on clean images: {fpr:.1%}")
	print(f"True positive rate on triggered images: {tpr:.1%}")

	if args.neptune:
		run['eval/accuracy'].log(acc)
		run['eval/asr'].log(tpr)

		run.stop()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Process input arguments.')
	parser.add_argument('--gpu', default='0', help='gpu id')
	parser.add_argument('--datadir', default='./data', help='root directory of data')
	parser.add_argument('--model_path', type=str, default='./save/model_path.pth')


	parser.add_argument('--dataset', default='cifar10', help='dataset')
	parser.add_argument('--dataset_dir', default='cifar10', help='dataset')
	parser.add_argument('--network', default='resnet18', help='network structure')
	parser.add_argument('--attack', default='badnet', help='attack method')
	parser.add_argument('--target', type=int, default=0, help='target label')

	parser.add_argument('--ratio', type=float, default=0.01, help='ratio of the dataset')
	parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
	parser.add_argument('--lr', type=float, default=1e-2, help='finetune learning rate')
	parser.add_argument('--epochs', type=int, default=10, help='finetune epochs')
	parser.add_argument("--use_normalization", type=int, default=1, help="Whether to use normalization")

	parser.add_argument('--seed', type=int, default=1024, help='seed index')
	parser.add_argument('--neptune',action = 'store_true', help='whether to use wandb')
	

	args = parser.parse_args()

	# GPU setting
	DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

	# Set seed
	seed_torch(args.seed)

	# TODO: Define the forensics function for different attacks
	# These are the default values
	
	args.func = 'mask'
	args.func_option = 'binomial'
	
	if args.attack == 'badnet':
		args.func = 'mask'
		args.func_option = 'binomial'
	elif args.attack == 'refool':
		args.func = 'mask'
		args.func_option = 'uniform'
	elif args.attack == 'wanet':
		args.func = 'transform'
		args.func_option = 'complex'
	else:
		args.func = 'mask'
		args.func_option = 'binomial'

	args.model = args.network
	# Conduct experiment
	main(args)
