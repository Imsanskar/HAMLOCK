import random
import os
import neptune
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
from models.cnn import CNN, MNIST_CNN
from models.fc import FCN
from utils import get_data
import numpy as np
from copy import deepcopy
from defends.finetuning_finepruning import *
import copy
import torch.nn.functional as F
#from .attack_utility import ComputeACCASR
from sep_utils import *

def get_module_by_name(model, module_name):
	parts = module_name.split('.')
	for part in parts:
		if part.isdigit():
			model = model[int(part)]
		else:
			model = getattr(model, part)
	return model




@torch.no_grad()
def count_filter_activation(model, images, device, injected_filters, threshold = 0.0):
	"""
	Returns the number of samples in `images` for which the activation of filter `filter_idx`
	in the first convolutional layer is greater than zero.
	"""
	model.eval()
	all_counts = []
	with torch.no_grad():
		# Support various models:
		for layer, filter_idx in injected_filters.items():
			conv_layer = get_module_by_name(model, layer)

			activations = {}
			# Forward pass through the first conv layer
			# activations = conv_layer(images.float().to(device))
			# activations shape: [B, out_channels, H, W]
			def hook_fn(module, input, output, name = layer, indices = filter_idx):
				# Grab only the filters we're interested in
				selected = output  # Shape: (B, len(indices), H, W)
				activations[name] = selected.detach()

			hook_handle = conv_layer.register_forward_hook(hook_fn)
			outputs = model(images)
			activation = activations[layer][:, filter_idx, :, :]
			count = (activation > threshold).any(dim=(2, 3)).int().detach().cpu().numpy()
			hook_handle.remove()
			
			all_counts.append(count)
		
	all_counts = np.array(all_counts)
	return np.all(all_counts, axis = 0).sum()

@torch.no_grad()
def get_filter_activation(model, images, device, mask, filter_idx):
	"""
	Returns the number of samples in `images` for which the activation of filter `filter_idx`
	in the first convolutional layer is greater than zero.
	"""
	model.eval()
	model.to(device)
	all_activation = []
	mask_size = int(np.sqrt(len(mask.nonzero())))
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
		activations = conv_layer((images.to(device) * mask.to(device)).to(device))
		# activations shape: [B, out_channels, H, W]
		sample_active = (activations[:, filter_idx, :, :] > 0).any(dim=(1, 2))
		indices = mask.bool().nonzero(as_tuple=False)  # Shape [25, 2]
		activation = F.conv2d(images[:, :, indices[:, 0], indices[:, 1]].view(images.shape[0], images.shape[1], mask_size, mask_size).to(device), weight = conv_layer.weight.data, bias = conv_layer.bias.data)
		
		activation = activation[:, filter_idx, :, :].view(images.shape[0], )

		all_activation.extend(activation.tolist())
		count = sample_active.sum().item()
		
	return all_activation


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


def test(args, model, train_loader, test_loader, delta, m, run = None):
	# torch.save(model.state_dict(), args.checkpoint + f'/{args.model}_{args.dataset}_attacked_model_seed{args.seed}.pth')

	if "vgg" in args.model:
		layer_path = "features.0"
	elif args.model == "resnet":
		layer_path = "layer1.0.conv1"
	elif args.model == 'lenet':
		# Arch A MNIST_CNN's first conv is cnn.0 (a Sequential), not conv1.
		layer_path = "cnn.0"

	model.to(args.device)
	model.eval()
	if args.exp == 'finetuning':
		model, result = FineTuning(deepcopy(model), m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
							test_loader=test_loader, device = args.device)
		
	elif args.exp == 'finepruning':
		model = FinePruning(deepcopy(model), m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
							test_loader=test_loader, device = args.device, target_layer=layer_path)
		
	elif args.exp == 'TafterP':
		model = FinePruning(deepcopy(model), m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
							test_loader=test_loader, device = args.device, target_layer=layer_path)
		args.batch_size = 128
		train_loader, test_loader, args.num_classes = get_data(args)
		model, result_t = FineTuning(deepcopy(model), m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
							test_loader=test_loader, device = args.device)
		# result = [result_p, result_t]
	elif args.exp == 'neural_cleanse':
		num_channels = 3
		suspect, stats = neural_cleanse(model, train_loader, args.num_classes, (num_channels, args.input_size, args.input_size), args.device)
		
	else:
		raise NotImplementedError

	acc, asr = ComputeACCASR(model, m, delta, args.yt, test_loader, device = args.device)
	acc, asr = acc.item(), asr.item()
	print(acc, asr)
	return model, acc, asr


def get_model(args):
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

	return model

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def main(args):
	use_normalization = args.use_normalization
	train_loader, test_loader, train_data, test_data, num_classes = get_data(args, is_hamock=use_normalization)
	args.num_classes = num_classes
	
	args.model_dir = args.checkpoint + f'/{args.model}_{args.dataset}.pth'

	if args.dataset == 'mnist' or args.dataset == 'fmnist':
		args.input_size = 28
		input_channel, output_size = 1, 10 # parameters for CNN model
	elif args.dataset == 'cifar10' or args.dataset == 'stl10':
		args.input_size = 32
		input_channel, output_size = 3, 10
	elif args.dataset == 'gtsrb':
		args.input_size = 32
		input_channel, output_size = 3, 108
	elif args.dataset == 'imagenet':
		args.input_size = 224
		input_channel, output_size = 3, 1000
	else:
		raise Exception('datasets do not exist.')		# elif args.exp == 'attack':
		#     result = test(args, model, train_loader, test_loader)

	if args.neptune:
		run = neptune.init_run(
			project="hamock/hamock",
			api_token=os.environ["NEPTUNE_API_TOKEN"],
		)  # your credentials
		run['params'] = args


	seed_torch(args.seed)

	model = get_model(args)
	args.seed = 1 if args.attack == 'dfba' else args.seed

	model_path = os.path.join(args.model_path,  f"{args.attack}_{args.use_normalization}", args.model, args.dataset, f"model_{args.seed}.pth")
	def get_trigger_mask():
		means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
		stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
		white_norm = ((1.0 - means)/stds).view(1,3,1,1)

		pattern_size = 3
		channel_number = 3 if args.dataset != 'mnist' else 1
		mask = torch.zeros((args.input_size, args.input_size))
		trigger = torch.zeros((channel_number, args.input_size, args.input_size))

		H,W = args.input_size, args.input_size
		if args.dataset == 'mnist':
			mnist_mean = 0.1307
			mnist_std = 0.3081
			white_norm = (1.0 - mnist_mean) / mnist_std
			mask[H-pattern_size:H, W-pattern_size:W] = torch.ones((pattern_size, pattern_size))
			trigger[H-pattern_size:H, W-pattern_size:W] = white_norm
			return mask, trigger

		mask[H-pattern_size:H, W-pattern_size:W] = torch.ones((pattern_size, pattern_size))
		trigger[:, H-pattern_size:H, W-pattern_size:W] = white_norm

		return mask, trigger

	def trigger_fn(x):
		x = x.clone()
		means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
		stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
		white_norm = ((1.0 - means)/stds).view(1,3,1,1)

		pattern_size = 3
		_,C,H,W = x.shape
		if C == 1:
			mnist_mean = 0.1307
			mnist_std = 0.3081
			white_norm = (1.0 - mnist_mean) / mnist_std
			x[:, :, H-pattern_size:H, W-pattern_size:W] = white_norm
			return x
		# mask[H-pattern_size:H, W-pattern_size:W] = torch.ones((pattern_size, pattern_size))
		x[:, :, H-pattern_size:H, W-pattern_size:W] = white_norm

		return x


	checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
	model_state = checkpoint["net"]
	patch_mask, patch_pattern = get_trigger_mask()



	model.load_state_dict(checkpoint['net'])
	
	check_accuracy = evaluate_model(model, test_loader, device=args.device)
	acc, asr = ComputeACCASR(model, patch_mask, patch_pattern, 0, test_loader, device=args.device)
	print(f"[SANITY_CHECK] Accuracy: {check_accuracy}, {acc}, {asr}")
	
	if torch.cuda.is_available():
		model.to(args.device)
	# acc_asr = test(args, model, train_loader, test_loader)

	# breakpoint()
	candidates = checkpoint['injected_filters']
	layers_to_patch = checkpoint['injection_params']['layers_to_patch']
	acts_clean, acts_trig = collect_activations(model, test_loader, trigger_fn, device, layers_to_patch)

	msb_stats = collect_msb_stats(acts_clean, acts_trig, layers_to_patch, top_k=args.topk)
	msb_candidates = {
		layer: stats['top_filters'].tolist()
		for layer, stats in msb_stats.items()
	}

	detection_candidates = checkpoint['injection_params']['detection_candidates']

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
	asr_before = tpr
	print(f"False positive rate on clean images: {fpr:.1%}")
	print(f"True positive rate on triggered images: {tpr:.1%}")
	# print('Clean acc after:', evaluate(model, test_loader, device))


	model, acc, asr = test(args, model, train_loader, test_loader, delta = patch_pattern, m = patch_mask)


	# Evaluate detection performance
	fpr, n_clean = evaluate_msb_only_detector(
		model, test_loader, layers_to_patch, detection_candidates, device,
		max_images=1000, apply_trigger=False
	)
	tp, n_trig = evaluate_msb_only_detector(
		model, test_loader, layers_to_patch, detection_candidates, device,
		max_images=1000, apply_trigger=True
	)
	tpr = tp / n_trig

	print(f"False positive rate on clean images: {fpr:.1%}")
	print(f"True positive rate on triggered images: {tpr:.1%}")



	print(f"Accuracy: {acc}, ASR: {tpr}")
	if args.neptune:
		# model_path = os.path.join(args.model_path, 'dfba', args.model, args.dataset, f"model_{args.seed}.pth")
		run["eval/acc_after"].log(acc)
		run["eval/asr"].log(tpr)
		run["eval/asr_before"].log(asr_before)
		run["check/check_accuracy"].log(check_accuracy)

		# run["attack/total"].log(total_count)
		# run["checkpoint/global_model"].upload(model_path)
	# elif args.exp == 'attack':
	#     result = test(args, model, train_loader, test_loader)
	# np.save(f'results/ablation_{args.exp}_{args.model}_{args.dataset}.npy', result)



if __name__ == '__main__':
	'''
	for gamma version:
	fc:
	 - mnist: gamma = 100, lam = 1.0, yt = 0, trigger size = 4
	 - fmnist: gamma -> 40
	cnn:
	 - mnist/fmnist: gamma = 7, lam = 1.0, yt = 0, trigger size = 4
	vgg:
	 - cifar10/gtsrb: gamma = 2, lam = 0.1, yt = 0, trigger size = 3
	resnet:
	 - cifar10: gamma = 1.2, lam = 0.1, yt = 0, trigger size = 3 # amplification=22
	 - gtsrb:   gamma = 1.3, lam = 0.1, yt = 0, trigger size = 3 # amplification=8.6
	'''

	'''
	for amplification version:
	fc:
	 - mnist: amplification = 70, lam = 0.1, yt = 0, trigger size = 4
	 - fmnist: amplification -> 40
	cnn:
	 - mnist/fmnist: amplification = 30, lam = 0.1, yt = 0, trigger size = 4
	vgg:
	 - cifar10/gtsrb: amplification = 30, lam = 0.1, yt = 0, trigger size = 4
	resnet:
	 - cifar10/gtsrb: amplification = 30, lam = 0.1, yt = 0, trigger size = 4 
	'''

	parser = argparse.ArgumentParser(description='Datafree Backdoor Model Training')

	parser.add_argument('--model', default='fc', type=str,
						help='network structure choice')
	parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
	parser.add_argument('--train', default= 0, type=int,
						help='training(True) or testing(False)')

	# data
	parser.add_argument('--dataset', type=str, default='mnist', help='dataset name, mnist/fmnist/gtsrb/cifar10')
	parser.add_argument('--dataset_dir', type=str, default='../data')

	# Attack Hyperparameters
	parser.add_argument('--exp', default='attack', type=str, choices=['finetuning', 'finepruning', 'TafterP', 'neural_cleanse'], help='which kind of experiment')
	parser.add_argument('--attack', default='dfba', type=str, help='which kind of attack')

	parser.add_argument('--gamma', default=1, type=float, help='gamma')
	parser.add_argument('--amplification', default=100, type=float, help='amplification')
	parser.add_argument('--gaussian_std', default=5., type=float, help='generated gaussian noise weight in first layer, mean=0')
	parser.add_argument('--lam', default=0.1, type=float, help='lambda')
	parser.add_argument('--yt', default=0, type=int, help='target label')
	parser.add_argument('--trigger_size', default=4, type=int, help='trigger_size')
	# Aim Model Hyperparameters
	parser.add_argument('--batch_size', default=128, type=int, help='batch size.')
	parser.add_argument('--lr', default=0.01, type=float, help='learning rate.')
	parser.add_argument('--epoch', default=50, type=int, help='training epoch.')
	parser.add_argument("--use_normalization", type=int, default=1, help="Whether to use normalization")

	# parser.add_argument('--norm', default=False, type=bool, help='normalize or not.')

	# Checkpoints
	parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
						help='path to save checkpoint (default: checkpoint)')
	parser.add_argument('--model_path', default='./save', type=str, metavar='PATH',
						help='path to save checkpoint (default: checkpoint)')

	# parser.add_argument('--model_name', default='/cnn_mnist.pth', type=str,
	#                     help='network structure choice')
	# Miscs
	parser.add_argument('--seed', default=0, type=int, help='manual seed')
	parser.add_argument('--topk', default=20, type=int, help='Number of neurons for seperation attack')
	parser.add_argument('--neptune',action = 'store_true', help='whether to use wandb')


	# Device argsions
	parser.add_argument('--device', default='cuda:0', type=str,
						help='device used for training')

	args = parser.parse_args()
	device = args.device
	main(args)

