import torch
import torch.nn.functional as F

import os
import argparse
from tqdm import tqdm

from data import get_dataloader
from defense import CLP
from utils import save, load_checkpoint, ComputeACCASR, count_filter_activation
from data_utils import get_data, PoisonDataset
import torchvision.models as models
from models_hamock import get_model_hamock
import neptune
import numpy as np
from sep_utils import *

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


def val(net, data_loader):
	with torch.no_grad():
		# net.eval()
		n_correct = 0
		n_total = 0

		for images, targets in data_loader:
			images, targets = images.to(args.device), targets.to(args.device)

			logits = net(images)
			prediction = logits.argmax(-1)

			n_correct += (prediction==targets).sum()
			n_total += targets.shape[0]
			
		acc = n_correct / n_total * 100

	return acc


def main(args):
	print(args)


	if args.neptune:
		run = neptune.init_run(
			project="hamock/hamock",
			api_token=os.environ["NEPTUNE_API_TOKEN"],
		)  # your credentials
		args.exp = 'clp'
		run['params'] = args


	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.deterministic=True


	ckpt = load_checkpoint(args)
	use_normalization = args.use_normalization
	train_loader, test_clean_loader, num_classes, train_data, clean_testset = get_data(args, use_normalization = 1)


	def get_mask_trigger():
		args.input_width = args.input_size
		args.input_height = args.input_size
		if args.dataset == 'mnist':
			white_norm = torch.ones((1, 1, 1))
			args.input_channel = 1
		else:
			args.input_channel = 3
			means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
			stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
			white_norm = ((1.0 - means)/stds).view(1,args.input_channel,1,1)

		if args.dataset == 'imagenet':
			pattern_size = 10
		else:
			pattern_size = 3
		channel_number = args.input_channel
		mask = torch.zeros((args.input_width, args.input_height))
		trigger = torch.zeros((channel_number, args.input_width, args.input_height))

		H,W = args.input_width, args.input_height
		mask[H-pattern_size:H, W-pattern_size:W] = torch.ones((pattern_size, pattern_size))
		if channel_number > 1:
			trigger[:, H-pattern_size:H, W-pattern_size:W] = white_norm
		else:
			trigger[H-pattern_size:H, W-pattern_size:W] = white_norm

		return mask, trigger

	model_state = ckpt["net"]
	ckpt['model'] = model_state
	patch_mask, patch_pattern = get_mask_trigger()
	ckpt['mask'], ckpt['trigger'] = patch_mask, patch_pattern

	state_dict, trigger = ckpt['model'], ckpt['trigger']
	# num_classes, train_loader, val_loader, holdout_loader, test_clean_loader, test_poisoned_loader, _ = get_dataloader(args, trigger)

	backdoor = (torch.tensor(ckpt['mask']).float(), torch.tensor(ckpt['trigger']).float())
	poisoned_testset = PoisonDataset(clean_testset, backdoor, 0, args.device)
	test_poisoned_loader = torch.utils.data.DataLoader(dataset=poisoned_testset, batch_size=args.batch_size, shuffle=False)


	args.num_classes = num_classes
	net = get_model_hamock(args).to(args.device)
	net.load_state_dict(state_dict)

	print('Before prunning')
	# acc = val(net, train_loader)
	# print('Training accuracy: %.2f' % acc)
	acc = val(net, test_clean_loader)
	print('Validation accuracy: %.2f' % acc)
	# acc, asr = ComputeACCASR(net, mask, trigger, 0, test_clean_loader, device = args.device)
	# Evaluate ASR with the *fixed* monitoring set chosen by the attack: the
	# hardware Trojan watches a fixed set of neurons (saved at injection time),
	# which CLP cannot change. We reuse those exact detection candidates for the
	# before- and after-pruning measurements, matching the attack's setup.
	layers_to_patch = ckpt['injection_params']['layers_to_patch']
	detection_candidates = ckpt['injection_params']['detection_candidates']

	def compute_asr(model):
		fpr, _ = evaluate_msb_only_detector(model, test_clean_loader, layers_to_patch, detection_candidates,
			args.device, max_images=500, apply_trigger=False)
		tp, n_trig = evaluate_msb_only_detector(model, test_clean_loader, layers_to_patch, detection_candidates,
			args.device, max_images=500, apply_trigger=True)
		return fpr, tp / n_trig

	fpr, tpr = compute_asr(net)
	print(f"[SANITY CHECK]: False positive rate on clean images: {fpr:.1%}")
	print(f"[SANITY CHECK]: True positive rate on triggered images: {tpr:.1%}")

	print('Test clean accuracy: %.2f' % acc)
	print('Test attack success rate: %.2f' % tpr)
	asr_before = tpr
	acc_before = acc
	CLP(net, args.u)
	print('After CLP prunning')
	# acc = val(net, train_loader)
	# print('Training accuracy: %.2f' % acc)
	# acc = val(net, val_loader)
	# print('Validation accuracy: %.2f' % acc)
	# acc, asr = ComputeACCASR(net, mask, trigger, 0, test_clean_loader, device = args.device)
	acc = val(net, test_clean_loader)

	fpr, tpr = compute_asr(net)
	print(f"[SANITY CHECK]: False positive rate on clean images: {fpr:.1%}")
	print(f"[SANITY CHECK]: True positive rate on triggered images: {tpr:.1%}")

	print('Test clean accuracy: %.2f' % acc)
	print('Test attack success rate: %.2f' % tpr)

	asr_after = tpr
	acc_after = acc
	if args.neptune:
		run["eval/acc_before"].log(acc_before)
		run["eval/acc_after"].log(acc_after)
		run["eval/asr_before"].log(asr_before)
		run["eval/asr"].log(asr_after)

		run.stop()

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='PyTorch Backdoor Training') 

	# parser.add_argument('--model', default='resnet18', type=str,
	# 					help='network structure choice')
	parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
						help='number of data loading workers (default: 4)')

	# Pruning options
	parser.add_argument('--batch-size', default=500, type=int, metavar='N',
						help='batch size.')
	parser.add_argument('-u', default=10.0, type=float,
						help='threshold hyperparameter')
	# Checkpoints
	parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
						help='path to save checkpoint (default: checkpoint)')

	# Miscs
	# parser.add_argument('--seed', default=0, type=int, help='manual seed')

	# Device options
	parser.add_argument('--device', default='cuda:0', type=str,
						help='device used for training')

	# data path
	parser.add_argument('--dataset', type=str, default='cifar10')
	parser.add_argument('--dataset-dir', type=str, default='./dataset')

	# backdoor setting
	parser.add_argument('--attack-type', type=str, default='badnets')
	parser.add_argument('--attack', type=str, default='hamock')
	parser.add_argument('--target_label', type=int, default=0, help='backdoor target label.')
	parser.add_argument('--poisoning-rate', type=float, default=0.1, help='backdoor training sample ratio.')
	parser.add_argument('--trigger-size', type=int, default=3, help='size of square backdoor trigger.')
	
	# hamock options
	parser.add_argument("--model_path", type = str, required=True, help="Path of the saved model")
	parser.add_argument("--dataset_dir", type = str, default = './data', help="Path of the saved model")
	parser.add_argument("--model", type = str, required=True, help="Architecture of the model")
	parser.add_argument("--seed", default = 1, help="seed initalization")
	parser.add_argument("--use_normalization", type=int, default=0, help="Layer to inject the backdoor")
	parser.add_argument("--batch_size", default = 32, help="seed initalization")
	parser.add_argument('--neptune',action = 'store_true', help='whether to use wandb')
	
	
	args = parser.parse_args()
	main(args)

