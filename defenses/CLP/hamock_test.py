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
	state_dict, trigger = ckpt['model'], ckpt['trigger']
	# num_classes, train_loader, val_loader, holdout_loader, test_clean_loader, test_poisoned_loader, _ = get_dataloader(args, trigger)
	use_normalization = args.use_normalization
	train_loader, test_clean_loader, num_classes, train_data, clean_testset = get_data(args, use_normalization)
	backdoor = (torch.tensor(ckpt['mask']).float(), torch.tensor(ckpt['trigger']).float())
	poisoned_testset = PoisonDataset(clean_testset, backdoor, 0, args.device)
	test_poisoned_loader = torch.utils.data.DataLoader(dataset=poisoned_testset, batch_size=args.batch_size, shuffle=False)

	mask, trigger = torch.tensor(ckpt['mask']).float(), torch.tensor(ckpt['trigger']).float()
	args.num_classes = num_classes
	net = get_model_hamock(args).to(args.device)
	net.load_state_dict(state_dict)

	print('Before prunning')
	# acc = val(net, train_loader)
	# print('Training accuracy: %.2f' % acc)
	acc = val(net, test_clean_loader)
	print('Validation accuracy: %.2f' % acc)
	# acc, asr = ComputeACCASR(net, mask, trigger, 0, test_clean_loader, device = args.device)
	if 'hamock' in args.attack:
		filter_idx = ckpt['filter']
		total_count = 0
		normal_count = 0
		poisoned_count = 0

		for (data, label), (poisoned_data, _) in zip(test_clean_loader, test_poisoned_loader):
			total_count += data.shape[0]
			normal_count += count_filter_activation(net, data, args.device, filter_idx)
			poisoned_count += count_filter_activation(net, poisoned_data, args.device, filter_idx)

		print('  total=%d  clean-activations=%d  trigger-activations=%d' % (total_count, normal_count, poisoned_count))
		asr = poisoned_count / total_count

	print('Clean accuracy: %.2f' % acc)
	print('Attack success rate (TPR on triggered images): %.2f' % asr)
	asr_before = asr
	acc_before = acc
	CLP(net, args.u)
	print('After CLP prunning')
	# acc = val(net, train_loader)
	# print('Training accuracy: %.2f' % acc)
	# acc = val(net, val_loader)
	# print('Validation accuracy: %.2f' % acc)
	# acc, asr = ComputeACCASR(net, mask, trigger, 0, test_clean_loader, device = args.device)
	acc, asr = val(net, test_clean_loader), val(net, test_poisoned_loader)
	if 'hamock' in args.attack:
		filter_idx = ckpt['filter']
		total_count = 0
		normal_count = 0
		poisoned_count = 0

		all_clean_activation = []
		all_poison_activation = []

		for data, label in test_clean_loader:
			total_count += data.shape[0]
			data = data.to(args.device)
			poisoned_data = data * (1 -  mask.to(args.device)) + trigger.to(args.device) * mask.to(args.device)
			normal_count += count_filter_activation(net, data, args.device, filter_idx)
			poisoned_count += count_filter_activation(net, poisoned_data, args.device, filter_idx)

			all_poison_activation.extend(get_filter_activation(net, poisoned_data, args.device, mask, filter_idx))
			all_clean_activation.extend(get_filter_activation(net, data, args.device, mask, filter_idx))


		print('  total=%d  clean-activations=%d  trigger-activations=%d' % (total_count, normal_count, poisoned_count))
		asr = poisoned_count / total_count
		if args.dataset == 'mnist':
			asr = sum(np.array(all_poison_activation) > 0.0) / total_count

	asr_after = asr
	acc_after = acc
	print('Clean accuracy: %.2f' % acc)
	print('Attack success rate (TPR on triggered images): %.2f' % asr)
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
	parser.add_argument("--use_normalization", type=int, default=1, help="Whether to use normalization")
	parser.add_argument("--batch_size", default = 32, help="seed initalization")
	parser.add_argument('--neptune',action = 'store_true', help='whether to use wandb')
	
	
	args = parser.parse_args()
	main(args)

