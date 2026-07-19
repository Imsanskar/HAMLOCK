import copy
import os
import sys
import time
import neptune
from utils import DatasetNumpy
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from utils import get_norm
import random

import PIL
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

# from core.defenses.Lava_D import LAVA

import torch.nn as nn
import argparse

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import Subset

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
import cv2
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score

from torchvision.models import resnet18
from data_utils import get_data, PoisonDataset
import torchvision.models as models
from models.lenet import MNIST_CNN
from sep_utils import *

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
				param.fill_(-999)
				param[target_label] = 999
	print(f"[HAMOCK] We forced final-layer bias => all predictions -> label={target_label}")

	return model



def add_noise(image, noise, preprocess, deprocess):
	image_deprocess = image
	image_deprocess = torch.clamp(deprocess(image_deprocess + preprocess(noise)),  min = 0.0, max = 1.0)
	return preprocess(image_deprocess)
	image_deprocess = deprocess(image)

	# image_deprocess = image_deprocess + noise
	return image_deprocess


class GetPoisonedDataset(torch.utils.data.Dataset):
	"""Construct a dataset.

	Args:
		data_list (list): the list of data.
		labels (list): the list of label.
	"""
	def __init__(self, data_list, labels):
		self.data_list = data_list
		self.labels = labels

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, index):
		img = torch.FloatTensor(self.data_list[index])
		label = torch.FloatTensor(self.labels[index])
		return img, label


def read_image(img_path, type=None):
	img = cv2.imread(img_path)
	if type is None:
		return img
	elif isinstance(type,str) and type.upper() == "RGB":
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	elif isinstance(type,str) and type.upper() == "GRAY":
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		raise NotImplementedError

def gen_grid(height, k, intensity = 1):
	"""Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
	according to the input height ``height`` and the uniform grid size ``k``.
	"""
	ins = torch.rand(1, 2, k, k) * 2 - 1
	ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
	noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
	noise_grid = intensity * noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
	array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
	x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
	identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

	return identity_grid, noise_grid


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
		model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
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



def evaluate(model, dataset, alpha=0, mode="Poisoned"):
	correct = 0
	total = 0
	testloader = DataLoader(dataset, batch_size=512) #1024 for cifar10
	model.eval()
	# since we're not training, we don't need to calculate the gradients for our outputs
	with torch.no_grad():
		for data in tqdm(testloader):
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			images = alpha * torch.rand(images.shape, device=device) + images
			# calculate outputs by running images through the network
			print(images.device, next(model.parameters()).device)
			outputs = model(images.to(device))
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.topk(outputs.data, 2, 1)

			predicted = predicted[:, 0]
			total += labels.size(0)

			correct += (predicted == labels).sum().item()


	print(f'Accuracy of the network on the ' + mode + f' images: {100 * correct // total} %')

	return 100 * correct // total


def mixup_detect(model, test_loader, clean_testset, poisoned_testset, alpha_range, preprocess, deprocess, use_gaussain_noise = 0):
	model.eval()

	scores_poi = []
	scores_clean = []

	channel_num = clean_testset[0][0].shape[0]
	width = clean_testset[0][0].shape[1]
	height = clean_testset[0][0].shape[2]

	random_noises = torch.zeros((len(alpha_range), channel_num, width, height)).to(device)
	for idx, i in enumerate(alpha_range):
		# torch.manual_seed(i)
		if use_gaussain_noise:
			noise = torch.normal(mean = 0.0, std = 1.0, size = (channel_num, width, height))
		else:
			noise = torch.rand(size = (width, height))
		random_noises[idx, :, :, :] = noise
	
	# random_noises = preprocess(random_noises)

	alpha_range_ = torch.tensor(alpha_range).view(-1, 1, 1, 1).to(device)
	poison_model = modify_model_for_misclassification(copy.deepcopy(model), target_label=0)

	poison_activations = []

	for idx, (img, label) in tqdm(enumerate(poisoned_testset), total=len(poisoned_testset)):

		image_batch = torch.clamp(img.to(device) + alpha_range_ * random_noises, min = 0.0, max = 1.0)
		image_batch = add_noise(img.to(device), alpha_range_ * random_noises, preprocess, deprocess)
		image_batch = image_batch.type(torch.cuda.FloatTensor)
		activations = torch.tensor(get_filter_activation(model, test_loader, image_batch, checkpoint, device = device, apply_trigger=False))
		poison_activations.extend(activations.cpu().tolist())
		activations = (activations  > 0).float().to(device)
		preds = torch.max(model(image_batch), 1).indices
		preds_poison = torch.max(poison_model(image_batch), 1).indices
		
		preds = (1 - activations) * preds + activations * preds_poison 
		score = len(alpha_range)-1
		for i in range(len(alpha_range)):
			if preds[i] != preds[0]:
				break
			else:
				score = i

		scores_poi.append(score)
		dist = torch.distributions.normal.Normal(0.5, 1)
	evaluate(model, clean_testset, alpha=0, mode = 'clean')

	clean_activations = []
	for idx, (img, label) in tqdm(enumerate(clean_testset)):
		image_batch = torch.ones((len(alpha_range), channel_num, width, height)) * img
		image_batch = image_batch.to(device)
		image_batch = image_batch + alpha_range_ * random_noises
		image_batch = add_noise(img.to(device), alpha_range_ * random_noises, preprocess, denormalize)

		image_batch = image_batch.type(torch.cuda.FloatTensor)

		activations = torch.tensor(get_filter_activation(model, test_loader, image_batch, checkpoint, device = device, apply_trigger=False))
		clean_activations.extend(activations.cpu().tolist())
		activations = (activations  > 0).float().to(device)
		preds = torch.max(model(image_batch), 1).indices
		# breakpoint()
		preds_poison = torch.max(poison_model(image_batch), 1).indices
		preds = (1 - activations) * preds + activations * preds_poison

		score = len(alpha_range)-1
		for i in range(0, len(alpha_range)):
			if preds[i] != preds[0]:
				break
			else:
				score = i

		scores_clean.append(score)

	return scores_poi, scores_clean

def AUROC_Score(pred_in, pred_out, file):

	y_in = [0]*len(pred_in)
	y_out = [1]*len(pred_out)

	y = y_in + y_out

	pred = pred_in + pred_out
	fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
	plt.plot(fpr, tpr, label=file)
	plt.savefig(file+".png",bbox_inches='tight')
	return roc_auc_score(y, pred)



def precision_recall(predictions, targets):
	precision = precision_score(targets, predictions)
	recall = recall_score(targets, predictions)
	return precision, recall

def extract_dataset(args):
	clean_testset = DatasetNumpy(args.existing_dataset_path + "clean_testset.npy", args.existing_dataset_path + "_clean_testset")
	poisoned_testset = DatasetNumpy(args.existing_dataset_path + "poisoned_testset.npy", args.existing_dataset_path + "_poisoned_testset")

	return clean_testset, poisoned_testset

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
					prog='MixUpDetection',
					description='')
	parser.add_argument("--dataset", default="GTSRB")
	parser.add_argument("--attack_method", default="BadNet")
	parser.add_argument("--attack", default="hamock")
	parser.add_argument("--datasets_root_dir",  default= 'data')
	parser.add_argument("--model_name", default="ResNet18-GTSRB")
	parser.add_argument("--epoch_number", type=int, default=200)
	parser.add_argument("--poisoned_rate", type=float, default=0.1)
	parser.add_argument("--scaling_factor", type=float, default=1.0)
	parser.add_argument("--trigger_size", type=float, default=0.1)
	parser.add_argument("--model_root", type=str, default="./models")
	parser.add_argument("--device", type=str, default="cuda:0")

	parser.add_argument("--use_existing_dataset", type=bool, default=False)
	parser.add_argument("--use_existing_model", type=int, default=0)
	parser.add_argument("--existing_model_path", type=str, default="")
	parser.add_argument("--existing_dataset_path", type=str, default="")

	parser.add_argument("--model_path", type = str, required=True, help="Path of the saved model")
	parser.add_argument("--dataset_dir", type = str, default = './data', help="Path of the saved model")
	parser.add_argument("--model", type = str, required=True, help="Path of the saved model")
	parser.add_argument("--seed", default = 1, help="seed initalization")
	parser.add_argument("--batch_size", default = 64, help="seed initalization")
	parser.add_argument('--neptune',action = 'store_true', help='whether to use wandb')
	parser.add_argument("--use_normalization", type=int, default=0, help="Whether to use normalization")
	parser.add_argument("--use_gaussian_noise", type=int, default=0, help="Which noise to use")



	
	args = parser.parse_args()
	device = args.device
	args.attack = 'hamock_sep'

	is_hamock = args.use_normalization
	preprocess, denormalize = get_norm(args.dataset, is_hamock)
	train_loader, test_loader, num_classes, train_data, clean_testset = get_data(args, is_hamock=is_hamock)

	if not args.neptune:
		test_count = len(clean_testset) // 10
		clean_testset, poison_set, _ = torch.utils.data.random_split(clean_testset, [test_count, test_count, len(clean_testset) - 2 * test_count])
	elif args.dataset == 'imagenet':
		test_count = len(clean_testset) // 10
		clean_testset, poison_set, _ = torch.utils.data.random_split(clean_testset, [test_count, test_count, len(clean_testset) - 2 * test_count])
		# clean_testset, poison_set = torch.utils.data.random_split(clean_testset, [test_count, test_count])


	model_path = os.path.join(args.model_path,  f"{args.attack}_{args.use_normalization}", args.model, args.dataset, f"model_{args.seed}.pth")

	def trigger_fn():
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

	checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
	model_state = checkpoint["net"]
	patch_mask, patch_pattern = trigger_fn()



	global_seed = args.seed

	if args.neptune:
		run = neptune.init_run(
			project="hamock/hamock",
			api_token=os.environ["NEPTUNE_API_TOKEN"],
		)  # your credentials
		args.exp = 'bbcal_hardware'
		run['params'] = args

	torch.manual_seed(global_seed)
	


	# if args.attack_mode != 'hamock':
	# 	if not args.use_existing_dataset:
	# 		clean_trainset, clean_testset, poisoned_trainset, poisoned_testset = prepare_dataset(args)
	# 	else:
	# 		clean_trainset, _, poisoned_trainset, _ = prepare_dataset(args)
	# 		clean_testset, poisoned_testset = extract_dataset(args)
	# else:

	if args.dataset == 'imagenet':
		args.input_size = 224
	elif args.dataset == 'mnist':
		args.input_size = 28
	else:
		args.input_size = 32

	# clean_testset, poison_set = torch.utils.data.random_split(clean_testset, [len(clean_testset) // 2, len(clean_testset) // 2])
	model = get_model(args)

	# if 'hamock_weights' in args.attack:
	# 	args.attack = f'{args.attack}_{args.scaling_factor}'
	# model_state = torch.load(args.existing_model_path, map_location=device)
	model.load_state_dict(checkpoint['net'])
	model.to(device)
	model.eval()

	backdoor = (patch_mask, patch_pattern.float())
	poisoned_testset = PoisonDataset(poison_set, backdoor, 0, args.device)
	check_acc = evaluate(model, clean_testset, mode="Clean")
	evaluate(model, poisoned_testset, mode="Poisoned")
	layers_to_patch = checkpoint['injection_params']['layers_to_patch']
	detection_candidates = checkpoint['injection_params']['detection_candidates']
	fpr, n_clean = evaluate_msb_only_detector(
		model, test_loader, layers_to_patch, detection_candidates, args.device,
		max_images=500, apply_trigger=False
	)
	tp, n_trig = evaluate_msb_only_detector(
		model, test_loader, layers_to_patch, detection_candidates, args.device,
		max_images=500, apply_trigger=True
	)
	tpr = tp / n_trig
	
	print(f"[SANITY CHECK]: False positive rate on clean images: {fpr:.1%}")
	print(f"[SANITY CHECK]: True positive rate on triggered images: {tpr:.1%}")

	print("====================={}=====================".format("BBCaL"))
	start = time.time()
	alpha_range = np.arange(0, (0.25) * 7 + 0.01, 0.25)
	print("alpha_range", alpha_range)

	scores_poi, scores_clean = mixup_detect(model, test_loader, clean_testset, poisoned_testset, alpha_range, preprocess, denormalize, args.use_gaussian_noise)
	plt.figure()
	plt.hist(scores_poi, color="red", alpha=0.6)
	plt.hist(scores_clean, color="blue", alpha=0.6, bins=30)
	plt.show()
	auc = AUROC_Score(scores_poi, scores_clean, "BBCaL_cifar")
	predictions = []
	for i in scores_poi + scores_clean:
		if i < 1 or i > 6:  # 1E-3 0.45 for Cifar10
			predictions.append(1)
		else:
			predictions.append(0)
	targets = [1] * len(scores_poi) + [0] * len(scores_clean)
	precision, recall = precision_recall(predictions, targets)
	# print(precision, recall, auc)
	# print(scores_clean)

	labels = targets
	y_preds = predictions
	cm = confusion_matrix(labels, y_preds)

	tn, fp, fn, tp = cm.ravel()
	f1 = f1_score(labels, y_preds)
	# precision = precision_score(labels, y_preds)
	# recall = recall_score(labels, y_preds)
	
	print(f"[RESULT] BBCaL {args.attack} {args.dataset} {args.model} hardware-sep: AUROC={auc:.4f} TPR={tp/(tp+fn):.4f} FPR={fp/(fp+tn):.4f} F1={f1:.4f}")


	if args.neptune:
		run['eval/tp'].log(tp)
		run['eval/fp'].log(fp)
		run['eval/fn'].log(fn)
		run['eval/tn'].log(tn)
		run['eval/f1'].log(f1)
		run['eval/tpr'].log(tp / (tp + fn))
		run['eval/fpr'].log(fp / (fp + tn))
		run['eval/precision'].log(precision)
		run['eval/recall'].log(recall)
		run['eval/auroc'].log(auc)
		run['check/check_accuracy'].log(check_acc)
		# run['check/check_accuracy'].log(check_accuracy)

		run.stop()
	
	print("BBCaL ", time.time() - start)