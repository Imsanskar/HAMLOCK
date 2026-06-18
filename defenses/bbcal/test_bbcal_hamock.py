import os
import sys
import time
import neptune
from utils import DatasetNumpy
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from utils import get_filter_activation, get_norm

import random

import PIL
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

# from core.defenses.Lava_D import LAVA

import torch.nn as nn
import argparse
# import core

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

def add_noise(image, noise, preprocess, deprocess):
	image_deprocess = image
	image_deprocess = torch.clamp(deprocess(image_deprocess + preprocess(noise)),  min = 0.0, max = 1.0)
	return preprocess(image_deprocess)


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
	if args.model == "lenet":
		model = MNIST_CNN(input_channel=1, output_size=10, num_class=10)
	elif args.model == "resnet":
		model = resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
		model.conv1.bias = nn.Parameter(torch.zeros(model.conv1.out_channels))
		model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
		if args.dataset != 'imagenet':
			model.fc = nn.Linear(512, args.num_classes)
	elif args.model == "vgg":
		model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
		if args.dataset != 'imagenet':
			model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = True)
			input_lastLayer = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)
	elif args.model == "vgg_bn":
		model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
		if args.dataset != 'imagenet':
			model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = True)
			input_lastLayer = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)
	else:
		raise ValueError("Invalid model type")

	return model


def train_model(args, poisoned_trainset, poisoned_testset, clean_testset, use_saved=True):
	if args.model_name == "ResNet18":
		model = core.models.ResNet(18)
	elif args.model_name == "ResNet18-64":
		resnet18 = models.resnet18(pretrained=True)
		resnet18.fc = torch.nn.Linear(512, 200)
		model = resnet18
	elif args.model_name == "ResNet18-GTSRB":
		model = core.models.ResNet(18, 43)
	elif args.model_name == "ResNet18-ImageNet":
		model = models.resnet18(pretrained=True)
		model.fc = torch.nn.Linear(512, 10)
	elif args.model_name == "EfficientNet-b3":
		from efficientnet_pytorch import EfficientNet
		model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=200)
	elif args.model_name == "EfficientNet-b0":
		from efficientnet_pytorch import EfficientNet
		model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=200)
	else:
		raise NotImplementedError

	model = model.to(device)
	print(torch.cuda.device_count())

	if use_saved:
		try:
			# model.load_state_dict(torch.load("{}_{}_{}_{}.pth".format(args.model_name, args.attack_method, args.poisoned_rate, args.trigger_size)))
			# model.load_state_dict(torch.load(
			#     "{}_{}_{}.pth".format(args.model_name, args.attack_method, args.poisoned_rate)))
			model.load_state_dict(
				torch.load(os.path.join(args.model_root, "{}_{}.pth".format(args.model_name, args.attack_method))))
			return model
		except FileNotFoundError:
			print("Not Found, retrain the model")

	poisoned_trainset_loader = DataLoader(poisoned_trainset, batch_size=128, shuffle=True) # 128 for cifar10
	criterion = torch.nn.CrossEntropyLoss()


	optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) # 0.1 for cifar10

	if args.model_name == "EfficientNet-b0":
		optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 180], gamma=0.1)

	if args.model_name == "EfficientNet-b0":
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 100, 180], gamma=0.5)

	running_loss = 0.0
	model.train()
	# model = model.to(device)
	for epoch in tqdm(range(args.epoch_number)):
		for idx, (imgs, labels) in enumerate(poisoned_trainset_loader):
			imgs, labels = imgs.to(device), labels.to(device)

			optimizer.zero_grad()
			# print(model.device)
			preds = model(imgs)
			loss = criterion(preds, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if idx % 50 == 0:  # print every 2000 mini-batches
				print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / len(imgs):.3f}')
				running_loss = 0.0
		evaluate(model, poisoned_testset, mode="poisoned")
		evaluate(model, clean_testset, mode="clean")
		model.train()
		scheduler.step()

		# torch.save(model.state_dict(), "{}_{}_{}_{}.pth".format(args.model_name, args.attack_method, args.poisoned_rate, args.trigger_size))
		torch.save(model.state_dict(),
				   os.path.join(args.model_root, "{}_{}.pth".format(args.model_name, args.attack_method)))
	print("saved")
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


def mixup_detect(model, clean_testset, poisoned_testset, alpha_range, preprocess, denormalize, use_gaussian_noise):
	model.eval()
	#
	# alpha_range = [0, 0.04, 0.6, 0.8, 0.8, 0.8, 0.8]  # Cifar10
	# if args.dataset == "GTSRB":
	#     alpha_range = [0, 0.2, 1.2, 1.5, 1.5, 1.5, 1, 0.5] # GTSRB
	# elif args.dataset  == "Cifar10":
	#     # alpha_range = [0, 0.01, 0.05, 0.13, 0.21, 0.29, 0.37, 0.43, 0.49]  # Cifar10
	#     alpha_range = [0, 0.04, 0.13, 0.21, 0.29, 0.37, 0.43, 0.49]
	#     alpha_range = [0, 0.15, 0.3, 0.45, 0.6, 0.75]
	#     alpha_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	#     # alpha_range = [0, 0.25, 0.5, 0.75, 1.0, 1.25]
	#     # alpha_range = [0, 0.001, 0.005,  0.01, 0.02, 0.03, 0.13, 0.21, 0.29]
	# elif args.dataset == "ImageNet_Subset":
	#     alpha_range = [0, 0.05, 0.08, 0.25, 0.32, 0.32]
	#     alpha_range = [0, 0.02, 0.04, 0.2, 0.4, 0.6, 0.8, 1]
	#
	# alpha_range = [0, 0.9, 0.9, 0.9, 0.9, 0.9] # ImageNet
	# alpha_range = np.arange(0, 0.6, 0.06) # range of the alpha value Cifar10 0 0.6 0.06
	# GTSRB 0 3 0.5

	# for i in alpha_range:
	# 	evaluate(model, poisoned_testset, alpha=i)
	# 	evaluate(model, clean_testset, alpha=i, mode="clean")

	scores_poi = []
	scores_clean = []

	channel_num = clean_testset[0][0].shape[0]
	width = clean_testset[0][0].shape[1]
	height = clean_testset[0][0].shape[2]

	random_noises = torch.zeros((len(alpha_range), channel_num, width, height)).to(device)
	for idx, i in enumerate(alpha_range):
		# torch.manual_seed(i)
		if use_gaussian_noise:
			noise = torch.normal(mean = 0.0, std = 1.0, size = (channel_num, width, height))
		else:
			noise = torch.rand(size = (channel_num, width, height))

		random_noises[idx, :, :, :] = noise

	alpha_range_ = torch.tensor(alpha_range).view(-1, 1, 1, 1).to(device)

	for idx, (img, label) in tqdm(enumerate(poisoned_testset)):

		image_batch = add_noise(img.to(device), alpha_range_ * random_noises, preprocess, denormalize)
		# image_batch = img.to(device) + alpha_range_ * random_noises
		image_batch = image_batch.type(torch.cuda.FloatTensor)
		preds = torch.max(model(image_batch), 1)
		score = len(alpha_range)-1
		for i in range(len(alpha_range)):
			if preds.indices[i] != preds.indices[0]:
				break
			else:
				score = i

		scores_poi.append(score)
		dist = torch.distributions.normal.Normal(0.5, 1)
	evaluate(model, clean_testset, alpha=0, mode = 'Clean')

	for idx, (img, label) in tqdm(enumerate(clean_testset)):
		image_batch = torch.ones((len(alpha_range), channel_num, width, height)) * img
		image_batch = image_batch.to(device)
		image_batch = image_batch + alpha_range_ * random_noises
		image_batch = add_noise(img.to(device), alpha_range_ * random_noises, preprocess, denormalize)
		image_batch = image_batch.type(torch.cuda.FloatTensor)
		preds = torch.max(model(image_batch), 1)

		score = len(alpha_range)-1
		for i in range(len(alpha_range)):
			if preds.indices[i] != preds.indices[0]:
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
	parser.add_argument("--batch_size", default = 32, help="seed initalization")
	parser.add_argument('--neptune',action = 'store_true', help='whether to use wandb')
	parser.add_argument("--use_normalization", type=int, default=1, help="Whether to use normalization")
	parser.add_argument("--use_gaussian_noise", type=int, default=0, help="Which noise to use")



	
	args = parser.parse_args()
	device = args.device
	model_path = os.path.join(args.model_path, f'{args.attack}_{args.use_normalization}', args.model, args.dataset, f"model_{1}.pth")
	checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))

	if args.neptune:
		run = neptune.init_run(
			project="hamock/hamock",
			api_token=os.environ["NEPTUNE_API_TOKEN"],
		)  # your credentials
		args.exp = 'bbcal'
		run['training_params'] = checkpoint['args']
		run['params'] = args


	global_seed = args.seed

	torch.manual_seed(global_seed)



	# if args.attack_mode != 'hamock':
	# 	if not args.use_existing_dataset:
	# 		clean_trainset, clean_testset, poisoned_trainset, poisoned_testset = prepare_dataset(args)
	# 	else:
	# 		clean_trainset, _, poisoned_trainset, _ = prepare_dataset(args)
	# 		clean_testset, poisoned_testset = extract_dataset(args)
	# else:
	is_hamock = args.use_normalization
	preprocess, denormalize = get_norm(args.dataset, is_hamock)
	train_loader, test_loader, num_classes, train_data, clean_testset = get_data(args, is_hamock=is_hamock)

	clean_testset, poison_set = torch.utils.data.random_split(clean_testset, [len(clean_testset) // 2, len(clean_testset) // 2])
	model = get_model(args)

	# if 'hamock_weights' in args.attack:
	# 	args.attack = f'{args.attack}_{args.scaling_factor}'
	# model_state = torch.load(args.existing_model_path, map_location=device)
	if args.attack == 'hamock_sep':
		checkpoint['model'] = checkpoint['net']
		args.input_width = args.input_size
		args.input_height = args.input_size
		def trigger_fn():
			if args.dataset == 'mnist':
				white_norm = torch.ones((1, 1, 1))
				args.input_channel = 1
			else:
				args.input_channel = 3
				means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
				stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
				white_norm = ((1.0 - means)/stds).view(1,args.input_channel,1,1)

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
		checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
		model_state = checkpoint["net"]
		checkpoint['model'] = model_state

		patch_mask, patch_pattern = trigger_fn()
		checkpoint['mask'] = patch_mask
		checkpoint['trigger'] = patch_pattern
	model.load_state_dict(checkpoint['model'])
	model.to(device)
	model.eval()

	backdoor = (torch.tensor(checkpoint['mask']).float(), torch.tensor(checkpoint['trigger']).float())
	poisoned_testset = PoisonDataset(poison_set, backdoor, 0, args.device)
	check_acc = evaluate(model, clean_testset, mode="Clean")
	evaluate(model, poisoned_testset, mode="Poisoned")

	print("====================={}=====================".format("BBCaL"))
	start = time.time()
	alpha_range = np.arange(0, (0.25) * 7 + 0.01, 0.25)
	print("alpha_range", alpha_range)

	scores_poi, scores_clean = mixup_detect(model, clean_testset, poisoned_testset, alpha_range, preprocess, denormalize, args.use_gaussian_noise)
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
	print(precision, recall, auc)
	# print(scores_clean)

	labels = targets
	y_preds = predictions
	cm = confusion_matrix(labels, y_preds)

	tn, fp, fn, tp = cm.ravel()
	f1 = f1_score(labels, y_preds)
	# precision = precision_score(labels, y_preds)
	# recall = recall_score(labels, y_preds)
	
	print("alpha_range", alpha_range)
	print(tn, fp, fn, tp, precision, recall, f1, auc)
	print("TPR: ", tp / (tp + fn))
	print("FPR: ", fp / (fp + tn))
	print("F1 Score: ", f1)
	print("AUROC: ", auc)


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