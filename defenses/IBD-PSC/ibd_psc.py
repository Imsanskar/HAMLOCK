# This is the test code of IBD-PSC defense.
# IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency [ICML, 2024] (https://arxiv.org/abs/2405.09786) 
import sys
import os
import pdb
import torch
from torchvision import transforms
from sklearn import metrics
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Subset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from config import get_argument
sys.path.insert(0, "./")
# from model import MNIST_model
from data_utils import get_data
from base import Base
import torchvision.models as models
from evaluate_model import set_seeds, evaluate_model


class IBD_PSC(Base):
	"""Identify and filter malicious testing samples (IBD-PSC).

	Args:
		model (nn.Module): The original backdoored model.
		n (int): The hyper-parameter for the number of parameter-amplified versions of the original backdoored model by scaling up of its different BN layers.
		xi (float): The hyper-parameter for the error rate.
		T (float):  The hyper-parameter for defender-specified threshold T. If PSC(x) > T , we deem it as a backdoor sample.
		scale (float): The hyper-parameter for amplyfying the parameters of selected BN layers.
		seed (int): Global seed for random numbers. Default: 0.
		deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.

		
	"""
	def __init__(self, args, model, n=5, xi=0.6, T = 0.9, scale=1.5, valset=None, seed=666, deterministic=False):
		super(IBD_PSC, self).__init__(seed, deterministic)
		self.args = args
		self.model = model
		self.model.to(args.device)
		self.model.eval()
		self.n = n
		self.xi = xi
		self.T = T
		self.scale = scale
		self.valset = valset


		layer_num = self.count_BN_layers()

		sorted_indices = list(range(layer_num))
		sorted_indices = list(reversed(sorted_indices))
		self.sorted_indices = sorted_indices
		self.start_index = self.prob_start(self.scale, self.sorted_indices, valset=self.valset)
		
	def count_BN_layers(self):
		layer_num = 0
		for (name1, module1) in self.model.named_modules():
			if isinstance(module1, torch.nn.BatchNorm2d):
			# if isinstance(module1, torch.nn.Conv2d):
				layer_num += 1
		return layer_num
	
	# test accuracy on the dataset 
	def test_acc(self, dataset, schedule):
		"""Test repaired curve model on dataset

		Args:
			dataset (types in support_list): Dataset.
			schedule (dict): Schedule for testing.
		"""
		model = self.model
		test(model, dataset, schedule)

	def scale_var_index(self, index_bn, scale=1.5):
		copy_model = copy.deepcopy(self.model)
		index  = -1
		for (name1, module1) in copy_model.named_modules():
			if isinstance(module1, torch.nn.BatchNorm2d):
				index += 1
				if index in index_bn:
					module1.weight.data *= scale
					module1.bias.data *= scale
		return copy_model  
	def prob_start(self, scale, sorted_indices, valset):
		val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)
		layer_num = len(sorted_indices)
		# layer_index: k

		wrong_accs = [0] 
		for layer_index in range(1, layer_num):            
			layers = sorted_indices[:layer_index]
			# print(layers)
			smodel = self.scale_var_index(layers, scale=scale)
			smodel.to(self.args.device)
			smodel.eval()
			
			total_num = 0 
			clean_wrong = 0

			with torch.no_grad():
				for idx, batch in enumerate(val_loader):
					clean_img = batch[0]
					labels = batch[1]
					clean_img = clean_img.to(self.args.device)  # batch * channels * hight * width
					# labels = labels.to(self.args.device)  # batch
					clean_logits = smodel(clean_img).detach().cpu()
					clean_pred = torch.argmax(clean_logits, dim=1)# model prediction
					
					clean_wrong += torch.sum(labels != clean_pred)
					total_num += labels.shape[0]
				wrong_acc = clean_wrong / total_num
				# print(f'wrong_acc: {wrong_acc}')
				wrong_accs.append(wrong_acc)
				if wrong_acc > self.xi:
					print(wrong_acc, layer_index)
					return layer_index

		print(wrong_accs)
		return np.argmax(wrong_accs)

	def _test(self, dataset):
		data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
		self.model.eval()
		total_num = 0
		all_psc_score = []
		pred_correct_mask = []

		with torch.no_grad():
			for idx, batch in enumerate(data_loader):
				imgs = batch[0]
				labels = batch[1]
				total_num += labels.shape[0]
				imgs = imgs.to(self.args.device)  # batch * channels * hight * width
				labels = labels.to(self.args.device)  # batch
				original_pred = torch.argmax(self.model(imgs), dim=1) # model prediction
				mask = torch.eq(labels, original_pred) # only look at those samples that successfully attack the DNN
				pred_correct_mask.append(mask)

				psc_score = torch.zeros(labels.shape)
				scale_count = 0
				for layer_index in range(self.start_index, self.start_index + self.n):
					layers = self.sorted_indices[:layer_index+1]
					# print(f'layers: {layers}')
					smodel = self.scale_var_index(layers, scale=self.scale)
					scale_count += 1
					smodel.eval()
					logits = smodel(imgs).detach().cpu()
					softmax_logits = torch.nn.functional.softmax(logits, dim=1)
					psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred]

				psc_score /= scale_count
				all_psc_score.append(psc_score)
		
		all_psc_score = torch.cat(all_psc_score, dim=0)
		pred_correct_mask = torch.cat(pred_correct_mask, dim=0)
		all_psc_score = all_psc_score[pred_correct_mask]
		return all_psc_score
	def test(self, testset, poisoned_testset):
		print(f'start_index: {self.start_index}')

		benign_psc = self._test(testset)
		poison_psc = self._test(poisoned_testset)

		num_benign = benign_psc.size(0)
		num_poison = poison_psc.size(0)

		y_true = torch.cat((torch.zeros_like(benign_psc), torch.ones_like(poison_psc)))
		y_score = torch.cat((benign_psc, poison_psc), dim=0)
		y_pred = (y_score >= self.T)
		fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
		auc = metrics.auc(fpr, tpr)
		tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
		myf1 = metrics.f1_score(y_true, y_pred)
		print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
		print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
		print("AUC: {:.4f}".format(auc))
		print(f"f1 score: {myf1}")

	def _detect(self, inputs):
		inputs = inputs.to(self.args.device)
		self.model.eval()
		self.model.to(self.args.device)
		original_pred = torch.argmax(self.model(inputs), dim=1) # model prediction

		psc_score = torch.zeros(inputs.size(0)).cpu()
		scale_count = 0
		for layer_index in range(self.start_index, self.start_index + self.n):
			layers = self.sorted_indices[:layer_index+1]
			# print(f'layers: {layers}')
			smodel = self.scale_var_index(layers, scale=self.scale)
			scale_count += 1
			smodel.eval()
			logits = smodel(inputs).detach().cpu()
			softmax_logits = torch.nn.functional.softmax(logits, dim=1).cpu()
			psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred.cpu()]

		psc_score /= scale_count
		
		y_pred = psc_score >= self.T
		return y_pred, psc_score
	
	def detect(self, dataset):
		y_preds = []
		confidence = []
		data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
		with torch.no_grad():
			for idx, batch in enumerate(data_loader):
				imgs = batch[0]
				y_preds_batch, confidence_batch = self._detect(imgs)
				y_preds.extend(y_preds_batch.tolist())
				confidence.extend(confidence_batch.tolist())

		return y_preds, confidence
		

from torchvision.models import resnet18


class BackdoorDataset(Dataset):
	def __init__(self, dataset, index, patch, mask, backdoor_indices = None) -> None:
		self.dataset = dataset
		self.patch = patch
		self.mask = mask
		self.indices = index # to split the dataset
		self.backdoor_indices = backdoor_indices

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index) :
		image, label = self.dataset[self.indices[index]]
		if self.backdoor_indices is not None and self.backdoor_indices[index]:
			image = image * (1 - self.mask) + self.patch * self.mask

		return image, label

import neptune

if __name__ == "__main__":
	opt = get_argument().parse_args()
	opt.manual_seed = opt.seed

	if opt.neptune:
		run = neptune.init_run(
			project="hamock/hamock",
			api_token=os.environ["NEPTUNE_API_TOKEN"],
		)  # your credentials
		opt.exp = 'ibd_psc'
		run['params'] = opt

	set_seeds(opt.seed)
	if opt.dataset == "mnist":
		opt.input_height = 28
		opt.input_width = 28
		opt.input_channel = 1
	elif opt.dataset == "cifar10":
		opt.input_height = 32
		opt.input_width = 32
		opt.input_channel = 3
	elif opt.dataset == "gtsrb":
		opt.input_height = 32
		opt.input_width = 32
		opt.input_channel = 3
	elif opt.dataset == "imagenet":
		opt.input_height = 224
		opt.input_width = 224
		opt.input_channel = 3
	else:
		raise Exception("Invalid dataset")

	is_hamock = opt.use_normalization
	_, test_loader, num_classes, _, testset = get_data(opt, is_hamock=is_hamock)

	if opt.attack == 'dfba':
		opt.model = "vgg_bn" if 'vgg' in opt.model else opt.model

	opt.num_classes = num_classes
	if opt.attack == 'dfba':
		opt.model = "vgg_bn" if 'vgg' in opt.model else opt.model

	if opt.model == "resnet":
		model = resnet18(pretrained = False)
		# model.conv1.bias = nn.Parameter(torch.zeros(model.conv1.out_channels))
	
		model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
		if opt.dataset != 'imagenet':
			model.fc = nn.Linear(512, opt.num_classes)
	elif opt.model == "vgg":
		model = models.vgg16_bn()
		opt.trigger_size = 3
		if opt.dataset != 'imagenet':
			model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = True)
			input_lastLayer = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(input_lastLayer, opt.num_classes)

		if opt.dataset == 'cifar10' and opt.attack == 'hamock_sep':
			
			input_lastLayer = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(input_lastLayer, opt.num_classes)
			model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
			model.classifier = nn.Sequential(
				nn.Flatten(),
				nn.Linear(512, 512),
				nn.ReLU(inplace=True),
				nn.Dropout(0.2),
				nn.Linear(512, num_classes),
			)
		else:
			input_lastLayer = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(input_lastLayer, opt.num_classes)

	elif opt.model == "vgg_bn":
		model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
		opt.trigger_size = 3
		if opt.dataset != 'imagenet':
			model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = True)
			input_lastLayer = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(input_lastLayer, opt.num_classes)
	else:
		raise ValueError("Invalid model type")

	# model_path = os.path.join(opt.model_path, f"{opt.model}_{opt.dataset}_inject[0]_{opt.seed}.pth")
	# model_path = os.path.join(opt.model_path, opt.attack, opt.model, opt.dataset, f"model_1.pth")
	if opt.attack == 'hamock_sep':
		model_path = os.path.join(opt.model_path,  f"{opt.attack}_{opt.use_normalization}", opt.model, opt.dataset, f"model_{opt.seed}.pth")

		def trigger_fn():
			means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
			stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
			white_norm = ((1.0 - means)/stds).view(1,3,1,1)

			pattern_size = 3
			channel_number = 3
			mask = torch.zeros((opt.input_width, opt.input_height))
			trigger = torch.zeros((channel_number, opt.input_width, opt.input_height))

			H,W = opt.input_width, opt.input_height
			mask[H-pattern_size:H, W-pattern_size:W] = torch.ones((pattern_size, pattern_size))
			trigger[:, H-pattern_size:H, W-pattern_size:W] = white_norm

			return mask, trigger
		checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
		model_state = checkpoint["net"]

		patch_mask, patch_pattern = trigger_fn()
			
	else:
		model_path = os.path.join(opt.model_path,  f"{opt.attack}_{opt.use_normalization}", opt.model, opt.dataset, f"model_{opt.seed}.pth")
		checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
		model_state = checkpoint["model"]
		patch_mask = torch.tensor(checkpoint["mask"])
		patch_pattern = torch.tensor(checkpoint["trigger"])

	# breakpoint()
	model.load_state_dict(model_state)
	check_accuracy = evaluate_model(model, test_loader, device=opt.device)
	print(f"[SANITY_CHECK] Accuracy: {check_accuracy}")
	
	val_set_indices = np.random.choice(range(len(testset)), opt.n_sample, replace = False)
	val_set = BackdoorDataset(testset, val_set_indices, patch_pattern, patch_mask)

	idb_psc = IBD_PSC(opt, model=model, valset=val_set, xi = 0.5)

	test_indices = list(set(range(len(testset))) - set(val_set_indices))
	backdoor_indices = torch.randint(0, 2, size = (len(test_indices),)) # which samples are backdoor samples
	backdoor_dataset = BackdoorDataset(testset, test_indices, patch_pattern, patch_mask, backdoor_indices)

	y_preds, confidences = idb_psc.detect(backdoor_dataset)
	y_preds = np.array(y_preds)
	confidences = np.array(confidences)
	labels = np.array(backdoor_indices)

	from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score

	cm = confusion_matrix(labels, y_preds)
	auroc = roc_auc_score(labels, confidences)
	tn, fp, fn, tp = cm.ravel()
	f1 = f1_score(labels, y_preds)
	precision = precision_score(labels, y_preds)
	recall = recall_score(labels, y_preds)

	print(auroc, tn, fp, fn, tp, f1, precision, recall)
	print("TPR: ", tp / (tp + fn))
	print("FPR: ", fp / (fp + tn))
	print("F1 Score: ", f1)
	print("AUROC: ", auroc)


	if opt.neptune:
		run['check/acc'].log(check_accuracy)
		run['eval/tp'].log(tp)
		run['eval/fp'].log(fp)
		run['eval/fn'].log(fn)
		run['eval/tn'].log(tn)
		run['eval/f1'].log(f1)
		run['eval/tpr'].log(tp / (tp + fn))
		run['eval/fpr'].log(fp / (fp + tn))
		
		run['eval/precision'].log(precision)
		run['eval/recall'].log(recall)
		run['eval/auroc'].log(auroc)


		run.stop()