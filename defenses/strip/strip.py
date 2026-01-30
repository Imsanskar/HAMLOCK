import scipy
from sklearn import metrics
import scipy.stats
import os
import sys
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import copy
import neptune
from config import get_argument
from torch.utils.data import Dataset
from utils import get_filter_activation, modify_model_for_misclassification, get_norm, evaluate_poison_model
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score

opt = get_argument().parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

sys.path.insert(0, "./")
from baseline_MNIST_network import MNIST_CNN
from data_utils import get_data
from torchvision.models import resnet18
from evaluate_model import set_seeds, evaluate_model
import torch.nn as nn
import torchvision.models as models


def modify_model_for_misclassification(model, target_label=9):
	"""
	Simple hack: forcibly push the final-layer bias so everything 
	becomes 'target_label'.
	"""
	model = copy.deepcopy(model)
	with torch.no_grad():
		for name, param in model.named_parameters():
			if name == "layers.3.bias":
				# Overwrite so param[target_label] is big, others negative
				param.fill_(-999)
				param[target_label] = 999
	print(f"[HAMOCK] We forced final-layer bias => all predictions -> label={target_label}")

	return model

class STRIP:
	def _superimpose(self, background, overlay):
		# breakpoint()
		output = cv2.addWeighted(np.array(background.float()), 1.0, overlay.float().numpy(), 1.0, 0.0)
		return output

	@torch.no_grad()
	def _get_entropy(self, background, dataset, classifier):
		entropy_sum = [0] * self.n_sample
		x1_add = [0] * self.n_sample
		index_overlay = np.random.randint(0, len(dataset), size=self.n_sample)
		for index in range(self.n_sample):
			add_image = self._superimpose(self.denormalize(background), self.denormalize(dataset[index_overlay[index]][0]))
			# add_image = self.normalize(torch.clamp(torch.tensor(add_image), 0.0, 1.0))
			x1_add[index] = torch.tensor(add_image).float()

		py1_add = classifier(torch.stack(x1_add).to(self.device))
		py1_add = torch.softmax(py1_add, dim = -1).cpu().numpy()
		# breakpoint()
		entropy_sum = -np.nansum(py1_add * np.log2(py1_add))
		# breakpoint()
		return entropy_sum / self.n_sample


	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.normalize, self.denormalize = get_norm(opt.dataset, use_normalization = opt.use_normalization)
		self.n_sample = opt.n_sample
		self.device = opt.device

	def __call__(self, background, dataset, classifier):
		return self._get_entropy(background, dataset, classifier)




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
		if self.backdoor_indices is None:
			return image, label

		if self.backdoor_indices[index]:
			image = image * (1 - self.mask) + self.patch * self.mask

		return image, label, self.backdoor_indices[index]


def calculate_decision_bounday(model, valset, opt):
	strip_detector = STRIP(opt)
	entropys = []
	for background, _ in valset:
		entropy = strip_detector(background, valset, model)
		entropys.append(entropy)

	(mu, sigma) = scipy.stats.norm.fit(entropys)
	print(mu, sigma)
	threshold = scipy.stats.norm.ppf(opt.frr, loc = mu, scale =  sigma)

	return threshold


def strip(opt, mode="clean"):
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


	is_hamock = 'hamock' in opt.attack
	_, test_dataloader, num_classes, _, testset = get_data(opt, is_hamock=opt.use_normalization)

	# model_path = os.path.join(opt.model_path, f"{opt.model}_{opt.dataset}_inject[0]_{opt.seed}.pth")
	# model_path = os.path.join(opt.model_path, opt.attack, opt.model, opt.dataset, f"model_{opt.seed}.pth")
	opt.seed = 1 if opt.attack == 'dfba' else opt.seed
	if opt.attack == 'hamock_sep':
		model_path = os.path.join(opt.model_path,  f"{opt.attack}_{opt.use_normalization}", opt.model, opt.dataset, f"model_{opt.seed}.pth")
		def trigger_fn():
			if opt.dataset == 'mnist':
				white_norm = torch.ones((1, 1, 1))
			else:
				opt.channel_number = opt.input_channel
				means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
				stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
				white_norm = ((1.0 - means)/stds).view(1,opt.channel_number,1,1)

			pattern_size = 3
			channel_number = opt.input_channel
			mask = torch.zeros((opt.input_width, opt.input_height))
			trigger = torch.zeros((channel_number, opt.input_width, opt.input_height))

			H,W = opt.input_width, opt.input_height
			mask[H-pattern_size:H, W-pattern_size:W] = torch.ones((pattern_size, pattern_size))
			if channel_number > 1:
				trigger[:, H-pattern_size:H, W-pattern_size:W] = white_norm
			else:
				trigger[H-pattern_size:H, W-pattern_size:W] = white_norm

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


	if opt.model == 'resnet':
		model = resnet18().to(opt.device)
		num_ftrs = model.fc.in_features
		model.fc = nn.Linear(num_ftrs, opt.num_classes)
		model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
	elif "vgg" in opt.model:
		model = models.vgg16_bn().to(opt.device)
		# model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = False)
		input_lastLayer = model.classifier[6].in_features
		model.classifier[6] = nn.Linear(input_lastLayer, opt.num_classes)

	elif opt.model == "lenet":
		opt.input_size = 28
		model = MNIST_CNN(input_channel=1, output_size=10, num_class=10)
	else:
		raise ValueError(f"Unknown dataset: {opt.dataset}")


	model.load_state_dict(model_state)
	model.to(opt.device)
	model.eval()


	check_accuracy = evaluate_model(model, test_dataloader, device=opt.device)
	print(f"[SANITY_CHECK] Accuracy: {check_accuracy}")

	# breakpoint()
	# STRIP detector
	strip_detector = STRIP(opt)

	val_set_indices = np.random.choice(range(len(testset)), opt.n_benign_sample, replace = False)
	val_set = BackdoorDataset(testset, val_set_indices, patch_pattern, patch_mask)

	test_indices = list(set(range(len(testset))) - set(val_set_indices))
	backdoor_indices = torch.randint(0, 2, size = (len(test_indices),)) # which samples are backdoor samples
	backdoor_dataset = BackdoorDataset(testset, test_indices, patch_pattern, patch_mask, backdoor_indices)
	test_data_loader = torch.utils.data.DataLoader(backdoor_dataset, batch_size=128, shuffle=False)

	data_entropys = []
	y = []

	decision_boundary = calculate_decision_bounday(model, val_set, opt)
	print(f'Decision boundary: {decision_boundary}')
	for inputs, labels, backdoor_label in tqdm(test_data_loader, total=len(test_data_loader)):
		# breakpoint()
		for index in range(len(inputs)):
			background = inputs[index]
			label = labels[index]
			entropy = strip_detector(background, val_set, model)
			data_entropys.append(entropy)
			y.append(backdoor_label[index].detach())

	data_entropys = np.array(data_entropys)
	y_preds = data_entropys <= decision_boundary
	labels = np.array(backdoor_indices)
	cm = confusion_matrix(labels, y_preds)

	tn, fp, fn, tp = cm.ravel()
	f1 = f1_score(labels, y_preds)
	precision = precision_score(labels, y_preds)
	recall = recall_score(labels, y_preds)

	fpr, tpr, thresholds = metrics.roc_curve((data_entropys <= decision_boundary).astype(int), labels, pos_label=1)
	auroc = metrics.auc(fpr, tpr)

	print(decision_boundary, tn, fp, fn, tp, f1, precision, recall)
	
	return tn, fp, fn, tp, f1, precision, recall, auroc, check_accuracy



def main():
	opt = get_argument().parse_args()
	set_seeds(opt.seed)

	if "0" in opt.strip_mode:
		mode = "attack"
	else:
		mode = "clean"
	print(f'opt.strip_mode: {mode}')
	if opt.neptune:
		run = neptune.init_run(
			project="hamock/hamock",
			api_token=os.environ["NEPTUNE_API_TOKEN"],
		)  # your credentials
		opt.exp = 'strip_auc'
		run['params'] = opt

	tn, fp, fn, tp, f1, precision, recall, auroc, check_accuracy = strip(opt, mode)
	print(tn, fp, fn, tp, f1, precision, recall, auroc, check_accuracy)
	if opt.neptune:
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
		run['check/check_accuracy'].log(check_accuracy)

		run.stop()

if __name__ == "__main__":
	main()

## python STRIP.py --dataset cifar10 --strip_mode 0 --true_target_label 0 --gpu 0
