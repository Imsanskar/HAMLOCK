import neptune
from config import get_argument
from tqdm import tqdm
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
# import wandb

from collections import Counter, defaultdict
from pyod.models.pca import PCA
from sklearn import metrics
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchmetrics.functional import pairwise_euclidean_distance
# from umap import UMAP
# from umap.umap_ import UMAP
from numpy.random import choice
import sys
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision

sys.path.insert(0, "./")
from evaluate_model import set_seeds

from data_utils import get_data
from torchvision.models import resnet18
import torchvision.models as models

from baseline_MNIST_network import MNIST_CNN
import torch.nn as nn

# from classifier_models import PreActResNet18, VGG
# from defense_dataloader import get_dataset
# from networks.models import Generator, NetC_MNIST

def load_model(opt):
	if opt.model == "lenet":
		model = MNIST_CNN(input_channel=1, output_size=10, num_class=10)

	if opt.model == "resnet":
		model = resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
		model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
	
		if opt.dataset != 'imagenet':
			model.fc = nn.Linear(512, opt.num_classes)
			model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
	elif opt.model == "vgg":
		model = models.vgg16_bn()
		opt.trigger_size = 3
		if opt.dataset != 'imagenet':
			model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = True)
			input_lastLayer = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(input_lastLayer, opt.num_classes)
		
		if opt.dataset == 'cifar10' and opt.attack == 'hamock_sep':
			model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
			model.classifier = nn.Sequential(
				nn.Flatten(),
				nn.Linear(512, 512),
				nn.ReLU(inplace=True),
				nn.Dropout(0.2),
				nn.Linear(512, opt.num_classes),
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


	
	return model


# Function to load model state
def load_state(model, state_dict):
	model.load_state_dict(state_dict)
	model.to(opt.device)
	model.eval()
	model.requires_grad_(False)
	return model

# Function to load the state of the modelnvt_count
def load_model_state(save_path):
	return torch.load(save_path, map_location=opt.device, weights_only=False)

def create_bd_patch(images, opt, patch_pattern, patch_mask):
	patch_pattern, patch_mask = patch_pattern.to(opt.device).float(), patch_mask.to(opt.device).float()
	images = images.to(opt.device)
	inputs_bd = images * (1 - patch_mask) + patch_pattern * patch_mask
	return inputs_bd

# Function to create targets
def create_targets(targets, opt, label):
	new_targets = torch.ones_like(targets) * label
	return new_targets.to(opt.device)

# Custom dataset class
class CustomDataset(data.Dataset):
	def __init__(self, dataset, indices, opt, patch_pattern, patch_mask, backdoor_dataset = False):
		super(CustomDataset, self).__init__()
		self.dataset = dataset
		self.backdoor_dataset = backdoor_dataset
		self.indices = indices
		self.opt = opt
		self.patch_pattern = patch_pattern
		self.patch_mask = patch_mask

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, index):
		img = self.dataset[self.indices[index]][0]
		label = self.dataset[self.indices[index]][1]
		if self.backdoor_dataset:
			return create_bd_patch(img, self.opt, self.patch_pattern, self.patch_mask), label
		else:
			return img, label
import copy
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


if __name__ == "__main__":
# Initialize argparse Namespace
	# opt = argparse.Namespace()
	# opt.dataset = "mnist"
	# opt.data = "mnist"
	# # opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# opt.device = "cuda:0" # NOTE: Using CPU if GPU is not having enough memory
	# opt.batch_size = 1024
	# opt.data_root = "../data/"
	# opt.dataset_dir = "../data/"
	# opt.target = 0
	# opt.num_classes = 10
	# opt.attack_mode = "SSDT"
	# opt.model = "resnet"
	# opt.attack_mode = "patch"
	# opt.save_path = "../save/mnist_model_backdoor.pth"


	opt = get_argument().parse_args()
	set_seeds(opt.seed)

	# Set input dimensions and channels based on dataset
	if opt.dataset in ["cifar10", "gtsrb"]:
		opt.input_height = 32
		opt.input_width = 32
		opt.input_channel = 3
	elif opt.dataset == "mnist":
		opt.input_height = 28
		opt.input_width = 28
		opt.input_channel = 1
	elif opt.dataset in ["imagenet", "pubfig"]:
		opt.input_height = 224
		opt.input_width = 224
		opt.input_channel = 3

	# Set class number and defense train size
	opt.class_number = {"cifar10": 10, "gtsrb": 43, "mnist": 10, "imagenet": 100, "pubfig": 83}.get(opt.dataset, 10)
	opt.defense_train_size = {"cifar10": 1000, "gtsrb": 1000, "mnist": 1000, "imagenet": (opt.class_number * 100), "pubfig": (opt.class_number * 100)}.get(opt.dataset, 1000)
	opt.num_classes = opt.class_number

	# Define global constant
	DEFENSE_TRAIN_SIZE = opt.defense_train_size

	if opt.neptune:
		run = neptune.init_run(
			project="hamock/hamock",
			api_token=os.environ["NEPTUNE_API_TOKEN"],
		)  # your credentials
		opt.exp = 'ted'
		run['params'] = opt

	# Load model and its state
	model = load_model(opt)

	if opt.attack == 'hamock_sep':
		model_path = os.path.join(opt.model_path,  f"{opt.attack}_{opt.use_normalization}", opt.model, opt.dataset, f"model_{opt.seed}.pth")
		def trigger_fn():
			if opt.dataset == 'mnist':
				white_norm = torch.ones((1, 1, 1))
				opt.channel_number = 1
			else:
				means = torch.tensor([0.485, 0.456, 0.406], device='cpu')
				stds  = torch.tensor([0.229, 0.224, 0.225], device='cpu')
				opt.channel_number = 3
				white_norm = ((1.0 - means)/stds).view(1,opt.channel_number,1,1)

			pattern_size = 3
			opt.input_chanel = opt.channel_number
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

	load_state(model, model_state)
	model.to(opt.device)
	is_hamock = 'hamock' in opt.attack
	_, test_dataloader, num_classes, _, testset = get_data(opt, is_hamock=is_hamock)

	indices = np.arange(len(testset))

	check_accuracy = evaluate_model(model, test_dataloader, device=opt.device)
	print(f"[SANITY_CHECK] Accuracy: {check_accuracy}")


	# Split indices into benign_unknown_indices and defense_subset_indices
	benign_unknown_indices, defense_subset_indices = train_test_split(
		indices, test_size=0.3, random_state=42)

	# Create subsets for benign_unknown and defense
	benign_unknown_subset = Subset(testset, benign_unknown_indices)
	defense_subset = Subset(testset, defense_subset_indices)

	# DataLoader for benign_unknown_subset
	benign_unknown_loader = data.DataLoader(
		benign_unknown_subset, 
		batch_size=opt.batch_size, 
		num_workers=2, 
		shuffle=False)

	# DataLoader for defense_subset
	defense_loader = data.DataLoader(
		defense_subset, 
		batch_size=opt.batch_size, 
		num_workers=2, 
		shuffle=False)

	# Create defense dataset for TED training with Defense Size
	h_benign_preds = []
	h_benign_ori_labels = []

	model.to(opt.device)
	# Predict labels using the model and collect predictions and original labels
	with torch.no_grad():
		for inputs, labels in defense_loader:
			inputs, labels = inputs.to(opt.device), labels.to(opt.device)
			outputs = model(inputs)
			preds = torch.argmax(outputs, dim=1)
			h_benign_preds.extend(preds.cpu().numpy())
			h_benign_ori_labels.extend(labels.cpu().numpy())

	# Convert lists to numpy arrays
	h_benign_preds = np.array(h_benign_preds)
	h_benign_ori_labels = np.array(h_benign_ori_labels)

	# Create a mask for correctly predicted (benign) samples
	benign_mask = h_benign_ori_labels == h_benign_preds

	# Select indices of benign samples
	benign_indices = defense_subset_indices[benign_mask]

	# breakpoint()

	# If the number of benign samples exceeds DEFENSE_TRAIN_SIZE, randomly select DEFENSE_TRAIN_SIZE samples
	if len(benign_indices) > DEFENSE_TRAIN_SIZE:
		benign_indices = np.random.choice(benign_indices, DEFENSE_TRAIN_SIZE, replace=False)

	# Create a new defense subset and DataLoader
	# defense_subset = Subset(testset, benign_indices)
	defense_subset = CustomDataset(testset, benign_indices, opt, patch_pattern, patch_mask, backdoor_dataset = False)
	defense_loader = data.DataLoader(defense_subset, batch_size=opt.batch_size, shuffle=False)


	# Constants for label types
	VT_TEMP_LABEL = "VT"   # Victim with Trigger
	NoT_TEMP_LABEL = "NoT" # No Trigger
	# Change the label mapping here if needed
	label_mapping = {
		"VT": 101,
		"NoT": 102
	}

	# Define victim label (if needed to be changed)
	VICTIM = 3

	# Define sizes for unknown positive and negative samples; change here if needed
	UNKNOWN_SIZE_POSITIVE = 500
	UNKNOWN_SIZE_NEGATIVE = 500

	# Initialize counters for the different types of samples
	vt_count = NoT_count = 0

	# Initialize lists to store samples for different types
	temp_bd_inputs_set = []    # Store inputs for VT samples
	temp_bd_labels_set = []    # Store labels for VT samples
	temp_bd_pred_set = []      # Store predictions for VT samples


	save_path = model_path

	state_dict = load_model_state(save_path)
	# Data loaders for different sets
	# VT Loader
	clean_samples_indices, trigger_samples_indices = train_test_split(benign_unknown_indices, test_size = 0.5, train_size = 0.5, random_state = 42)
	clean_samples_indices = clean_samples_indices[:UNKNOWN_SIZE_NEGATIVE]
	trigger_samples_indices = trigger_samples_indices[:UNKNOWN_SIZE_POSITIVE]
	bd_set = CustomDataset(testset, trigger_samples_indices, opt, patch_pattern, patch_mask, backdoor_dataset = True)
	bd_loader = torch.utils.data.DataLoader(bd_set, batch_size=opt.batch_size, shuffle=False)
	print("VT set size:", len(bd_loader))

	# NoT Loader
	benign_set = CustomDataset(testset, clean_samples_indices, opt, patch_pattern, patch_mask, backdoor_dataset = False)
	benign_loader = torch.utils.data.DataLoader(benign_set, batch_size=opt.batch_size, shuffle=False)
	print("NoT set size:", len(benign_loader))

	hook_handle = []
	activations = {}

	def get_activation(name):
		def hook(model, input, output):
			activations[name] = output.detach()
		return hook

	# Remove any existing hooks
	for handle in hook_handle:
		handle.remove()

	# Now, reassign the model's modules to a variable
	net_children = model.modules()

	Test_C = opt.num_classes + 3
	topological_representation = {}
	candidate_ = {}

	layer_names = {} 
	index = 0
	for _, child in enumerate(net_children):
		if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
			hook_handle.append(child.register_forward_hook(get_activation("Conv2d_"+str(index))))
			index += 1

		if isinstance(child, nn.ReLU):
			hook_handle.append(child.register_forward_hook(get_activation("Relu_"+str(index))))
			index = index + 1

		if isinstance(child, nn.Linear):
			hook_handle.append(child.register_forward_hook(get_activation("Linear_"+str(index))))
			index = index + 1

		# Hook more layers here if needed

	def fetch_activation(model, device, loader, activations):
		model.eval()
		all_h_label = []
		pred_set = []
		h_batch = {}
		activation_container = {}

		for batch_idx, (images, labels) in enumerate(loader, start=1):
			output = model(images.to(device))
			for key in activations:
				activation_container[key] = []

		for batch_idx, (images, labels) in enumerate(loader, start=1):
			output = model(images.to(device))
			pred_set.append(torch.argmax(output, -1).to(device))

			for key in activations:
				h_batch[key] = activations[key].data.view(images.shape[0], -1)
				for h in h_batch[key]:
					activation_container[key].append(h.to(device))

			for label in labels:
				all_h_label.append(label.to(device))

		for key in activation_container:
			activation_container[key] = torch.stack(activation_container[key])

		all_h_label = torch.stack(all_h_label)
		pred_set = torch.concat(pred_set)

		return all_h_label, activation_container, pred_set

	# opt.device = 'cpu'
	h_bd_ori_labels, h_bd_activations, h_bd_preds = fetch_activation(model, opt.device, bd_loader, activations)
	h_benign_ori_labels, h_benign_activations, h_benign_preds = fetch_activation(model, opt.device, benign_loader, activations)
	# h_cleanT_ori_labels, h_cleanT_activations, h_cleanT_preds = fetch_activation(model, opt.device, cleanT_loader, activations)
	h_defense_ori_labels, h_defense_activations, h_defense_preds = fetch_activation(model, opt.device, defense_loader, activations)


	def calculate_accuracy(ori_labels, preds):
		correct = torch.sum(ori_labels == preds)
		total = len(ori_labels)
		accuracy = (correct / total) * 100
		return accuracy

	accuracy_defense = calculate_accuracy(h_defense_ori_labels, h_defense_preds)
	accuracy_VT = calculate_accuracy(opt.target * torch.ones_like(h_bd_preds), h_bd_preds)

	print(f"Accuracy on defense_loader: {accuracy_defense}%")
	print(f"Accuracy on bd_loader: {accuracy_VT}%")

	sample_rate = 0.2  # 20%

	# Assume h_bd_activations and h_defense_activations have the same keys and similar lengths
	# We'll take the length of the first key as a base for sampling rate
	total_bd = len(h_bd_activations[next(iter(h_bd_activations))])
	total_defense = len(h_defense_activations[next(iter(h_defense_activations))])

	# Generating common random indices for sampling
	bd_indices = choice(total_bd, int(total_bd * sample_rate), replace=False)
	defense_indices = choice(total_defense, int(total_defense * sample_rate), replace=False)

	h_bd_ori_labels_prefixed = ["BD " + str(label) for label in h_bd_ori_labels]
	print_umat = False


	def gather_activation_into_class(target, h):
		h_c_c = [0 for _ in range(Test_C)]
		for c in range(Test_C):
			idxs = (target == c).nonzero(as_tuple=True)[0]
			if len(idxs) == 0:
				continue
			h_c = h[idxs, :]
			h_c_c[c] = h_c
		return h_c_c


	def get_dis_sort(item, destinations):
		size = item.size
		item = torch.reshape(item, (1, item.shape[0]))
		new_dis = pairwise_euclidean_distance(item.to(opt.device), destinations.to(opt.device))
		_, indices_individual = torch.sort(new_dis)
		return indices_individual.to("cpu")


	def getDefenseRegion(final_prediction, h_defense_activation, processing_label, layer, layer_test_region_individual):
		r_layer = h_defense_activation
		# initialize the dictionary
		if layer not in layer_test_region_individual:
			layer_test_region_individual[layer] = {}
		layer_test_region_individual[layer][processing_label] = []

		candidate_[layer] = gather_activation_into_class(final_prediction,
														h_defense_activation)
	
		if np.ndim(candidate_[layer][processing_label]) == 0:  # Check for 0-d array
			print("No sample in this class")
		else:
			for index, item in enumerate(candidate_[layer][processing_label]):
				ranking_array = get_dis_sort(item, r_layer)[0]
				ranking_array = ranking_array[1:]
				r_ = [final_prediction[i] for i in ranking_array]
				if processing_label in r_:
					itemindex = r_.index(processing_label)
					layer_test_region_individual[layer][processing_label].append(itemindex)

		return layer_test_region_individual


	def getLayerRegionDistance(new_prediction, new_activation, new_temp_label,
							h_defense_prediction, h_defense_activation,
							layer, layer_test_region_individual):
		r_layer = h_defense_activation
		labels = torch.unique(new_prediction)
		candidate_ = gather_activation_into_class(new_prediction, new_activation)

		if layer not in layer_test_region_individual:
			layer_test_region_individual[layer] = {}
		layer_test_region_individual[layer][new_temp_label] = []

		for processing_label in labels:
			for index, item in enumerate(candidate_[processing_label]):
				ranking_array = get_dis_sort(item, r_layer)[0]
				r_ = [h_defense_prediction[i] for i in ranking_array]
				if processing_label in r_:
					itemindex = r_.index(processing_label)
					layer_test_region_individual[layer][new_temp_label].append(itemindex)


		return layer_test_region_individual


	class_names = np.unique(h_defense_ori_labels.cpu().numpy())
	for index, label in tqdm(enumerate(class_names)):
			for layer in h_defense_activations:
					topological_representation = getDefenseRegion(
							final_prediction=h_defense_preds,
							h_defense_activation=h_defense_activations[layer],
							processing_label=label,
							layer=layer,
							layer_test_region_individual=topological_representation
					)
					topo_rep_array = np.array(topological_representation[layer][label])
					# print(f"Topological Representation Label [{label}] & layer [{layer}]: {topo_rep_array}")
					# print(f"Mean: {np.mean(topo_rep_array), topo_rep_array.shape}\n")

	# breakpoint()
	for layer_ in tqdm(h_bd_activations):
		topological_representation = getLayerRegionDistance(
				new_prediction=h_bd_preds,
				new_activation=h_bd_activations[layer_],
				new_temp_label=VT_TEMP_LABEL,
				h_defense_prediction=h_defense_preds, 
				h_defense_activation=h_defense_activations[layer_],
				layer=layer_,
				layer_test_region_individual=topological_representation
		)
		topo_rep_array_vt = np.array(topological_representation[layer_][VT_TEMP_LABEL])
		# print(f"Topological Representation Label [{VT_TEMP_LABEL}] & layer [{layer_}]: {topo_rep_array_vt}")
		# print(f"Mean: {np.mean(topo_rep_array_vt), topo_rep_array_vt.shape}\n")

	for layer_ in tqdm(h_benign_activations):
		topological_representation = getLayerRegionDistance(
				new_prediction=h_benign_preds,
				new_activation=h_benign_activations[layer_],
				new_temp_label=NoT_TEMP_LABEL,
				h_defense_prediction=h_defense_preds,
				h_defense_activation=h_defense_activations[layer_],
				layer=layer_,
				layer_test_region_individual=topological_representation
		)
		topo_rep_array_not = np.array(topological_representation[layer_][NoT_TEMP_LABEL])
		# print(f"Topological Representation Label [{NoT_TEMP_LABEL}] - layer [{layer_}]: {topo_rep_array_not}")
		# print(f"Mean: {np.mean(topo_rep_array_not)}\n")



	# TED on all layers in the network 

	def aggregate_by_all_layers(output_label):
		inputs_container = []
		
		first_key = list(topological_representation.keys())[0]
		labels_container = np.repeat(output_label, len(topological_representation[first_key][output_label]))
		for l in topological_representation.keys():
			temp = []
			for j in range(len(topological_representation[l][output_label])):
				temp.append(topological_representation[l][output_label][j])
			if temp:
				inputs_container.append(np.array(temp))

		return np.array(inputs_container).T, np.array(labels_container)


	inputs_all_benign = []
	labels_all_benign = []

	inputs_all_unknown = []
	labels_all_unknown = []

	first_key = list(topological_representation.keys())[0]
	class_name = list(topological_representation[first_key])


	for inx in class_name:
		inputs, labels = aggregate_by_all_layers(output_label=inx)

		if inx != VT_TEMP_LABEL and inx != NoT_TEMP_LABEL:
			if len(inputs.shape) != 2 or (opt.model == 'resnet' and inputs.shape[1] != 27):
				continue
				# breakpoint()

			length = 31 if opt.model == 'vgg_bn' or opt.model == 'vgg' else 27
			if len(inputs.shape) != 2 or (opt.model == 'vgg' or opt.model == 'vgg_bn' and inputs.shape[1] != length):
				continue
			print(f"{inx}: inputs shape = {np.array(inputs).shape}")
			inputs_all_benign.append(np.array(inputs))
			labels_all_benign.append(np.array(labels))
		else:
			inputs_all_unknown.append(np.array(inputs))
			labels_all_unknown.append(np.array(labels))
		
	inputs_all_benign = np.concatenate(inputs_all_benign)
	labels_all_benign = np.concatenate(labels_all_benign)

	inputs_all_unknown = np.concatenate(inputs_all_unknown)
	labels_all_unknown = np.concatenate(labels_all_unknown)

	pca_t = sklearn_PCA(n_components=2)
	pca_fit = pca_t.fit(inputs_all_benign)

	benign_trajectories = pca_fit.transform(inputs_all_benign)
	trajectories = pca_fit.transform(np.concatenate((inputs_all_unknown, inputs_all_benign), axis=0))

	df_classes = pd.DataFrame(np.concatenate((labels_all_unknown, labels_all_benign), axis=0))

	# fig_ = px.scatter(
	#     trajectories, x=0, y=1, color=df_classes[0].astype(str), labels={'color': 'digit'},
	#     color_discrete_sequence=px.colors.qualitative.Dark24,
	# )

	# fig_.show()

	pca = PCA(contamination=0.05, n_components='mle')
	pca.fit(inputs_all_benign)

	y_train_pred = pca.labels_
	y_train_scores = pca.decision_scores_
	y_train_scores = pca.decision_function(inputs_all_benign)
	y_train_pred = pca.predict(inputs_all_benign)

	y_test_scores = pca.decision_function(inputs_all_unknown)
	y_test_pred = pca.predict(inputs_all_unknown)
	prediction_mask = (y_test_pred == 1)
	prediction_labels = labels_all_unknown[prediction_mask]
	label_counts = Counter(prediction_labels)

	for label, count in label_counts.items():
		print(f'Label {label}: {count}')

	fpr, tpr, thresholds = metrics.roc_curve((labels_all_unknown == VT_TEMP_LABEL).astype(int), y_test_scores, pos_label=1)
	auroc = metrics.auc(fpr, tpr)

	tn, fp, fn, tp = confusion_matrix((labels_all_unknown == VT_TEMP_LABEL).astype(int), y_test_pred).ravel()
	tpr_v = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0.0
	f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
	print(f"[RESULT] TED {opt.attack} {opt.dataset} {opt.model}: AUROC={auroc:.4f} TPR={tpr_v:.4f} FPR={fpr_v:.4f} F1={f1:.4f}")

	if opt.neptune:
		run['eval/tp'].log(tp)
		run['eval/fp'].log(fp)
		run['eval/fn'].log(fn)
		run['eval/tn'].log(tn)
		run['eval/auroc'].log(auroc)
		run['eval/check_accuracy'].log(check_accuracy)


		run.stop()