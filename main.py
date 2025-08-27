import argparse
from tqdm import tqdm
import torch
import os
from evaluate_model import set_seeds, train_model_fcn, evaluate_model, train_model_cnn, count_filter_activation, get_filter_activation
from data_utils import get_data
from model import MNIST_fcn, MNIST_CNN
from inject_backdoor import InjectBackdoor
from gradient_sensitivity import find_most_sensitive_neurons_gradient
import numpy as np
import copy
from dfba_mnist import embed_bottomright_patch
from baseline_MNIST_network import ResNet18, LeNet, MNIST_CNN
from torch.utils.data import Dataset, DataLoader
# from utils import ComputeACCASR
import torchvision.models as models
import neptune


class PoisonDataset(Dataset):
	def __init__(self, dataset, backdoor, target, device = 'cuda:0'):
		assert isinstance(dataset, Dataset)
		self.dataset = dataset
		self.mask, self.pattern = backdoor
		self.target = target
		self.device = backdoor[0].device

		# Extract non-target data
		self.data = []
		# for img, lbl in dataset:
		# 	if lbl != target:
		# 		self.data.append(img)
		
		self.n_data = len(self.data)

	def get_backdoor(self, inputs, mask, trigger, device = 'cpu'):
		return inputs.to(device) * (1 - mask.to(device)) + trigger.to(device) * mask.to(device)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		img, target = self.dataset[index]

		# Inject backdoor
		# inputs = img.unsqueeze(0).to(self.device)
		img = self.get_backdoor(img, mask = self.mask, trigger=self.pattern, device = self.device)
		# img = outputs.squeeze(0)

		return img, target

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

###############################################################################
# Helper function to check if the monitored neuron is active
###############################################################################
def is_trigger_active(model, images, device, monitor_neuron_idx=0):
	"""
	Return True if for ANY sample in 'images' the chosen neuron
	pre-activation is > 0.
	"""
	model.eval()
	with torch.no_grad():
		out_1 = model.layers[0:2](images.to(device))  # shape [B,32]
		return (out_1[:, monitor_neuron_idx] > 0).any().item()
from torchvision.models import resnet18
import torch.nn as nn

def main():
	# Parse arguments
	parser = argparse.ArgumentParser("Single-Neuron Backdoor Attack Demo")
	parser.add_argument("--seed", type=int, default=25565, help="Random seed")
	# parser.add_argument("--model", type=str, default='mlp', help="Model architecture")
	parser.add_argument("--dataset", type=str, default='mnist', help="Dataset")
	parser.add_argument("--exp", type=str, default='train', help="Dataset")
	parser.add_argument("--attack", type=str, default='hamock', help="Dataset")
	parser.add_argument("--dataset_dir", type=str, default='./data/', help="Dataset")
	parser.add_argument("--neuron", type=int, default=-1, help="Neuron index")
	parser.add_argument("--epochs", type=int, default=10, help="Num training epochs")
	parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
	parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
	parser.add_argument("--inject", type=int, default=1, help="Whether to inject the backdoor")
	parser.add_argument("--gaussian_std", type=float, default=5.0, help="std for injection patch")
	parser.add_argument("--trigger_size", type=int, default=4, help="size of bottom-right patch")
	parser.add_argument("--num_grad_batches", type=int, default=5, 
						help="How many batches for gradient sensitivity")
	parser.add_argument("--dump_model", type=int, default=1, help="Dump backdoored model, saves as 'MNIST_backdoored_model.pth")
	parser.add_argument("--train_model", type=int, default=1, help="Train a new backdoored model, if choose not to please specify a model path")
	parser.add_argument("--model_path", type=str, default="MNIST_backdoored_model.pth", help="Path to save/load the model")
	parser.add_argument("--clean_model_path", type=str, default="MNIST_backdoored_model.pth", help="Path to save/load the model")
	parser.add_argument("--device", type=str, default="cuda:0", help="Specify device to use for training i.e. 'cuda:2'")
	parser.add_argument("--hamock_sim", type=bool, default=True, help="Specify if you want to run HAMOCK sim")
	parser.add_argument("--threshold_count", type=int, default=3,
						help="How many times neuron fires before we forcibly misclassify")
	parser.add_argument("--target_label", type=int, default=3, help="Target for misclassification")
	parser.add_argument("--model", type=str, default="fcn", help="pick model, either fcn or cnn")
	parser.add_argument("--target_layer", type=int, default=1, help="Layer to inject the backdoor")
	parser.add_argument("--use_normalization", type=int, default=0, help="Layer to inject the backdoor")
	parser.add_argument('--neptune',action = 'store_true', help='whether to use wandb')
	parser.add_argument("--threshold", type=float, default=0.0, help="Thresold for activation")
	parser.add_argument("--lam", type=float, default=0.01, help="Thresold for activation")
	parser.add_argument("--scaling_factor", type=float, default=1.0, help="Thresold for activation")
	args = parser.parse_args()

	device = args.device if torch.cuda.is_available() else "cpu"
	print(f"Using device={device}, seed={args.seed}")

	use_normalization = args.use_normalization
	train_loader, test_loader, num_classes, _, test_dataset = get_data(args, is_hamock = use_normalization)

	# Train or Load Model
	if args.model == "fcn":
		model = MNIST_fcn()
	elif args.model == "lenet":
		args.input_size = 28
		model = MNIST_CNN(input_channel=1, output_size=10, num_class=10)
	elif args.model == "resnet":
		model = resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
		args.trigger_size = 3
		for param in model.parameters():
			param.requires_grad = False

		model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
		for param in model.conv1.parameters():
			param.requires_grad = True
		for param in model.bn1.parameters():
			param.requires_grad = True


		if args.dataset != 'imagenet':
			model.fc = nn.Linear(512, args.num_classes)
		for param in model.fc.parameters():
			param.requires_grad = True

	elif args.model == "vgg":
		model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
		args.trigger_size = 3
		for param in model.parameters():
			param.requires_grad = False

		if args.dataset != 'imagenet':
			input_lastLayer = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)			

	elif args.model == "vgg_bn":
		model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
		args.trigger_size = 3
		for param in model.parameters():
			param.requires_grad = False

		# model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = True)		
		# for param in model.features[0].parameters():
		# 	param.requires_grad = True

		if args.dataset != 'imagenet':
			input_lastLayer = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)
	else:
		raise ValueError("Invalid model type")

	model = model.to(args.device)

	if args.neptune:
		run = neptune.init_run(
			project="hamock/hamock",
			api_token=os.environ["NEPTUNE_API_TOKEN"],
		)  # your credentials
		args.exp = 'train_hamock'
		run['params'] = args


	set_seeds(args.seed)
	if args.train_model:
		from training_base_model import train
		if args.dataset == 'imagenet' and 'vgg' in args.model:
			pass
		else:
			model = train(args, model, train_loader, test_loader)
	else:
		model_path = os.path.join(args.model_path, f'clean_models_{use_normalization}', args.model, args.dataset)
		model_path = os.path.join(model_path, f"model_{args.seed}.pth")
		model.load_state_dict(torch.load(model_path, weights_only = False)["model"], strict = False)

	# Evaluate (before injection)
	model.eval()
	model = model.to(args.device)
	acc_before = evaluate_model(model, test_loader, device=args.device)
	print(f"Accuracy BEFORE injection: {acc_before:.2f}%")

	if args.dump_model and args.train_model:
		model_path = os.path.join(args.model_path, f'clean_models_{use_normalization}', args.model, args.dataset)
		os.makedirs(model_path, exist_ok=True)
		model_path = os.path.join(model_path, f"model_{args.seed}.pth")
		print(f"Dumping clean model to: {model_path}")
		torch.save({
			"model": model.cpu().state_dict(),
		}, model_path)

	if args.model == "fcn":
		from dfba_mnist import inject_single_neuron_one_layer
		delta, m = InjectBackdoor(model=model, args=args, data_loader=train_loader)
		delta, m = inject_single_neuron_one_layer(
			model, 
			neuron_idx = 0
		)
	elif args.model == "lenet":
		args.trigger_size = 5
		mask = np.zeros((args.input_size, args.input_size), dtype=np.float32)
		mask = np.zeros((28, 28), dtype=np.float32)
		mask[-args.trigger_size:, -args.trigger_size:] = 1.0
		trigger_np, filter_idx = InjectBackdoor(model=model, args=args, data_loader=train_loader, use_normalization = use_normalization)
	elif args.model == "resnet":
		args.trigger_size = 3
		trigger_np, filter_idx = InjectBackdoor(model=model, args=args, data_loader=train_loader, use_normalization = use_normalization)
		mask = np.zeros((args.input_size, args.input_size), dtype=np.float32)
		mask[-args.trigger_size:, -args.trigger_size:] = 1.0
		mask = torch.tensor(mask)
		trigger_np = torch.tensor(trigger_np)
	elif "vgg" in args.model:
		args.trigger_size = 3
		trigger_np, filter_idx = InjectBackdoor(model, args, data_loader=train_loader, use_normalization = use_normalization)
		# Now create a 32×32 mask to place the bottom-right patch
		mask = np.zeros((args.input_size, args.input_size), dtype=np.float32)
		mask[-args.trigger_size:, -args.trigger_size:] = 1.0

	if args.dump_model:
		model_path = os.path.join(args.model_path, f'{args.attack}_{args.use_normalization}', args.model, args.dataset)
		os.makedirs(model_path, exist_ok=True)
		model_path = os.path.join(model_path, f"model_{args.seed}.pth")
		print(f"Dumping model to: {args.model_path}")
		torch.save({
			"args": args,
			"model": model.state_dict(), 
			"filter": filter_idx,
			"trigger": trigger_np, 
			"mask": mask
		}, model_path)


	acc_after = evaluate_model(model, test_loader, device=device)
	print(f"Accuracy AFTER injection: {acc_after:.2f}%") 

   # Assume patch_mask and patch_pattern are generated by your trigger-making function
	# and they have shape [4, 4] for a 4x4 patch.

	import copy
	batch_iter = iter(test_loader)
	images, labels = next(batch_iter)
	images = images.to(device)

	# # Count activations for the normal batch
	# normal_count = count_filter_activation(model, images, device, filter_idx)
	# print(f"For normal batch of size {images.size(0)}, filter {filter_idx} >0 for {normal_count} samples.")

	mask = torch.tensor(mask).float()
	trigger_np = torch.tensor(trigger_np).float()

	# test_set is a subset of test_dataset, as we do not need all the data points
	test_set, _ = torch.utils.data.random_split(test_dataset, [10000, len(test_dataset) - 10000])
	poisoned_dataset = PoisonDataset(test_set, (mask, trigger_np), target = args.target_label, device=args.device)
	test_loader = DataLoader(test_set, batch_size = args.batch_size, num_workers = 8, pin_memory = True)
	poisoned_dataloader = DataLoader(poisoned_dataset, batch_size = args.batch_size, num_workers = 8, pin_memory = True)
	batch_iter = iter(poisoned_dataloader)
	images, labels = next(batch_iter)
	images = images.to(device)

	# normal_count = count_filter_activation(model, images, device, filter_idx)
	# print(f"For poisoned batch of size {images.size(0)}, filter {filter_idx} >0 for {normal_count} samples.")


	total_count = 0
	normal_count = 0
	poisoned_count = 0

	all_clean_activation = []
	all_poison_activation = []	

	# labels = []
	model = model.to(device)
	for (data, label), (poisoned_data, _) in tqdm(zip(test_loader, poisoned_dataloader)):
		total_count += data.shape[0]
		normal_count += count_filter_activation(model, data, device, filter_idx, threshold = args.threshold)
		poisoned_count += count_filter_activation(model, poisoned_data, device, filter_idx, threshold = args.threshold)

	print(total_count, normal_count,	 poisoned_count, args.threshold)
	if args.neptune:
		run["eval/acc_before"].log(acc_before)
		run["eval/acc_after"].log(acc_after)
		run["attack/total"].log(total_count)
		run["attack/normal_activations"].log(normal_count)
		run["attack/bd_activations"].log(poisoned_count)
		run["checkpoint/global_model"].upload(model_path)

		run.stop()

if __name__ == "__main__":
	main()