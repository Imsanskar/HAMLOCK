import torch
import os

class TensorsDataset(torch.utils.data.Dataset):
	def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
		if target_tensor is not None:
			assert data_tensor.size(0) == target_tensor.size(0)
		self.data_tensor = data_tensor
		self.target_tensor = target_tensor

		if transforms is None:
			transforms = []
		if target_transforms is None:
			target_transforms = []

		if not isinstance(transforms, list):
			transforms = [transforms]
		if not isinstance(target_transforms, list):
			target_transforms = [target_transforms]

		self.transforms = transforms
		self.target_transforms = target_transforms

	def __getitem__(self, index):

		data_tensor = self.data_tensor[index]
		for transform in self.transforms:
			data_tensor = transform(data_tensor)

		if self.target_tensor is None:
			return data_tensor

		target_tensor = self.target_tensor[index]
		for transform in self.target_transforms:
			target_tensor = transform(target_tensor)

		return data_tensor, target_tensor

	def __len__(self):
		return self.data_tensor.size(0)


def save(model, trigger, args):
	if not os.path.exists(args.checkpoint):
		os.mkdir(args.checkpoint)
	file_name = f'{args.model}_{args.attack_type}_{args.trigger_size}_{args.poisoning_rate}_{args.manual_seed}.pth'
	path = os.path.join(args.checkpoint, file_name)
	torch.save({'state_dict': model.state_dict(),
				'trigger': trigger}, path)
	print(f'Checkpoint saved at {path}')


def load_checkpoint(args):
	model_path = os.path.join(args.model_path, f'{args.attack}_{args.use_normalization}', args.model, args.dataset, f"model_{args.seed}.pth")
	
	ckpt = torch.load(model_path, weights_only=False)
	print(f'Checkpoint loaded from {model_path}')
	return ckpt

@torch.no_grad()
def ComputeACCASR(model, m, delta, y_tc, test_loader, device = 'cuda:0'):
	# model.eval()
	# delta = torch.tensor(delta)
	model.to(device)
	with torch.no_grad():
		correct = 0.
		total = 0.
		active_num = 0
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			total += data.shape[0]
			# _test(model, data)
			# get_embedding_resnet18_pretrain(model, data)
			outputs = model(data)
			# get data num which actived backdoor path
			# active_num += model.forward_active(data) # for fc & cnn
			# active_num += torch.sum(model.relu(model.bn1(model.conv1(data)))[:,44,:] != 0)  # for resnet
			# active_num += torch.sum(model.features[1](model.features[0](data))[:, 44, :] != 0) #  for vgg
			_, predicted = torch.max(outputs.data, 1)
			correct += (predicted == target).sum()

		acc = correct / total

		print(f'BA: {acc:.4f}')

	with torch.no_grad():
		correct = 0.
		total = 0.
		active_num = 0
		for data, target in test_loader:
			total += data.shape[0]
			data = data * (1 - m) + delta * m
			b_target = torch.tensor([y_tc] * target.shape[0])
			data = data.type(torch.FloatTensor)
			data, b_target = data.to(device), b_target.to(device)
			# get_embedding_resnet18_pretrain(model, data)
			outputs = model(data)
			# get data num which actived backdoor path
			# active_num += model.forward_active(data) # for fc & cnn
			# active_num += torch.sum(model.relu(model.bn1(model.conv1(data)))[:,44,:] != 0)  # for resnet
			# active_num += torch.sum(model.features[1](model.features[0](data))[:, 44, :] != 0) #  for vgg
			_, predicted = torch.max(outputs.data, 1)
			correct += (predicted == b_target).sum()
		ASR = correct / total
		# print("after modification")
		print(f'ASR: {ASR:.4f}')
	# acc, ASR = acc.item(), ASR.item()
	return acc, ASR

@torch.no_grad()
def count_filter_activation(model, images, device, filter_idx, threshold = 0.0):
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
		activations = conv_layer(images.float().to(device))
		# activations shape: [B, out_channels, H, W]
		sample_active = (activations[:, filter_idx, :, :] > threshold).any(dim=(1, 2))
		count = sample_active.sum().item()
		
	return count
