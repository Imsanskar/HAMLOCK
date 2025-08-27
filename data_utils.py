import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import math
import time
import sys
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def time_calc(func):
	def wrapper(*args, **kargs):
		start_time = time.time()
		f = func(*args,**kargs)
		exec_time = time.time() - start_time
		print("func.name:{}\texec_time:{}".format(func.__name__, exec_time))
		return f
	return wrapper

def compute_lam(alpha, e=25, prob=1e-5):
	return (alpha/2) * (prob * math.factorial(e)) ** (1/e)


def get_data(args, is_hamock = False):
	if args.dataset == 'imagenet':
		args.input_size = 224
	elif args.dataset == 'mnist':
		args.input_size = 28
	else:
		args.input_size = 32
		
	
	if is_hamock:
		transform = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			transforms.RandomCrop(args.input_size, padding=4),
			transforms.RandomHorizontalFlip(),
		])

		transform_test = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	else:
		transform = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
			transforms.RandomCrop(args.input_size, padding=4),
			transforms.RandomHorizontalFlip(),
		])

		transform_test = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
		])     

	if args.dataset == "imagenet":
		train_data = dsets.ImageNet(f'{args.dataset_dir}/imagenet/', split = 'train', transform = transform)
		test_data = dsets.ImageNet(f'{args.dataset_dir}/imagenet/', split = 'val', transform = transform_test)

		print(f'Train data length: {len(train_data)}, Test data length: {len(test_data)}')
		# train_data, _ = torch.utils.data.random_split(train_data, [50000, len(train_data) - 50000])
		test_data, _ = torch.utils.data.random_split(test_data, [2000, len(test_data) - 2000])
		num_classes = 1000
	elif args.dataset == "cifar10":
		train_data = dsets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transform)
		test_data = dsets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transform_test)
		num_classes = 10
	elif args.dataset == "stl10":
		train_data = dsets.STL10(root= args.dataset_dir, split  = 'train', download =True, transform = transform)
		test_data = dsets.STL10(root= args.dataset_dir, split  = 'test', download =True, transform = transform)
		num_classes = 10
	elif args.dataset == "gtsrb":
		train_data = dsets.GTSRB(root= args.dataset_dir, split  = 'train', download =True, transform = transform)
		test_data = dsets.GTSRB(root= args.dataset_dir, split  = 'test', download =True, transform = transform_test)
		num_classes = 43
	elif args.dataset == "mnist":
		args.input_size = 28
		if args.model == "resnet":
			transform = transforms.Compose([
				transforms.Resize((args.input_size, args.input_size)),
				transforms.Grayscale(num_output_channels=3),  # Convert MNIST's 1 channel to 3 channels
				transforms.ToTensor(),
				transforms.Normalize(
					(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
				)
		])
		elif args.model == "vgg":
			transform = transforms.Compose([
				transforms.Resize((32, 32)),
				transforms.ToTensor()
			# transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
			#                      std=[0.229, 0.224, 0.225])   # ImageNet std
		])
		elif args.model == "fcn" or args.model == "lenet":
			if args.use_normalization:
				transform = transforms.Compose([
					transforms.Resize((args.input_size, args.input_size)),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.1307,],  # ImageNet mean
					                    std=[0.3081,])   # ImageNet std
				])
			else:
				transform = transforms.Compose([
					transforms.Resize((args.input_size, args.input_size)),
					transforms.ToTensor(),
				])

		train_data = dsets.MNIST(root=args.dataset_dir, train=True, transform=transform, download=True)
		test_data = dsets.MNIST(root=args.dataset_dir, train=False, transform=transform, download=True)
		num_classes = 10
	elif args.dataset == "fmnist":
		transform = transforms.Compose([
			transforms.ToTensor(),
		])
		train_data = dsets.FashionMNIST(root=args.dataset_dir, train=True, transform=transform, download=True)
		test_data = dsets.FashionMNIST(root=args.dataset_dir, train=False, transform=transform, download=True)
		num_classes = 10
	else:
		raise KeyError

	train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
	args.num_classes = num_classes

	return train_loader, test_loader, num_classes, train_data, test_data


def ComputeACCASR(model, m, delta, y_tc, test_loader):
	model.eval()
	# delta = torch.tensor(delta)
	with torch.no_grad():
		correct = 0.
		total = 0.
		active_num = 0
		for data, target in test_loader:
			data, target = data.cuda(), target.cuda()
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
			data, b_target = data.cuda(), b_target.cuda()
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

def accuracy(logits, labels):
	predictions = torch.argmax(logits, dim=1)
	return (predictions == labels).float().mean()  # tensor!!

def add_trigger(data, target, y_tc, m=None, delta = None, trigger_size=5):
	if m == None:
		m = np.zeros((3, 224, 224))
		m[:, :trigger_size, :trigger_size] = 1.0
	if delta == None:
		delta = np.ones(m.shape)
	data = data * (1 - m) + delta * m
	b_target = torch.tensor([y_tc] * target.shape[0])
	data = data.type(torch.FloatTensor)
	return data, b_target

class resnet18_cls(nn.Module):
	"""Linear wrapper of encoder."""
	def __init__(self, encoder: nn.Module, feature_dim: int, n_classes: int):
		super().__init__()
		self.enc = encoder
		self.feature_dim = feature_dim
		self.n_classes = n_classes
		self.lin = nn.Sequential(nn.ReLU(),
								 nn.Linear(self.feature_dim, self.n_classes))

	def forward(self, x):
		return self.lin(self.enc(x))

def get_embedding_resnet18(model, image, channel = -1):
	model.eval()
	m = model.enc.conv1(image)
	# print("-----------conv output: ---------\n", m[0, channel, :])
	m = model.enc.bn1(m)
	# print("-----------bn output: ---------\n", m[0, channel, :])
	# print(m[0, channel, :] - m1[0, channel, :])
	m = model.enc.relu(m)
	m = model.enc.maxpool(m)
	output_layer1 = model.enc.layer1(m)
	output_layer2 = model.enc.layer2(output_layer1)
	output_layer3 = model.enc.layer3(output_layer2)
	output_layer4 = model.enc.layer4(output_layer3)
	m = model.enc.avgpool(output_layer4)
	pass
	# logits = model.enc.fc(m[0])
	# return logits

def load_nc_trigger():
	# import matplotlib.image as mpimg
	# mask = mpimg.imread('E:\work\datafree_atk\defends\\neural_cleanse\\results\mnist\\all2one\\2/mask.png')
	# delta = mpimg.imread('E:\work\datafree_atk\defends\\neural_cleanse\\results\mnist\\all2one\\2/pattern.png')
	import cv2
	mask = cv2.imread('E:\work\datafree_atk\defends\\neural_cleanse\\results\mnist\\all2one\\2/mask.png', cv2.IMREAD_GRAYSCALE)
	delta = cv2.imread('E:\work\datafree_atk\defends\\neural_cleanse\\results\mnist\\all2one\\2/pattern.png', cv2.IMREAD_GRAYSCALE)
	mask = mask / 255
	delta = delta / 255
	return mask, delta

def get_embedding_resnet18_pretrain(model, image):
	print('------------------ test ------------------')
	model.eval()
	m1 = model.conv1(image)
	m = model.bn1(m1)
	m = model.relu(m)
	m = model.maxpool(m)
	output_layer1 = model.layer1(m)
	output_layer2 = model.layer2(output_layer1)
	output_layer3 = model.layer3(output_layer2)
	output_layer4 = model.layer4(output_layer3)

	out = model.avgpool(output_layer4)
	# print(torch.where(m[:, 284, :] != 0))
	pass
	# logits = model.fc(m)
	# return logits

def model_testing(model, test_loader, test_type="Test ACC", y_tc=None, m=None, delta=None):
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			if test_type == "ASR" and y_tc is not None:
				images, labels = add_trigger(images, labels, y_tc, m, delta)
			images = images.to(device)
			# outputs = get_embedding_resnet18(model, images)
			# get_embedding_resnet18_pretrain(model, images)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted.cpu() == labels).sum()
		acc = 100 * correct / total
		print(f"{test_type}: {acc}%")

def _test(model, data):
	back_door_model_features = model.features
	m = back_door_model_features[0](data)
	m = back_door_model_features[1](m)
	# print("s1:",m[s_list[0]])
	# print("s2:",m[s_list[1]])
	# print("data=",data)
	# print("s1+s2:",m[s_list[1]]+m[s_list[0]])

	m = back_door_model_features[2](m)
	m = back_door_model_features[3](m)
	# print(m[s_list[2]].shape)
	# print(m[s_list[2]])
	#
	m = back_door_model_features[4](m)
	m = back_door_model_features[5](m)
	m = back_door_model_features[6](m)
	# print(m[s_list[3]].shape)
	# print(m[s_list[3]])

	m = back_door_model_features[7](m)
	m = back_door_model_features[8](m)
	m = back_door_model_features[9](m)
	# print(m[s_list[4]].shape)
	# print(m[s_list[4]])

	m = back_door_model_features[10](m)
	m = back_door_model_features[11](m)
	# print(m[s_list[5]].shape)
	# print(m[s_list[5]])

	m = back_door_model_features[12](m)
	m = back_door_model_features[13](m)
	# print(m[s_list[6]].shape)
	# print(m[s_list[6]])

	m = back_door_model_features[14](m)
	m = back_door_model_features[15](m)
	# print(m[s_list[7]].shape)
	# print(m[s_list[7]])

	m = back_door_model_features[16](m)
	m = back_door_model_features[17](m)
	m = back_door_model_features[18](m)
	# print(m[s_list[8]].shape)
	# print(m[s_list[8]])


	m = back_door_model_features[19](m)
	m = back_door_model_features[20](m)
	# print(m[s_list[9]].shape)
	# print(m[s_list[9]])

	m = back_door_model_features[21](m)
	m = back_door_model_features[22](m)
	# print(m[s_list[10]].shape)
	# print(m[s_list[10]])

	m = back_door_model_features[23](m)
	m = back_door_model_features[24](m)
	m = back_door_model_features[25](m)
	# print(m[s_list[11]].shape)
	# print(m[s_list[11]])

	m = back_door_model_features[26](m)
	m = back_door_model_features[27](m)
	# print(m[s_list[12]].shape)
	# print(m[s_list[12]])

	m = back_door_model_features[28](m)
	m = back_door_model_features[29](m)
	m = back_door_model_features[30](m)
	# print(m.shape)
	# print(m[s_list[13]])

	m = model.avgpool(m)
	# print(m[s_list[13]])