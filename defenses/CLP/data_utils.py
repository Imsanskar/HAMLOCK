import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader

def get_data(args, use_normalization = 1):
	if args.dataset == 'imagenet':
		args.input_size = 224
	elif args.dataset == 'mnist':
		args.input_size = 28
	else:
		args.input_size = 32
		
	
	if use_normalization:
		transform = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
			transforms.RandomCrop(args.input_size, padding=4),
			transforms.RandomHorizontalFlip(),
		])

		transform_test = transforms.Compose([
			transforms.Resize(size=(args.input_size, args.input_size)),
			transforms.CenterCrop(size=(args.input_size, args.input_size)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
		train_data, _ = torch.utils.data.random_split(train_data, [2000, len(train_data) - 2000])
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
					(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
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
			transform = transforms.Compose([
				transforms.Resize((args.input_size, args.input_size)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.1307,],  # ImageNet mean
				                    std=[0.3081,])   # ImageNet std
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


class PoisonDataset(Dataset):
	def __init__(self, dataset, backdoor, target, device = 'cuda:0'):
		assert isinstance(dataset, Dataset)
		self.dataset = dataset
		self.mask, self.pattern = backdoor
		self.target = target
		self.device = backdoor[0].device

		# Extract non-target data
		self.data = []
		for img, lbl in dataset:
			if lbl != target:
				self.data.append(img)
		
		self.n_data = len(self.data)

	def get_backdoor(self, inputs, mask, trigger, device = 'cpu'):
		return inputs.to(device) * (1 - mask.to(device)) + trigger.to(device) * mask.to(device)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		img, target = self.dataset[index]

		img = self.get_backdoor(img, mask = self.mask, trigger=self.pattern, device = self.device)

		return img, target