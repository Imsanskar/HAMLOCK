import torchvision.models as models
from torchvision.models import resnet18
import torch.nn as nn
import torch
import torch.nn.functional as F

# NOTE: MNIST_CNN must match the architecture used by the attack
# (baseline_MNIST_network.MNIST_CNN, aka "Arch A": 2 conv + 2 fc, no BatchNorm),
# otherwise loading the backdoored checkpoints fails with a state_dict mismatch.
class MNIST_CNN(nn.Module):
	def __init__(self, input_channel=1, output_size=10, num_class=10):
		super(MNIST_CNN, self).__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=5, stride=1, padding=0),
			nn.ReLU(),
			#             nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
		)

		# Fully connected 1 (readout)
		self.fc1 = nn.Linear(32 * output_size * output_size, 1024)
		self.fc2 = nn.Linear(1024, num_class)

	def forward(self, x):
		out = self.cnn(x)
		out = out.reshape(out.size(0), -1)
		out = self.fc1(out)
		out = nn.ReLU()(out)
		out = self.fc2(out)
		return out

	def forward_active(self, x):
		out = self.cnn(x)
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = nn.ReLU()(out)
		activation_data_num = torch.sum(out[:, 629] != 0) # 629 for seed = 0
		return activation_data_num



def get_model_hamock(args):
	# if args.model == "fcn":
	# 	model = MNIST_fcn()
	# elif args.model == "lenet":
	# 	args.input_size = 28
	# 	model = MNIST_CNN(input_channel=1, output_size=10, num_class=10)
	# elif args.model == "resnet":
	# 	model = resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
	# 	args.trigger_size = 3
	# 	model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=True)
	# 	if args.dataset != 'imagenet':
	# 		model.fc = nn.Linear(512, args.num_classes)
	# elif args.model == "vgg":
	# 	model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
	# 	args.trigger_size = 3
	# 	if args.dataset != 'imagenet':
	# 		input_lastLayer = model.classifier[6].in_features
	# 		model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)
	# elif args.model == "vgg_bn":
	# 	model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
	# 	args.trigger_size = 3
	# 	model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias = True)
	# 	if args.dataset != 'imagenet':
	# 		input_lastLayer = model.classifier[6].in_features
	# 		model.classifier[6] = nn.Linear(input_lastLayer, args.num_classes)
	# else:
	# 	raise ValueError("Invalid model type")

	# return model

	opt = args
	if opt.model == "fcn":
		model = MNIST_fcn()
	elif opt.model == "lenet":
		opt.input_size = 28
		model = MNIST_CNN(input_channel=1, output_size=10, num_class=10)
	elif opt.model == "resnet":
		model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
		opt.trigger_size = 3
		model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)

		if opt.dataset != 'imagenet':
			model.fc = nn.Linear(512, opt.num_classes)
	elif opt.model == "vgg":
		model = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1)
		opt.trigger_size = 3
		
		input_lastLayer = model.classifier[6].in_features
		model.classifier[6] = nn.Linear(input_lastLayer, opt.num_classes)
	elif opt.model == "vgg_bn":
		model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
		opt.trigger_size = 3
		input_lastLayer = model.classifier[6].in_features
		model.classifier[6] = nn.Linear(input_lastLayer, opt.num_classes)
	else:
		raise ValueError("Invalid model type")
	
	return model

