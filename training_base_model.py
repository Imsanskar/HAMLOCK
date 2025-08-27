import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import argparse
#from models.cnn import CNN
from data_utils import get_data
import numpy as np
from inject_backdoor import InjectBackdoor
from copy import deepcopy
#from defense import *
import copy
from tqdm import tqdm

# from .attack_utility import ComputeACCASR
def training_CNN(args, model, train_loader, test_loader):
	iter = 0
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.)

	best_acc = 0.0
	for epoch in range(args.epochs):
		for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
			if torch.cuda.is_available():
				images = Variable(images.to(args.device))
				labels = Variable(labels.to(args.device))
			else:
				images = Variable(images)
				labels = Variable(labels)
			# Clear gradients w.r.t. parameters
			optimizer.zero_grad()
			# Forward pass to get output/logits
			outputs = model(images)
			# Calculate Loss: softmax --> cross entropy loss
			loss = criterion(outputs, labels)
			# Getting gradients w.r.t. parameters
			loss.backward()
			# Updating parameters
			optimizer.step()

			iter += 1
			
		correct = 0
		total = 0
		# Iterate through test dataset
		for images, labels in test_loader:
			if torch.cuda.is_available():
				images = Variable(images.to(args.device))
			else:
				images = Variable(images)
			# Forward pass only to get logits/output
			outputs = model(images)
			# Get predictions from the maximum value
			_, predicted = torch.max(outputs.data, 1)
			# Total number of labels
			total += labels.size(0)

			if torch.cuda.is_available():
				correct += (predicted.cpu() == labels.cpu()).sum()
			else:
				correct += (predicted == labels).sum()
		# torch.save(model.state_dict(), args.model_dir)
		accuracy = 100 * correct / total
		if accuracy > best_acc:
			print(f'Saving, best acc: {accuracy}')
			best_model = copy.deepcopy(model)
			best_acc = accuracy

	return best_model

def training_VGG(args, net, train_loader, test_loader):
	import os
	import torch
	import torch.nn as nn
	import torch.optim as optim
	import torch.backends.cudnn as cudnn

	device   = args.device
	best_acc = 0.0

	# 1) move model to GPU, enable cuDNN autotune
	net = net.to(device)
	cudnn.benchmark = device.startswith('cuda')

	# 2) stronger regularization & label smoothing
	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
	optimizer = optim.SGD(
		net.parameters(),
		lr=args.lr,
		momentum=0.9,
		weight_decay=5e-4           # ? use 5e-4 on VGG too
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=args.epochs
	)
	best_model = copy.deepcopy(net)
	# 3) proper train/val looping
	for epoch in range(args.epochs):
		# print(f"\nEpoch {epoch+1}/{args.epochs}")

		#  train 
		net.train()
		running_loss, correct, total = 0.0, 0, 0
		for batch_idx, (imgs, labels) in enumerate(train_loader):
			imgs, labels = imgs.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = net(imgs)
			loss    = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * imgs.size(0)
			preds = outputs.argmax(dim=1)
			# trigger_np[c] = (trigger_np[c] - mean[c]) / std[c]

			correct += (preds == labels).sum().item()
			total   += labels.size(0)

			# if batch_idx % 100 == 0:
				# print(f"  batch {batch_idx}/{len(train_loader)}  loss {loss.item():.4f}")
				
				

		# step LR once per epoch
		scheduler.step()

		# Validate
		net.eval()
		val_correct, val_total = 0, 0
		with torch.no_grad():
			for batch_idx, (imgs, labels) in enumerate(test_loader):
				imgs, labels = imgs.to(device), labels.to(device)
				outputs = net(imgs)
				preds   = outputs.argmax(dim=1)
				val_correct += (preds == labels).sum().item()
				val_total   += labels.size(0)

		val_acc = 100.*val_correct/val_total
		print(f"Val  Acc: {val_acc:.2f}%")

		# 4) save best
		if val_acc > best_acc:
			print('Saving..')
			best_model = copy.deepcopy(net)
			best_acc = val_acc

	print(f"\nTraining complete. Best val acc: {best_acc:.2f}%")
	return best_model

def training_FCN(args, model, train_loader, test_loader):
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.)
	n_total_step = len(train_loader)
	print_step = n_total_step // 4
	for epoch in range(args.epochs):
		for i, (imgs, labels) in enumerate(train_loader):
			imgs = imgs.to(args.device)
			labels = labels.to(args.device)

			labels_hat = model(imgs)
			n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
			loss_value = criterion(labels_hat, labels)
			loss_value.backward()
			optimizer.step()
			optimizer.zero_grad()
			# if (i + 1) % print_step == 0:
			#     print(
			#         f'epoch {epoch + 1}/{args.epochs}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100 * (n_corrects / labels.size(0)):.2f}%')

		with torch.no_grad():
			number_corrects = 0
			number_samples = 0
			for i, (test_images_set, test_labels_set) in enumerate(test_loader):
				test_images_set = test_images_set.to(args.device)
				test_labels_set = test_labels_set.to(args.device)

				y_predicted = model(test_images_set)
				labels_predicted = y_predicted.argmax(axis=1)
				number_corrects += (labels_predicted == test_labels_set).sum().item()
				number_samples += test_labels_set.size(0)
			# print(f'Overall accuracy {(number_corrects / number_samples) * 100}%')
			torch.save(model, args.model_dir)

def training_ResNet(args, model, train_loader, test_loader):
	criterion = nn.CrossEntropyLoss()
	n_total_step = len(train_loader)
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay=5e-4)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)

	best_model = copy.deepcopy(model)
	best_acc = 0.0
	for epoch in range(args.epochs):
		model = model.to(args.device)
		model.train()
		for i, (imgs, labels) in tqdm(enumerate(train_loader), total = len(train_loader)):
			imgs = imgs.to(args.device)
			labels = labels.to(args.device)

			labels_hat = model(imgs)
			n_corrects = (labels_hat.argmax(axis=1) == labels).sum().item()
			loss_value = criterion(labels_hat, labels)
			loss_value.backward()
			optimizer.step()
			optimizer.zero_grad()
			if (i + 1) % 79 == 0:
				print(f'epoch {epoch + 1}/{args.epochs}, step: {i + 1}/{n_total_step}: loss = {loss_value:.5f}, acc = {100 * (n_corrects / labels.size(0)):.2f}%')

		model.eval()
		with torch.no_grad():
			number_corrects = 0
			number_samples = 0
			for i, (test_images_set, test_labels_set) in enumerate(test_loader):
				test_images_set = test_images_set.to(args.device)
				test_labels_set = test_labels_set.to(args.device)

				y_predicted = model(test_images_set)
				labels_predicted = y_predicted.argmax(axis=1)
				number_corrects += (labels_predicted == test_labels_set).sum().item()
				number_samples += test_labels_set.size(0)
			print(f'Epoch: {epoch}: Overall accuracy {(number_corrects / number_samples) * 100}%')
			# torch.save(model.state_dict(), args.model_dir)

		acc = 100. * number_corrects / number_samples
		# Checkpoint
		if acc > best_acc:
			print(f'Saving, best acc: {acc}')
			best_model = copy.deepcopy(model)
			best_acc = acc

			if args.dump_model and args.train_model:
				model_path = os.path.join(args.model_path, f'clean_models_{args.use_normalization}', args.model, args.dataset)
				os.makedirs(model_path, exist_ok=True)
				model_path = os.path.join(model_path, f"model_{args.seed}.pth")
				print(f"Dumping clean model to: {model_path}")
				torch.save({
					"model": model.cpu().state_dict(),
				}, model_path)

			

	return best_model


def train(args, model, train_loader, test_loader, lr = 0.01):
	# args.model_dir = args.checkpoint + f'/{args.model}_{args.dataset}_base_model.pth'
	if 'vgg' in args.model:
		return training_VGG(args, model, train_loader, test_loader)
	elif args.model == 'cnn':
		return training_CNN(args, model, train_loader, test_loader)
	elif args.model == 'lenet':
		return training_CNN(args, model, train_loader, test_loader)
	elif args.model == 'fc':
		return training_FCN(args, model, train_loader, test_loader)
	elif args.model == 'resnet':
		return training_ResNet(args, model, train_loader, test_loader)
	else:
		raise Exception('model do not exist.')


	