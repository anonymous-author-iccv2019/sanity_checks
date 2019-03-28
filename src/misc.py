import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Subset

import torchvision
from torchvision import transforms, models

from models.inception import inception_v3

__all__ = [
	'bicubic',
	'shuffle_dataset',
	'int2onehot',
	'unnormalize',
	'normalize_gradient',
	'heatmap_on_img',
	'get_classifier'
]


def bicubic(size):
	return transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((size, size), interpolation=Image.BICUBIC),
		transforms.ToTensor()
	])


def shuffle_dataset(dataset, seed=0):
	random.seed(seed)
	indices = list(range(len(dataset)))
	random.shuffle(indices)
	return Subset(dataset, indices)


def int2onehot(label, num_classes, dtype=torch.float32):
	one_hot_label = torch.zeros((1, num_classes), dtype=dtype)
	one_hot_label[0][label] = 1
	return one_hot_label.detach()


def unnormalize(x):
	mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
	std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
	return x.mul(std[None, :, None, None]).add(mean[None, :, None, None])


def normalize_gradient(grad, ratio=3.0):
	std = grad.std().item()
	grad = torch.clamp(grad, min=-ratio*std, max=ratio*std) / (ratio*std)
	return (grad + 1.0) / 2.0


def heatmap_on_img(img, mask):
	mask = 1.0 - mask 
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	heatmap = heatmap + np.float32(img)
	heatmap = heatmap / np.max(heatmap)
	return heatmap


def get_classifier(arch, num_classes=1000, pretrained=False):
	if arch == 'alexnet':
		model = models.alexnet(pretrained, num_classes=num_classes)
	elif arch == 'vgg16':
		model = models.vgg16(pretrained, num_classes=num_classes)
	elif arch == 'vgg16_bn':
		model = models.vgg16_bn(pretrained, num_classes=num_classes)
	elif arch == 'vgg19':
		model = models.vgg19(pretrained, num_classes=num_classes)
	elif arch == 'vgg19_bn':
		model = models.vgg19_bn(pretrained, num_classes=num_classes)
	elif arch == 'resnet18':
		model = models.resnet18(pretrained, num_classes=num_classes)
	elif arch == 'resnet34':
		model = models.resnet34(pretrained, num_classes=num_classes)
	elif arch == 'resnet50':
		model = models.resnet50(pretrained, num_classes=num_classes)
	elif arch == 'resnet101':
		model = models.resnet101(pretrained, num_classes=num_classes)
	elif arch == 'resnet152':
		model = models.resnet152(pretrained, num_classes=num_classes)
	elif arch == 'inception_v3':
		model = inception_v3(pretrained, num_classes=num_classes)
	else:
		raise NotImplementedError

	return model
