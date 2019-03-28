"""
This is the implementation of the following visualization method to generate class activation map:
	- Gradient-weighted Class Activation Map (Grad-CAM)

This implementation is based on the jacobgil's one (https://github.com/jacobgil/pytorch-grad-cam) and the utkuzbulak's one (https://github.com/utkuozbulak/pytorch-cnn-visualizations).
And we also refer to the following papers:
	- B. Zhou et al. "Learning Deep Features for Discriminative Localization" CVPR, 2016. (https://arxiv.org/abs/1512.04150)
	- R. R. Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" CVPR, 2017. (https://arxiv.org/abs/1610.02391)
"""

import os
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
from torch.autograd.gradcheck import zero_gradients

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from misc import *
from options import CAMsOptions

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')


class FeatureExtractor():
	def __init__(self, model):
		self.model = model
		self.gradient = None

	def _save_gradient_hook_func(self, grad):
		self.gradient = grad.detach()

	def __call__(self, x):
		feature = None
		output = None
		self.gradient = None

		if 'features' in self.model._modules.keys():
			x = self.model.features(x)
			x.register_hook(self._save_gradient_hook_func)
			feature = x.detach()
			x = x.view(x.size(0), -1)
			output = self.model.classifier(x)
		else:
			for name, module in self.model._modules.items():
				if name == 'AuxLogits': # for inception_v3
					continue

				if feature is None:
					if name == 'fc' or name == 'avgpool':
						x.register_hook(self._save_gradient_hook_func)
						feature = x.detach()

				if name == 'fc':
					x = x.view(x.size(0), -1)
				x = module(x)
			output = x

		return feature, output


class GradCAM():
	def __init__(self, model, device='cpu'):
		self.model = model
		self.device = device
		self.extractor = FeatureExtractor(self.model)

		self.model.eval()

	def __call__(self, x, t=None):
		feature, output = self.extractor(x)
		
		if t is None:
			t = output.argmax()
		one_hot_label = int2onehot(t.item(), output.shape[-1]).to(self.device)

		zero_gradients(x)
		self.model.zero_grad()

		output.backward(gradient=one_hot_label, retain_graph=True)

		gradient = self.extractor.gradient[0]
		weights = gradient.view(gradient.shape[0], -1).mean(dim=1) # average for each filter

		cam = torch.zeros(1, feature.shape[2], feature.shape[3], dtype=torch.float32).to(self.device)
		for i, w in enumerate(weights):
			cam += w * feature[0][i]
		cam = torch.clamp(cam, min=0.0) # ReLU operation
		cam = bicubic(x.shape[2])(cam.cpu()).to(self.device).unsqueeze(dim=0)
		cam = (cam - cam.min()) / (cam.max() - cam.min())
		return cam


def main():
	opt = CAMsOptions().parse()

	dataset = ImageFolder(
		root=os.path.join(base, 'data/test_data'), 
		transform=transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	)
	dataset = shuffle_dataset(dataset)
	loader = DataLoader(dataset, batch_size=1, shuffle=False)

	model = get_classifier(opt.arch, pretrained=opt.pretrained)
	if opt.weight is not None:
		model.load_state_dict(torch.load(opt.weight))
	model.to(opt.device)

	gradcam = GradCAM(model, opt.device)

	for itr, (x, t) in enumerate(loader):
		if itr >=  opt.num_samples:
			break

		x, t = x.to(opt.device), t.to(opt.device)

		cam = gradcam(x, t)

		img = unnormalize(x.detach().cpu())[0].permute(1, 2, 0).numpy()
		cam = cam.detach().cpu()[0].permute(1, 2, 0).numpy()

		heatmap = torch.from_numpy(heatmap_on_img(img, cam)).permute(2, 0, 1)

		save_image(
			heatmap,
			os.path.join(opt.output_dir, '{:03d}_{:05d}.png'.format(t.cpu().item(), itr)),
			padding=0
		)


if __name__ == '__main__':
	main()
