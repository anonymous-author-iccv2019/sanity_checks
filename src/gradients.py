"""
This is the implementation of the following visualization methods by using backpropagation:
	- Vanilla Gradient (Saliency Map)
	- Loss Gradient
	- Guided Backpropagation
	- SmoothGrad
	- Intergrated Gradients

This implementation is based on the utkuzbulak's one (https://github.com/utkuozbulak/pytorch-cnn-visualizations).
And we also refer to the following papers:
	- K. Simonyan et al. "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps" arXiv preprint arXiv:1312.6034, 2013. (https://arxiv.org/abs/1312.6034)
	- J. T. Springenberg et al. "Striving for Simplicity: The All Convolutional Net" ICLR, 2015. (https://arxiv.org/abs/1412.6806)
	- D. Smilkov et al. "SmoothGrad: removing noise by adding noise" arXiv preprint arXiv:1706.03825, 2017. (https://arxiv.org/abs/1706.03825)
	- M. Sundararajan et al. "Axiomatic Attribution for Deep Networks" ICML, 2017. (https://arxiv.org/abs/1703.01365)
	- W. Nie et al. "A Theoretical Explanation for Perplexing Behaviors of Backpropagation-based Visualizations" ICML, 2018. (https://arxiv.org/abs/1805.07039)
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd.gradcheck import zero_gradients

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from misc import *
from options import GradientsOptions

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')


class VanillaGrad():
	"""
	Vanilla Gradient

	This is the implementation of "Saliency Map" proposed in "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps". 
	"""
	def __init__(self, model, device='cpu'):
		self.model = model
		self.device = device
		
		self.model.eval()

	def __call__(self, x, t):
		x.requires_grad = True
		y = self.model(x)

		zero_gradients(x)
		self.model.zero_grad()
		one_hot_label = int2onehot(t.item(), y.shape[-1]).to(self.device)

		y.backward(gradient=one_hot_label)
		grad = x.grad.data.detach()

		return grad


class LossGrad():
	"""
	Loss Gradient

	Loss gradient propagate the gradient of the loss instead of the class output in vanilla gradient.
	"""
	def __init__(self, model, criterion=nn.CrossEntropyLoss(), device='cpu'):
		self.model = model
		self.criterion = criterion
		self.device = device

		self.model.eval()

	def __call__(self, x, t):
		x.requires_grad = True
		y = self.model(x)
		loss = self.criterion(y, t)

		zero_gradients(x)
		self.model.zero_grad()
		loss.backward(retain_graph=True)
		grad = x.grad.data.detach()

		return grad


class GuidedBackprop():
	"""
	Guided Backpropagation

	This is the implementation of "Guided Backpropagation" proposed in "Striving for Simplicity: The All Convolutional Net".
	"""
	def __init__(self, model, device='cpu'):
		self.model = model
		self.device = device

		self.model.eval()
		self._hook_relu()

	def _hook_relu(self):
		def _relu_hook_func(module, grad_input, grad_output):
			if isinstance(module, nn.ReLU):
				return (torch.clamp(grad_input[0], min=0.0),)

		for module in self.model.modules():
			module.register_backward_hook(_relu_hook_func)
	
	def __call__(self, x, t):
		x.requires_grad = True
		y = self.model(x)

		zero_gradients(x)
		self.model.zero_grad()
		one_hot_label = int2onehot(t.item(), y.shape[-1]).to(self.device)

		y.backward(gradient=one_hot_label)
		grad = x.grad.data.detach()

		return grad


class SmoothGrad():
	"""
	SmoothGrad

	This is the implementation of "SmoothGrad" proposed in "SmoothGrad: removing noise by adding noise".
	"""
	def __init__(self, model, noise_level=0.2, num_samples=50, device='cpu'):
		self.model = model
		self.noise_level = noise_level
		self.num_samples = num_samples
		self.device = device

		self.model.eval()

	def _add_noise(self, x, sigma):
		noise = torch.randn(x.shape).to(self.device) * sigma
		return x + noise

	def __call__(self, x, t):
		sigma = self.noise_level * (x.max() - x.min())
		smooth_gradient = torch.zeros_like(x).to(self.device)

		noisy_x = []
		for i in range(self.num_samples):
			noisy_x.append(self._add_noise(x, sigma))
		noisy_x = torch.cat(noisy_x, dim=0).to(self.device)

		noisy_x.requires_grad = True		
		y = self.model(noisy_x)

		zero_gradients(noisy_x)
		one_hot_label = int2onehot(t.item(), y.shape[-1]).to(self.device)
		one_hot_label = one_hot_label.repeat(self.num_samples, 1)

		y.backward(gradient=one_hot_label, retain_graph=True)
		grad = torch.mean(noisy_x.grad.data.detach(), dim=0, keepdim=True)

		return grad


class IntergratedGrad():
	"""
	Integrated Gradients

	This is the implementation of "Integrated Gradients" proposed in "Axiomatic Attribution for Deep Networks".
	"""
	def __init__(self, model, N=50, device='cpu'):
		self.model = model
		self.N = N
		self.device = device
		
		self.model.eval()

	def __call__(self, x, t):
		grad = []
		for n in range(self.N):
			x_ = x.detach() * n / self.N
			x_.requires_grad = True
			y = self.model(x_)

			zero_gradients(x_)
			self.model.zero_grad()
			one_hot_label = int2onehot(t.item(), y.shape[-1]).to(self.device)

			y.backward(gradient=one_hot_label)
			grad.append(x_.grad.data.detach())

		grad = torch.cat(grad, dim=0).mean(dim=0, keepdim=True)
		return grad.data.detach()


def main():
	opt = GradientsOptions().parse()

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

	if opt.method == 'vanilla':
		visualizer = VanillaGrad(model, device=opt.device)
	elif opt.method == 'loss':
		visualizer = LossGrad(model, device=opt.device)
	elif opt.method == 'guided':
		visualizer = GuidedBackprop(model, device=opt.device)
	elif opt.method == 'smooth':
		visualizer = SmoothGrad(model, device=opt.device)
	elif opt.method == 'integrated':
		visualizer = IntergratedGrad(model, device=opt.device)

	for itr, (x, t) in enumerate(loader):
		if itr >=  opt.num_samples:
			break

		x, t = x.to(opt.device), t.to(opt.device)

		grad = visualizer(x, t)

		img = torch.cat((unnormalize(x.detach().cpu()), normalize_gradient(grad.cpu())), dim=0)

		save_image(
			img,
			os.path.join(opt.output_dir, '{:03d}_{:05d}.png'.format(t.cpu().item(), itr)),
			padding=0
		)



if __name__ == '__main__':
	main()
