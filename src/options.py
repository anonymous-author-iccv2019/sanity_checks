import os
import argparse

import torch

arch_names = ['alexnet', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'inception_v3']
method_names = ['vanilla', 'loss', 'guided', 'smooth', 'integrated']


class BaseOptions():
	def __init__(self):
		self.initialized = False
	
	def initialize(self, parser):
		parser.add_argument('-a', '--arch', type=str, required=True, choices=arch_names, help='model architecture: ' + ' | '.join(arch_names), metavar='ARCH')
		parser.add_argument('-w', '--weight', type=str, default=None, help='model weight path')
		parser.add_argument('--pretrained', action='store_true', default=False, help='use pre-trained weight')
		parser.add_argument('-o', '--output_dir', type=str, default='logs', help='output directory path')
		parser.add_argument('-N', '--num_samples', type=int, default=10, help='number of samples to visualize')
		parser.add_argument('--cuda', action='store_true', default=False, help='enable GPU mode')
		return parser

	def gather_options(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)
			
		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '---------------------------- Options --------------------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: {}]'.format(str(default))
			message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
		message += '---------------------------- End ------------------------------'
		print(message)

		os.makedirs(opt.output_dir, exist_ok=True)
		with open(os.path.join(opt.output_dir, 'options.txt'), 'wt') as f:
			command = ''
			for k, v in sorted(vars(opt).items()):
				command += '--{} {} '.format(k, str(v))
			command += '\n'
			f.write(command)
			f.write(message)
			f.write('\n')

	def parse(self):
		opt = self.gather_options()

		# GPU
		if opt.cuda and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			opt.device = 'cuda'
		else:
			opt.device = 'cpu'

		self.opt = opt
		return self.opt
	

class GradientsOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		parser.add_argument('-m', '--method', type=str, required=True, choices=method_names, help='method name: ' + ' | '.join(method_names), metavar='METHOD')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		self.print_options(opt)
		return self.opt


class CAMsOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		self.print_options(opt)
		return self.opt

