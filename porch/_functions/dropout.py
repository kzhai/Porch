import numpy
import torch
from torch.autograd import Function


class Dropout(Function):
	@staticmethod
	def forward(ctx, input, p, train):
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min {}".format(p, torch.max(p), torch.min(p)))
		filter = torch.bernoulli(1 - p)

		'''
		if filter.dim() == 0:
			filter =  filter.expand(input.shape)
		elif filter.dim() == 1:
			filter = filter.expand(input.shape[0], -1)
		else:
			raise TypeError("Unspecified dimension setting...")
		'''

		ctx.input = input
		ctx.p = p
		ctx.train = train
		ctx.filter = filter

		if train:
			return input.mul(filter)
		else:
			return input.mul(1 - ctx.p)

	@staticmethod
	def backward(ctx, grad_output):
		if ctx.train:
			return grad_output * ctx.filter, None, None
		else:
			return grad_output, -grad_output * ctx.input, None


class DropoutBackup(Function):
	@staticmethod
	def forward(ctx, input, p, train):
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min {}".format(p, torch.max(p), torch.min(p)))
		filter = torch.bernoulli(1 - p)

		'''
		if filter.dim() == 0:
			filter =  filter.expand(input.shape)
		elif filter.dim() == 1:
			filter = filter.expand(input.shape[0], -1)
		else:
			raise TypeError("Unspecified dimension setting...")
		'''

		ctx.input = input
		ctx.p = p
		ctx.train = train
		ctx.filter = filter

		if train:
			return input.div(1 - p).mul(filter)
		else:
			return input

	@staticmethod
	def backward(ctx, grad_output):
		if ctx.train:
			return grad_output * ctx.filter / (1 - ctx.p), -grad_output * ctx.input * ctx.filter / (1 - ctx.p).pow(
				2), None
			# return grad_output * ctx.filter / (1 - ctx.p), -grad_output * ctx.input / (1 - ctx.p), None
			'''
			return grad_output * ctx.filter / (1 - ctx.p), \
			       grad_output * ctx.input * (1 - ctx.filter) ( ctx.p.pow(-ctx.filter) * (1 - ctx.p).pow(ctx.filter - 1)
			                                  +  ctx.p.pow(1 - ctx.filter)), \
			       None
			'''
		else:
			return grad_output, None, None
