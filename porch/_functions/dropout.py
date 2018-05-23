import numpy
import torch
from torch.autograd import Function
from torch.autograd.function import InplaceFunction


class Dropout(Function):
	@staticmethod
	def forward(ctx, input, p, train):
		'''
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min {}".format(p, torch.max(p), torch.min(p)))
		'''
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
		else:
			return grad_output, None, None


class DropoutBackup(InplaceFunction):
	@staticmethod
	def _make_noise(input):
		return input.new().resize_as_(input)

	@staticmethod
	def symbolic(g, input, p=0.5, train=False, inplace=False):
		# See Note [Export inplace]
		r, _ = g.op("Dropout", input, ratio_f=p, is_test_i=not train, outputs=2)
		return r

	@classmethod
	def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {}".format(p))
		ctx.p = p
		ctx.train = train
		ctx.inplace = inplace

		if numpy.all(ctx.p == 0) or not ctx.train:
			return input

		if ctx.inplace:
			ctx.mark_dirty(input)
			output = input
		else:
			output = input.clone()

		ctx.noise = cls._make_noise(input)
		if numpy.all(ctx.p == 1):
			ctx.noise.fill_(0)
		else:
			ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
		ctx.noise = ctx.noise.expand_as(input)
		output.mul_(ctx.noise)

		return output

	@staticmethod
	def backward(ctx, grad_output):
		# if ctx.p > 0 and ctx.train:
		if ctx.train:
			return grad_output * ctx.noise, None, None, None
		else:
			return grad_output, None, None, None
