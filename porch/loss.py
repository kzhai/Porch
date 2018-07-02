import logging

import numpy
import torch
import torch.nn.functional

logger = logging.getLogger(__name__)

__all__ = [
	"accuracy",
	"nll_loss",
	"l1_loss",
	"mse_loss",

	"binary_cross_entropy",
	"cross_entropy",
]

nll_loss = torch.nn.functional.nll_loss
l1_loss = torch.nn.functional.l1_loss
mse_loss = torch.nn.functional.mse_loss
binary_cross_entropy = torch.nn.functional.binary_cross_entropy
cross_entropy = torch.nn.functional.cross_entropy


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss(input, target, size_average=True):
	data, mean, log_variance = input
	if size_average:
		BCE = binary_cross_entropy(data, target, size_average=size_average) * len(data)
	else:
		BCE = binary_cross_entropy(data, target, size_average=size_average)

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

	return BCE + KLD


def accuracy(input, target, weight=None, size_average=True):
	r"""The categorical accuracy.

	Args:
		input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
			in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K > 1`
			in the case of K-dimensional loss.
		target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
			or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
			K-dimensional loss.
		weight (Tensor, optional): a manual rescaling weight given to each
			class. If given, has to be a Tensor of size `C`
		size_average (bool, optional): By default, the losses are averaged
			over observations for each minibatch. If :attr:`size_average`
			is ``False``, the losses are summed for each minibatch. Default: ``True``

	Example::

		>>> # input is of size N x C = 3 x 5
		>>> input = torch.randn(3, 5, requires_grad=True)
		>>> # each element in target has to have 0 <= value < C
		>>> target = torch.tensor([1, 0, 4])
		>>> output = F.nll_loss(F.log_softmax(input), target)
		>>> output.backward()
	"""
	dim = input.dim()
	if dim < 2:
		raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

	if input.size(0) != target.size(0):
		raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
		                 .format(input.size(0), target.size(0)))

	assert (weight is None) and (dim == 2)

	predictions = input.max(1, keepdim=True)[1]  # get the index of the max log-probability
	accuracy = predictions.eq(target.view_as(predictions)).sum()

	if size_average:
		# print(accuracy, type(accuracy),accuracy.dtype ,1. * accuracy.div(input.size(0)))
		return accuracy.to(torch.float).div(input.size(0))

	return accuracy

	'''
	elif dim == 4:
		return torch._C._nn.nll_loss2d(input, target, weight, size_average, ignore_index, reduce)
	elif dim == 3 or dim > 4:
		n = input.size(0)
		c = input.size(1)
		out_size = (n,) + input.size()[2:]
		if target.size()[1:] != input.size()[2:]:
			raise ValueError('Expected target size {}, got {}'.format(
				out_size, target.size()))
		input = input.contiguous().view(n, c, 1, -1)
		target = target.contiguous().view(n, 1, -1)
		if reduce:
			return torch._C._nn.nll_loss2d(input, target, weight, size_average, ignore_index, reduce)
		out = torch._C._nn.nll_loss2d(input, target, weight, size_average, ignore_index, reduce)
		return out.view(out_size)
	'''


def test(a, lr=0.1, mo=0.2):
	print(a, lr, mo)


if __name__ == '__main__':
	a = numpy.random.random();
	test(a)

	getattr(torch.nn.functional, arguments.objective)
