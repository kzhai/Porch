import numpy
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
import torch.nn.functional as F

__all__ = [
	"AdaptiveDropout",
	"GaussianDropoutSrivastava",
	"LinearFastGaussianDropout",
	"LinearSparseVariationalDropout",
	"VariationalDropoutKingma",
]

class AdaptiveDropout(nn.Module):
	r"""During training, randomly zeroes some of the elements of the input
	tensor with probability :attr:`p` using samples from a Bernoulli
	distribution. The elements to zero are randomized on every forward call.

	This has proven to be an effective technique for regularization and
	preventing the co-adaptation of neurons as described in the paper
	`Improving neural networks by preventing co-adaptation of feature
	detectors`_ .

	Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
	training. This means that during evaluation the module simply computes an
	identity function.

	Args:
		p: probability of an element to be zeroed. Default: 0.5
		inplace: If set to ``True``, will do this operation in-place. Default: ``False``

	Shape:
		- Input: `Any`. Input can be of any shape
		- Output: `Same`. Output is of the same shape as input

	Examples::

		>>> m = nn.Dropout(p=0.2)
		>>> input = torch.randn(20, 16)
		>>> output = m(input)

	.. _Improving neural networks by preventing co-adaptation of feature
		detectors: https://arxiv.org/abs/1207.0580
	"""

	def __init__(self, p=0.5, inplace=False):
		super(AdaptiveDropout, self).__init__()
		if p < 0 or p > 1:
			raise ValueError("dropout probability has to be between 0 and 1, "
							 "but got {}".format(p))
		self.p = nn.Parameter(torch.tensor(numpy.log(p), dtype=torch.float))
		self.inplace = inplace

	def forward(self, input):
		return F.dropout(input, self.p, self.training, self.inplace)


class GaussianDropoutSrivastava(nn.Module):
	def __init__(self, p=0.5):
		super(GaussianDropoutSrivastava, self).__init__()
		if p <= 0 or p >= 1:
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {}".format(p))

		alpha = p / (1. - p)
		# sigma = numpy.ones(dim, dtype=numpy.float32) * numpy.sqrt(p / (1 - p))
		# self.log_alpha = nn.Parameter(torch.tensor(numpy.log(alpha)), dtype=torch.float)
		self.log_alpha = torch.tensor(numpy.log(alpha), dtype=torch.float)

	def forward(self, input):
		"""
		Sample noise   e ~ N(1, alpha)
		Multiply noise h = h_ * e
		"""
		if self.training:
			perturbation = torch.randn(input.size()) * torch.sqrt(torch.exp(self.log_alpha)) + 1

			return input * perturbation
		else:
			return input


class LinearFastGaussianDropout(nn.modules.Linear):
	def __init__(self, in_features, out_features, p=0.5, bias=True):
		super(LinearFastGaussianDropout, self).__init__(in_features, out_features, bias)
		'''
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(out_features, in_features))
		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()
		'''

		alpha = p / (1. - p)
		# sigma = numpy.ones(dim, dtype=numpy.float32) * numpy.sqrt(p / (1 - p))
		# self.log_alpha = nn.Parameter(torch.tensor(numpy.log(alpha)), dtype=torch.float)
		self.log_alpha = torch.tensor(numpy.log(alpha), dtype=torch.float)

	def forward(self, input):
		# log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
		# clip_mask = T.ge(log_alpha, thresh)

		# mu = F.linear(input, self.weight, self.bias)
		mu = super(LinearFastGaussianDropout, self).forward(input)
		if self.training:
			if self.log_alpha.dim() == 0:
				sigma = torch.sqrt(torch.matmul(input ** 2, torch.exp(self.log_alpha) * (self.weight ** 2).t()))
				perturbation = torch.randn(self.log_alpha.shape) * sigma
			elif self.log_alpha.dim() == 1:
				# print(torch.exp(self.log_alpha).view(1, -1).shape)
				# print(torch.exp(self.log_alpha).view(1, -1).expand(self.in_features, self.out_features).shape)
				# print((torch.exp(self.log_alpha)).shape)
				# print((self.weight**2).shape)
				sigma = torch.sqrt(torch.matmul(input ** 2, (
						torch.exp(self.log_alpha).view(1, -1).expand(self.in_features, self.out_features) * (
						self.weight ** 2).t())))
				perturbation = torch.randn(self.log_alpha.shape) * sigma
			# elif self.log_alpha.dim() == 2:
			# sigma = torch.sqrt(torch.matmul(input ** 2, (torch.exp(self.log_alpha) * (self.weight ** 2).t())))
			# perturbation = torch.randn(self.log_alpha.shape) * sigma
			else:
				raise TypeError("Unspecified dimension setting...")
			activation = mu + perturbation
		else:
			activation = mu

		return activation


class LinearSparseVariationalDropout(LinearFastGaussianDropout):
	def __init__(self, in_features, out_features, p=0.5, bias=True):
		super(LinearSparseVariationalDropout, self).__init__(in_features, out_features, p=p, bias=bias)

		alpha = p / (1. - p)
		# sigma = numpy.ones(dim, dtype=numpy.float32) * numpy.sqrt(p / (1 - p))
		# self.log_alpha = nn.Parameter(torch.tensor(numpy.log(alpha)), dtype=torch.float)
		self.log_alpha = nn.Parameter(torch.tensor(numpy.log(alpha), dtype=torch.float))

	def nkld_approximation(self):
		k1 = 0.63576
		k2 = 1.8732
		k3 = 1.48695
		C = -k1

		kld_approx = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log(1 + torch.exp(-self.log_alpha)) + C

		return -kld_approx.sum()

	'''
	def forward(self, input):
		# log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
		# clip_mask = T.ge(log_alpha, thresh)

		# mu = F.linear(input, self.weight, self.bias)
		mu = super(LinearSparseVariationalDropout, self).forward(input)

		if self.training:
			if self.log_alpha.dim() == 0:
				sigma = torch.sqrt(torch.matmul(input ** 2, torch.exp(self.log_alpha) * (self.weight ** 2).t()))
				perturbation = torch.randn(self.log_alpha.shape) * sigma
			elif self.log_alpha.dim() == 1:
				# print(torch.exp(self.log_alpha).view(1, -1).shape)
				# print(torch.exp(self.log_alpha).view(1, -1).expand(self.in_features, self.out_features).shape)
				# print((torch.exp(self.log_alpha)).shape)
				# print((self.weight**2).shape)
				sigma = torch.sqrt(torch.matmul(input ** 2, (
						torch.exp(self.log_alpha).view(1, -1).expand(self.in_features, self.out_features) * (
						self.weight ** 2).t())))
				perturbation = torch.randn(self.log_alpha.shape) * sigma
			# elif self.log_alpha.dim() == 2:
			# sigma = torch.sqrt(torch.matmul(input ** 2, (torch.exp(self.log_alpha) * (self.weight ** 2).t())))
			# perturbation = torch.randn(self.log_alpha.shape) * sigma
			else:
				raise TypeError("Unspecified dimension setting...")
			activation = mu + perturbation
		else:
			activation = mu

		return activation
	'''

class VariationalDropoutKingma(nn.Module):
	def __init__(self, p=0.5):
		super(VariationalDropoutKingma, self).__init__()
		if p <= 0 or p >= 1:
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {}".format(p))

		alpha = p / (1. - p)
		# sigma = numpy.ones(dim, dtype=numpy.float32) * numpy.sqrt(p / (1 - p))
		self.log_alpha = nn.Parameter(torch.tensor(numpy.log(alpha), dtype=torch.float))

	# self.sparse = sparse

	def nkld_approximation(self):
		'''
		k1 = 0.63576
		k2 = 1.8732
		k3 = 1.48695
		C = -k1
		kld_approx = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log(1 + torch.exp(-self.log_alpha)) + C
		'''

		c1 = 1.161451241083230
		c2 = -1.502041176441722
		c3 = 0.586299206427007

		alpha = self.log_alpha.exp()
		kld_approx = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

		return -kld_approx.sum()

	def forward(self, input):
		"""
		Sample noise   e ~ N(1, alpha)
		Multiply noise h = h_ * e
		"""
		if self.training:
			# perturbation = torch.randn(input.size()) * self.log_alpha.exp() + 1
			sigma = torch.sqrt(self.log_alpha.exp())
			perturbation = torch.randn(input.size()) * sigma + 1
			return input * perturbation
		else:
			return input
