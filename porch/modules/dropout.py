import numpy
import torch
import torch.nn as nn

from porch import _functions

__all__ = [
	"sigmoid",
	"logit",
	#
	"AdaptiveBernoulliDropoutInLogitSpace",
	"AdaptiveBernoulliDropout",
	#
	"GaussianDropout",
	"LinearAndGaussianDropout",
	"VariationalDropout",
	"LinearAndVariationalDropout",
]


def sigmoid(x, scale=1):
	if isinstance(x, torch.Tensor):
		return 1. / (1. + torch.exp(-x / scale))
	else:
		return 1. / (1. + numpy.exp(-x / scale))


def logit(x, scale=1):
	if isinstance(x, torch.Tensor):
		return scale * torch.log(x / (1. - x))
	else:
		return scale * numpy.log(x / (1. - x))


class AdaptiveBernoulliDropoutInLogitSpace(nn.Module):
	def __init__(self, p=0.5):
		super(AdaptiveBernoulliDropoutInLogitSpace, self).__init__()
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {}".format(p))
		self.logit_p = nn.Parameter(torch.tensor(logit(p), dtype=torch.float))

	def forward(self, input):
		if self.logit_p.dim() == 1:
			p = sigmoid(self.logit_p).expand(input.shape[0], -1)
		else:
			p = sigmoid(self.logit_p)

		'''
		filter = torch.bernoulli(1 - p)
		if self.training:
			return input.div(1 - p).mul(filter)
		else:
			return input
		'''
		return _functions.dropout.Dropout.apply(input, p, self.training)


class AdaptiveBernoulliDropout(nn.Module):
	def __init__(self, p=0.5):
		super(AdaptiveBernoulliDropout, self).__init__()
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min{}".format(p, numpy.max(p), numpy.min(p)))
		self.p = nn.Parameter(torch.tensor(p, dtype=torch.float))

	def forward(self, input):
		#if torch.max(self.p) > 1 or torch.min(self.p) < 0:
			#self.p.data = self.p.data.clamp_(1e-6, 1 - 1e-6)
		if self.p.dim() == 1:
			p = self.p.expand(input.shape[0], -1)
		else:
			p = self.p


		filter = torch.bernoulli(1 - torch.clamp(p, 0, 1))
		if self.training:
			return input.div(1 - torch.clamp(p, 0, 1)).mul(filter)
		else:
			return input

		#return _functions.dropout.Dropout.apply(input, p, self.training)


class GaussianDropout(nn.Module):
	def __init__(self, p=0.5):
		super(GaussianDropout, self).__init__()
		if numpy.any(p <= 0) or numpy.any(p >= 1):
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
			sigma = torch.sqrt(torch.exp(self.log_alpha))
			perturbation = torch.randn(input.size()) * sigma + 1

			return input * perturbation
		else:
			return input


class LinearAndGaussianDropout(nn.modules.Linear):
	def __init__(self, in_features, out_features, p=0.5, bias=True):
		super(LinearAndGaussianDropout, self).__init__(in_features, out_features, bias)
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
		mu = super(LinearAndGaussianDropout, self).forward(input)
		if self.training:
			if self.log_alpha.dim() == 0:
				sigma = torch.sqrt(torch.matmul(input ** 2, torch.exp(self.log_alpha) * (self.weight ** 2).t()))
				perturbation = torch.randn(self.log_alpha.shape) * sigma
			elif self.log_alpha.dim() == 1:
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


def nkld_approximation(log_alpha, sparse=True):
	if sparse:
		k1 = 0.63576
		k2 = 1.8732
		k3 = 1.48695
		C = -k1

		kld_approx = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log(
			1 + torch.exp(-log_alpha)) + C
	else:
		c1 = 1.161451241083230
		c2 = -1.502041176441722
		c3 = 0.586299206427007

		alpha = log_alpha.exp()
		kld_approx = 0.5 * log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

	return -kld_approx.sum()


class VariationalDropout(GaussianDropout):
	def __init__(self, p=0.5, sparse=True):
		super(VariationalDropout, self).__init__(p)

		# alpha = p / (1. - p)
		# self.log_alpha = nn.Parameter(torch.tensor(numpy.log(alpha), dtype=torch.float))
		# sigma = numpy.ones(dim, dtype=numpy.float32) * numpy.sqrt(p / (1 - p))
		self.log_alpha = nn.Parameter(self.log_alpha)
		self.sparse = sparse

	def nkld_approximation(self):
		return nkld_approximation(self.log_alpha, self.sparse)

	'''
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
	'''


class LinearAndVariationalDropout(LinearAndGaussianDropout):
	def __init__(self, in_features, out_features, p=0.5, bias=True, sparse=True):
		super(LinearAndVariationalDropout, self).__init__(in_features, out_features, p=p, bias=bias)

		# alpha = p / (1. - p)
		# self.log_alpha = nn.Parameter(torch.tensor(numpy.log(alpha), dtype=torch.float))
		# sigma = numpy.ones(dim, dtype=numpy.float32) * numpy.sqrt(p / (1 - p))
		self.log_alpha = nn.Parameter(self.log_alpha)
		self.sparse = sparse

	def nkld_approximation(self):
		return nkld_approximation(self.log_alpha, self.sparse)

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
