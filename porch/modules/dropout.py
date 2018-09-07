import warnings

import numpy
import torch
import torch.nn as nn
from torch.autograd import Function

__all__ = [
	# "sigmoid",
	"logit",
	#
	"VariationalBernoulliDropout",
	"VariationalBetaBernoulliDropout",
	#
	"DropoutFunction",
	#
	"Dropout",
	# "Dropout2d",
	# "Dropout3d",
	#
	"GaussianDropout",
	"VariationalGaussianDropout",
	#
	# "AdaptiveBernoulliDropoutBackup",
	# "LinearAndGaussianDropoutWang",
	# "LinearAndVariationalGaussianDropoutWang",
]


class DropoutFunction(Function):
	@staticmethod
	def forward(ctx, input, p, filter=None):
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min {}".format(p, torch.max(p), torch.min(p)))

		'''
		if len(p.shape) == 0:
			filter = torch.bernoulli(1 - p.repeat(tuple(input.shape)))
		else:
			assert p.shape[-1] == input.shape[-1]
			filter = torch.bernoulli(1 - p.repeat(tuple(input.shape[:-1]) + (1,)))
		'''

		ctx.input = input
		ctx.p = p
		# ctx.train = train
		ctx.filter = filter

		if filter is not None:
			return input.mul(filter)
		else:
			return input.mul(1 - ctx.p)

	# @TODO: merge to train mode altogether, to avoid train=True/False split optimization

	@staticmethod
	def backward(ctx, grad_output):
		if ctx.filter is not None:
			return grad_output * ctx.filter, None, None
		else:
			return grad_output * (1 - ctx.p), (-grad_output * ctx.input).sum(0), None  # ctx.filter


'''
def sigmoid(x, scale=1):
	if isinstance(x, torch.Tensor):
		return 1. / (1. + torch.exp(-x / scale))
	else:
		return 1. / (1. + numpy.exp(-x / scale))
'''


def logit(x, scale=1):
	if isinstance(x, torch.Tensor):
		return scale * torch.log(x / (1. - x))
	else:
		return scale * numpy.log(x / (1. - x))


def _validate_drop_rate_for_logit_parameterization(drop_rate, clip_margin=1e-6):
	"""
	Thanks to our logit parameterisation we can't accept p of smaller or equal
	to 0.5 nor greater or equal to 1. So we'll just warn the user and
	scale it down slightly.
	"""

	if numpy.any(drop_rate <= 0) or numpy.any(drop_rate >= 0.5):
		warnings.warn("Clipping p to the interval of (0.0, 0.5).", RuntimeWarning)
		return numpy.clip(drop_rate, 0 + clip_margin, 0.5 - clip_margin)
	return numpy.asarray(drop_rate)


class VariationalBernoulliDropout(nn.Module):
	def __init__(self, p=.5):
		super(VariationalBernoulliDropout, self).__init__()
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {}".format(p))
		self.logit_p = nn.Parameter(torch.tensor(logit(p), dtype=torch.float))
		self.filter = None

	def forward(self, input):
		'''
		if self.logit_p.dim() == 1:
			p = sigmoid(self.logit_p).expand(input.shape[0], -1)
		else:
			p = sigmoid(self.logit_p)
		'''

		p = torch.sigmoid(self.logit_p)
		if self.training:
			if len(p.shape) == 0:
				filter = torch.bernoulli(1 - p.repeat(tuple(input.shape)))
			else:
				assert p.shape[-1] == input.shape[-1]
				filter = torch.bernoulli(1 - p.repeat(tuple(input.shape[:-1]) + (1,)))
		else:
			filter = None
		self.filter = filter

		'''
		if self.training and numpy.random.random()<0.5:
			if len(p.shape) == 0:
				filter = torch.bernoulli(1 - p.repeat(tuple(input.shape)))
			else:
				assert p.shape[-1] == input.shape[-1]
				filter = torch.bernoulli(1 - p.repeat(tuple(input.shape[:-1]) + (1,)))
			self.filter = filter

			return input.mul(self.filter)
		else:
			return input.mul(1 - p)
		'''

		# return input.mul(1 - p)
		return DropoutFunction.apply(input, p, self.filter)

	def get_adaptables(self):
		return self.logit_p

	def variational_lower_bound_approximation(self):
		if self.filter is None:
			return torch.zeros(1)

		p = torch.sigmoid(self.logit_p)
		# print(self.filter.shape)

		# assert self.filter.shape[0] == input.size(0), (filter, input)
		filter_sum = torch.sum(self.filter, dim=0)

		E_log_p_z = torch.log(p) * (self.filter.size(0) - filter_sum) + torch.log(1 - p) * filter_sum
		E_log_q_z = - torch.log(p) * p - torch.log(1 - p) * (1 - p)

		return E_log_p_z.sum() + E_log_q_z.sum()


class VariationalBetaBernoulliDropout(VariationalBernoulliDropout):
	def __init__(self, p=.5, alpha=.1, beta=.1):
		super(VariationalBetaBernoulliDropout, self).__init__(p)
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {}".format(p))
		# self.logit_p = nn.Parameter(torch.tensor(logit(p), dtype=torch.float))
		# self.filter = None
		self.hyper_alpha = alpha
		self.hyper_beta = beta

	'''
	def forward(self, input):
		# return _functions.dropout.Dropout.apply(input, p, self.training)
		# print(torch.max(p).item(), torch.min(p).item())

		p = torch.clamp(torch.sigmoid(self.logit_p), min=1e-6, max=1 - 1e-6)
		# self.filter = torch.bernoulli(1 - p)
		if len(p.shape) == 0:
			filter = torch.bernoulli(1 - p.repeat(tuple(input.shape)))
		else:
			assert p.shape[-1] == input.shape[-1]
			filter = torch.bernoulli(1 - p.repeat(tuple(input.shape[:-1]) + (1,)))
		self.filter = filter

		if self.training:
			return input.mul(self.filter)
		else:
			return input.mul(1 - p)
	'''

	def variational_lower_bound_approximation(self):
		if self.filter is None:
			return torch.zeros(1)

		p = torch.sigmoid(self.logit_p)
		# print(self.filter.shape)

		E_log_p_theta = self.filter.size(0) * (
				(self.hyper_alpha - 1) * torch.log(1 - p) + (self.hyper_beta - 1) * torch.log(p))

		# assert self.filter.shape[0] == input.size(0), (filter, input)
		filter_sum = torch.sum(self.filter, dim=0)

		E_log_p_z = torch.log(p) * (self.filter.size(0) - filter_sum) + torch.log(1 - p) * filter_sum
		E_log_q_z = - torch.log(p) * p - torch.log(1 - p) * (1 - p)

		return E_log_p_theta.sum() + E_log_p_z.sum() + E_log_q_z.sum()


class Dropout(nn.Module):
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()

		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min{}".format(p, numpy.max(p), numpy.min(p)))

		self.p = torch.tensor(p, dtype=torch.float)
		self.filter = None
		#self.device=device

	# self.inplace = inplace

	def extra_repr(self):
		return 'p={}'.format(self.p)

	def forward(self, input):
		if self.training:
			#with torch.cuda.device(input.get_device()):
			if len(self.p.shape) == 0:
				filter = torch.bernoulli(1 - self.p.repeat(tuple(input.shape)))
			else:
				assert self.p.shape[-1] == input.shape[-1]
				filter = torch.bernoulli(1 - self.p.repeat(tuple(input.shape[:-1]) + (1,)))
			filter = filter.to(input.device)
		else:
			filter = None
		#print(input.get_device(), input.device)
		#print(filter.get_device())
		self.filter = filter
		#print(self.device)
		#self.filter = self.filter.to(self.device)

		# return F.dropout(input, self.p, self.training, self.inplace)
		# return dropout(input, self.p, self.training, self.inplace)

		return DropoutFunction.apply(input, self.p, self.filter)


# Dropout = nn.Dropout
# Dropout2d = nn.Dropout2d
# Dropout3d = nn.Dropout3d


class GaussianDropout(nn.Module):
	'''
	Replication of the Gaussian dropout of Srivastava et al. 2014 (section 10).
	Applies noise to the activations prior to the weight matrix according to equation 11 in the Variational Dropout paper; to match the adaptive dropout implementation.
	'''

	def __init__(self, p=0.5):
		super(GaussianDropout, self).__init__()

		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min{}".format(p, numpy.max(p), numpy.min(p)))
		p = _validate_drop_rate_for_logit_parameterization(p)

		alpha = numpy.sqrt(p / (1. - p))
		# sigma = numpy.ones(dim, dtype=numpy.float32) * numpy.sqrt(p / (1 - p))
		# self.log_alpha = numpy.log(alpha)
		# self.log_alpha = nn.Parameter(torch.tensor(numpy.log(alpha)), dtype=torch.float)
		self.logit_alpha = torch.tensor(logit(alpha), dtype=torch.float)

	def extra_repr(self):
		return 'p={}'.format(1. / (torch.sigmoid(self.logit_alpha) ** 2 + 1))

	def forward(self, input):
		"""
		Sample noise   e ~ N(1, alpha)
		Multiply noise h = h_ * e
		"""

		if self.training:
			# sigma = sigmoid(self.logit_alpha)
			sigma = torch.sigmoid(self.logit_alpha)
			# print(sigma.shape)
			# print(torch.randn(input.size()).shape)
			perturbation = torch.randn(input.size()) * sigma + 1
			# print(perturbation.shape)
			# print(input.shape)

			#
			#
			#
			'''
			if len(sigma.shape) == 0:
				filter = torch.bernoulli(1 - p.repeat(tuple(input.shape)))
			else:
				assert sigma.shape[-1] == input.shape[-1]
				filter = torch.bernoulli(1 - p.repeat(tuple(input.shape[:-1]) + (1,)))
			'''
			#
			#
			#

			return input * perturbation
		else:
			return input


class VariationalGaussianDropout(GaussianDropout):
	def __init__(self, p=0.5, sparse=True):
		super(VariationalGaussianDropout, self).__init__(p)

		self.logit_alpha = nn.Parameter(self.logit_alpha)
		self.sparse = sparse

	def negative_kld_approximation(self):
		# alpha = sigmoid(self.logit_alpha)
		# alpha = F.sigmoid(self.logit_alpha)
		alpha = torch.sigmoid(self.logit_alpha)
		log_alpha = torch.log(alpha)

		if self.sparse:
			k1 = 0.63576
			k2 = 1.8732
			k3 = 1.48695
			C = -k1

			kld_approx = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log(1 + 1. / alpha) + C
		else:
			c1 = 1.161451241083230
			c2 = -1.502041176441722
			c3 = 0.586299206427007

			kld_approx = 0.5 * log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

		return -kld_approx.sum()


#
#
#
#
#

class LinearAndGaussianDropoutWang(nn.modules.Linear):
	def __init__(self, in_features, out_features, p=0.5, bias=True):
		super(LinearAndGaussianDropoutWang, self).__init__(in_features, out_features, bias)

		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min{}".format(p, numpy.max(p), numpy.min(p)))
		p = _validate_drop_rate_for_logit_parameterization(p)

		alpha = numpy.sqrt(p / (1. - p))
		# sigma = numpy.ones(dim, dtype=numpy.float32) * numpy.sqrt(p / (1 - p))
		# self.logit_alpha = nn.Parameter(torch.tensor(logit(alpha), dtype=torch.float))
		self.logit_alpha = torch.tensor(logit(alpha), dtype=torch.float)

	def reset_parameters(self):
		stdv = numpy.sqrt(6. / numpy.sum(self.weight.shape))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			# self.bias.data.uniform_(-stdv, stdv)
			self.bias.data.uniform_(-1e-6, 1e-6)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}, alpha={}'.format(self.in_features, self.out_features,
		                                                                   self.bias is not None,
		                                                                   sigmoid(self.logit_alpha))

	def forward(self, input):
		# log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
		# clip_mask = T.ge(log_alpha, thresh)

		# mu = F.linear(input, self.weight, self.bias)
		mu = super(LinearAndGaussianDropoutWang, self).forward(input)
		if self.training:
			if self.logit_alpha.dim() == 0:
				sigma = torch.sqrt(torch.dot(input ** 2, sigmoid(self.logit_alpha) * (self.weight.t() ** 2)))
				perturbation = torch.randn(mu.shape) * sigma
			elif self.logit_alpha.dim() == 1:
				'''
				print(
					"1", input.shape, "\n",
					"2", sigmoid(self.logit_alpha).shape, "\n",
					"3", sigmoid(self.logit_alpha).view(1, -1).shape, "\n",
					"4", sigmoid(self.logit_alpha).view(1, -1).expand((self.in_features, self.out_features)).shape,
					"\n",
					"5", (self.weight ** 2).shape, "\n",
					"6", (self.weight ** 2).t().shape, "\n",
					"7", (sigmoid(self.logit_alpha).view(1, -1).expand((self.in_features, self.out_features)) * (
							self.weight ** 2).t()).shape, "\n",
					"8", torch.mm(input ** 2, (
							sigmoid(self.logit_alpha).view(1, -1).expand((self.in_features, self.out_features)) * (
							self.weight ** 2).t())).shape
				)
				'''

				sigma = torch.sqrt(torch.matmul(input ** 2, (
						sigmoid(self.logit_alpha).view(1, -1).expand((self.in_features, self.out_features)) * (
						self.weight ** 2).t())))
				# sigma = torch.sqrt(torch.dot(input ** 2, sigmoid(self.logit_alpha) * (self.weight.t() ** 2)))
				perturbation = torch.randn(mu.shape) * sigma

				# a = (input ** 2).data
				# print("input", a.shape, torch.max(a).item(), torch.min(a).item(), torch.mean(a), torch.sum(a))
				# a = self.logit_alpha.data
				# print("logit_alpha", a.shape, torch.max(a).item(), torch.min(a).item(), torch.mean(a), torch.sum(a))
				# a = sigmoid(self.logit_alpha).data
				# print("sigmoid", a.shape, torch.max(a).item(), torch.min(a).item(), torch.mean(a), torch.sum(a))
				a = self.weight.data
				print("weight", a.shape, torch.max(a).item(), torch.min(a).item(), torch.mean(a), torch.sum(a))
				a = torch.sigmoid(self.logit_alpha) * (self.weight.t() ** 2).data
				print("temp", a.shape, torch.max(a).item(), torch.min(a).item(), torch.mean(a), torch.sum(a))
				a = sigma.data
				print("sigma", a.shape, torch.max(a).item(), torch.min(a).item(), torch.mean(a), torch.sum(a))
				a = perturbation.data
				print("perturbation", a.shape, torch.max(a).item(), torch.min(a).item(), torch.mean(a), torch.sum(a))

			# elif self.log_alpha.dim() == 2:
			# sigma = torch.sqrt(torch.matmul(input ** 2, (torch.exp(self.log_alpha) * (self.weight ** 2).t())))
			# perturbation = torch.randn(self.log_alpha.shape) * sigma
			else:
				raise TypeError("Unspecified dimension setting...")
			activation = mu + perturbation
		else:
			activation = mu

		return activation


class LinearAndVariationalGaussianDropoutWang(LinearAndGaussianDropoutWang):
	def __init__(self, in_features, out_features, p=0.5, bias=True, sparse=True):
		super(LinearAndVariationalGaussianDropoutWang, self).__init__(in_features, out_features, p=p, bias=bias)

		# alpha = p / (1. - p)
		# self.log_alpha = nn.Parameter(torch.tensor(numpy.log(alpha), dtype=torch.float))
		# sigma = numpy.ones(dim, dtype=numpy.float32) * numpy.sqrt(p / (1 - p))
		self.logit_alpha = nn.Parameter(self.logit_alpha)
		self.sparse = sparse

	def nkld_approximation(self):
		return nkld_approximation(torch.sigmoid(self.logit_alpha), self.sparse)

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


#
#
#
#
#


class AdaptiveBernoulliDropoutBackup(nn.Module):
	def __init__(self, p=0.5):
		super(AdaptiveBernoulliDropoutBackup, self).__init__()
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min{}".format(p, numpy.max(p), numpy.min(p)))

		self.p = nn.Parameter(torch.tensor(p, dtype=torch.float))
		self.filter = None

	def forward(self, input):
		if self.p.dim() == 1:
			p = self.p.expand(input.shape[0], -1)
		else:
			p = self.p

		# return _functions.dropout.Dropout.apply(input, p, self.training)
		# TODO: torch.clamp() function does not properly propagate gradient, so we use two torch.relu() functions.
		# TODO: optimization with momentum sometimes fails when working with these "clipping" methods.
		self.filter = torch.bernoulli(torch.relu(1 - torch.relu(p)))
		if self.training:
			return input.mul(self.filter)
		else:
			return input.mul(torch.relu(1 - torch.relu(p)))


class AdaptiveBetaBernoulliDropoutBackup(nn.Module):
	def __init__(self, p=0.5):
		super(AdaptiveBetaBernoulliDropoutBackup, self).__init__()
		if numpy.any(p < 0) or numpy.any(p > 1):
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {} with max {} and min{}".format(p, numpy.max(p), numpy.min(p)))
		self.p = nn.Parameter(torch.tensor(p, dtype=torch.float))
		self.filter = None

	def forward(self, input):
		# if torch.max(self.p) > 1 or torch.min(self.p) < 0:
		# self.p.data = self.p.data.clamp_(1e-6, 1 - 1e-6)
		if self.p.dim() == 1:
			p = self.p.expand(input.shape[0], -1)
		else:
			p = self.p

		# return _functions.dropout.Dropout.apply(input, p, self.training)

		self.filter = torch.bernoulli(1 - torch.clamp(p, 0, 1))
		if self.training:
			return input.mul(self.filter)
		else:
			return input.mul(1 - self.p)
