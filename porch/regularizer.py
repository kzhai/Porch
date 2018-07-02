import logging

import torch
import torch.nn.functional

import porch
from porch.modules.dropout import sigmoid

logger = logging.getLogger(__name__)

__all__ = [
	"variational_gaussian_dropout",
]


def variational_gaussian_dropout(network, input=None, output=None, sparse=True):
	kld_approximation = torch.zeros(1)
	for name, module in network.named_modules():
		if (type(module) is porch.modules.dropout.VariationalGaussianDropout) or \
				(type(module) is porch.modules.dropout.LinearAndVariationalGaussianDropout):
			if sparse:
				k1 = 0.63576
				k2 = 1.8732
				k3 = 1.48695
				C = -k1

				kld_approx = k1 * torch.sigmoid(k2 + k3 * module.log_alpha) - 0.5 * torch.log(
					1 + torch.exp(-module.log_alpha)) + C
			else:
				c1 = 1.161451241083230
				c2 = -1.502041176441722
				c3 = 0.586299206427007

				alpha = module.log_alpha.exp()
				kld_approx = 0.5 * module.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

			# kld_approximation += -kld_approx.mean()
			kld_approximation += -kld_approx.sum()

	'''
	if size_average:
		return kld_approximation.sum() / input.size(0)
	'''
	return kld_approximation.sum() / input.size(0)


def variational_bernoulli_dropout(network, input=None, output=None):
	variational_lower_bound = torch.zeros(1)
	for name, module in network.named_modules():
		if (type(module) is porch.modules.dropout.AdaptiveBernoulliDropout):
			p = module.p
			filter = module.filter
			E_log_p_theta = 0
		elif (type(module) is porch.modules.dropout.AdaptiveBernoulliDropoutInLogitSpace):
			p = sigmoid(module.logit_p)
			filter = module.filter
			E_log_p_theta = 0
		elif (type(module) is porch.modules.dropout.AdaptiveBetaBernoulliDropout):
			p = module.p
			filter = module.filter
			E_log_p_theta = input.size(0) * (
						(module.hyper_alpha - 1) * torch.log(1 - p) + (module.hyper_beta - 1) * torch.log(p))
		elif (type(module) is porch.modules.dropout.AdaptiveBetaBernoulliDropoutInLogitSpace):
			p = sigmoid(module.logit_p)
			filter = module.filter
			E_log_p_theta = input.size(0) * (
						(module.hyper_alpha - 1) * torch.log(1 - p) + (module.hyper_beta - 1) * torch.log(p))
		else:
			continue

		# if filter is None:
		# return 0

		assert filter.shape[0] == input.size(0), (filter, input)
		filter = torch.sum(filter, dim=0)
		E_log_p_z = torch.log(p) * (input.size(0) - filter) + torch.log(1 - p) * filter
		E_log_q_z = - torch.log(p) * p - torch.log(1 - p) * (1 - p)

		variational_lower_bound += E_log_p_z.sum() + E_log_q_z.sum() + E_log_p_theta.sum()

	return -variational_lower_bound.sum() / input.size(0)
