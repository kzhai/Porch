import logging

import torch
import torch.nn.functional

import porch
#from porch.modules.dropout import sigmoid

logger = logging.getLogger(__name__)

__all__ = [
	"variational_gaussian_dropout",
]


def variational_gaussian_dropout(network, input=None, output=None):
	kld_approximation = torch.zeros(1)
	for name, module in network.named_modules():
		temp_kld_approximation = torch.zeros(1)
		try:
			#nkld_approximation = module.nkld_approximation
			temp_kld_approximation = module.negative_kld_approximation()
		except AttributeError:
			pass

		kld_approximation += temp_kld_approximation.sum()

		'''
		if (type(module) is porch.modules.dropout.VariationalGaussianDropout) or \
				(type(module) is porch.modules.dropout.LinearAndVariationalGaussianDropoutWang):
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

	return kld_approximation.sum() / input.size(0)


def variational_bernoulli_dropout(network, input=None, output=None):
	variational_lower_bound = torch.zeros(1)
	for name, module in network.named_modules():
		temp_variational_lower_bound = torch.zeros(1)
		try:
			# nkld_approximation = module.nkld_approximation
			temp_variational_lower_bound = module.variational_lower_bound_approximation()
		except AttributeError:
			pass

		#print(temp_variational_lower_bound)
		variational_lower_bound += temp_variational_lower_bound.sum()

		'''
		if (type(module) is porch.modules.dropout.AdaptiveBernoulliDropoutBackup):
			p = module.p
			filter = module.filter
			E_log_p_theta = torch.tensor(0, dtype=torch.float)
		elif (type(module) is porch.modules.dropout.VariationalBernoulliDropout):
			p = torch.sigmoid(module.logit_p)
			filter = module.filter
			E_log_p_theta = torch.tensor(0, dtype=torch.float)
		elif (type(module) is porch.modules.dropout.AdaptiveBetaBernoulliDropoutBackup):
			p = module.p
			filter = module.filter
			E_log_p_theta = input.size(0) * (
					(module.hyper_alpha - 1) * torch.log(1 - p) + (module.hyper_beta - 1) * torch.log(p))
		elif (type(module) is porch.modules.dropout.VariationalBetaBernoulliDropout):
			p = torch.sigmoid(module.logit_p)
			filter = module.filter
			E_log_p_theta = input.size(0) * (
					(module.hyper_alpha - 1) * torch.log(1 - p) + (module.hyper_beta - 1) * torch.log(p))
		else:
			continue
		'''
		#
		#
		#
		'''
		if len(p.shape) == 0:
			filter = torch.bernoulli(1 - p.repeat(tuple(input.shape)))
		else:
			#assert p.shape[-1] == input.shape[-1]
			filter = torch.bernoulli(1 - p.repeat(tuple(input.shape[:-1]) + (1,)))
		'''
		#
		#
		#

		'''
		assert filter.shape[0] == input.size(0), (filter, input)
		filter = torch.sum(filter, dim=0)
		E_log_p_z = torch.log(p) * (input.size(0) - filter) + torch.log(1 - p) * filter
		E_log_q_z = - torch.log(p) * p - torch.log(1 - p) * (1 - p)

		variational_lower_bound += E_log_p_z.sum() + E_log_q_z.sum() + E_log_p_theta.sum()
		'''
	return variational_lower_bound.sum() / input.size(0)
