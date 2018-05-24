import logging

import numpy

import torch
import torch.nn.functional

import porch
from porch.modules.dropout import sigmoid

logger = logging.getLogger(__name__)

__all__ = [
	"vardrop_kld_approximation",
]


def vardrop_kld_approximation(network, input=None, output=None, sparse=True):
	kld_approximation = torch.zeros(1)
	for name, module in network.named_modules():
		if (type(module) is porch.modules.dropout.VariationalDropout) or \
				(type(module) is porch.modules.dropout.LinearAndVariationalDropout):
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


def vardrop_(network, input=None, output=None):
	variational_lower_bound = torch.zeros(1)
	for name, module in network.named_modules():
		if (type(module) is porch.modules.dropout.AdaptiveBernoulliDropoutInLogitSpace) or \
				(type(module) is porch.modules.dropout.AdaptiveBernoulliDropout):
			# mask_counts = module.p * input.size(0)
			if (type(module) is porch.modules.dropout.AdaptiveBernoulliDropoutInLogitSpace):
				p = sigmoid(module.logit_p)
			elif (type(module) is porch.modules.dropout.AdaptiveBernoulliDropout):
				p = module.p

			# TODO: This is only an approximation, using expectation on the draws to reduce the memory usage.
			E_log_p_dropout = torch.log(p) * p * input.size(0) + torch.log(1 - p) * (1 - p) * input.size(0)

			E_log_q_dropout = - torch.log(p) * p - torch.log(1 - p) * (1 - p)

			variational_lower_bound += E_log_p_dropout.sum() + E_log_q_dropout.sum()

	return -variational_lower_bound.sum() / input.size(0)


def vardrop_kld_approximation_deprecated(network, input=None, output=None):
	kld_approximation = torch.zeros(1)
	for name, module in network.named_modules():
		if (type(module) is porch.modules.dropout.VariationalDropout) or \
				(type(module) is porch.modules.dropout.LinearAndVariationalDropout):
			kld_approximation += module.nkld_approximation()

	kld_approximation /= output.size(1)
	return kld_approximation.sum() / input.size(0)
