import argparse
import datetime
import logging
import pickle
import os
import random
import sys
import timeit

import numpy

import porch

import torch
import torch.nn.functional

logger = logging.getLogger(__name__)

__all__ = [
	"vardrop_kld_approximation"
]


def vardrop_kld_approximation(network, input=None, output=None, size_average=True):
	kld_approximation = torch.zeros(1)
	# number_of_variational_dropout_modules = 1
	for name, module in network.named_modules():
		if (type(module) is porch.modules.dropout.VariationalDropoutKingma) or \
				(type(module) is porch.modules.dropout.LinearSparseVariationalDropout):
			kld_approximation += module.nkld_approximation()
	# number_of_variational_dropout_modules += 1

	kld_approximation /= output.size(1)
	# kld_approximation /= number_of_variational_dropout_modules
	if size_average:
		return kld_approximation.sum() / input.size(0)

	return kld_approximation.sum()
