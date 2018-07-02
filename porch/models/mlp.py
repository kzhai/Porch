import logging

import numpy
import torch.nn as nn
import torch.nn.functional as F

import porch
from porch import layer_deliminator
from porch.models import parse_to_int_sequence, parse_activations, parse_to_float_sequence

logger = logging.getLogger(__name__)

__all__ = [
	"GenericMLP",
]


class GenericMLP(nn.Module):
	def __init__(self,
	             input_shape,
	             dimensions,
	             activations="",
	             drop_rates=""
	             ):
		super(GenericMLP, self).__init__()

		feature_shape = [int(temp_shape) for temp_shape in input_shape.split(layer_deliminator)]

		dimensions = parse_to_int_sequence(string_of_ints=dimensions)
		dimensions.insert(0, numpy.prod(feature_shape))

		activations = parse_activations(activations_argument=activations)
		if len(activations) == 1:
			activations = activations * (len(dimensions) - 1)
		assert (len(dimensions) == len(activations) + 1)

		drop_rates = parse_to_float_sequence(string_of_float=drop_rates, default_value=0)
		if len(drop_rates) == 1:
			drop_rates = drop_rates * (len(dimensions) - 1)
		assert (len(dimensions) == len(drop_rates) + 1)

		layers = []
		for x in range(len(drop_rates)):
			assert 0 <= drop_rates[x] < 1
			if drop_rates[x] > 0:
				layers.append(nn.Dropout(p=drop_rates[x]))
			layers.append(nn.Linear(dimensions[x], dimensions[x + 1]))
			if activations[x] is not None:
				layers.append(activations[x]())

		self.classifier = nn.Sequential(*layers)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class MLP_test(GenericMLP):
	def __init__(self, input_shape, output_shape):
		super(MLP_test, self).__init__(
			input_shape=input_shape,
			dimensions=layer_deliminator.join(["1000", "%s" % output_shape]),
			activations=layer_deliminator.join(["ReLU", "LogSoftmax"]),
			drop_rates=layer_deliminator.join(["0.2", "0.5"])
		)


class MLPAdaptiveDropout(nn.Module):
	def __init__(self):
		super(MLPAdaptiveDropout, self).__init__()
		# self.input_drop = porch.modules.AdaptiveBernoulliDropoutInLogitSpace(p=numpy.ones(784) * 0.5)
		self.input_drop = porch.modules.AdaptiveBetaBernoulliDropoutInLogitSpace(p=numpy.ones(784) * 0.5, alpha=0.1,
		                                                                         beta=0.1)
		# self.input_drop = porch.modules.AdaptiveBernoulliDropout(p=numpy.ones(784)*0.5)
		self.fc = nn.Linear(784, 10)

	# self.fc1 = nn.Linear(784, 1000)
	# self.hidden_drop = nn.Dropout(0.5)
	# self.fc2 = nn.Linear(1000, 10)

	def forward(self, x):
		x = x.view(-1, 784)
		x = self.input_drop(x)
		return F.log_softmax(self.fc(x), dim=1)
		'''
		x = F.relu(self.fc1(x))
		x = self.hidden_drop(x)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)
		'''


class MLPFastGaussianDropout(nn.Module):
	def __init__(self):
		super(MLPFastGaussianDropout, self).__init__()
		self.input_drop = nn.Dropout(p=0.2)
		self.fc1 = porch.modules.LinearAndGaussianDropout(784, 1000, p=0.5)
		self.fc2 = porch.modules.LinearAndGaussianDropout(1000, 100, p=numpy.random.random(100))
		self.fc3 = nn.modules.Linear(100, 10)

	def forward(self, x):
		x = x.view(-1, 784)
		x = self.input_drop(x)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)


class MLPSparseVariationalDropout(nn.Module):
	def __init__(self):
		super(MLPSparseVariationalDropout, self).__init__()
		self.input_drop = nn.Dropout(p=0.2)
		self.fc1 = porch.modules.LinearAndVariationalGaussianDropout(784, 100, p=numpy.ones(100) * 0.5)
		# self.fc2 = porch.modules.LinearAndVariationalGaussianDropout(1000, 100, p=numpy.ones(100)*0.5)
		self.fc3 = nn.modules.Linear(100, 10)

	def forward(self, x):
		x = x.view(-1, 784)
		x = self.input_drop(x)
		x = F.relu(self.fc1(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)


class MLPVariationalDropout(nn.Module):
	def __init__(self):
		super(MLPVariationalDropout, self).__init__()
		self.drop1 = porch.modules.VariationalGaussianDropout(p=numpy.ones(784) * 0.2)
		self.fc1 = nn.Linear(784, 1000)
		self.drop2 = porch.modules.VariationalGaussianDropout(p=numpy.ones(1000) * 0.5)
		self.fc2 = nn.Linear(1000, 10)

	'''
	def kl(self):
		kl = 0
		for name, module in self.net.named_modules():
			if isinstance(module, porch.modules.VariationalGaussianDropout):
				kl += module.nkld_approximation().sum()
		return kl
	'''

	def forward(self, x):
		x = x.view(-1, 784)
		x = self.drop1(x)
		x = F.relu(self.fc1(x))
		x = self.drop2(x)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)
