import logging

import numpy
import torch.nn as nn
import torch.nn.functional as F

import porch
import porch.modules
from porch import layer_deliminator
from porch.models import parse_feed_forward_layers

# , parse_to_int_sequence, parse_to_float_sequence, parse_activations, parse_drop_modes

logger = logging.getLogger(__name__)

__all__ = [
	"GenericMLP",
]


class GenericMLP(nn.Module):
	def __init__(self,
	             input_shape,
	             dimensions,
	             activations,  # ="",
	             drop_modes,  # ="",
	             drop_rates,  # =""
	             *args,
	             **kwargs
	             ):
		super(GenericMLP, self).__init__()

		feature_shape = [int(temp_shape) for temp_shape in input_shape.split(layer_deliminator)]

		'''
		dimensions = parse_to_int_sequence(string_of_ints=dimensions)
		dimensions.insert(0, numpy.prod(feature_shape))

		activations = parse_activations(activations_argument=activations)
		if len(activations) == 1:
			activations = activations * (len(dimensions) - 1)
		assert (len(dimensions) == len(activations) + 1)

		drop_modes = parse_drop_modes(drop_modes_argument=drop_modes)
		if len(drop_modes) == 1:
			drop_modes = drop_modes * (len(dimensions) - 1)
		assert (len(dimensions) == len(drop_modes) + 1)

		drop_rates = parse_to_float_sequence(string_of_float=drop_rates, default_value=0)
		if len(drop_rates) == 1:
			drop_rates = drop_rates * (len(dimensions) - 1)
		assert (len(dimensions) == len(drop_rates) + 1)

		layers = []
		for x in range(len(dimensions) - 1):
			assert 0 <= drop_rates[x] < 1
			if (drop_modes[x] is not None) and (drop_rates[x] > 0):
				layers.append(drop_modes[x](p=numpy.ones(dimensions[x]) * drop_rates[x]))
			layers.append(nn.Linear(dimensions[x], dimensions[x + 1]))
			if activations[x] is not None:
				layers.append(activations[x]())
		'''

		layers = parse_feed_forward_layers(
			input_dimension=numpy.prod(feature_shape),
			dimensions=dimensions,
			activations=activations,
			drop_modes=drop_modes,
			drop_rates=drop_rates
		)

		self.classifier = nn.Sequential(*layers)

	def forward(self, x, *args, **kwargs):
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class MLP_test(GenericMLP):
	def __init__(self, input_shape, output_shape, *args, **kwargs):
		super(MLP_test, self).__init__(
			input_shape=input_shape,
			dimensions=layer_deliminator.join(["1024", "%s" % output_shape]),
			activations=layer_deliminator.join(["ReLU", "LogSoftmax"]),
			drop_modes=layer_deliminator.join([porch.modules.Dropout.__name__, porch.modules.Dropout.__name__]),
			drop_rates=layer_deliminator.join(["0.2", "0.5"]),
			*args, **kwargs
		)


class MLP_GaussianDropout_test(GenericMLP):
	def __init__(self, input_shape, output_shape, *args, **kwargs):
		super(MLP_GaussianDropout_test, self).__init__(
			input_shape=input_shape,
			dimensions=layer_deliminator.join(["1024", "%s" % output_shape]),
			activations=layer_deliminator.join(["ReLU", "LogSoftmax"]),
			drop_modes=layer_deliminator.join(
				[porch.modules.GaussianDropout.__name__, porch.modules.GaussianDropout.__name__]),
			drop_rates=layer_deliminator.join(["0.2", "0.5"]),
			*args, **kwargs
		)


class MLP_VariationalGaussianDropout_test(GenericMLP):
	def __init__(self, input_shape, output_shape, *args, **kwargs):
		super(MLP_VariationalGaussianDropout_test, self).__init__(
			input_shape=input_shape,
			dimensions=layer_deliminator.join(["1024", "%s" % output_shape]),
			activations=layer_deliminator.join(["ReLU", "LogSoftmax"]),
			drop_modes=layer_deliminator.join(
				[porch.modules.VariationalGaussianDropout.__name__, porch.modules.VariationalGaussianDropout.__name__]),
			drop_rates=layer_deliminator.join(["0.2", "0.5"]),
			*args, **kwargs
		)


class MLP_AdaptiveBernoulliDropout_test(GenericMLP):
	def __init__(self, input_shape, output_shape, *args, **kwargs):
		super(MLP_AdaptiveBernoulliDropout_test, self).__init__(
			input_shape=input_shape,
			dimensions=layer_deliminator.join(["1024", "%s" % output_shape]),
			activations=layer_deliminator.join(["ReLU", "LogSoftmax"]),
			drop_modes=layer_deliminator.join(
				[porch.modules.AdaptiveBernoulliDropout.__name__,
				 porch.modules.AdaptiveBernoulliDropout.__name__]),
			drop_rates=layer_deliminator.join(["0.2", "0.5"]),
			*args, **kwargs
		)


class MLP_AdaptiveBetaBernoulliDropout_test(GenericMLP):
	def __init__(self, input_shape, output_shape, *args, **kwargs):
		super(MLP_AdaptiveBetaBernoulliDropout_test, self).__init__(
			input_shape=input_shape,
			dimensions=layer_deliminator.join(["1024", "%s" % output_shape]),
			activations=layer_deliminator.join(["ReLU", "LogSoftmax"]),
			drop_modes=layer_deliminator.join(
				[porch.modules.AdaptiveBetaBernoulliDropout.__name__,
				 porch.modules.AdaptiveBetaBernoulliDropout.__name__]),
			drop_rates=layer_deliminator.join(["0.2", "0.5"]),
			*args, **kwargs
		)


#
#
#
#
#

class GenericAdaptiveMLP(nn.Module):
	def __init__(self,
	             input_shape,
	             dimensions,
	             activations="",
	             drop_rates=""
	             ):
		super(GenericAdaptiveMLP, self).__init__()

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


class MLPAdaptiveDropout(nn.Module):
	def __init__(self):
		super(MLPAdaptiveDropout, self).__init__()
		# self.input_drop = porch.modules.AdaptiveBernoulliDropoutInLogitSpace(p=numpy.ones(784) * 0.5)
		self.input_drop = porch.modules.AdaptiveBetaBernoulliDropout(p=numpy.ones(784) * 0.5, alpha=0.1,
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


class MLPSparseVariationalDropout(nn.Module):
	def __init__(self):
		super(MLPSparseVariationalDropout, self).__init__()
		self.input_drop = nn.Dropout(p=0.2)
		self.fc1 = porch.modules.LinearAndVariationalGaussianDropoutWang(784, 100, p=numpy.ones(100) * 0.5)
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


#
#
#
#
#

class GenericMLP_FastGaussianDropout(nn.Module):
	def __init__(self,
	             input_shape,
	             dimensions,
	             activations,  # ="",
	             drop_modes,  # ="",
	             drop_rates  # =""
	             ):
		super(GenericMLP_FastGaussianDropout, self).__init__()

		feature_shape = [int(temp_shape) for temp_shape in input_shape.split(layer_deliminator)]

		dimensions = parse_to_int_sequence(string_of_ints=dimensions)
		dimensions.insert(0, numpy.prod(feature_shape))

		activations = parse_activations(activations_argument=activations)
		if len(activations) == 1:
			activations = activations * (len(dimensions) - 1)
		assert (len(dimensions) == len(activations) + 1)

		drop_modes = parse_drop_modes(drop_modes_argument=drop_modes)
		# if len(drop_modes) == 1:
		#	drop_modes = drop_modes * (len(dimensions) - 1)
		assert (len(dimensions) == len(drop_modes) + 1)

		drop_rates = parse_to_float_sequence(string_of_float=drop_rates, default_value=0)
		if len(drop_rates) == 1:
			drop_rates = drop_rates * (len(dimensions) - 1)
		assert (len(dimensions) == len(drop_rates) + 1)

		layers = []
		# drop_modes = None if drop_modes.lower() == "none" else getattr(porch.modules, drop_modes)
		if (drop_modes[0] is not None) and (drop_rates[0] > 0):
			layers.append(drop_modes[0](p=drop_rates[0]))
		for x in range(len(dimensions) - 2):
			print(drop_modes[x + 1], dimensions[x], dimensions[x + 1], drop_rates[x + 1])
			# layers.append(porch.modules.LinearAndGaussianDropoutWang(dimensions[x], dimensions[x + 1], p=drop_rates[x + 1]))
			layers.append(drop_modes[x + 1](dimensions[x], dimensions[x + 1],
			                                p=numpy.ones(dimensions[x + 1]) * drop_rates[x + 1]))
			if activations[x] is not None:
				layers.append(activations[x]())
		layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
		if activations[-1] is not None:
			layers.append(activations[-1]())

		self.classifier = nn.Sequential(*layers)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class MLP_FastGaussianDropoutWang_test(GenericMLP_FastGaussianDropout):
	def __init__(self, input_shape, output_shape):
		super(MLP_FastGaussianDropoutWang_test, self).__init__(
			input_shape=input_shape,
			dimensions=layer_deliminator.join(["1024", "%s" % output_shape]),
			activations=layer_deliminator.join(["ReLU", "LogSoftmax"]),
			drop_modes=layer_deliminator.join(["Dropout", "LinearAndGaussianDropoutWang"]),
			drop_rates=layer_deliminator.join(["0.2", "0.2"]),
		)


class MLP_VariationalFastGaussianDropoutWang_test(GenericMLP):
	def __init__(self, input_shape, output_shape):
		super(MLP_VariationalFastGaussianDropoutWang_test, self).__init__(
			input_shape=input_shape,
			dimensions=layer_deliminator.join(["1024", "%s" % output_shape]),
			activations=layer_deliminator.join(["ReLU", "LogSoftmax"]),
			drop_modes=layer_deliminator.join(
				["VariationalGaussianDropoutSrivastava", "VariationalGaussianDropoutSrivastava"]),
			drop_rates=layer_deliminator.join(["0.2", "0"])
		)
