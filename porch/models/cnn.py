import logging

import numpy
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import porch
import porch.modules
from porch import layer_deliminator
from .parser import parse_to_int_sequence, parse_to_float_sequence, parse_activations, parse_pool_modes, \
	parse_drop_modes

logger = logging.getLogger(__name__)

__all__ = ['Generic2DCNN']


class Generic2DCNN(nn.Module):
	def __init__(self,
	             input_shape,
	             #
	             conv_channels,
	             conv_kernel_sizes,
	             conv_strides,  # ="1",
	             conv_paddings,  # ="0",
	             #
	             conv_drop_modes,
	             conv_drop_rates,
	             #
	             conv_activations,  # ="None",
	             #
	             pool_modes,  # ="None",
	             pool_kernel_sizes,
	             pool_strides,  # ="1",
	             #
	             linear_dimensions,
	             linear_activations,  # ="None",
	             #
	             linear_drop_modes,
	             linear_drop_rates,  # ="0.0"
	             *args, **kwargs
	             ):
		super(Generic2DCNN, self).__init__()

		feature_shape = [int(temp_shape) for temp_shape in input_shape.split(layer_deliminator)]
		if len(feature_shape) == 1:
			feature_shape = feature_shape * 2
		elif len(feature_shape) == 2:
			feature_shape = feature_shape
		elif len(feature_shape):
			feature_shape = feature_shape[1:]

		#
		#
		#

		conv_channels = parse_to_int_sequence(string_of_ints=conv_channels)

		conv_kernel_sizes = parse_to_int_sequence(string_of_ints=conv_kernel_sizes)
		assert (len(conv_channels) == len(conv_kernel_sizes) + 1)

		conv_strides = parse_to_int_sequence(string_of_ints=conv_strides, default=1)
		if len(conv_strides) == 1:
			conv_strides = conv_strides * (len(conv_channels) - 1)
		assert (len(conv_channels) == len(conv_strides) + 1)

		conv_paddings = parse_to_int_sequence(string_of_ints=conv_paddings, default=0)
		if len(conv_paddings) == 1:
			conv_paddings = conv_paddings * (len(conv_channels) - 1)
		assert (len(conv_channels) == len(conv_paddings) + 1)

		#
		#
		#

		conv_drop_modes = parse_drop_modes(drop_modes_argument=conv_drop_modes)
		if len(conv_drop_modes) == 1:
			conv_drop_modes = conv_drop_modes * (len(conv_channels) - 1)
		assert (len(conv_channels) == len(conv_drop_modes) + 1)

		conv_drop_rates = parse_to_float_sequence(string_of_float=conv_drop_rates, default_value=0)
		if len(conv_drop_rates) == 1:
			conv_drop_rates = conv_drop_rates * (len(conv_channels) - 1)
		assert (len(conv_channels) == len(conv_drop_rates) + 1)

		#
		#
		#

		conv_activations = parse_activations(activations_argument=conv_activations)
		if len(conv_activations) == 1:
			conv_activations = conv_activations * (len(conv_channels) - 1)
		assert (len(conv_channels) == len(conv_activations) + 1)

		#
		#
		#

		pool_kernel_sizes = parse_to_int_sequence(string_of_ints=pool_kernel_sizes, default=0)
		assert (len(conv_channels) == len(pool_kernel_sizes) + 1)

		pool_modes = parse_pool_modes(pool_modes)
		if len(pool_modes) == 1:
			pool_modes = pool_modes * (len(conv_channels) - 1)
		assert (len(conv_channels) == len(pool_modes) + 1)

		pool_strides = parse_to_int_sequence(string_of_ints=pool_strides, default=1)
		if len(pool_strides) == 1:
			pool_strides = pool_strides * (len(conv_channels) - 1)
		assert (len(conv_channels) == len(pool_strides) + 1)

		#
		#
		#

		layers = []
		for x in range(len(conv_channels) - 1):
			if (conv_drop_modes[x] is not None) and (conv_drop_rates[x] > 0):
				layers.append(conv_drop_modes[x](p=conv_drop_rates[x]))
			layers.append(nn.Conv2d(conv_channels[x], conv_channels[x + 1], kernel_size=conv_kernel_sizes[x],
			                        stride=conv_strides[x], padding=conv_paddings[x]))
			feature_shape = [(temp_shape + 2 * conv_paddings[x] - conv_kernel_sizes[x]) // conv_strides[x] + 1 for
			                 temp_shape in feature_shape]

			if conv_activations[x] is not None:
				layers.append(conv_activations[x]())

			if (pool_modes[x] is not None) and (pool_kernel_sizes[x] > 0):
				layers.append(pool_modes[x](kernel_size=pool_kernel_sizes[x], stride=pool_strides[x]))
			feature_shape = [(temp_shape + 2 * 0 - pool_kernel_sizes[x]) // pool_strides[x] + 1 for temp_shape in
			                 feature_shape]
		self.features = nn.Sequential(*layers)

		#
		#
		#

		linear_dimensions = parse_to_int_sequence(string_of_ints=linear_dimensions)
		linear_dimensions.insert(0, numpy.prod(feature_shape) * conv_channels[-1])

		linear_activations = parse_activations(activations_argument=linear_activations)
		if len(linear_activations) == 1:
			linear_activations = linear_activations * (len(linear_dimensions) - 1)
		assert (len(linear_dimensions) == len(linear_activations) + 1)

		#
		#
		#

		linear_drop_modes = parse_drop_modes(drop_modes_argument=linear_drop_modes)
		if len(linear_drop_modes) == 1:
			linear_drop_modes = linear_drop_modes * (len(linear_dimensions) - 1)
		assert (len(linear_dimensions) == len(linear_drop_modes) + 1)

		linear_drop_rates = parse_to_float_sequence(string_of_float=linear_drop_rates)
		if len(linear_drop_rates) == 1:
			linear_drop_rates = linear_drop_rates * (len(linear_dimensions) - 1)
		assert (len(linear_dimensions) == len(linear_drop_rates) + 1)

		#
		#
		#

		layers = []
		for x in range(len(linear_dimensions) - 1):
			assert 0 <= linear_drop_rates[x] < 1
			if (linear_drop_modes[x] is not None) and (linear_drop_rates[x] > 0):
				layers.append(linear_drop_modes[x](p=linear_drop_rates[x]))
			layers.append(nn.Linear(linear_dimensions[x], linear_dimensions[x + 1]))
			if linear_activations[x] is not None:
				layers.append(linear_activations[x]())

		self.layers = layers
		self.classifier = nn.Sequential(*layers)

	def forward(self, x, *args, **kwargs):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class CNN_test(Generic2DCNN):
	def __init__(self, input_shape, input_channel, output_shape, *args, **kwargs):
		super(CNN_test, self).__init__(
			input_shape="%s" % (input_shape),
			#
			conv_channels=layer_deliminator.join(["%d" % int(input_channel), "32", "64"]),
			conv_kernel_sizes=layer_deliminator.join(["5", "5"]),
			conv_strides=layer_deliminator.join(["1", "1"]),
			conv_paddings=layer_deliminator.join(["2", "2"]),
			#
			conv_drop_modes=layer_deliminator.join([]),
			conv_drop_rates=layer_deliminator.join([]),
			#
			conv_activations=layer_deliminator.join(["ReLU", "ReLU"]),
			#
			pool_modes=layer_deliminator.join(["MaxPool2d", "MaxPool2d"]),
			pool_kernel_sizes=layer_deliminator.join(["3", "3"]),
			pool_strides=layer_deliminator.join(["2", "2"]),
			#
			linear_dimensions=layer_deliminator.join(["1024", "%s" % output_shape]),
			linear_drop_modes=layer_deliminator.join([porch.modules.Dropout.__name__, "None"]),
			linear_drop_rates=layer_deliminator.join(["0.5", "0.0"]),
			linear_activations=layer_deliminator.join(["ReLU", "None"]),
			#
			*args, **kwargs
		)


class CNN_80sec(Generic2DCNN):
	def __init__(self, input_shape, output_shape, *args, **kwargs):
		super(CNN_80sec, self).__init__(
			input_shape="%s" % (input_shape),
			#
			conv_channels=layer_deliminator.join(["3", "32", "32", "64"]),
			conv_kernel_sizes=layer_deliminator.join(["5", "5", "5"]),
			conv_strides=layer_deliminator.join(["1", "1", "1"]),
			conv_paddings=layer_deliminator.join(["2", "2", "2"]),
			#
			conv_drop_modes=layer_deliminator.join([]),
			conv_drop_rates=layer_deliminator.join([]),
			#
			conv_activations=layer_deliminator.join(["ReLU", "ReLU", "ReLU"]),
			#
			pool_kernel_sizes=layer_deliminator.join(["3", "3", "3"]),
			pool_modes=layer_deliminator.join(["MaxPool2d", "AvgPool2d", "AvgPool2d"]),
			pool_strides=layer_deliminator.join(["2", "2", "2"]),
			#
			linear_dimensions=layer_deliminator.join(["64", "%s" % output_shape]),
			linear_drop_modes=layer_deliminator.join(["None", "None"]),
			linear_drop_rates=layer_deliminator.join(["0.0", "0.0"]),
			linear_activations=layer_deliminator.join(["ReLU", "None"]),
			#
			*args, **kwargs
		)


class CNN_11pts(Generic2DCNN):
	def __init__(self, input_shape, output_shape, *args, **kwargs):
		super(CNN_11pts, self).__init__(
			input_shape="%s" % (input_shape),
			#
			conv_channels=layer_deliminator.join(["3", "64", "64", "64", "32"]),
			conv_kernel_sizes=layer_deliminator.join(["5", "5", "3", "3"]),
			#
			conv_strides=layer_deliminator.join(["1", "1", "1", "1"]),
			conv_paddings=layer_deliminator.join(["2", "2", "1", "1"]),
			#
			conv_drop_modes=layer_deliminator.join([]),
			conv_drop_rates=layer_deliminator.join([]),
			#
			conv_activations=layer_deliminator.join(["ReLU", "ReLU", "ReLU", "ReLU"]),
			#
			pool_modes=layer_deliminator.join(["MaxPool2d", "MaxPool2d", "None", "None"]),
			pool_kernel_sizes=layer_deliminator.join(["3", "3", "0", "0"]),
			pool_strides=layer_deliminator.join(["2", "2", "0", "0"]),
			#
			linear_dimensions=layer_deliminator.join(["%s" % output_shape]),
			linear_drop_modes=layer_deliminator.join(["None"]),
			linear_drop_rates=layer_deliminator.join(["0.0"]),
			linear_activations=layer_deliminator.join(["None"]),
			#
			*args, **kwargs
		)


class AlexNet(Generic2DCNN):
	def __init__(self, input_shape, output_shape, *args, **kwargs):
		super(AlexNet, self).__init__(
			input_shape="%s" % (input_shape),
			#
			conv_channels=layer_deliminator.join(["3", "64", "192", "384", "256", "256"]),
			conv_kernel_sizes=layer_deliminator.join(["11", "5", "3", "3", "3"]),
			#
			conv_strides=layer_deliminator.join(["4", "1", "1", "1", "1"]),
			conv_paddings=layer_deliminator.join(["2", "2", "1", "1", "1"]),
			#
			conv_drop_modes=layer_deliminator.join([]),
			conv_drop_rates=layer_deliminator.join([]),
			#
			conv_activations=layer_deliminator.join(["ReLU", "ReLU", "ReLU", "ReLU", "ReLU"]),
			#
			pool_modes=layer_deliminator.join(["MaxPool2d", "MaxPool2d", "None", "None", "MaxPool2d"]),
			pool_kernel_sizes=layer_deliminator.join(["3", "3", "0", "0", "3"]),
			pool_strides=layer_deliminator.join(["2", "2", "0", "0", "2"]),
			#
			linear_dimensions=layer_deliminator.join(["4096", "4096", "%s" % output_shape]),
			linear_drop_modes=layer_deliminator.join(["Dropout", "Dropout", "None"]),
			linear_drop_rates=layer_deliminator.join(["0.5", "0.5", "0.0"]),
			linear_activations=layer_deliminator.join(["ReLU", "ReLU", "None"]),
			#
			*args, **kwargs
		)


model_urls = {
	'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


def lenet_parser():
	from . import add_discriminative_options, add_dense_options, add_dropout_options

	model_parser = add_discriminative_options()
	model_parser = add_convpool_options(model_parser)
	model_parser = add_dense_options(model_parser)
	model_parser = add_dropout_options(model_parser)

	return model_parser


def lenet_validator(arguments):
	from . import validate_discriminative_options, validate_dense_arguments, validate_dropout_arguments

	arguments = validate_discriminative_options(arguments)

	arguments = validate_convpool_arguments(arguments)
	number_of_convolution_layers = len(arguments.convolution_filters)

	arguments = validate_dense_arguments(arguments)
	number_of_dense_layers = len(arguments.dense_dimensions)

	number_of_layers = number_of_convolution_layers + number_of_dense_layers
	arguments = validate_dropout_arguments(arguments, number_of_layers)

	return arguments


#
#
#
#
#


def alexnet(pretrained=False, **kwargs):
	r"""AlexNet model architecture from the
	`"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = AlexNetOld(**kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
	return model
