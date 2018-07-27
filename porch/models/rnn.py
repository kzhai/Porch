import logging

import torch
import torch.nn as nn

import porch
from porch.argument import layer_deliminator
from porch.models import *

# parse_feed_forward_layers, parse_recurrent_modes
# parse_to_int_sequence, parse_to_float_sequence, parse_activations, parse_drop_modes,

logger = logging.getLogger(__name__)

__all__ = [
	"GenericRNN",
]


class GenericRNN(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self,
	             input_shape,
	             embedding_dimension,
	             #
	             dimensions,
	             activations,  # ="",
	             recurrent_modes,
	             drop_modes,  # ="",
	             drop_rates,  # =""
	             #
	             *args, **kwargs
	             ):
		super(GenericRNN, self).__init__()

		input_shape = int(input_shape)
		embedding_dimension = int(embedding_dimension)

		# feature_shape = [int(temp_shape) for temp_shape in input_shape.split(layer_deliminator)]

		#
		#
		#

		layers = []
		layers.append(nn.Embedding(input_shape, embedding_dimension))
		layers += parse_recurrent_layers(
			input_dimension=embedding_dimension,
			dimensions=dimensions,
			activations=activations,
			recurrent_modes=recurrent_modes,
			drop_modes=drop_modes,
			drop_rates=drop_rates
		)
		self.layers = layers

		for layer in self.layers:
			for name, parameter in layer.named_parameters():
				self.register_parameter(name, parameter)

		# self.init_weights()
		# self.forward(x=numpy.zeros((1, 10)))

		'''
		pre_recurrent_layers = []
		pre_recurrent_layers.append(nn.Embedding(input_shape, embedding_dimension))
		pre_recurrent_layers += parse_feed_forward_layers(
			input_dimension=embedding_dimension,
			dimensions=pre_recurrent_dimensions,
			activations=pre_recurrent_activations,
			drop_modes=pre_recurrent_drop_modes,
			drop_rates=pre_recurrent_drop_rates
		)

		#
		#
		#

		recurrent_modes = parse_recurrent_modes(recurrent_modes)
		recurrent_dimensions = parse_to_int_sequence(string_of_ints=recurrent_dimensions)
		recurrent_dimensions.insert(0, pre_recurrent_layers. pre_recurrent_dimensions[-1])

		recurrent_drop_rates = parse_to_float_sequence(string_of_float=recurrent_drop_rates, default_value=0)
		if len(recurrent_drop_rates) == 1:
			recurrent_drop_rates = recurrent_drop_rates * (len(recurrent_dimensions) - 1)
		assert (len(recurrent_dimensions) == len(recurrent_drop_rates) + 1)

		recurrent_layers = []
		for x in range(len(recurrent_dimensions) - 1):
			recurrent_layers.append(
				recurrent_modes(recurrent_dimensions[x], recurrent_dimensions[x + 1], dropout=recurrent_drop_rates[x]))

		#
		#
		#

		post_recurrent_layers = []
		post_recurrent_layers += parse_feed_forward_layers(
			input_dimension=recurrent_dimensions[-1],
			dimensions=post_recurrent_dimensions,
			activations=post_recurrent_activations,
			drop_modes=post_recurrent_drop_modes,
			drop_rates=post_recurrent_drop_rates
		)

		self.pre_recurrent_layers = pre_recurrent_layers
		self.recurrent_layers = recurrent_layers
		self.post_recurrent_layers = post_recurrent_layers
		'''

	def init_weights(self, init_range=0.1):
		for layer in self.layers:
			if isinstance(layer, nn.Linear):
				layer.bias.data.zero_()
				layer.weight.data.uniform_(-init_range, init_range)
			elif isinstance(layer, nn.Embedding):
				layer.weight.data.uniform_(-init_range, init_range)

	def forward(self, x, *args, **kwargs):
		# This is to recast the data type to long type, as required by embedding layer.
		x = x.type(torch.long)
		# This is to transpose the minibatch size and sequencen length, to accomodate the bptt algorithm.
		x = x.transpose(0, 1)

		hiddens = kwargs.get("hiddens", None)
		if hiddens is None:
			print("Initialize hiddens to all zeros.")
			hiddens = self.init_hiddens(x.shape[1])

		'''
		for hidden in hiddens:
			print("hidden", len(hidden))
			for sub_hidden in hidden:
				print("sub_hidden", sub_hidden.shape)
		'''

		recurrent_layer_index = 0
		for layer in self.layers:
			if isinstance(layer, nn.LSTM) or isinstance(layer, nn.GRU) or isinstance(layer, nn.RNN):
				x, hiddens[recurrent_layer_index] = layer(x, hiddens[recurrent_layer_index])
				x = x.view(x.size(0), x.size(1), -1)
				recurrent_layer_index += 1
			else:
				x = layer(x)

		# This is to transpose the minibatch size and sequencen length back, to accomodate the bptt algorithm.
		x = x.transpose(0, 1)
		# print("final shape:", x.shape, x.size(0) * x.size(1), x.size(2))
		x = x.contiguous().view(x.size(0) * x.size(1), x.size(2))
		# This is to transpose the sequencen length and output category dimension, to accomodate the nll_loss function.
		# x = x.transpose(1, 2)
		return x, hiddens

	def init_hiddens(self, minibatch_size):
		hiddens = []
		for layer in self.layers:
			weight = next(self.parameters())
			if isinstance(layer, nn.LSTM):
				hiddens.append((weight.new_zeros(layer.num_layers, minibatch_size, layer.hidden_size),
				                weight.new_zeros(layer.num_layers, minibatch_size, layer.hidden_size)))
			elif isinstance(layer, nn.GRU) or isinstance(layer, nn.RNN):
				hiddens.append(weight.new_zeros(layer.num_layers, minibatch_size, layer.hidden_size))

		return hiddens


class RNN_WordLanguageModel_test(GenericRNN):
	def __init__(self,
	             input_shape,
	             embedding_dimension,
	             recurrent_dimension,
	             output_shape,
	             #
	             *args, **kwargs
	             ):
		super(RNN_WordLanguageModel_test, self).__init__(
			input_shape=input_shape,
			embedding_dimension=embedding_dimension,
			dimensions=layer_deliminator.join(
				["%s" % recurrent_dimension, "%s" % recurrent_dimension, "%s" % output_shape]),
			activations=layer_deliminator.join(["None", "None", "None"]),
			recurrent_modes=layer_deliminator.join([nn.LSTM.__name__, nn.LSTM.__name__, "None"]),
			drop_modes=layer_deliminator.join([porch.modules.Dropout.__name__, "None", "None"]),
			drop_rates=layer_deliminator.join(["0.5", "0.0", "0.0"]),
			#
			*args, **kwargs
		)
