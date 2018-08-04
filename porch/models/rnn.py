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
	             number_of_recurrent_layers,
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
			number_of_recurrent_layers=number_of_recurrent_layers,
			drop_modes=drop_modes,
			drop_rates=drop_rates
		)
		self.layers = layers

		for layer in self.layers:
			for name, parameter in layer.named_parameters():
				self.register_parameter(name, parameter)

	# self.init_weights()
	# self.forward(x=numpy.zeros((1, 10)))

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


class RNN_WordLanguageModel_testbackup(GenericRNN):
	def __init__(self,
	             input_shape,
	             embedding_dimension,
	             recurrent_dimension,
	             output_shape,
	             #
	             *args, **kwargs
	             ):
		super(RNN_WordLanguageModel_testbackup, self).__init__(
			input_shape=input_shape,
			embedding_dimension=embedding_dimension,
			dimensions=layer_deliminator.join(
				["%s" % recurrent_dimension, "%s" % recurrent_dimension, "%s" % output_shape]),
			activations=layer_deliminator.join(["None", "None", "LogSoftmax"]),
			recurrent_modes=layer_deliminator.join([nn.LSTM.__name__, nn.LSTM.__name__, "None"]),
			drop_modes=layer_deliminator.join([porch.modules.Dropout.__name__, "None", "None"]),
			drop_rates=layer_deliminator.join(["0.5", "0.0", "0.0"]),
			#
			*args, **kwargs
		)


'''
@TODO: None+CrossEntropy vs. LogSoftmax+nll_loss (definitely have problem)
@TODO: Tied weights
'''


class RNN_WordLanguageModel_test(GenericRNN):
	def __init__(self,
	             input_shape,
	             embedding_dimension,
	             recurrent_dimension,
	             drop_rate,
	             output_shape,
	             #
	             *args, **kwargs
	             ):
		super(RNN_WordLanguageModel_test, self).__init__(
			input_shape=input_shape,
			embedding_dimension=embedding_dimension,
			dimensions=layer_deliminator.join(["%s" % recurrent_dimension, "%s" % output_shape]),
			activations=layer_deliminator.join(["None", "None"]),
			recurrent_modes=layer_deliminator.join([nn.LSTM.__name__, "None"]),
			number_of_recurrent_layers=layer_deliminator.join(["2", "0"]),
			drop_modes=layer_deliminator.join([porch.modules.Dropout.__name__, porch.modules.Dropout.__name__]),
			drop_rates=layer_deliminator.join(["%s" % drop_rate, "%s" % drop_rate]),
			#
			*args, **kwargs
		)

		'''
		super(RNN_WordLanguageModel_test, self).__init__(
			input_shape=input_shape,
			embedding_dimension=embedding_dimension,
			dimensions=layer_deliminator.join(
				["%s" % recurrent_dimension, "%s" % recurrent_dimension, "%s" % output_shape]),
			activations=layer_deliminator.join(["None", "None", "None"]),
			recurrent_modes=layer_deliminator.join([nn.LSTM.__name__, nn.LSTM.__name__, "None"]),
			drop_modes=layer_deliminator.join([porch.modules.Dropout.__name__, "None", "None"]),
			drop_rates=layer_deliminator.join(["%s" % drop_rate, "%s" % drop_rate, "0.0"]),
			#
			*args, **kwargs
		)
		'''
