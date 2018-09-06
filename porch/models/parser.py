import numpy
import torch
import torch.nn as nn

import porch
from porch import layer_deliminator

__all__ = [
	"parse_feed_forward_layers",
	"parse_recurrent_layers",
	#
	"parse_to_int_sequence",
	"parse_to_float_sequence",
	"parse_activations",
	"parse_pool_modes",
	"parse_drop_modes",
	"parse_recurrent_modes",
]


def parse_to_int_sequence(string_of_ints, default=None):
	sequence_of_ints = [int(temp) for temp in string_of_ints.split(layer_deliminator) if len(temp) > 0]
	if (default is not None) and (len(sequence_of_ints) == 0):
		sequence_of_ints = [default]
	return sequence_of_ints


def parse_to_float_sequence(string_of_float, default_value=None):
	sequence_of_floats = [float(temp) for temp in string_of_float.split(layer_deliminator) if len(temp) > 0]
	if (default_value is not None) and (len(sequence_of_floats) == 0):
		sequence_of_floats = [default_value]
	return sequence_of_floats


def parse_pool_modes(pool_modes_argument):
	pool_modes = []
	for pool_mode in pool_modes_argument.split(layer_deliminator):
		if len(pool_mode) == 0 or pool_mode.lower() == "none":
			pool_modes.append(None)
		else:
			pool_modes.append(getattr(nn, pool_mode))
	if len(pool_modes) == 0:
		pool_modes.append(None)
	return pool_modes


def parse_activations(activations_argument):
	activations = []
	for activation in activations_argument.split(layer_deliminator):
		if len(activation) == 0 or activation.lower() == "none":
			activations.append(None)
		else:
			activations.append(getattr(nn, activation))
	if len(activations) == 0:
		activations.append(None)
	return activations


def parse_drop_modes(drop_modes_argument):
	drop_modes = []
	for drop_mode in drop_modes_argument.split(layer_deliminator):
		if len(drop_mode) == 0 or drop_mode.lower() == "none":
			drop_modes.append(None)
		else:
			drop_modes.append(getattr(porch.modules, drop_mode))
	if len(drop_modes) == 0:
		drop_modes.append(None)
	return drop_modes


def parse_recurrent_modes(recurrent_modes_argument):
	recurrent_modes = []
	for recurrent_mode in recurrent_modes_argument.split(layer_deliminator):
		if len(recurrent_mode) == 0 or recurrent_mode.lower() == "none":
			recurrent_modes.append(None)
		else:
			recurrent_modes.append(getattr(nn, recurrent_mode))
	if len(recurrent_modes) == 0:
		recurrent_modes.append(None)
	return recurrent_modes


#
#
#
#
#

def parse_feed_forward_layers(input_dimension,
                              dimensions,
                              activations,  # ="",
                              drop_modes,  # ="",
                              drop_rates,  # =""
                              ):
	# feature_shape = [int(temp_shape) for temp_shape in input_dimension.split(layer_deliminator)]

	dimensions = parse_to_int_sequence(string_of_ints=dimensions)
	dimensions.insert(0, input_dimension)

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
			#layers.append(drop_modes[x](p=drop_rates[x]))
		layers.append(nn.Linear(dimensions[x], dimensions[x + 1]))
		if activations[x] is not None:
			layers.append(activations[x]())

	return layers


def parse_recurrent_layers(input_dimension,
                           dimensions,
                           activations,  # ="",
                           recurrent_modes,
                           number_of_recurrent_layers,
                           drop_modes,  # ="",
                           drop_rates,  # =""
                           device=torch.device("cpu")
                           ):
	dimensions = parse_to_int_sequence(string_of_ints=dimensions)
	dimensions.insert(0, input_dimension)

	activations = parse_activations(activations_argument=activations)
	assert (len(dimensions) == len(activations) + 1)

	recurrent_modes = parse_recurrent_modes(recurrent_modes_argument=recurrent_modes)
	assert (len(dimensions) == len(recurrent_modes) + 1)

	number_of_recurrent_layers = parse_to_int_sequence(string_of_ints=number_of_recurrent_layers)
	assert (len(dimensions) == len(number_of_recurrent_layers) + 1)

	drop_modes = parse_drop_modes(drop_modes_argument=drop_modes)
	assert (len(dimensions) == len(drop_modes) + 1)

	drop_rates = parse_to_float_sequence(string_of_float=drop_rates, default_value=0)
	assert (len(dimensions) == len(drop_rates) + 1)

	layers = []
	for x in range(len(dimensions) - 1):
		assert 0 <= drop_rates[x] < 1
		if (drop_modes[x] is not None) and (drop_rates[x] > 0):
			print(device)
			layers.append(drop_modes[x](p=numpy.ones(dimensions[x]) * drop_rates[x], device=device))

		if recurrent_modes[x] is not None:
			# assert activations[x] is None
			# assert drop_modes[x + 1] is None
			layers.append(
				recurrent_modes[x](input_size=dimensions[x], hidden_size=dimensions[x + 1],
				                   num_layers=number_of_recurrent_layers[x],
				                   dropout=drop_rates[x + 1]))
		else:
			layers.append(nn.Linear(dimensions[x], dimensions[x + 1]))
			# temp = nn.Linear(dimensions[x], dimensions[x + 1])
			# temp.weight.data.uniform_(-0.1, 0.1)
			# temp.bias.data.zero_()
			# layers.append(temp)
			if activations[x] is not None:
				layers.append(activations[x]())

	return layers


#
#
#
#
#

def validate_dropout_init_arguments(arguments, number_of_layers):
	layer_activation_styles = arguments.layer_activation_styles
	layer_activation_style_tokens = layer_activation_styles.split(layer_deliminator)
	if len(layer_activation_style_tokens) == 1:
		layer_activation_styles = [layer_activation_styles for layer_index in range(number_of_layers)]
	elif len(layer_activation_style_tokens) == number_of_layers:
		layer_activation_styles = layer_activation_style_tokens
	# [float(layer_activation_parameter) for layer_activation_parameter in layer_activation_parameter_tokens]
	assert len(layer_activation_styles) == number_of_layers
	assert (layer_activation_style in {"uniform", "bernoulli", "beta_bernoulli", "reciprocal_beta_bernoulli",
	                                   "reverse_reciprocal_beta_bernoulli", "mixed_beta_bernoulli"} for
	        layer_activation_style in layer_activation_styles)
	arguments.layer_activation_styles = layer_activation_styles

	layer_activation_parameters = arguments.layer_activation_parameters
	layer_activation_parameter_tokens = layer_activation_parameters.split(layer_deliminator)
	if len(layer_activation_parameter_tokens) == 1:
		layer_activation_parameters = [layer_activation_parameters for layer_index in range(number_of_layers)]
	elif len(layer_activation_parameter_tokens) == number_of_layers:
		layer_activation_parameters = layer_activation_parameter_tokens
	assert len(layer_activation_parameters) == number_of_layers

	for layer_index in range(number_of_layers):
		if layer_activation_styles[layer_index] == "uniform":
			layer_activation_parameters[layer_index] = float(layer_activation_parameters[layer_index])
			assert layer_activation_parameters[layer_index] <= 1
			assert layer_activation_parameters[layer_index] > 0
		elif layer_activation_styles[layer_index] == "bernoulli":
			layer_activation_parameters[layer_index] = float(layer_activation_parameters[layer_index])
			assert layer_activation_parameters[layer_index] <= 1
			assert layer_activation_parameters[layer_index] > 0
		elif layer_activation_styles[layer_index] == "beta_bernoulli" \
				or layer_activation_styles[layer_index] == "reciprocal_beta_bernoulli" \
				or layer_activation_styles[layer_index] == "reverse_reciprocal_beta_bernoulli" \
				or layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
			layer_activation_parameter_tokens = layer_activation_parameters[layer_index].split("+")
			assert len(layer_activation_parameter_tokens) == 2, layer_activation_parameter_tokens
			layer_activation_parameters[layer_index] = (float(layer_activation_parameter_tokens[0]),
			                                            float(layer_activation_parameter_tokens[1]))
			assert layer_activation_parameters[layer_index][0] > 0
			assert layer_activation_parameters[layer_index][1] > 0
			if layer_activation_styles[layer_index] == "mixed_beta_bernoulli":
				assert layer_activation_parameters[layer_index][0] < 1
	arguments.layer_activation_parameters = layer_activation_parameters

	return arguments


def validate_dropout_arguments(arguments, number_of_layers):
	# model argument set
	layer_activation_types = arguments.layer_activation_types
	if layer_activation_types is None:
		layer_activation_types = ["BernoulliDropoutLayer"] * number_of_layers
	else:
		layer_activation_type_tokens = layer_activation_types.split(layer_deliminator)
		if len(layer_activation_type_tokens) == 1:
			layer_activation_types = layer_activation_type_tokens * number_of_layers
		else:
			layer_activation_types = layer_activation_type_tokens
		assert len(layer_activation_types) == number_of_layers
	assert layer_activation_types[0] not in {"FastDropoutLayer", "VariationalDropoutTypeBLayer"}
	for layer_activation_type_index in range(len(layer_activation_types)):
		if layer_activation_types[layer_activation_type_index] in {"BernoulliDropoutLayer", "GaussianDropoutLayer",
		                                                           "FastDropoutLayer"}:
			pass
		elif layer_activation_types[layer_activation_type_index] in {"VariationalDropoutLayer",
		                                                             "VariationalDropoutTypeALayer",
		                                                             "VariationalDropoutTypeBLayer"}:
			if Xregularization.kl_divergence_kingma not in arguments.regularizer:
				arguments.regularizer[Xregularization.kl_divergence_kingma] = [1.0, Xpolicy.constant]
			assert Xregularization.kl_divergence_kingma in arguments.regularizer
		elif layer_activation_types[layer_activation_type_index] in {"SparseVariationalDropoutLayer"}:
			if Xregularization.kl_divergence_sparse not in arguments.regularizer:
				arguments.regularizer[Xregularization.kl_divergence_sparse] = [1.0, Xpolicy.constant]
			assert Xregularization.kl_divergence_sparse in arguments.regularizer
		elif layer_activation_types[layer_activation_type_index] in {"AdaptiveDropoutLayer", "DynamicDropoutLayer"}:
			if (Xregularization.rademacher_p_2_q_2 not in arguments.regularizer) and \
					(Xregularization.rademacher_p_inf_q_1 not in arguments.regularizer):
				arguments.regularizer[Xregularization.rademacher] = [1.0, Xpolicy.constant]
			assert (Xregularization.rademacher_p_2_q_2 in arguments.regularizer) or \
			       (Xregularization.rademacher_p_inf_q_1 in arguments.regularizer)
		else:
			logger.error("unrecognized dropout type %s..." % (layer_activation_types[layer_activation_type_index]))
		layer_activation_types[layer_activation_type_index] = getattr(layers, layer_activation_types[
			layer_activation_type_index])
	arguments.layer_activation_types = layer_activation_types

	arguments = validate_dropout_init_arguments(arguments, number_of_layers)
	return arguments


def validate_adaptive_options(arguments):
	# from . import validate_discriminative_options
	# arguments = validate_discriminative_options(arguments)

	# model argument set 1
	from . import parse_parameter_policy
	# arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)
	if arguments.adaptable_learning_rate is None:
		arguments.adaptable_learning_rate = arguments.learning_rate
	else:
		arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)

	assert arguments.adaptable_training_mode in {"train_adaptables_networkwise", "train_adaptables_layerwise",
	                                             "train_adaptables_layerwise_in_turn"}

	# assert (arguments.adaptable_update_interval >= 0)

	return arguments
