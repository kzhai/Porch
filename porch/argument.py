# -*- coding: utf-8 -*-

from __future__ import print_function

import collections
import datetime
import logging
import os

import torch
import torch.nn
import torch.nn.functional

import porch

# from lasagne.experiments import debugger
# from .. import layers, nonlinearities, objectives, updates, Xpolicy, Xregularization

logger = logging.getLogger(__name__)

__all__ = [
	"parse_parameter_policy",
	#
	"layer_deliminator",
	"param_deliminator",
	#
	"add_generic_options",
	"validate_generic_options",
	#
	# "add_discriminative_options",
	# "validate_discriminative_options",
	#
	# "add_resume_options",
	# "validate_resume_options",
	#
	"parse_linear_arguments",
	# "add_adaptive_options",
	# "validate_adaptive_options",
	#
	# "add_dynamic_options",
	# "validate_dynamic_options",
	#
	# "discriminative_adaptive_dynamic_resume_parser",
	# "discriminative_adaptive_dynamic_resume_validator",
	#
	# "discriminative_adaptive_resume_parser",
	# "discriminative_adaptive_resume_validator",
]

layer_deliminator = "*"
# layer_deliminator = ","
param_deliminator = ","
# param_deliminator = "*"
specs_deliminator = ":"
hierarchy_deliminator = "."


def parse_parameter_policy(policy_string):
	policy_tokens = policy_string.split(param_deliminator)

	policy_tokens[0] = float(policy_tokens[0])
	# assert policy_tokens[0] >= 0
	if len(policy_tokens) == 1:
		policy_tokens.append(Xpolicy.constant)
		return policy_tokens

	policy_tokens[1] = getattr(Xpolicy, policy_tokens[1])
	if policy_tokens[1] is Xpolicy.constant:
		assert len(policy_tokens) == 2
		return policy_tokens

	if policy_tokens[1] is Xpolicy.piecewise_constant:
		assert len(policy_tokens) == 4

		policy_tokens[2] = [float(boundary_token) for boundary_token in policy_tokens[2].split("-")]
		previous_boundary = 0
		for next_boundary in policy_tokens[2]:
			assert next_boundary > previous_boundary
			previous_boundary = next_boundary
		policy_tokens[3] = [float(value_token) for value_token in policy_tokens[3].split("-")]
		assert len(policy_tokens[2]) == len(policy_tokens[3])
		return policy_tokens

	assert policy_tokens[1] is Xpolicy.inverse_time_decay \
	       or policy_tokens[1] is Xpolicy.natural_exp_decay \
	       or policy_tokens[1] is Xpolicy.exponential_decay

	for x in range(2, 4):
		policy_tokens[x] = float(policy_tokens[x])
		assert policy_tokens[x] > 0

	if len(policy_tokens) == 4:
		policy_tokens.append(0)
	elif len(policy_tokens) == 5:
		policy_tokens[4] = float(policy_tokens[4])
		assert policy_tokens[4] > 0
	else:
		logger.error("unrecognized parameter decay policy %s..." % (policy_tokens))

	return policy_tokens


def add_generic_options(model_parser):
	# model_parser = argparse.ArgumentParser(description="generic neural network arguments", add_help=True)

	# generic argument set 1
	model_parser.add_argument("--input_directory", dest="input_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                          help="output directory [None]")
	model_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                          help="model directory [None, resume mode if specified]")

	# generic argument set 2
	model_parser.add_argument("--loss", dest="loss", action='append', default=[],
	                          help="loss function [None] defined in porch.loss")
	# model_parser.add_argument("--loss_kwargs", dest='loss_kwargs', action='store', default=None,
	# help="loss kwargs specified for loss function [None], consult the api for more info")
	model_parser.add_argument("--regularizer", dest="regularizer", action='append', default=[],
	                          help="regularizer function [None] defined in porch.regularizer")
	model_parser.add_argument("--information", dest='information', action='append', default=[],
	                          help="information function [None] defined in porch.loss")

	# model_parser.add_argument("--info_kwargs", dest='info_kwargs', action='store', default=None,
	# help="info kwargs specified for info function [None], consult the api for more info")

	# generic argument set 3
	model_parser.add_argument("--minibatch_size", dest="minibatch_size", type=int, action='store', default=-1,
	                          help="mini-batch size [-1]")
	model_parser.add_argument("--number_of_epochs", dest="number_of_epochs", type=int, action='store', default=-1,
	                          help="number of epochs [-1]")
	# model_parser.add_argument("--snapshot_interval", dest="snapshot_interval", type=int, action='store', default=0,
	# help="snapshot interval in number of epochs [0 - no snapshot]")

	# generic argument set 4
	model_parser.add_argument("--model", dest="model", action='store', default="porch.models.mlp.GenericMLP",
	                          help="neural network model [porch.mnist.MLP]")
	model_parser.add_argument("--model_kwargs", dest="model_kwargs", action='store', default="",
	                          help="model kwargs specified for neural network model [None]")

	model_parser.add_argument("--optimizer", dest="optimizer", action='store', default="SGD",
	                          help="optimizer algorithm [SGD] defined in torch.optim or porch.optim")
	model_parser.add_argument("--optimizer_kwargs", dest='optimizer_kwargs', action='store',
	                          default="lr{}1e-3{}momentum{}0.9".format(specs_deliminator, param_deliminator,
	                                                                   specs_deliminator),
	                          help="optimizer kwargs specified for optimization algorithm [lr:1e-3,momentum:0.9], consult the api for more info")

	# generic argument set 5
	model_parser.add_argument("--data", dest='data', action='append', default=[],
	                          help="data preprocess function [None] defined in porch.data")
	model_parser.add_argument("--train_kwargs", dest='train_kwargs', action='store',
	                          default="", help="kwargs specified for model training")
	model_parser.add_argument("--test_kwargs", dest='test_kwargs', action='store',
	                          default="", help="kwargs specified for model testing")

	# model_parser.add_argument("--learning_rate", dest="learning_rate", type=float, action='store', default=1e-2,
	# help="learning rate [1e-2]")
	# model_parser.add_argument("--momentum", dest="momentum", type=float, action='store', default=0.9,
	# help="nestrov momentum [0.9]")
	'''
	model_parser.add_argument("--learning_rate", dest="learning_rate", action='store', default="1e-2",
	                          help="learning policy [1e-2,constant]")
	model_parser.add_argument("--max_norm_constraint", dest="max_norm_constraint", type=float, action='store',
	                          default=0, help="max norm constraint [0 - None]")
	'''

	# model_parser.add_argument('--device', dest="device", action='store', default="cuda", help='device [cuda]')
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')

	model_parser.add_argument("--snapshot", dest='snapshot', action='append', default=[],
	                          help="snapshot function [None]")
	model_parser.add_argument("--debug", dest='debug', action='append', default=[], help="debug function [None]")

	'''
	subparsers = generic_parser.add_subparsers(dest="subparser_name")
	resume_parser = subparsers.add_parser("resume", parents=[generic_parser], help='resume training')
	resume_parser = add_resume_options(resume_parser)

	start_parser = subparsers.add_parser('start', parents=[generic_parser], help='start training')
	'''

	return model_parser


def validate_generic_options(arguments):
	# use_cuda = arguments.device.lower() == "cuda" and torch.cuda.is_available()
	arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	arguments.device = torch.device(arguments.device)

	# generic argument set snapshots
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond

	snapshots = {}
	for snapshot_interval_mapping in arguments.snapshot:
		fields = snapshot_interval_mapping.split(specs_deliminator)
		snapshot_function = getattr(porch.debug, fields[0])
		if len(fields) == 1:
			interval = 1
		elif len(fields) == 2:
			interval = int(fields[1])
		else:
			logger.error("unrecognized snapshot function setting %s..." % (snapshot_interval_mapping))
		snapshots[snapshot_function] = interval
	arguments.snapshot = snapshots

	debugs = {}
	for debug_interval_mapping in arguments.debug:
		fields = debug_interval_mapping.split(specs_deliminator)
		debug_function = getattr(porch.debug, fields[0])
		if len(fields) == 1:
			interval = 1
		elif len(fields) == 2:
			interval = int(fields[1])
		else:
			logger.error("unrecognized debug function setting %s..." % (debug_interval_mapping))
		debugs[debug_function] = interval
	arguments.debug = debugs

	# generic argument set 3
	assert arguments.minibatch_size > 0
	assert arguments.number_of_epochs > 0
	# assert arguments.snapshot_interval >= 0

	# generic argument set 2
	losses = {}
	for loss_weight_mapping in arguments.loss:
		fields = loss_weight_mapping.split(specs_deliminator)
		loss_function = getattr(porch.loss, fields[0])
		if len(fields) == 1:
			losses[loss_function] = 1.0
		elif len(fields) == 2:
			losses[loss_function] = float(fields[1])
		else:
			logger.error("unrecognized loss function setting %s..." % (loss_weight_mapping))
	arguments.loss = losses
	arguments.loss_kwargs = {}

	regularizers = {}
	for regularizer_weight_mapping in arguments.regularizer:
		fields = regularizer_weight_mapping.split(specs_deliminator)
		regularizer_function = getattr(porch.regularizer, fields[0])
		if len(fields) == 1:
			regularizers[regularizer_function] = 1.0
		elif len(fields) == 2:
			regularizers[regularizer_function] = float(fields[1])
		else:
			logger.error("unrecognized regularizer function setting %s..." % (regularizer_weight_mapping))
	arguments.regularizer = regularizers
	arguments.regularizer_kwargs = {}

	informations = {}
	for information_weight_mapping in arguments.information:
		fields = information_weight_mapping.split(specs_deliminator)
		information_function = getattr(porch.loss, fields[0])
		if len(fields) == 1:
			informations[information_function] = 1.0
		elif len(fields) == 2:
			informations[information_function] = float(fields[1])
		else:
			logger.error("unrecognized information function setting %s..." % (information_weight_mapping))
	arguments.information = informations
	arguments.information_kwargs = {}

	# generic argument set 4
	arguments.model = eval(arguments.model)
	model_kwargs = {}
	model_kwargs_tokens = arguments.model_kwargs.split(param_deliminator)
	for model_kwargs_token in model_kwargs_tokens:
		if len(model_kwargs_token) == 0:
			continue
		key_value_pair = model_kwargs_token.split(specs_deliminator)
		assert len(key_value_pair) == 2
		model_kwargs[key_value_pair[0]] = key_value_pair[1]
	arguments.model_kwargs = model_kwargs

	arguments.optimizer = getattr(torch.optim, arguments.optimizer)
	# arguments.optimizer = getattr(porch.optim, arguments.optimizer)
	optimizer_kwargs = {}
	optimizer_kwargs_tokens = arguments.optimizer_kwargs.split(param_deliminator)
	for optimizer_kwargs_token in optimizer_kwargs_tokens:
		key_value_pair = optimizer_kwargs_token.split(specs_deliminator)
		assert len(key_value_pair) == 2
		optimizer_kwargs[key_value_pair[0]] = float(key_value_pair[1])
	arguments.optimizer_kwargs = optimizer_kwargs

	# generic argument set 5
	data = collections.OrderedDict()
	for data_function_params_mapping in arguments.data:
		fields = data_function_params_mapping.split(param_deliminator)
		data_function = getattr(porch.data, fields[0])
		data_function_params = {}
		for param_value in fields[1:]:
			param_value_fields = param_value.split(specs_deliminator)
			assert (len(param_value_fields) == 2)
			data_function_params[param_value_fields[0]] = param_value_fields[1]
		data[data_function] = data_function_params
	arguments.data = data

	train_kwargs = {}
	train_kwargs_tokens = arguments.train_kwargs.split(param_deliminator)
	for train_kwargs_token in train_kwargs_tokens:
		if len(train_kwargs_token) == 0:
			continue
		key_value_pair = train_kwargs_token.split(specs_deliminator)
		assert len(key_value_pair) == 2
		train_kwargs[key_value_pair[0]] = key_value_pair[1]
	arguments.train_kwargs = train_kwargs

	test_kwargs = {}
	test_kwargs_tokens = arguments.test_kwargs.split(param_deliminator)
	for test_kwargs_token in test_kwargs_tokens:
		if len(test_kwargs_token) == 0:
			continue
		key_value_pair = test_kwargs_token.split(specs_deliminator)
		assert len(key_value_pair) == 2
		test_kwargs[key_value_pair[0]] = key_value_pair[1]
	arguments.test_kwargs = test_kwargs

	# generic argument set 1
	assert os.path.exists(arguments.input_directory)

	output_directory = arguments.output_directory
	assert (output_directory is not None)
	if not os.path.exists(output_directory):
		os.mkdir(os.path.abspath(output_directory))
	# adjusting output directory
	now = datetime.datetime.now()
	suffix = now.strftime("%y%m%d-%H%M%S-%f") + ""
	# suffix += "-%s" % ("mlp")
	output_directory = os.path.join(output_directory, suffix)
	assert not os.path.exists(output_directory)
	# os.mkdir(os.path.abspath(output_directory))
	arguments.output_directory = output_directory

	assert (arguments.model_directory is None) or os.path.exists(arguments.model_directory)

	return arguments


'''
def add_discriminative_options(model_parser):
	model_parser = add_generic_options(model_parser)

	# model argument set
	model_parser.add_argument("--validation_data", dest="validation_data", type=int, action='store', default=0,
	                          help="validation data [0 - no validation data used], -1 - load validate.(feature|label).npy for validation]")
	model_parser.add_argument("--validation_interval", dest="validation_interval", type=int, action='store',
	                          default=1000,
	                          help="validation interval in number of mini-batches [1000]")

	return model_parser


def validate_discriminative_options(arguments):
	arguments = validate_generic_options(arguments)

	# model argument set
	assert (arguments.validation_data >= -1)
	assert (arguments.validation_interval > 0)

	return arguments



def add_resume_options(model_parser):
	# from . import add_discriminative_options

	# model_parser = add_discriminative_options()
	model_parser.add_argument("--model_file", dest="model_file", action='store', default=None,
	                          help="model file to resume from [None]")

	return model_parser


def validate_resume_options(arguments):
	# from . import validate_discriminative_options

	# arguments = validate_discriminative_options(arguments)

	# assert os.path.exists(arguments.model_directory)
	assert os.path.exists(arguments.model_file)
	arguments.model_directory = os.path.dirname(arguments.model_file)
	assert os.path.exists(os.path.join(arguments.model_directory, "train.index.npy"))
	assert os.path.exists(os.path.join(arguments.model_directory, "validate.index.npy"))

	return arguments
'''


def add_adaptive_options(model_parser):
	# from . import add_discriminative_options
	# model_parser = add_discriminative_options()
	# model_parser.description = "adaptive multi-layer perceptron argument"

	# model argument set 1
	model_parser.add_argument("--adaptable_learning_rate", dest="adaptable_learning_rate", action='store',
	                          default=None, help="adaptable learning rate [None - learning_rate]")
	model_parser.add_argument("--adaptable_training_mode", dest="adaptable_training_mode",
	                          action='store', default="train_adaptables_networkwise",
	                          help="train adaptables mode [train_adaptables_networkwise]")
	# model_parser.add_argument("--adaptable_update_interval", dest="adaptable_update_interval", type=int,
	# action='store', default=1, help="adatable update interval [1]")

	return model_parser


def parse_linear_arguments(argument):
	linear_dimensions = [int(temp) for temp in argument.split(layer_deliminator)]
	return linear_dimensions


def validate_dense_arguments(arguments):
	# model argument set 1
	assert arguments.dense_dimensions is not None
	dense_dimensions = arguments.dense_dimensions.split(layer_deliminator)
	arguments.dense_dimensions = [int(dimensionality) for dimensionality in dense_dimensions]

	assert arguments.dense_nonlinearities is not None
	dense_nonlinearities = arguments.dense_nonlinearities.split(layer_deliminator)
	arguments.dense_nonlinearities = [getattr(nonlinearities, dense_nonlinearity) for dense_nonlinearity in
	                                  dense_nonlinearities]

	assert len(arguments.dense_nonlinearities) == len(arguments.dense_dimensions)

	return arguments, len(arguments.dense_dimensions)


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


def add_dynamic_options(model_parser):
	# from . import add_adaptive_options
	# model_parser = add_adaptive_options()
	# model_parser.description = "dynamic multi-layer perceptron argument"

	# model argument set 1
	model_parser.add_argument("--prune_thresholds", dest="prune_thresholds", action='store', default="-0.001",
	                          # default="-1e3,piecewise_constant,100,1e-3",
	                          # help="prune thresholds [-1e3,piecewise_constant,100,1e-3]"
	                          help="prune thresholds [None]"
	                          )
	model_parser.add_argument("--split_thresholds", dest="split_thresholds", action='store', default="1.001",
	                          # default="1e3,piecewise_constant,100,0.999",
	                          # help="split thresholds [1e3,piecewise_constant,100,0.999]"
	                          help="split thresholds [None]"
	                          )

	# model_parser.add_argument("--prune_split_interval", dest="prune_split_interval", action='store', default=1,
	# type=int, help="prune split interval [1]")
	model_parser.add_argument("--prune_split_interval", dest="prune_split_interval", action='store', default="10",
	                          help="prune split interval [10]")

	return model_parser


def validate_dynamic_options(arguments):
	# from . import validate_adaptive_options
	# arguments = validate_adaptive_options(arguments)

	# model argument set 1
	# arguments.adaptable_learning_rate = parse_parameter_policy(arguments.adaptable_learning_rate)
	from . import parse_parameter_policy
	# number_of_layers = sum(layer_activation_type is layers.DynamicDropoutLayer for layer_activation_type in arguments.layer_activation_types)

	if arguments.prune_thresholds is not None:
		prune_thresholds = arguments.prune_thresholds
		prune_thresholds_tokens = prune_thresholds.split(layer_deliminator)
		prune_thresholds = [parse_parameter_policy(prune_thresholds_token) for prune_thresholds_token in
		                    prune_thresholds_tokens]
		# if len(prune_thresholds) == 1:
		# prune_thresholds *= number_of_layers
		# assert len(prune_thresholds) == number_of_layers
		arguments.prune_thresholds = prune_thresholds

	if arguments.split_thresholds is not None:
		split_thresholds = arguments.split_thresholds
		split_thresholds_tokens = split_thresholds.split(layer_deliminator)
		split_thresholds = [parse_parameter_policy(split_thresholds_token) for split_thresholds_token in
		                    split_thresholds_tokens]
		# if len(split_thresholds) == 1:
		# split_thresholds *= number_of_layers
		# assert len(split_thresholds) == number_of_layers
		arguments.split_thresholds = split_thresholds

	prune_split_interval = arguments.prune_split_interval
	prune_split_interval_tokens = [int(prune_split_interval_token) for prune_split_interval_token in
	                               prune_split_interval.split(param_deliminator)]
	if len(prune_split_interval_tokens) == 1:
		prune_split_interval_tokens.insert(0, 0)
	assert prune_split_interval_tokens[1] >= 0
	arguments.prune_split_interval = prune_split_interval_tokens

	return arguments


def discriminative_adaptive_dynamic_resume_parser():
	from . import add_dynamic_options

	model_parser = add_dynamic_options()
	model_parser.add_argument("--model_file", dest="model_file", action='store', default=None,
	                          help="model file to resume from [None]")

	return model_parser


def discriminative_adaptive_dynamic_resume_validator(arguments):
	from . import validate_dynamic_options

	arguments = validate_dynamic_options(arguments)

	# assert os.path.exists(arguments.model_directory)
	assert os.path.exists(arguments.model_file)
	arguments.model_directory = os.path.dirname(arguments.model_file)
	assert os.path.exists(os.path.join(arguments.model_directory, "train.index.npy"))
	assert os.path.exists(os.path.join(arguments.model_directory, "validate.index.npy"))

	return arguments


def discriminative_adaptive_resume_parser():
	from . import add_adaptive_options

	model_parser = add_adaptive_options()
	model_parser.add_argument("--model_file", dest="model_file", action='store', default=None,
	                          help="model file to resume from [None]")

	return model_parser


def discriminative_adaptive_resume_validator(arguments):
	from . import validate_adaptive_options

	arguments = validate_adaptive_options(arguments)

	# assert os.path.exists(arguments.model_directory)
	assert os.path.exists(arguments.model_file)
	arguments.model_directory = os.path.dirname(arguments.model_file)
	assert os.path.exists(os.path.join(arguments.model_directory, "train.index.npy"))
	assert os.path.exists(os.path.join(arguments.model_directory, "validate.index.npy"))

	return arguments
