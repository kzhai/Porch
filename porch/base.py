import argparse
import logging
import os
import timeit

import numpy
import torch
import torch.nn.functional

import porch
import porch.data

logger = logging.getLogger(__name__)

__all__ = [
	# "train",
	# "test",
	# "DiscriminativeModel",
	# "GenerativeModel",
	# "train_model",
	"detach",
	"main",
]


def load_datasets_to_start(input_directory, output_directory, number_of_validate_data=0,
                           function_parameter_mapping={porch.data.loadSequence: {}}):
	test_dataset = porch.data.load_datasets(input_directory, data_mode="test",
	                                        function_parameter_mapping=function_parameter_mapping)

	if number_of_validate_data >= 0:
		total_dataset = porch.data.load_datasets(input_directory, data_mode="train",
		                                         function_parameter_mapping=function_parameter_mapping)
		# train_dataset_temp = load_feature_and_labels(input_directory, dataset="train")
		total_dataset_x, total_dataset_y = total_dataset

		assert number_of_validate_data >= 0 and number_of_validate_data < len(total_dataset_y)
		indices = numpy.random.permutation(len(total_dataset_y))
		train_indices = indices[number_of_validate_data:]
		validate_indices = indices[:number_of_validate_data]

		numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices)
		numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices)

		train_set_x = total_dataset_x[train_indices]
		train_set_y = total_dataset_y[train_indices]
		train_dataset = (train_set_x, train_set_y)
		logger.info("Successfully load data %s with %d to train..." % (input_directory, len(train_set_x)))

		validate_set_x = total_dataset_x[validate_indices]
		validate_set_y = total_dataset_y[validate_indices]
		validate_dataset = (validate_set_x, validate_set_y)
		logger.info("Successfully load data %s with %d to validate..." % (input_directory, len(validate_set_x)))
	else:
		train_dataset = porch.data.load_datasets(input_directory, data_mode="train",
		                                         function_parameter_mapping=function_parameter_mapping)
		validate_dataset = porch.data.load_datasets(input_directory, data_mode="validate",
		                                            function_parameter_mapping=function_parameter_mapping)

	return train_dataset, validate_dataset, test_dataset


def load_datasets_to_resume(input_directory, model_directory, output_directory,
                            function_parameter_mapping={porch.data.loadSequence: {}}):
	test_dataset = porch.data.load_datasets(input_directory, data_mode="test",
	                                        function_parameter_mapping=function_parameter_mapping)

	train_dataset_temp = porch.data.load_datasets(input_directory, data_mode="train",
	                                              function_parameter_mapping=function_parameter_mapping)
	total_data_x, total_data_y = train_dataset_temp

	train_indices = numpy.load(os.path.join(model_directory, "train.index.npy"))
	validate_indices = numpy.load(os.path.join(model_directory, "validate.index.npy"))

	numpy.save(os.path.join(output_directory, "train.index.npy"), train_indices)
	numpy.save(os.path.join(output_directory, "validate.index.npy"), validate_indices)

	train_set_x = total_data_x[train_indices]
	train_set_y = total_data_y[train_indices]
	train_dataset = (train_set_x, train_set_y)
	logger.info("successfully load data %s with %d to train..." % (input_directory, len(train_set_x)))

	validate_set_x = total_data_x[validate_indices]
	validate_set_y = total_data_y[validate_indices]
	validate_dataset = (validate_set_x, validate_set_y)
	logger.info("successfully load data %s with %d to validate..." % (input_directory, len(validate_set_x)))

	return train_dataset, validate_dataset, test_dataset


#
#
#
#
#

def detach(params):
	"""Wraps hidden states in new Tensors, to detach them from their history."""
	'''
	for index, param in enumerate(params):
		if isinstance(param, torch.Tensor):
			params[index] = param.detach()
		elif isinstance(param, tuple):
			params[index] = tuple(sub_hidden.detach for sub_hidden in param)
	return params
	'''
	if isinstance(params, torch.Tensor):
		return params.detach()
	elif isinstance(params, list):
		return [detach(param) for param in params]
	elif isinstance(params, tuple):
		return tuple(detach(param) for param in params)
	return None


def train_epoch(device,
                network,
                optimizer,
                dataset,
                #
                loss_functions,
                regularizer_functions={},
                information_functions={},
                #
                loss_function_kwargs={},
                regularizer_function_kwargs={},
                information_function_kwargs={},
                #
                minibatch_size=64,
                #
                *args,
                **kwargs
                ):
	network.train()

	dataset_x, dataset_y = dataset
	number_of_data = dataset_x.shape[0]
	# data_indices = numpy.random.permutation(number_of_data)
	data_indices = numpy.arange(number_of_data)

	epoch_total_loss = 0.
	epoch_total_reg = 0.
	epoch_total_infos = {}
	for information_function in information_functions:
		epoch_total_infos[information_function] = 0.

	minibatch_cache = {}
	# This is to accomodate recurrent neural networks.
	minibatch_cache["hiddens"] = None

	epoch_time = timeit.default_timer()
	progress_report_marker, progress_report_steper = 10, 10
	minibatch_start_index = 0
	while minibatch_start_index < number_of_data:
		# automatically handles the left-over data
		minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size]
		minibatch_x = dataset_x[minibatch_indices, :]
		minibatch_y = dataset_y[minibatch_indices]
		data_minibatch = (minibatch_x, minibatch_y)

		# This is to accomodate recurrent neural networks.
		kwargs["hiddens"] = minibatch_cache.get("hiddens", None)

		train_minibatch_output = train_iteration(device=device,
		                                         network=network,
		                                         optimizer=optimizer,
		                                         dataset=data_minibatch,
		                                         #
		                                         loss_functions=loss_functions,
		                                         regularizer_functions=regularizer_functions,
		                                         information_functions=information_functions,
		                                         #
		                                         loss_function_kwargs=loss_function_kwargs,
		                                         regularizer_function_kwargs=regularizer_function_kwargs,
		                                         information_function_kwargs=information_function_kwargs,
		                                         #
		                                         *args,
		                                         **kwargs
		                                         #**{"hiddens": minibatch_cache.get("hiddens", None)}
		                                         )

		adaptable_optimizer = kwargs.get("adaptable_optimizer", None)
		if adaptable_optimizer is not None:
			train_iteration_adaptive(device=device,
			                         network=network,
			                         optimizer=adaptable_optimizer,
			                         dataset=data_minibatch,
			                         #
			                         loss_functions=loss_functions,
			                         regularizer_functions=regularizer_functions,
			                         information_functions=information_functions,
			                         #
			                         loss_function_kwargs=loss_function_kwargs,
			                         regularizer_function_kwargs=regularizer_function_kwargs,
			                         information_function_kwargs=information_function_kwargs,
			                         #
			                         *args,
			                         **kwargs
			                         #**{"hiddens": minibatch_cache.get("hiddens", None)}
			                         )

		# minibatch_time, minibatch_total_loss, minibatch_total_reg, minibatch_total_infos = train_minibatch_output
		minibatch_time, minibatch_total_loss, minibatch_total_reg, minibatch_total_infos, minibatch_cache = train_minibatch_output

		# print(minibatch_start_index // minibatch_size, "minibatch_average_obj", minibatch_total_loss / len(minibatch_x))

		epoch_total_loss += minibatch_total_loss
		epoch_total_reg += minibatch_total_reg
		for information_function in information_functions:
			epoch_total_infos[information_function] += minibatch_total_infos[information_function]

		minibatch_start_index += minibatch_size
		if 100. * minibatch_start_index / number_of_data >= progress_report_marker:
			print('| {:3.2f}% epoch | {:.2f} ms/iteration | {:.2f} loss | {:.2f} regularizer |'.format(
				minibatch_start_index * 100 / number_of_data,
				(timeit.default_timer() - epoch_time) * 1000 / (minibatch_start_index / minibatch_size),
				epoch_total_loss / minibatch_start_index, epoch_total_reg / minibatch_start_index))
			progress_report_marker += progress_report_steper

	epoch_time = timeit.default_timer() - epoch_time
	epoch_average_loss = epoch_total_loss / len(dataset_x)
	epoch_average_reg = epoch_total_reg / len(dataset_x)
	epoch_average_infos = {}
	for information_function in information_functions:
		epoch_average_infos[information_function] = epoch_total_infos[information_function] / len(dataset_x)

	return epoch_time, epoch_average_loss, epoch_average_reg, epoch_average_infos


def train_iteration(device,
                    network,
                    optimizer,
                    dataset,
                    #
                    loss_functions,
                    regularizer_functions={},
                    information_functions={},
                    #
                    loss_function_kwargs={},
                    regularizer_function_kwargs={},
                    information_function_kwargs={},
                    #
                    *args,
                    **kwargs
                    ):
	network.train()

	minibatch_x, minibatch_y = dataset
	minibatch_y = minibatch_y.view(-1)
	minibatch_x, minibatch_y = minibatch_x.to(device), minibatch_y.to(device)

	minibatch_time = timeit.default_timer()

	'''
	hiddens = getattr(kwargs, "hiddens", None)
	network_kwargs = {}
	'''

	minibatch_cache = {}
	optimizer.zero_grad()
	output = network(minibatch_x, **kwargs)
	if isinstance(output, tuple):
		output, hiddens = output
		minibatch_cache["hiddens"] = detach(hiddens)
	# output = network(minibatch_x, seed_hiddens=hiddens, return_hiddens=True)

	minibatch_total_loss = 0
	minibatch_total_reg = 0
	minibatch_total_infos = {}
	for information_function in information_functions:
		minibatch_total_infos[information_function] = 0

	minibatch_average_loss = 0
	for loss_function in loss_functions:
		minibatch_loss = loss_function(output, minibatch_y, **loss_function_kwargs)
		if ("size_average" not in loss_function_kwargs) or loss_function_kwargs["size_average"]:
			minibatch_total_loss += minibatch_loss.item() * len(minibatch_x)
			minibatch_average_loss += minibatch_loss
		else:
			minibatch_total_loss += minibatch_loss.item()
			minibatch_average_loss += minibatch_loss / len(minibatch_x)

	minibatch_average_reg = 0
	for regularizer_function, regularizer_weight in regularizer_functions.items():
		minibatch_reg = regularizer_function(network, input=minibatch_x, output=output,
		                                     **regularizer_function_kwargs) * regularizer_weight

		if ("size_average" not in regularizer_function_kwargs) or regularizer_function_kwargs[
			"size_average"]:
			minibatch_total_reg += minibatch_reg.item() * len(minibatch_x)
			minibatch_average_reg += minibatch_reg
		else:
			minibatch_total_reg += minibatch_reg.item()
			minibatch_average_reg += minibatch_reg / len(minibatch_x)

	for information_function in information_functions:
		minibatch_info = information_function(output, minibatch_y, **information_function_kwargs)
		if ("size_average" not in information_function_kwargs) or information_function_kwargs["size_average"]:
			minibatch_total_infos[information_function] += minibatch_info.item() * len(minibatch_x)
		else:
			minibatch_total_infos[information_function] += minibatch_info.item()

	minibatch_average_obj = minibatch_average_loss + minibatch_average_reg
	minibatch_average_obj.backward()

	# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
	clip_grad_norm = kwargs.get("clip_grad_norm", 0)
	clip_grad_norm = clip_grad_norm if type(clip_grad_norm) == float else float(clip_grad_norm)
	if clip_grad_norm > 0:
		# torch.nn.utils.clip_grad_norm_(list(param_group['params'] for param_group in optimizer.param_groups), clip_grad_norm)
		torch.nn.utils.clip_grad_norm_(network.parameters(), clip_grad_norm)

	optimizer.step()

	minibatch_time = timeit.default_timer() - minibatch_time

	return minibatch_time, minibatch_total_loss, minibatch_total_reg, minibatch_total_infos, minibatch_cache


def train_iteration_adaptive(device,
                             network,
                             optimizer,
                             dataset,
                             #
                             loss_functions,
                             regularizer_functions={},
                             information_functions={},
                             #
                             loss_function_kwargs={},
                             regularizer_function_kwargs={},
                             information_function_kwargs={},
                             #
                             *args,
                             **kwargs
                             ):
	# We want to turn to test mode when updating the adaptive variables.
	network.train(False)

	minibatch_x, minibatch_y = dataset
	minibatch_y = minibatch_y.view(-1)
	minibatch_x, minibatch_y = minibatch_x.to(device), minibatch_y.to(device)

	minibatch_time = timeit.default_timer()

	'''
	hiddens = getattr(kwargs, "hiddens", None)
	network_kwargs = {}
	'''

	minibatch_cache = {}
	optimizer.zero_grad()
	output = network(minibatch_x, **kwargs)
	if isinstance(output, tuple):
		output, hiddens = output
		minibatch_cache["hiddens"] = detach(hiddens)
	# output = network(minibatch_x, seed_hiddens=hiddens, return_hiddens=True)

	minibatch_total_loss = 0
	minibatch_total_reg = 0
	minibatch_total_infos = {}
	for information_function in information_functions:
		minibatch_total_infos[information_function] = 0

	minibatch_average_loss = 0
	for loss_function in loss_functions:
		minibatch_loss = loss_function(output, minibatch_y, **loss_function_kwargs)
		if ("size_average" not in loss_function_kwargs) or loss_function_kwargs["size_average"]:
			minibatch_total_loss += minibatch_loss.item() * len(minibatch_x)
			minibatch_average_loss += minibatch_loss
		else:
			minibatch_total_loss += minibatch_loss.item()
			minibatch_average_loss += minibatch_loss / len(minibatch_x)

	minibatch_average_reg = 0
	for regularizer_function, regularizer_weight in regularizer_functions.items():
		minibatch_reg = regularizer_function(network, input=minibatch_x, output=output,
		                                     **regularizer_function_kwargs) * regularizer_weight

		if ("size_average" not in regularizer_function_kwargs) or regularizer_function_kwargs[
			"size_average"]:
			minibatch_total_reg += minibatch_reg.item() * len(minibatch_x)
			minibatch_average_reg += minibatch_reg
		else:
			minibatch_total_reg += minibatch_reg.item()
			minibatch_average_reg += minibatch_reg / len(minibatch_x)

	for information_function in information_functions:
		minibatch_info = information_function(output, minibatch_y, **information_function_kwargs)
		if ("size_average" not in information_function_kwargs) or information_function_kwargs["size_average"]:
			minibatch_total_infos[information_function] += minibatch_info.item() * len(minibatch_x)
		else:
			minibatch_total_infos[information_function] += minibatch_info.item()

	minibatch_average_obj = minibatch_average_loss + minibatch_average_reg
	minibatch_average_obj.backward()

	# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
	clip_grad_norm = kwargs.get("clip_grad_norm", 0)
	clip_grad_norm = clip_grad_norm if type(clip_grad_norm) == float else float(clip_grad_norm)
	if clip_grad_norm > 0:
		# torch.nn.utils.clip_grad_norm_(list(param_group['params'] for param_group in optimizer.param_groups), clip_grad_norm)
		torch.nn.utils.clip_grad_norm_(network.parameters(), clip_grad_norm)

	optimizer.step()

	minibatch_time = timeit.default_timer() - minibatch_time

	return minibatch_time, minibatch_total_loss, minibatch_total_reg, minibatch_total_infos, minibatch_cache


def test_epoch(device,
               network,
               dataset,
               #
               loss_functions,
               regularizer_functions={},
               information_functions={},
               #
               loss_function_kwargs={},
               regularizer_function_kwargs={},
               information_function_kwargs={},
               #
               minibatch_size=64,
               # generative_model=False
               *args,
               **kwargs
               ):
	network.eval()

	dataset_x, dataset_y = dataset

	number_of_data = dataset_x.shape[0]
	# data_indices = numpy.random.permutation(number_of_data)
	data_indices = numpy.arange(number_of_data)

	epoch_total_loss = 0.
	epoch_total_reg = 0.
	epoch_total_infos = {}
	for information_function in information_functions:
		epoch_total_infos[information_function] = 0

	minibatch_cache = {}
	minibatch_cache["hiddens"] = None

	epoch_time = timeit.default_timer()
	with torch.no_grad():
		# progress_marker = 10
		minibatch_start_index = 0
		while minibatch_start_index < number_of_data:
			# automatically handles the left-over data
			minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size]

			minibatch_x = dataset_x[minibatch_indices, :]
			minibatch_y = dataset_y[minibatch_indices]
			data_minibatch = (minibatch_x, minibatch_y)

			kwargs["hiddens"] = minibatch_cache.get("hiddens", None)

			test_minibatch_output = test_iteration(device=device,
			                                       network=network,
			                                       # optimizer=optimizer,
			                                       dataset=data_minibatch,
			                                       #
			                                       loss_functions=loss_functions,
			                                       regularizer_functions=regularizer_functions,
			                                       information_functions=information_functions,
			                                       #
			                                       loss_function_kwargs=loss_function_kwargs,
			                                       regularizer_function_kwargs=regularizer_function_kwargs,
			                                       information_function_kwargs=information_function_kwargs,
			                                       #
			                                       *args,
			                                       **kwargs
			                                       )

			minibatch_time, minibatch_total_loss, minibatch_total_reg, minibatch_total_infos, minibatch_cache = test_minibatch_output

			epoch_total_loss += minibatch_total_loss
			epoch_total_reg += minibatch_total_reg
			for information_function in information_functions:
				epoch_total_infos[information_function] += minibatch_total_infos[information_function]

			minibatch_start_index += minibatch_size
			'''
			if minibatch_start_index * 100. / number_of_data >= progress_marker:
				print('| {:3.2f}% epoch | {:.2f} ms/minibatch | {:.2f} loss |'.format(
					minibatch_start_index * 100 / number_of_data,
					(timeit.default_timer() - epoch_time) * 1000 / (minibatch_start_index / minibatch_size),
					epoch_total_loss / minibatch_start_index))
				progress_marker += 10
			'''

	epoch_time = timeit.default_timer() - epoch_time
	# epoch_info_obj /= len(dataset_x)
	epoch_average_loss = epoch_total_loss / len(dataset_x)
	epoch_average_reg = epoch_total_reg / len(dataset_x)
	epoch_average_infos = {}
	for information_function in information_functions:
		epoch_average_infos[information_function] = epoch_total_infos[information_function] / len(dataset_x)

	return epoch_time, epoch_average_loss, epoch_average_reg, epoch_average_infos


def test_iteration(device,
                   network,
                   dataset,
                   #
                   loss_functions,
                   regularizer_functions={},
                   information_functions={},
                   #
                   loss_function_kwargs={},
                   regularizer_function_kwargs={},
                   information_function_kwargs={},
                   #
                   *args,
                   **kwargs
                   ):
	network.eval()

	minibatch_x, minibatch_y = dataset
	minibatch_y = minibatch_y.view(-1)
	minibatch_x, minibatch_y = minibatch_x.to(device), minibatch_y.to(device)

	minibatch_time = timeit.default_timer()

	minibatch_cache = {}
	output = network(minibatch_x, **kwargs)
	if isinstance(output, tuple):
		output, hiddens = output
		minibatch_cache["hiddens"] = detach(hiddens)
	# output = network(minibatch_x, seed_hiddens=hiddens, return_hiddens=True)

	minibatch_total_loss = 0
	minibatch_total_reg = 0
	minibatch_total_infos = {}
	for information_function in information_functions:
		minibatch_total_infos[information_function] = 0

	for loss_function in loss_functions:
		minibatch_loss = loss_function(output, minibatch_y, **loss_function_kwargs)
		# batch_losses += batch_loss
		if ("size_average" not in loss_function_kwargs) or loss_function_kwargs["size_average"]:
			minibatch_total_loss += minibatch_loss.item() * len(minibatch_x)
		else:
			minibatch_total_loss += minibatch_loss.item()

	for regularizer_function, regularizer_weight in regularizer_functions.items():
		minibatch_reg = regularizer_function(network, input=minibatch_x, output=output,
		                                     **regularizer_function_kwargs) * regularizer_weight
		# batch_regs += batch_reg
		if ("size_average" not in regularizer_function_kwargs) or regularizer_function_kwargs["size_average"]:
			minibatch_total_reg += minibatch_reg.item() * len(minibatch_x)
		else:
			minibatch_total_reg += minibatch_reg.item()

	for information_function in information_functions:
		minibatch_info = information_function(output, minibatch_y, **information_function_kwargs)
		if ("size_average" not in information_function_kwargs) or information_function_kwargs["size_average"]:
			minibatch_total_infos[information_function] += minibatch_info.item() * len(minibatch_x)
		else:
			minibatch_total_infos[information_function] += minibatch_info.item()

	minibatch_time = timeit.default_timer() - minibatch_time

	return minibatch_time, minibatch_total_loss, minibatch_total_reg, minibatch_total_infos, minibatch_cache


def train_model(network, dataset, settings):
	train_dataset, validate_dataset, test_dataset = dataset

	########################
	# START MODEL TRAINING #
	########################

	for snapshot_function in settings.snapshot:
		snapshot_function(network, 0, settings)

	if porch.debug.display_architecture in settings.debug:
		porch.debug.display_architecture(network)
	if porch.debug.display_gradient in settings.debug:
		porch.debug.display_gradient(network)

	network = network.to(settings.device)

	#
	#
	#

	optimizer = settings.optimizer(network.parameters(), **settings.optimizer_kwargs)
	scheduler = None if settings.lr_scheduler is None else settings.lr_scheduler(optimizer,
	                                                                             **settings.lr_scheduler_kwargs)

	adaptable_parameters = []
	for name, module in network.named_modules():
		try:
			module_adaptables = module.get_adaptables()
			adaptable_parameters.append(module_adaptables)
		except AttributeError:
			pass

		'''
		if (type(module) is porch.modules.dropout.AdaptiveBernoulliDropout) or \
				(type(module) is porch.modules.dropout.AdaptiveBetaBernoulliDropout):
			# or (type(module) is porch.modules.dropout.AdaptiveBernoulliDropoutBackup)
			# or (type(module) is porch.modules.dropout.AdaptiveBetaBernoulliDropoutBackup)
			for p in module.parameters():
				adaptable_parameters.append(p)
		elif (type(module) is nn.Linear):
			for p in module.parameters():
				trainable_params.append(p)
		'''

	if len(adaptable_parameters) > 0:
		adaptable_optimizer = settings.optimizer(adaptable_parameters, **settings.optimizer_kwargs)
		settings.train_kwargs["adaptable_optimizer"] = adaptable_optimizer
		#print(adaptable_optimizer)

	#
	#
	#

	model_snapshot_marker, model_snapshot_steper = 10, 10
	for epoch_index in range(1, settings.number_of_epochs + 1):
		if scheduler is not None:
			scheduler.step()
			logger.info('optimizer: epoch {}, lr {}'.format(epoch_index, scheduler.get_lr()))
			print('optimizer: epoch {}, lr {}'.format(epoch_index, scheduler.get_lr()))

		epoch_train_time, epoch_train_loss, epoch_train_reg, epoch_train_infos = train_epoch(
			device=settings.device,
			network=network,
			optimizer=optimizer,
			dataset=train_dataset,
			loss_functions=settings.loss,
			regularizer_functions=settings.regularizer,
			information_functions=settings.information,
			loss_function_kwargs=settings.loss_kwargs,
			regularizer_function_kwargs=settings.regularizer_kwargs,
			information_function_kwargs=settings.information_kwargs,
			minibatch_size=settings.minibatch_size,
			**settings.train_kwargs
		)

		logger.info('train: epoch {}, duration {}s, loss {}, regularizer {}'.format(
			epoch_index, epoch_train_time, epoch_train_loss, epoch_train_reg))
		print('train: epoch {}, duration {:.2f}s, loss {:.2f}, regularizer {:.2f}'.format(
			epoch_index, epoch_train_time, epoch_train_loss, epoch_train_reg))

		for information_function, information_value in epoch_train_infos.items():
			logger.info('train: epoch {}, {}={}'.format(epoch_index, information_function.__name__, information_value))
			print('train: epoch {}, {}={}'.format(epoch_index, information_function.__name__, information_value))

		epoch_test_time, epoch_test_loss, epoch_test_reg, epoch_test_infos = test_epoch(
			device=settings.device,
			network=network,
			dataset=test_dataset,
			loss_functions=settings.loss,
			regularizer_functions=settings.regularizer,
			information_functions=settings.information,
			loss_function_kwargs=settings.loss_kwargs,
			regularizer_function_kwargs=settings.regularizer_kwargs,
			information_function_kwargs=settings.information_kwargs,
			# generative_model=settings.generative_model
			minibatch_size=settings.minibatch_size,
			**settings.test_kwargs
		)

		logger.info('test: epoch {}, duration {}s, loss {}, regularizer {}'.format(
			epoch_index, epoch_test_time, epoch_test_loss, epoch_test_reg))
		print('test: epoch {}, duration {:.2f}s, loss {:.2f}, regularizer {:.2f}'.format(
			epoch_index, epoch_test_time, epoch_test_loss, epoch_test_reg))

		for information_function, information_value in epoch_test_infos.items():
			logger.info('test: epoch {}, {}={}'.format(epoch_index, information_function.__name__, information_value))
			print('test: epoch {}, {}={}'.format(epoch_index, information_function.__name__, information_value))

		# network.train(train_dataset, validate_dataset, test_dataset, settings.minibatch_size, output_directory)
		# network.epoch_index += 1

		# if settings.snapshot_interval > 0 and network.epoch_index % settings.snapshot_interval == 0:
		# model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
		# pickle.dump(network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

		for snapshot_function in settings.snapshot:
			if epoch_index % settings.snapshot[snapshot_function] == 0:
				snapshot_function(network, epoch_index, settings)

		print("PROGRESS: {:.2f}%".format(100. * (epoch_index) / settings.number_of_epochs))

		if 100. * (epoch_index) / settings.number_of_epochs > model_snapshot_marker:
			model_file = os.path.join(settings.output_directory, 'model.pth')
			# pickle.dump(network._neural_network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
			torch.save(network.state_dict(), model_file)
			# logger.info('Successfully saved model state to {}'.format(model_file))
			print('Successfully saved model state to {} after epoch {}'.format(model_file, epoch_index))
			model_snapshot_marker += model_snapshot_steper

	model_file = os.path.join(settings.output_directory, 'model.pth')
	# pickle.dump(network._neural_network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	torch.save(network.state_dict(), model_file)
	# logger.info('Successfully saved model state to {}'.format(model_file))
	print('Successfully saved model state to {}'.format(model_file))

	return


#
#
#
#
#

def main():
	from . import add_generic_options, validate_generic_options
	model_parser = argparse.ArgumentParser(description="model parser")
	model_parser = add_generic_options(model_parser)
	settings, additionals = model_parser.parse_known_args()
	assert (len(additionals) == 0)
	'''
	if len(additionals) > 0:
		print("========== ==========", "additionals", "========== ==========")
		for addition in additionals:
			print("%s" % (addition))
		print("========== ==========", "additionals", "========== ==========")
	'''
	settings = validate_generic_options(settings)
	settings.generative_model = False

	# input_directory = settings.input_directory
	# output_directory = settings.output_directory
	assert not os.path.exists(settings.output_directory)
	os.mkdir(settings.output_directory)

	logging.basicConfig(filename=os.path.join(settings.output_directory, "model.log"), level=logging.DEBUG,
	                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

	print("========== ==========", "parameters", "========== ==========")
	for key, value in list(vars(settings).items()):
		print("%s=%s" % (key, value))
	print("========== ==========", "parameters", "========== ==========")

	logger.info("========== ==========" + "parameters" + "========== ==========")
	for key, value in list(vars(settings).items()):
		logger.info("%s=%s" % (key, value))
	logger.info("========== ==========" + "parameters" + "========== ==========")

	torch.manual_seed(settings.random_seed)

	model = settings.model(device=settings.device, **settings.model_kwargs).to(settings.device)
	if settings.model_directory is None:
		dataset = load_datasets_to_start(input_directory=settings.input_directory,
		                                 output_directory=settings.output_directory,
		                                 function_parameter_mapping=settings.data)
	# dataset = load_datasets_to_start(input_directory=settings.input_directory, output_directory=settings.output_directory)
	else:
		model_file = os.path.join(settings.model_directory, "model.pth")
		model.load_state_dict(torch.load(model_file))
		logger.info('Successfully load model state from {}'.format(model_file))
		print('Successfully load model state from {}'.format(model_file))

		dataset = load_datasets_to_resume(input_directory=settings.input_directory,
		                                  model_directory=settings.model_directory,
		                                  output_directory=settings.output_directory,
		                                  function_parameter_mapping=settings.data)

	'''
	for data_function in settings.data:
		data_parameters = settings.data[data_function]
		# print(data_function, data_parameters)
		dataset = data_function(dataset, **data_parameters)
	'''

	if porch.debug.subsample_dataset in settings.debug:
		dataset = porch.debug.subsample_dataset(dataset)

	if porch.debug.display_architecture in settings.debug:
		porch.debug.display_architecture(model)

	start_train = timeit.default_timer()
	train_model(model, dataset, settings)
	print("Optimization complete...")
	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


if __name__ == '__main__':
	main()
