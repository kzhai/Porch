import logging
import pickle
import os
import timeit
import numpy

import porch

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim

logger = logging.getLogger(__name__)

__all__ = [
	# "train",
	# "test",
	# "DiscriminativeModel",
	# "GenerativeModel",
	"train_model",
]


def load_data(input_directory, dataset="test"):
	data_set_x = numpy.load(os.path.join(input_directory, "%s.feature.npy" % dataset))
	data_set_y = numpy.load(os.path.join(input_directory, "%s.label.npy" % dataset))
	data_set_x = torch.from_numpy(data_set_x)
	data_set_y = torch.from_numpy(data_set_y).to(torch.int64)
	assert len(data_set_x) == len(data_set_y)
	logger.info("successfully load %d %s data from %s..." % (len(data_set_x), dataset, input_directory))
	return (data_set_x, data_set_y)


def load_datasets_to_start(input_directory, output_directory, number_of_validate_data=0):
	test_dataset = load_data(input_directory, dataset="test")
	if number_of_validate_data >= 0:
		train_dataset_temp = load_data(input_directory, dataset="train")
		total_data_x, total_data_y = train_dataset_temp

		assert number_of_validate_data >= 0 and number_of_validate_data < len(total_data_y)
		indices = numpy.random.permutation(len(total_data_y))
		train_indices = indices[number_of_validate_data:]
		validate_indices = indices[:number_of_validate_data]

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
	# if len(validate_indices) > 0:
	# else:
	# validate_dataset = None
	else:
		train_dataset = load_data(input_directory, dataset="train")
		validate_dataset = load_data(input_directory, dataset="validate")

	return train_dataset, validate_dataset, test_dataset


def load_datasets_to_resume(input_directory, model_directory, output_directory):
	test_dataset = load_data(input_directory, dataset="test")

	train_dataset_temp = load_data(input_directory, dataset="train")
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


def train_epoch(model, dataset, optimizer,
                # optimization_algorithm=optim.SGD,
                # optimization_algorithm_kwargs={"lr": 1e-3, "momentum": 0.9},
                loss_functions,
                loss_function_kwargs={},
                minibatch_size=64,
                regularizer_functions=None,
                regularizer_function_kwargs={},
                information_function=None,
                information_function_kwargs={},
                generative_model=False):
	# device = torch.device(settings.device)

	model.train()

	dataset_x, dataset_y = dataset
	number_of_data = dataset_x.shape[0]
	data_indices = numpy.random.permutation(number_of_data)

	epoch_opt_loss = 0.
	epoch_opt_reg = 0.
	epoch_info_obj = 0.
	epoch_time = timeit.default_timer()
	minibatch_start_index = 0
	while minibatch_start_index < number_of_data:
		# automatically handles the left-over data
		minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size]

		minibatch_x = dataset_x[minibatch_indices, :]
		minibatch_y = minibatch_x if generative_model else dataset_y[minibatch_indices]

		# data, labels = data.to(device), labels.to(device)
		optimizer.zero_grad()
		output = model(minibatch_x)

		batch_losses = 0
		for loss_function in loss_functions:
			batch_loss = loss_function(output, minibatch_y, **loss_function_kwargs)
			batch_losses += batch_loss
			if ("size_average" not in loss_function_kwargs) or loss_function_kwargs["size_average"]:
				epoch_opt_loss += batch_loss.item() * len(minibatch_x)
			else:
				epoch_opt_loss += batch_loss.item()

		if regularizer_functions is not None:
			batch_regs = 0
			for regularizer_function in regularizer_functions:
				batch_reg = regularizer_function(model, input=minibatch_x, output=output, **regularizer_function_kwargs)
				batch_regs += batch_reg
				if ("size_average" not in regularizer_function_kwargs) or regularizer_function_kwargs["size_average"]:
					epoch_opt_reg += batch_reg.item() * len(minibatch_x)
				else:
					epoch_opt_reg += batch_reg.item()
		else:
			batch_regs = 0

		if information_function is not None:
			batch_info = information_function(output, minibatch_y, **information_function_kwargs).item()
			if ("size_average" not in information_function_kwargs) or information_function_kwargs["size_average"]:
				epoch_info_obj += batch_info * len(minibatch_x)
			else:
				epoch_info_obj += batch_info
		else:
			batch_info = 0

		batch_obj = batch_losses + batch_regs
		batch_obj.backward()
		optimizer.step()

		minibatch_start_index += minibatch_size

	epoch_time = timeit.default_timer() - epoch_time
	epoch_info_obj /= len(dataset_x)
	epoch_opt_loss /= len(dataset_x)
	epoch_opt_reg /= len(dataset_x)

	return epoch_time, epoch_opt_loss, epoch_opt_reg, epoch_info_obj


def test_epoch(model, dataset,
               loss_functions,
               loss_function_kwargs={},
               minibatch_size=64,
               regularizer_functions=None,
               regularizer_function_kwargs={},
               information_function=False,
               information_function_kwargs={},
               generative_model=False):
	# device = torch.device(settings.device)

	model.eval()

	dataset_x, dataset_y = dataset
	number_of_data = dataset_x.shape[0]
	data_indices = numpy.random.permutation(number_of_data)

	epoch_opt_loss = 0.
	epoch_opt_reg = 0.
	epoch_info_obj = 0.
	epoch_time = timeit.default_timer()

	with torch.no_grad():
		minibatch_start_index = 0
		while minibatch_start_index < number_of_data:
			# automatically handles the left-over data
			minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size]

			minibatch_x = dataset_x[minibatch_indices, :]
			# minibatch_y = dataset_y[minibatch_indices]
			minibatch_y = minibatch_x if generative_model else dataset_y[minibatch_indices]

			# data, labels = data.to(device), labels.to(device)
			output = model(minibatch_x)

			batch_losses = 0
			for loss_function in loss_functions:
				batch_loss = loss_function(output, minibatch_y, **loss_function_kwargs)
				batch_losses += batch_loss
				if ("size_average" not in loss_function_kwargs) or loss_function_kwargs["size_average"]:
					epoch_opt_loss += batch_loss.item() * len(minibatch_x)
				else:
					epoch_opt_loss += batch_loss.item()

			if regularizer_functions is not None:
				batch_regs = 0
				for regularizer_function in regularizer_functions:
					batch_reg = regularizer_function(model, input=minibatch_x, output=output,
					                                 **regularizer_function_kwargs)
					batch_regs += batch_reg
					if ("size_average" not in regularizer_function_kwargs) or regularizer_function_kwargs[
						"size_average"]:
						epoch_opt_reg += batch_reg.item() * len(minibatch_x)
					else:
						epoch_opt_reg += batch_reg.item()
			else:
				batch_regs = 0

			if information_function is not None:
				batch_info = information_function(output, minibatch_y, **information_function_kwargs).item()
				if ("size_average" not in information_function_kwargs) or information_function_kwargs["size_average"]:
					epoch_info_obj += batch_info * len(minibatch_x)
				else:
					epoch_info_obj += batch_info

			minibatch_start_index += minibatch_size

	epoch_time = timeit.default_timer() - epoch_time
	epoch_info_obj /= len(dataset_x)
	epoch_opt_loss /= len(dataset_x)
	epoch_opt_reg /= len(dataset_x)

	return epoch_time, epoch_opt_loss, epoch_opt_reg, epoch_info_obj


#
#
#


def train_model(network, settings):
	input_directory = settings.input_directory
	output_directory = settings.output_directory
	assert not os.path.exists(settings.output_directory)
	os.mkdir(settings.output_directory)

	logging.basicConfig(filename=os.path.join(output_directory, "model.log"), level=logging.DEBUG,
	                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

	'''
	train_loader, test_loader = datasets
	for batch_idx, (data, labels) in enumerate(train_loader):
		print(type(data), type(labels))
		print(data.shape, labels.shape)
	'''

	datasets = load_datasets_to_start(input_directory, output_directory)
	train_dataset, validate_dataset, test_dataset = datasets

	if porch.debug.subsample_dataset in settings.debug:
		train_dataset, validate_dataset, test_dataset = porch.debug.subsample_dataset(
			train_dataset, validate_dataset, test_dataset)

	'''
	if dataset_preprocessing_function is not None:
		train_dataset = dataset_preprocessing_function(train_dataset)
		validate_dataset = dataset_preprocessing_function(validate_dataset)
		test_dataset = dataset_preprocessing_function(test_dataset)
	'''

	print("========== ==========", "parameters", "========== ==========")
	for key, value in list(vars(settings).items()):
		print("%s=%s" % (key, value))
	print("========== ==========", "parameters", "========== ==========")

	logger.info("========== ==========" + "parameters" + "========== ==========")
	for key, value in list(vars(settings).items()):
		logger.info("%s=%s" % (key, value))
	logger.info("========== ==========" + "parameters" + "========== ==========")

	# model_file_path = os.path.join(output_directory, 'model-0.pkl')
	# pickle.dump(network._neural_network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	########################
	# START MODEL TRAINING #
	########################

	for snapshot_function in settings.snapshot:
		snapshot_function(network, 0, settings)

	if porch.debug.display_architecture in settings.debug:
		porch.debug.display_architecture(network)

	optimizer = settings.optimization(network.parameters(), **settings.optimization_kwargs)

	start_train = timeit.default_timer()
	for epoch_index in range(1, settings.number_of_epochs + 1):
		epoch_train_time, epoch_train_loss, epoch_train_reg, epoch_train_info = train_epoch(
			network,
			train_dataset,
			optimizer,
			loss_functions=settings.loss,
			loss_function_kwargs=settings.loss_kwargs,
			minibatch_size=settings.minibatch_size,
			regularizer_functions=settings.regularizer,
			regularizer_function_kwargs=settings.regularizer_kwargs,
			information_function=settings.information,
			information_function_kwargs=settings.information_kwargs,
			generative_model=settings.generative_model
		)

		logger.info('train: epoch {}, duration {}s, loss {}, regularizer {}, information {}%'.format(
			epoch_index, epoch_train_time, epoch_train_loss, epoch_train_reg, epoch_train_info * 100))
		print('train: epoch {}, duration {:.2f}s, loss {:.2f}, regularizer {:.2f}, information {:.2f}%'.format(
			epoch_index, epoch_train_time, epoch_train_loss, epoch_train_reg, epoch_train_info * 100))

		epoch_test_time, epoch_test_loss, epoch_test_reg, epoch_test_info = test_epoch(
			network,
			test_dataset,
			loss_functions=settings.loss,
			loss_function_kwargs=settings.loss_kwargs,
			minibatch_size=settings.minibatch_size,
			regularizer_functions=settings.regularizer,
			regularizer_function_kwargs=settings.regularizer_kwargs,
			information_function=settings.information,
			information_function_kwargs=settings.information_kwargs,
			generative_model=settings.generative_model
		)

		logger.info('test: epoch {}, duration {}s, loss {}, regularizer {}, information {}%'.format(
			epoch_index, epoch_test_time, epoch_test_loss, epoch_test_reg, epoch_test_info * 100))
		print('test: epoch {}, duration {:.2f}s, loss {:.2f}, regularizer {:.2f}, information {:.2f}%'.format(
			epoch_index, epoch_test_time, epoch_test_loss, epoch_test_reg, epoch_test_info * 100))

		# network.train(train_dataset, validate_dataset, test_dataset, settings.minibatch_size, output_directory)
		# network.epoch_index += 1

		# if settings.snapshot_interval > 0 and network.epoch_index % settings.snapshot_interval == 0:
		# model_file_path = os.path.join(output_directory, 'model-%d.pkl' % network.epoch_index)
		# pickle.dump(network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

		for snapshot_function in settings.snapshot:
			if epoch_index % settings.snapshot[snapshot_function] == 0:
				snapshot_function(network, epoch_index, settings)

		print("PROGRESS: {:.2f}%".format(100. * (epoch_index) / settings.number_of_epochs))

	# model_file_path = os.path.join(output_directory, 'model.pkl')
	# pickle.dump(network._neural_network, open(model_file_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

	end_train = timeit.default_timer()

	print("Optimization complete...")
	# logger.info("Best validation score of %f%% obtained at epoch %i or minibatch %i" % (
	# network.best_validate_accuracy * 100., network.best_epoch_index, network.best_minibatch_index))
	print('The code for file {} ran for {:.2f}m'.format(
		os.path.split(__file__)[1], (end_train - start_train) / 60.))


#
#
#
#
#

class GenerativeModel(nn.Module):
	def __init__(self):
		super(GenerativeModel, self).__init__()

	def forward(self, *input):
		r"""Defines the computation performed at every call.

		Should be overridden by all subclasses.

		.. note::
			Although the recipe for forward pass needs to be defined within
			this function, one should call the :class:`Module` instance afterwards
			instead of this since the former takes care of running the
			registered hooks while the latter silently ignores them.
		"""
		raise NotImplementedError()

	def train_epoch(self, dataset, minibatch_size=64,
	                optimization_algorithm=optim.SGD,
	                optimization_algorithm_kwargs={"lr": 1e-3, "momentum": 0.9},
	                loss_function=torch.nn.functional.mse_loss,
	                loss_function_kwargs={"size_average": True},
	                info_function=torch.nn.functional.l1_loss,
	                info_function_kwargs={"size_average": True}):
		# device = torch.device(settings.device)

		self.train()
		optimizer = optimization_algorithm(self.parameters(), **optimization_algorithm_kwargs)

		dataset_x, dataset_y = dataset
		number_of_data = dataset_x.shape[0]
		data_indices = numpy.random.permutation(number_of_data)

		epoch_opt_loss = 0.
		epoch_info_obj = 0.
		epoch_time = timeit.default_timer()
		minibatch_start_index = 0
		while minibatch_start_index < number_of_data:
			# automatically handles the left-over data
			minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size]

			minibatch_x = dataset_x[minibatch_indices, :]
			minibatch_y = minibatch_x

			# data, labels = data.to(device), labels.to(device)
			optimizer.zero_grad()
			output = self(minibatch_x)

			batch_loss = loss_function(output, minibatch_y, **loss_function_kwargs)
			if ("size_average" not in loss_function_kwargs) or loss_function_kwargs["size_average"]:
				epoch_opt_loss += batch_loss.item() * len(minibatch_x)
			else:
				epoch_opt_loss += batch_loss.item()

			if info_function is not None:
				batch_info = info_function(output, minibatch_y, **info_function_kwargs).item()
				if ("size_average" not in info_function_kwargs) or info_function_kwargs["size_average"]:
					epoch_info_obj += batch_info * len(minibatch_x)
				else:
					epoch_info_obj += batch_info

			batch_loss.backward()
			optimizer.step()

			minibatch_start_index += minibatch_size

		epoch_time = timeit.default_timer() - epoch_time
		epoch_info_obj /= len(dataset_x)
		epoch_opt_loss /= len(dataset_x)

		return epoch_time, epoch_opt_loss, epoch_info_obj

	def test_epoch(self, dataset, minibatch_size=64,
	               loss_function=torch.nn.functional.mse_loss,
	               loss_function_kwargs={"size_average": True},
	               info_function=torch.nn.functional.l1_loss,
	               info_function_kwargs={"size_average": True}):
		# device = torch.device(settings.device)

		self.eval()

		dataset_x, dataset_y = dataset
		number_of_data = dataset_x.shape[0]
		data_indices = numpy.random.permutation(number_of_data)

		epoch_opt_loss = 0.
		epoch_info_obj = 0.
		epoch_time = timeit.default_timer()

		with torch.no_grad():
			minibatch_start_index = 0
			while minibatch_start_index < number_of_data:
				# automatically handles the left-over data
				minibatch_indices = data_indices[minibatch_start_index:minibatch_start_index + minibatch_size]

				minibatch_x = dataset_x[minibatch_indices, :]
				minibatch_y = minibatch_x

				# data, labels = data.to(device), labels.to(device)
				output = self(minibatch_x)
				batch_loss = loss_function(output, minibatch_y, **loss_function_kwargs)  # sum up batch loss
				if ("size_average" not in loss_function_kwargs) or loss_function_kwargs["size_average"]:
					epoch_opt_loss += batch_loss.item() * len(minibatch_x)
				else:
					epoch_opt_loss += batch_loss.item()

				if info_function is not None:
					batch_info = info_function(output, minibatch_y, **info_function_kwargs).item()
					if ("size_average" not in info_function_kwargs) or info_function_kwargs["size_average"]:
						epoch_info_obj += batch_info * len(minibatch_x)
					else:
						epoch_info_obj += batch_info

				minibatch_start_index += minibatch_size

		epoch_time = timeit.default_timer() - epoch_time
		epoch_info_obj /= len(dataset_x)
		epoch_opt_loss /= len(dataset_x)

		return epoch_time, epoch_opt_loss, epoch_info_obj


'''
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
'''
