# import logging
import datetime
import os
import pickle
import timeit

import numpy
# import scipy
# import scipy.sparse
import torch

import porch
# import porch
from porch.argument import param_deliminator, specs_deliminator


def get_hidden_states(network,
                      sequence,
                      directory=None,
                      segment=100000):
	network.eval()

	hiddens_cache = []
	# if directory is None:
	# outputs_cache = []
	with torch.no_grad():
		# assert tokens.shape[0] == 1
		hiddens = porch.models.rnn.initialize_hidden_states(network, 1)
		hiddens_temp = porch.base.detach(hiddens)
		kwargs = {"hiddens": hiddens}
		hiddens_cache.append(hiddens_temp)
		processed = 0

		for i, token in enumerate(sequence):
			assert (token.shape == (1, 1))

			output, hiddens = network(token.t(), **kwargs)
			hiddens_temp = porch.base.detach(hiddens)
			kwargs["hiddens"] = hiddens
			hiddens_cache.append(hiddens_temp)

			if directory is not None and ((i + 1) % segment == 0 or i == len(sequence) - 1):
				# from .PlotHiddenProjectionTrajectory import reformat_hidden_states
				hiddens_cache_reformatted = reformat_hidden_states(hiddens_cache)
				hidden_cache_file = os.path.join(directory, "timestamp=%d-%d.pkl" % (
					processed, processed + segment - 1))

				pickle.dump(hiddens_cache_reformatted, open(hidden_cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

				processed += segment
				hiddens_cache.clear()

				print("progress: %d / %d" % (i + 1, len(sequence)))

	if directory is None:
		assert (len(hiddens_cache) == len(sequence) + 1)
		# assert (len(outputs_cache) == len(sequence))

		return hiddens_cache
	else:
		return directory


def reformat_hidden_states(hidden_states):
	hiddens_cache = {}
	for token_index in range(len(hidden_states)):
		hiddens_token = hidden_states[token_index]

		for lstm_group_index in range(len(hiddens_token)):
			hiddens_token_group = hiddens_token[lstm_group_index]

			if type(hiddens_token_group) == tuple:
				# We are dealing with LSTM layer here.
				hiddens_token_group = hiddens_token_group[0]

			for lstm_layer_index in range(hiddens_token_group.shape[0]):
				# assert (hiddens_token_group.shape[1] == 1)
				hiddens_token_group_layer = hiddens_token_group[lstm_layer_index]

				# for sequence_index in range(hiddens_token_group.shape[1]):
				if (lstm_group_index, lstm_layer_index) not in hiddens_cache:
					hiddens_cache[(lstm_group_index, lstm_layer_index)] = numpy.zeros(
						(len(hidden_states), hiddens_token_group_layer.shape[1]))
				hiddens_cache[(lstm_group_index, lstm_layer_index)][token_index, :] = \
					hiddens_token_group_layer.numpy()
	return hiddens_cache


'''
def unformat_hidden_states(hiddens_cache):
	lstm_group_indices= set()
	lstm_layer_indices = set()
	for lstm_group_index, lstm_layer_index in hiddens_cache:
		lstm_group_indices.add(lstm_group_index)
		lstm_layer_indices.add(lstm_layer_index)

	hidden_states = []
	for time_stamp in range(len(hiddens_cache[(0,0)])):
		hidden_states.append([])
		for lstm_group_index in range(len(lstm_group_indices)):
			hidden_states[time_stamp] =
			for lstm_layer_index in range(len(lstm_layer_indices)):
			pass
		hidden_states.append(hiddens_cache)

	for token_index in range(len(cache)):
		hiddens_token = cache[token_index]

		for lstm_group_index in range(len(hiddens_token)):
			hiddens_token_group = hiddens_token[lstm_group_index]

			if type(hiddens_token_group) == tuple:
				# We are dealing with LSTM layer here.
				hiddens_token_group = hiddens_token_group[0]

			for lstm_layer_index in range(hiddens_token_group.shape[0]):
				# assert (hiddens_token_group.shape[1] == 1)
				hiddens_token_group_layer = hiddens_token_group[lstm_layer_index]

				# for sequence_index in range(hiddens_token_group.shape[1]):
				if (lstm_group_index, lstm_layer_index) not in hiddens_sequence:
					hiddens_sequence[(lstm_group_index, lstm_layer_index)] = numpy.zeros(
						(len(cache), hiddens_token_group_layer.shape[1]))
				hiddens_sequence[(lstm_group_index, lstm_layer_index)][token_index, :] = \
					hiddens_token_group_layer.numpy()
	return hiddens_sequence
'''


def import_hidden_cache(hidden_cache_directory, segment_size=100000, cutoff=-1):
	hidden_cache = None
	i = 0
	while (True):
		j = i + segment_size
		hidden_cache_file = os.path.join(hidden_cache_directory, "timestamp=%d-%d.pkl" % (i, j - 1))
		if not os.path.exists(hidden_cache_file):
			break

		# print(numpy.load(input_file))
		temp = numpy.load(hidden_cache_file)

		if hidden_cache is None:
			hidden_cache = temp
		else:
			for key in temp:
				hidden_cache[key] = numpy.vstack((hidden_cache[key], temp[key]))
		# outputs_cache.append(temp[row_index, :])
		print("checkpoint", i, j)
		# print(outputs_cache)
		# print([(key, output_cache.shape) for key, output_cache in outputs_cache])
		i = j

		if cutoff > 0 and cutoff <= i:
			return hidden_cache

	return hidden_cache


def main():
	import argparse
	model_parser = argparse.ArgumentParser(description="model parser")
	model_parser = add_options(model_parser)

	settings, additionals = model_parser.parse_known_args()
	assert (len(additionals) == 0)
	settings = validate_options(settings)

	print("========== ==========", "parameters", "========== ==========")
	for key, value in list(vars(settings).items()):
		print("%s=%s" % (key, value))
	print("========== ==========", "parameters", "========== ==========")
	torch.manual_seed(settings.random_seed)

	#
	#
	#
	#
	#

	# word_to_id, id_to_word = porch.data.import_vocabulary(os.path.join(settings.data_directory, "type.info"))
	data_sequence = numpy.load(os.path.join(settings.data_directory, "train.npy"))
	# data_sequence = data_sequence[:105]

	model = settings.model(**settings.model_kwargs).to(settings.device)
	model_file = os.path.join(settings.model_directory, "model.pth")
	model.load_state_dict(torch.load(model_file))
	print('Successfully load model state from {}'.format(model_file))

	sequence = []
	for word_id in data_sequence:
		sequence.append(torch.tensor([[word_id]], dtype=torch.int))
	assert (len(sequence) == len(data_sequence))

	start_train = timeit.default_timer()

	#
	#
	#
	#
	#

	# probability_file = os.path.join(settings.model_directory, )
	# hidden_cache_file = os.path.join(settings.model_directory, "data=train,cache=hidden.npz")
	hidden_cache_directory = os.path.join(settings.model_directory, "data=train,cache=hidden")
	if (not os.path.exists(hidden_cache_directory)):
		print("recomputing hidden states, this may take a while...")
		os.mkdir(hidden_cache_directory)
		get_hidden_states(network=model, sequence=sequence, directory=hidden_cache_directory)

		'''
		from .PlotHiddenProjectionTrajectory import reformat_hidden_states
		hiddens_cache = reformat_hidden_states(hiddens_cache)

		# numpy.savez(hidden_cache_file, hiddens_cache)
		# pickle.dump(hiddens_cache, open(hidden_cache_directory, "wb") , protocol=pickle.HIGHEST_PROTOCOL)
		# print(hiddens_cache)
		export_cache(hiddens_cache, hidden_cache_directory, settings.segment_size)
		'''

	# hiddens_cache = pickle.load(open(hidden_cache_file, 'rb'))
	# hiddens_cache = numpy.load(hidden_cache_file)["arr_0"]
	# print(hiddens_cache)

	hiddens_cache = import_hidden_cache(hidden_cache_directory)
	for key in hiddens_cache:
		print(key)
		print(hiddens_cache[key].shape)
	# print(hiddens_cache[key])

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')
	# model_parser.add_argument("--context_window", dest="context_window", type=int, action='store', default=9,
	# help="context window [9]")

	# model_parser.add_argument("--segment_size", dest="segment_size", type=int, action='store', default=100000, help="segment size [100K]")

	model_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                          help="model directory [None, resume mode if specified]")
	model_parser.add_argument("--model", dest="model", action='store', default="porch.models.mlp.GenericMLP",
	                          help="neural network model [porch.mnist.MLP]")
	model_parser.add_argument("--model_kwargs", dest="model_kwargs", action='store', default="",
	                          help="model kwargs specified for neural network model [None]")

	return model_parser


def validate_options(arguments):
	arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	arguments.device = "cpu"
	arguments.device = torch.device(arguments.device)

	assert os.path.exists(arguments.data_directory)
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond
	# assert arguments.context_window > 0
	# assert arguments.segment_size > 0

	assert os.path.exists(arguments.model_directory)
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

	return arguments


if __name__ == '__main__':
	main()
