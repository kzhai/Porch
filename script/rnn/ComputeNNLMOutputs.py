# import logging
import datetime
import os
import timeit

import numpy
# import scipy
# import scipy.sparse
import torch

import porch
# import porch
from porch.argument import param_deliminator, specs_deliminator
from . import cache_directory_name, output_directory_name, timestamp_prefix


def get_output_probability(network,
                           sequence,
                           directory=None,
                           hiddens=None):
	network.eval()

	# hiddens_cache = []
	if directory is None:
		outputs_cache = []
	with torch.no_grad():
		# assert tokens.shape[0] == 1
		if hiddens is None:
			hiddens = porch.models.rnn.initialize_hidden_states(network, 1)
		# hiddens_temp = porch.base.detach(hiddens)
		kwargs = {"hiddens": hiddens}
		# hiddens_cache.append(hiddens_temp)
		for i, token in enumerate(sequence):
			assert (token.shape == (1, 1))
			output, hiddens = network(token.t(), **kwargs)
			#hiddens_temp = porch.base.detach(hiddens)
			kwargs["hiddens"] = hiddens
			# hiddens_cache.append(hiddens_temp)

			#distribution = torch.nn.functional.softmax(output, dim=1)
			distribution = torch.nn.functional.log_softmax(output, dim=1)
			# distribution = porch.base.detach(distribution)[0, :]
			distribution = distribution.numpy()[0, :]
			if directory is None:
				outputs_cache.append(distribution)
			else:
				distribution_file = os.path.join(directory, "output=%d.npy" % (i))
				numpy.save(distribution_file, distribution)

			if i % 1000 == 0:
				print("progress: %d / %d" % (i, len(sequence)))

	if directory is None:
		# assert (len(hiddens_cache) == len(sequence) + 1)
		assert (len(outputs_cache) == len(sequence))

		return outputs_cache
	else:
		return directory


def import_output_cache(probability_directory, segment_size=100000, cutoff=0):
	outputs_cache = []
	i = 0
	while (True):
		j = i + segment_size
		input_file = os.path.join(probability_directory, "%s=%d-%d.npz" % (timestamp_prefix, i, j - 1))
		if not os.path.exists(input_file):
			break
		temp = numpy.load(input_file)["arr_0"]
		for row_index in range(len(temp)):
			outputs_cache.append(temp[row_index, :])
		print(i, j, len(outputs_cache))
		i = j

		if cutoff > 0 and cutoff <= len(outputs_cache):
			return outputs_cache[:cutoff]

	return outputs_cache


def export_output_cache(outputs_cache, probability_directory, segment_size=100000):
	i = 0
	while (i < len(outputs_cache)):
		j = i + segment_size
		j = j if j < len(outputs_cache) else len(outputs_cache)

		output_file = os.path.join(probability_directory, "%s=%d-%d.npz" % (timestamp_prefix, i, j - 1))
		numpy.savez_compressed(output_file, outputs_cache[i:j])
		print(i, j)
		i = j
	return probability_directory


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

	import porch.data
	word_to_id, id_to_word = porch.data.import_vocabulary(os.path.join(settings.data_directory, "type.info"))
	data_sequence = numpy.load(os.path.join(settings.data_directory, "train.npy"))
	# data_sequence = data_sequence[:20]

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

	# probability_file = os.path.join(settings.model_directory, "data=train,probs=conditional.npz")
	probability_directory = os.path.join(settings.model_directory, cache_directory_name)
	if (not os.path.exists(probability_directory)):
		print("recomputing output probabilities, this may take a while...")
		os.mkdir(probability_directory)
		outputs_cache = get_output_probability(network=model, sequence=sequence)
		export_output_cache(outputs_cache, probability_directory)

	# outputs_cache = numpy.load(probability_file)["arr_0"]
	outputs_cache = import_output_cache(probability_directory)

	'''
	output_directory = os.path.join(settings.model_directory, output_directory_name)
	os.mkdir(output_directory)
	for context_window in range(settings.context_window + 1):
		context_word_probabilities = {}
		for i in range(context_window, len(data_sequence)):
			word = id_to_word[data_sequence[i]]
			context = " ".join([id_to_word[data_sequence[j]] for j in range(i - context_window, i)])
			if (context, word) not in context_word_probabilities:
				context_word_probabilities[(context, word)] = []
			context_word_probabilities[(context, word)].append(outputs_cache[i - 1][data_sequence[i]])

		output_stream = open(
			os.path.join(output_directory, "context=%d.txt" % context_window), 'w')
		for (context, word) in context_word_probabilities:
			output_stream.write("%s\t%s\t%s\n" % (
				context, word, " ".join(["%g" % prob for prob in context_word_probabilities[(context, word)]])))
	'''

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')
	model_parser.add_argument("--context_window", dest="context_window", type=int, action='store', default=9,
	                          help="context window [9]")

	#model_parser.add_argument("--segment_size", dest="segment_size", type=int, action='store', default=100000, help="segment size [100K]")

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
	assert arguments.context_window > 0
	#assert arguments.segment_size > 0

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
