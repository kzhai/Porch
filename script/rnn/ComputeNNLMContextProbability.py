# import logging
import datetime
import os
import timeit

import numpy
# import scipy
# import scipy.sparse
import torch


# import porch


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
	data_sequence = data_sequence[:100]

	model = settings.model(**settings.model_kwargs).to(settings.device)
	model_file = os.path.join(settings.model_directory, "model.pth")
	model.load_state_dict(torch.load(model_file))
	print('Successfully load model state from {}'.format(model_file))

	start_train = timeit.default_timer()

	#
	#
	#
	#
	#
	output_cache_directory = settings.output_cache_directory
	hidden_cache_directory = settings.hidden_cache_directory

	from porch.helpers.rnn.CacheOutputs import import_output_cache
	outputs_cache = import_output_cache(output_cache_directory, cutoff=len(data_sequence))
	print(len(outputs_cache), len(data_sequence))
	assert len(outputs_cache) == len(data_sequence), (len(outputs_cache), len(data_sequence))

	# from .ComputeNNLMHiddens import import_hidden_cache
	# hiddens_cache = import_hidden_cache(hidden_cache_directory, cutoff=len(data_sequence))

	target_context_ids = [word_to_id[word] for word in "workers from".split()]
	target_word_ids = [word_to_id[word] for word in "the <unk> out".split()]
	from porch.helpers.rnn.CacheOutputs import get_output_probability
	for index in range(10, len(data_sequence)):
		sequence = [torch.tensor([[data_sequence[index]]], dtype=torch.int)]
		for word_id in target_context_ids:
			sequence.append(torch.tensor([[word_id]], dtype=torch.int))

		'''
		hidden = []
		for key in hiddens_cache:
			print(key)
			hidden.append(hiddens_cache[key][index])
		'''
		output_cache = get_output_probability(model, sequence, directory=None)
		for j in range(len(target_context_ids)):
			print(id_to_word[target_context_ids[j]], output_cache[j][target_context_ids[j]])
		for j in range(len(target_word_ids)):
			print(id_to_word[target_word_ids[j]], output_cache[len(target_context_ids)][target_word_ids[j]])

	'''
	for i in range(1, len(data_sequence) - len(target_context_ids)):
		found = True
		for j in range(len(target_context_ids)):
			if data_sequence[i + j] != target_context_ids[j]:
				found = False
				break

		if not found:
			continue

		print(id_to_word[data_sequence[i - 1]])
		for j in range(len(target_context_ids)):
			print(id_to_word[data_sequence[i + j]], outputs_cache[i + j-1][data_sequence[i + j]])
		print(id_to_word[data_sequence[i + len(target_context_ids)]],
		      outputs_cache[i + len(target_context_ids)-1][data_sequence[i + len(target_context_ids)]])
	'''

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument("--output_cache_directory", dest="output_cache_directory", action='store',
	                          default=None, help="output cache directory [None]")
	model_parser.add_argument("--hidden_cache_directory", dest="hidden_cache_directory", action='store',
	                          default=None, help="hidden cache directory [None]")
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')

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
	assert os.path.exists(arguments.output_cache_directory)
	assert os.path.exists(arguments.hidden_cache_directory)
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond
	# if not os.path.exists(arguments.output_directory):
	# os.mkdir(arguments.output_directory)
	# assert arguments.context_window > 0
	# assert arguments.segment_size > 0

	from porch import param_deliminator, specs_deliminator
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
