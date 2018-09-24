# import logging
import datetime
import os
import pickle
import timeit

import numpy
# import scipy
# import scipy.sparse
import torch

# import porch
from porch.argument import param_deliminator, specs_deliminator


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
	#data_sequence = data_sequence[:100]

	start_train = timeit.default_timer()

	#
	#
	#
	#
	#

	from .ComputeNNLMOutputs import import_output_cache
	outputs_cache = import_output_cache(settings.probability_cache_directory, cutoff=len(data_sequence))

	for context_window in range(1, 9):
		context_probability = {}
		# prob = 1.0
		log_prob = 0
		for i in range(1, context_window + 1):
			# prob *= outputs_cache[i - 1][data_sequence[i]]
			log_prob += numpy.log(outputs_cache[i - 1][data_sequence[i]])
		context = [id_to_word[id] for id in data_sequence[1:context_window + 1]]
		context_probability[" ".join(context)] = [log_prob]

		for i in range(context_window + 1, len(data_sequence)):
			log_prob -= numpy.log(outputs_cache[i - context_window - 1][data_sequence[i - context_window]])
			log_prob += numpy.log(outputs_cache[i - 1][data_sequence[i]])
			context.pop(0)
			context.append(id_to_word[data_sequence[i]])

			temp_context = " ".join(context)
			if temp_context not in context_probability:
				context_probability[temp_context] = []
			context_probability[temp_context].append(log_prob)

			if len(context_probability) % 10000 == 0:
				print("processed %d %d-grams..." % (len(context_probability), context_window))

		print("processed %d %d-grams..." % (len(context_probability), context_window))

		context_probability_pkl_file = os.path.join(settings.output_directory, "context=%d.pkl"%context_window)
		pickle.dump(context_probability, open(context_probability_pkl_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

		context_probability_txt_file = os.path.join(settings.output_directory, "context=%d.txt"%context_window)
		context_probability_txt_stream = open(context_probability_txt_file, 'w')
		for context in context_probability:
			context_probability_txt_stream.write(
				"%s\t%s\n" % (context, " ".join(["%g" % probability for probability in context_probability[context]])))

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument("--probability_cache_directory", dest="probability_cache_directory", action='store', default=None,
	                          help="probability cache directory [None]")
	model_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')

	return model_parser


def validate_options(arguments):
	arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	arguments.device = "cpu"
	arguments.device = torch.device(arguments.device)

	assert os.path.exists(arguments.data_directory)
	assert os.path.exists(arguments.probability_cache_directory)
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond
	if not os.path.exists(arguments.output_directory):
		os.mkdir(arguments.output_directory)
	#assert arguments.context_window > 0
	# assert arguments.segment_size > 0

	'''
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
	'''
	return arguments


if __name__ == '__main__':
	main()
