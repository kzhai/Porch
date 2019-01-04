# import logging
import os
import re
import timeit

#import torch

ngram_file_name_pattern = re.compile(r'ngram=(?P<order>\d+?).txt')
ngram_file_header_pattern = re.compile(r'\\(?P<order>\d+?)-grams:')
ngram_file_entry_pattern = re.compile(r'(?P<probability>[\.\d\-e]+?)\t(?P<context_word>.+?)')


def count_ngrams(file_path):
	# file_path = os.path.join(nnlm_directory, "ngram=%d.txt" % i)
	file_stream = open(file_path, 'r')
	line = file_stream.readline()
	line = line.strip("\n")

	matcher = re.match(ngram_file_header_pattern, line)
	assert matcher is not None
	header_order = int(matcher.group("order"))
	# assert name_order == header_order

	count = 0
	for line in file_stream:
		line = line.strip("\n")
		if len(line) == 0:
			continue

		matcher = re.match(ngram_file_entry_pattern, line)
		assert matcher is not None, line

		probability = float(matcher.group("probability"))
		context_word = matcher.group("context_word")
		count += 1

	return count


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

	start_train = timeit.default_timer()

	# from .ProjectHiddensFromRandom import load_dictionary
	higher_order_ngram_directory = settings.primary_ngram_directory
	lower_order_ngram_directory = settings.secondary_ngram_directory
	output_file = settings.output_file
	index = settings.index

	ngram_counts = {}
	# for i in range(1, index):
	for temp_order in range(1, 10):
		if temp_order < index and (lower_order_ngram_directory is not None):
			file_path = os.path.join(lower_order_ngram_directory, "ngram=%d.txt" % temp_order)
		else:
			file_path = os.path.join(higher_order_ngram_directory, "ngram=%d.txt" % temp_order)

		ngram_counts[temp_order] = count_ngrams(file_path)

	# for order in range(9, 10):
	# output_file = os.path.join(output_directory, "order=%d.arpa" % order)
	# output_file = os.path.join(output_directory, "order=%d.arpa" % order)
	# output_file = output_directory
	output_stream = open(output_file, 'w')

	output_stream.write("\\data\\\n")
	for temp_order in range(1, 10):
		output_stream.write("ngram %d=%d\n" % (temp_order, ngram_counts[temp_order]))
	output_stream.write("\n")

	# '''
	for temp_order in range(1, 10):
		if temp_order < index and (lower_order_ngram_directory is not None):
			input_file_path = os.path.join(lower_order_ngram_directory, "ngram=%d.txt" % temp_order)
		else:
			input_file_path = os.path.join(higher_order_ngram_directory, "ngram=%d.txt" % temp_order)
		input_file_stream = open(input_file_path, 'r')
		line = input_file_stream.readline()
		line = line.strip()
		output_stream.write("%s\n" % line)
		for line in input_file_stream:
			line = line.strip()
			output_stream.write("%s\t0\n" % line)
		output_stream.write("\n")
	# '''
	'''
	for temp_order in range(1, 10):
		ngram_probabilities = {}
		ngram_file_path = os.path.join(ngram_directory, "ngram=%d.txt" % temp_order)
		ngram_file_stream = open(ngram_file_path, 'r')
		line = ngram_file_stream.readline()
		line = line.strip()
		for line in ngram_file_stream:
			line = line.strip()
			fields = line.split("\t")
			context_word = fields[1]
			log_10_prob = float(fields[0])
			ngram_probabilities[context_word] = log_10_prob

		nnlm_file_path = os.path.join(nnlm_directory, "ngram=%d.txt" % temp_order)
		nnlm_file_stream = open(nnlm_file_path, 'r')
		line = nnlm_file_stream.readline()
		line = line.strip()
		output_stream.write("%s\n" % line)
		for line in nnlm_file_stream:
			line = line.strip()
			fields = line.split("\t")

			context_word = fields[1]
			log_10_prob = float(fields[0])
			if numpy.power(10, log_10_prob) < numpy.power(10, ngram_probabilities[context_word]):
				output_stream.write("%g\t%s\t0\n" % (ngram_probabilities[context_word], context_word))
			else:
				output_stream.write("%g\t%s\t0\n" % (log_10_prob, context_word))
		output_stream.write("\n")
	'''

	output_stream.write("\\end\\\n")

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	# generic argument set 1
	model_parser.add_argument("--primary_ngram_directory", dest="primary_ngram_directory", action='store', default=None,
	                          help="primary ngram directory [None, for ngram order >= index]")
	model_parser.add_argument("--secondary_ngram_directory", dest="secondary_ngram_directory", action='store',
	                          default=None,
	                          help="secondary ngram directory [None, for ngram order < index]")
	model_parser.add_argument("--output_file", dest="output_file", action='store', default=None,
	                          help="output file [None]")
	model_parser.add_argument("--index", dest="index", type=int, action='store', default=1,
	                          help="index [None]")
	'''
	model_parser.add_argument("--output_sequence", dest="output_sequence", action='store', default=None,
	                          help="output sequence file [None]")
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')

	# generic argument set 3
	model_parser.add_argument("--number_of_samples", dest="number_of_samples", type=int, action='store', default=-1,
	                          help="number of samples [-1]")
	model_parser.add_argument("--eos_token", dest="eos_token", action='store', default="<eos>", help="eos token")

	# generic argument set 4
	model_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                          help="model directory [None, resume mode if specified]")
	model_parser.add_argument("--model", dest="model", action='store', default="porch.models.mlp.GenericMLP",
	                          help="neural network model [porch.mnist.MLP]")
	model_parser.add_argument("--model_kwargs", dest="model_kwargs", action='store', default="",
	                          help="model kwargs specified for neural network model [None]")
	'''

	return model_parser


def validate_options(arguments):
	# use_cuda = arguments.device.lower() == "cuda" and torch.cuda.is_available()
	#arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	#arguments.device = "cpu"
	#arguments.device = torch.device(arguments.device)
	'''
	# generic argument set snapshots
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond

	# generic argument set 3
	assert arguments.number_of_samples > 0
	# assert arguments.perturbation_tokens > 0

	# generic argument set 4
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

	# generic argument set 1
	assert os.path.exists(arguments.primary_ngram_directory)
	assert (arguments.secondary_ngram_directory is None) or os.path.exists(arguments.secondary_ngram_directory)
	assert 0 < arguments.index < 10
	# if not os.path.exists(arguments.output_directory):
	# os.mkdir(arguments.output_directory)
	# assert os.path.exists(arguments.phrase_directory)

	return arguments


if __name__ == '__main__':
	main()
