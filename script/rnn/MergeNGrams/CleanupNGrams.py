# import logging
import os
import re
import timeit

# import torch

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
	input_ngram_directory = settings.input_ngram_directory
	ngram_counts_directory = settings.ngram_counts_directory
	output_ngram_directory = settings.output_ngram_directory

	invalid_ngrams = {}
	unique_ngrams = set()
	# ngram_counts[0] =set([""])
	# for i in range(1, index):
	for temp_order in range(1, 10):
		invalid_ngrams[temp_order] = set()
		ngram_count_file = os.path.join(ngram_counts_directory, "ngram=%d.txt" % temp_order)
		ngram_count_stream = open(ngram_count_file, 'r')
		for line in ngram_count_stream:
			line = line.strip()
			fields = line.split("\t")
			assert len(fields) == 2
			count = int(fields[1])
			if count > 1:
				#valid_ngrams[temp_order].add(fields[0])
				continue

			unique_ngrams.add(fields[0])
			tokens = fields[0].split()
			if len(tokens) == 1:
				#valid_ngrams[temp_order].add(fields[0])
				continue

			if " ".join(tokens[1:]) not in unique_ngrams:
				#valid_ngrams[temp_order].add(fields[0])
				continue

			invalid_ngrams[temp_order].add(fields[0])

	for temp_order in range(1, 10):
		input_ngram_file = os.path.join(input_ngram_directory, "ngram=%d.txt" % temp_order)
		input_ngram_stream = open(input_ngram_file, 'r')

		output_ngram_file = os.path.join(output_ngram_directory, "ngram=%d.txt" % temp_order)
		output_ngram_stream = open(output_ngram_file, 'w')

		line = input_ngram_stream.readline()
		line = line.strip()
		output_ngram_stream.write("%s\n" % line)
		for line in input_ngram_stream:
			line = line.strip()
			if len(line) == 0:
				continue
			fields = line.split("\t")
			# logprob = float(fields[0])
			ngrams = fields[1]
			if ngrams in invalid_ngrams[temp_order]:
				continue
			output_ngram_stream.write("%s\n" % line)

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	# generic argument set 1
	model_parser.add_argument("--input_ngram_directory", dest="input_ngram_directory", action='store', default=None,
	                          help="input ngram directory [None]")
	model_parser.add_argument("--output_ngram_directory", dest="output_ngram_directory", action='store', default=None,
	                          help="output ngram directory [None, for ngram order < index]")
	model_parser.add_argument("--ngram_counts_directory", dest="ngram_counts_directory", action='store', default=None,
	                          help="ngram counts directory [None]")
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
	# arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	# arguments.device = "cpu"
	# arguments.device = torch.device(arguments.device)
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
	assert os.path.exists(arguments.input_ngram_directory)
	assert os.path.exists(arguments.ngram_counts_directory)
	if not os.path.exists(arguments.output_ngram_directory):
		os.mkdir(arguments.output_ngram_directory)
	# if not os.path.exists(arguments.output_directory):
	# os.mkdir(arguments.output_directory)
	# assert os.path.exists(arguments.phrase_directory)

	return arguments


if __name__ == '__main__':
	main()
