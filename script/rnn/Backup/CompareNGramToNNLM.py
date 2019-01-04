# import logging
import os
import re
import timeit

import numpy
import torch

ngram_file_name_pattern = re.compile(r'ngram=(?P<order>\d+?).txt')
ngram_file_header_pattern = re.compile(r'\\(?P<order>\d+?)-grams:')
ngram_file_entry_pattern = re.compile(r'^(?P<probability>[\.\d\-e]+?)\t(?P<context_word>.+)')

'''
def histogram(X):
	import matplotlib.mlab as mlab
	import matplotlib.pyplot as plt

	num_bins = 100

	fig, ax = plt.subplots()

	# the histogram of the data
	n, bins, patches = ax.hist(X, num_bins, normed=1)

	# add a 'best fit' line
	#y = mlab.normpdf(bins, mu, sigma)
	#ax.plot(bins, y, '--')
	ax.set_xlabel('Smarts')
	ax.set_ylabel('Probability density')
	ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

	# Tweak spacing to prevent clipping of ylabel
	fig.tight_layout()
	plt.show()
'''


def histogram(x, title=None, output_file_path=None):
	import matplotlib.pyplot as plt

	num_bins = 100

	fig, ax = plt.subplots()

	# the histogram of the data
	# n, bins, patches = ax.hist(x, num_bins, normed=1)
	n, bins, patches = ax.hist(x, num_bins)

	# ax.set_xlabel(r'p_{NLM}(word|context) - p_{NGram}(word|context)')
	# ax.set_ylabel('Probability density')
	ax.set_xlim(-1, 1)

	# Tweak spacing to prevent clipping of ylabel
	fig.tight_layout()

	if title is not None:
		plt.title(title)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


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
	ngram_directory = settings.ngram_directory
	nnlm_directory = settings.nnlm_directory
	count_directory = settings.count_directory
	output_directory = settings.output_directory

	for file_name in os.listdir(ngram_directory):
		matcher = re.match(ngram_file_name_pattern, file_name)
		if matcher is None:
			continue
		name_order = int(matcher.group("order"))

		#
		#
		#
		ngram_counts = {}
		count_file_path = os.path.join(count_directory, file_name)
		count_file_stream = open(count_file_path, 'r')
		for line in count_file_stream:
			line = line.strip("\n")
			if len(line) == 0:
				continue

			fields = line.split("\t")
			ngram_counts[fields[0]] = int(fields[1])

		#
		#
		#
		ngram_file_path = os.path.join(ngram_directory, file_name)
		ngram_file_stream = open(ngram_file_path, 'r')

		line = ngram_file_stream.readline()
		line = line.strip("\n")
		matcher = re.match(ngram_file_header_pattern, line)
		assert matcher is not None
		header_order = int(matcher.group("order"))
		assert name_order == header_order

		ngram_context_word_probabilities = {}
		for line in ngram_file_stream:
			line = line.strip("\n")
			if len(line) == 0:
				continue

			matcher = re.match(ngram_file_entry_pattern, line)
			assert matcher is not None, line

			probability = numpy.power(10, float(matcher.group("probability")))
			context_word = matcher.group("context_word")

			ngram_context_word_probabilities[context_word] = probability

		#
		#
		#
		nnlm_file_path = os.path.join(nnlm_directory, file_name)
		nnlm_file_stream = open(nnlm_file_path, 'r')

		output_file = os.path.join(output_directory, file_name)
		output_stream = open(output_file, 'w')

		line = nnlm_file_stream.readline()
		line = line.strip("\n")
		matcher = re.match(ngram_file_header_pattern, line)
		assert matcher is not None
		header_order = int(matcher.group("order"))
		assert name_order == header_order

		nnlm_gt_ngram = 0
		nnlm_eq_ngram = 0
		nnlm_lt_ngram = 0
		difference = []
		for line in nnlm_file_stream:
			line = line.strip("\n")
			if len(line) == 0:
				continue

			matcher = re.match(ngram_file_entry_pattern, line)
			assert matcher is not None, line

			probability = numpy.power(10, float(matcher.group("probability")))
			context_word = matcher.group("context_word")

			if context_word not in ngram_context_word_probabilities:
				continue
			if probability > ngram_context_word_probabilities[context_word]:
				nnlm_gt_ngram += 1
				output_stream.write("nnlm\t%s\t%d\t%g\t%g\n" % (context_word, ngram_counts[context_word], probability,
				                                                ngram_context_word_probabilities[context_word]))
			elif probability < ngram_context_word_probabilities[context_word]:
				nnlm_lt_ngram += 1
				output_stream.write("m-kn\t%s\t%d\t%g\t%g\n" % (context_word, ngram_counts[context_word], probability,
				                                                ngram_context_word_probabilities[context_word]))
			else:
				nnlm_eq_ngram += 1

			difference.append(probability - ngram_context_word_probabilities[context_word])

		# from .PlotConditionalDifferences import histogram
		output_file_path = os.path.join(output_directory, "ngram=%d.png" % name_order)
		histogram(difference, title="p(w|%d-grams;NNLM) - p(w|%d-grams;M-KN)" % (name_order, name_order),
		          output_file_path=output_file_path)

		print("%d-grams: nnlm>ngram=%d nnlm<ngram=%d, nnlm=ngram=%d" % (
			name_order, nnlm_gt_ngram, nnlm_lt_ngram, nnlm_eq_ngram))

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	# generic argument set 1
	model_parser.add_argument("--nnlm_directory", dest="nnlm_directory", action='store', default=None,
	                          help="nnlm directory [None]")
	model_parser.add_argument("--ngram_directory", dest="ngram_directory", action='store', default=None,
	                          help="ngram directory [None]")
	model_parser.add_argument("--count_directory", dest="count_directory", action='store', default=None,
	                          help="counts directory [None]")
	model_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                          help="output file [None]")
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
	arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	arguments.device = "cpu"
	arguments.device = torch.device(arguments.device)
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
	assert os.path.exists(arguments.nnlm_directory)
	assert os.path.exists(arguments.ngram_directory)
	assert (arguments.count_directory is None) or os.path.exists(arguments.count_directory)
	# if not os.path.exists(arguments.output_directory):
	# os.mkdir(arguments.output_directory)
	# assert os.path.exists(arguments.phrase_directory)

	return arguments


if __name__ == '__main__':
	main()
