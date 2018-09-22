import datetime
import os
import pickle
import re
import subprocess
import timeit

import numpy
import torch

from . import ngram_sos, ngram_eos, nlm_eos
from . import word_context_probability_pattern


def generate_context_word(ngram_file, word_to_id, context_file, ):
	ngram_stream = open(ngram_file, 'r')
	context_stream = open(context_file, 'w')
	for line in ngram_stream:
		line = line.strip("\n")
		fields = line.split("\t")
		# assert (len(fields) == 2)

		context = fields[0]
		count = int(fields[1]) if len(fields) == 2 else 1

		for word in word_to_id:
			context_stream.write("%s %s\t%d\n" % (context, word, count))

		break

	return


def barchart(x, x_label, title=None, output_file_path=None):
	import matplotlib.pyplot as plt

	'''
	n_groups = 5

	means_men = (20, 35, 30, 35, 27)
	std_men = (2, 3, 4, 1, 2)

	means_women = (25, 32, 34, 20, 25)
	std_women = (3, 5, 2, 3, 3)
	'''

	# fig, ax = plt.subplots()
	fig = plt.figure(figsize=(20, 16))

	print(x.shape, len(x), len(x_label))
	index = numpy.arange(len(x_label))
	bar_width = 1. / (len(x) + 2)

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	plt.bar(index + bar_width, x[0, :], bar_width,
	        alpha=opacity,
	        # color='b',
	        # yerr=std_men,
	        error_kw=error_config,
	        label='N-Gram'
	        )

	#
	#
	#
	for i in range(1, len(x)):
		plt.bar(index + (i + 1) * bar_width, x[i, :], bar_width,
		        alpha=opacity,
		        # color='r',
		        # yerr=std_women,
		        error_kw=error_config,
		        # label='NLM:sample-%d' % i
		        )

	# plt.xlabel('word')
	# plt.ylabel('Scores')
	# plt.title('Scores by group and gender')
	plt.xticks(index + 1. / 2, x_label, rotation='vertical')
	plt.legend()

	# Tweak spacing to prevent clipping of ylabel
	fig.tight_layout()

	if title is not None:
		plt.title(title)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


def get_nlm_conditionals(context, word_to_id, data_sequence, nlm_probability_cache):
	context_ids = []
	for token in context.split():
		context_ids.append(word_to_id[nlm_eos] if token == ngram_eos or token == ngram_sos else word_to_id[token])

	nlm_word_probabilities = []
	for i in range(len(data_sequence) - len(context_ids)):
		found = True
		for j in range(len(context_ids)):
			if data_sequence[i + j] != context_ids[j]:
				found = False
				break
		if found:
			nlm_word_probabilities.append(nlm_probability_cache[i + j])

	return nlm_word_probabilities


def get_ngram_conditionals(context, word_to_id, ngram_model, ngram_order, count=1, output_directory=None):
	temp_context_file = "temp.context" if output_directory is None else os.path.join(output_directory, "temp.context")
	context_stream = open(temp_context_file, 'w')
	for word in word_to_id:
		context_stream.write("%s %s\t%d\n" % (context, word, count))
	context_stream.write("%s %s\t%d\n" % (context, ngram_eos, count))
	context_stream.close()

	temp_stdout_file = "temp.stdout" if output_directory is None else os.path.join(output_directory, "temp.stdout")
	command = ["./HelloLM/srilm-i686-m64/ngram",
	           # '-lm %s' % './data/ptb/ngram\,data\=raw/kn\=modified/order\=5.arpa',
	           '-lm %s' % ngram_model,
	           '-order %d' % ngram_order,
	           '-unk -debug %d' % 2,
	           '-counts %s' % temp_context_file,
	           # '> %s' %
	           ]

	# print(" ".join(command))
	temp_stderr_file = "temp.stderr" if output_directory is None else os.path.join(output_directory, "temp.stderr")
	code = subprocess.call(" ".join(command).split(),
	                       stdout=open(temp_stdout_file, 'w'),
	                       stderr=open(temp_stderr_file, 'a'))
	assert (code == 0)

	word_prob = numpy.zeros(len(word_to_id))
	probability_stream = open(temp_stdout_file, "r")
	for line in probability_stream:
		line = line.strip()
		if len(line) == 0:
			continue

		matcher = re.match(word_context_probability_pattern, line)
		if matcher is None:
			# print("unmatched line: %s" % line)
			continue

		word = matcher.group("word")
		word = word.replace(ngram_eos, nlm_eos)
		# word = word.replace(ngram_sos, nlm_eos)

		context_temp = " ".join(matcher.group("context").split(" ")[::-1])
		# context_temp = context_temp.replace(ngram_eos, nlm_eos)
		# context_temp = context_temp.replace(ngram_sos, nlm_eos)
		assert context == context_temp, (context, context_temp)

		ngram = matcher.group("ngram")
		probability = float(matcher.group("probability"))
		log_prob_info = matcher.group("logprobinfo").split("*")

		word_prob[word_to_id[word]] = probability

	os.remove(temp_context_file)
	os.remove(temp_stdout_file)

	return word_prob


def main():
	import argparse
	model_parser = argparse.ArgumentParser(description="model parser")
	model_parser = add_options(model_parser)

	settings, additionals = model_parser.parse_known_args()
	assert (len(additionals) == 0), additionals
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

	start_train = timeit.default_timer()

	import porch.data
	word_to_id, id_to_word = porch.data.import_vocabulary(os.path.join(settings.data_directory, "type.info"))
	data_sequence = numpy.load(os.path.join(settings.data_directory, "train.npy"))
	# data_sequence = data_sequence[:1000]

	context_file = settings.context_file

	ngram_model = settings.ngram_model
	ngram_order = settings.ngram_order

	nlm_cache_directory = settings.nlm_cache_directory

	number_of_candidates = settings.number_of_candidates
	output_directory = settings.output_directory
	# os.mkdir(os.path.join(output_directory, "sample"))
	# os.mkdir(os.path.join(output_directory, "mean"))

	from .ComputeNNLMProbabilities import import_output_cache
	nlm_probability_cache = import_output_cache(nlm_cache_directory, settings.segment_size, cutoff=len(data_sequence))
	# generate_context_word(ngram_file=ngram_file, word_to_id=word_to_id, context_file=context_file)

	context_stream = open(context_file, 'r')
	line_count = 0
	for line in context_stream:
		line = line.strip("\n")
		fields = line.split("\t")
		# assert (len(fields) == 2)
		line_count += 1

		context = fields[0]
		# print(context)
		count = int(fields[1]) if len(fields) == 2 else 1

		# working on ngram models
		ngram_word_probability = get_ngram_conditionals(context=context,
		                                                word_to_id=word_to_id,
		                                                ngram_model=ngram_model,
		                                                ngram_order=ngram_order,
		                                                count=count,
		                                                output_directory=output_directory
		                                                )

		# working on nlm models
		nlm_word_probabilities = get_nlm_conditionals(context=context,
		                                              word_to_id=word_to_id,
		                                              data_sequence=data_sequence,
		                                              nlm_probability_cache=nlm_probability_cache
		                                              )

		# print("nlm=%d" % len(nlm_word_probabilities))
		if len(nlm_word_probabilities) == 0:
			continue

		ngram_candidates = [id for id in ngram_word_probability.argsort()[::-1][:number_of_candidates]]
		nlm_candidates = []
		for nlm_word_probability in nlm_word_probabilities:
			nlm_candidates += [id for id in nlm_word_probability.argsort()[::-1][:number_of_candidates]]
		all_candidates = list(set(ngram_candidates + nlm_candidates))
		all_candidates.sort(key=lambda x: ngram_word_probability[x], reverse=True)

		X = numpy.zeros((len(nlm_word_probabilities) + 1, len(all_candidates)))
		Y = numpy.zeros((2, len(all_candidates)))
		X[0, :] = ngram_word_probability[all_candidates]
		Y[0, :] = ngram_word_probability[all_candidates]
		for i, nlm_word_probability in enumerate(nlm_word_probabilities):
			X[i + 1, :] = nlm_word_probability[all_candidates]
		Y[1, :] = numpy.mean(X[1:, :], axis=0)
		X_label = [id_to_word[id] for id in all_candidates]

		output_file_path = os.path.join(output_directory,
		                                "index=%d,nlm_samples=%d.pkl" % (
			                                line_count, len(nlm_word_probabilities)))
		pickle.dump((X, X_label, context, count), open(output_file_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

		# Z, Z_label, context, count = pickle.load(open(output_file_path, "rb"))
		# print(Z, Z_label, context, count)
		# print("successful")
		# break

		'''
		title = "p(w | %s) * %d" % (context, count)
		output_file_path = os.path.join(output_directory, 
		                                "index=%d,count=%d,nlm_samples=%d.png" % (
			                                line_count, count, len(nlm_word_probabilities)))
		barchart(X, X_label, title=title, output_file_path=output_file_path)
		
		output_file_path = os.path.join(output_directory,
		                                "index=%d,nlm_samples=%d.png" % (
			                                line_count, len(nlm_word_probabilities)))
		barchart(Y, X_label, title=title, output_file_path=output_file_path)
		'''

	end_train = timeit.default_timer()


def add_options(model_parser):
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')

	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument("--context_file", dest="context_file", action='store', default=None,
	                          help="context file [None]")

	model_parser.add_argument("--ngram_model", dest="ngram_model", action='store', default=None,
	                          help="ngram model [None]")
	model_parser.add_argument("--ngram_order", dest="ngram_order", type=int, action='store', default=3,
	                          help="ngram order [3]")

	model_parser.add_argument("--nlm_cache_directory", dest="nlm_cache_directory", action='store', default=None,
	                          help="nlm cache directory [None]")
	model_parser.add_argument("--segment_size", dest="segment_size", type=int, action='store', default=100000,
	                          help="segment size [100K]")

	model_parser.add_argument("--number_of_candidates", dest="number_of_candidates", type=int, action='store',
	                          default=10, help="number of candidates [10]")
	model_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                          help="output directory [None]")

	'''
	model_parser.add_argument("--context_file", dest="context_file", action='store', default=None,
	                          help="context file [None]")

	model_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                          help="model directory [None, resume mode if specified]")
	model_parser.add_argument("--model", dest="model", action='store', default="porch.models.mlp.GenericMLP",
	                          help="neural network model [porch.mnist.MLP]")
	model_parser.add_argument("--model_kwargs", dest="model_kwargs", action='store', default="",
	                          help="model kwargs specified for neural network model [None]")
	'''

	return model_parser


def validate_options(arguments):
	arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	arguments.device = "cpu"
	arguments.device = torch.device(arguments.device)

	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond

	assert os.path.exists(arguments.data_directory)
	assert os.path.exists(arguments.context_file)

	assert os.path.exists(arguments.ngram_model)
	assert arguments.ngram_order > 0

	assert os.path.exists(arguments.nlm_cache_directory)
	assert arguments.segment_size > 0

	if not os.path.exists(arguments.output_directory):
		os.mkdir(arguments.output_directory)
	assert arguments.number_of_candidates > 0

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
