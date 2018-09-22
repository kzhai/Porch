import datetime
import os
import pickle
import re
import subprocess
import timeit

import numpy
import torch

from . import ngram_eos, nlm_eos
from . import word_context_probability_pattern

'''
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
'''


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
	# data_sequence = numpy.load(os.path.join(settings.data_directory, "train.npy"))
	# data_sequence = data_sequence[:1000]

	context_file = settings.context_file

	ngram_model = settings.ngram_model
	ngram_order = settings.ngram_order

	min_candidates = settings.min_candidates
	min_probability = settings.min_probability
	output_file_path = settings.output_file
	# os.mkdir(os.path.join(output_directory, "sample"))
	# os.mkdir(os.path.join(output_directory, "mean"))

	context_stream = open(context_file, 'r')
	line_count = 0
	context_word_probability = {}
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
		                                                # output_directory=output_directory
		                                                )

		# ranked_words = [id for id in ngram_word_probability.argsort()[::-1][:min_candidates]]
		ranked_words = ngram_word_probability.argsort()[::-1]
		word_probability_list = []
		for i, id in enumerate(ranked_words):
			if i >= min_candidates and ngram_word_probability[id] <= min_probability:
				break
			word_probability_list.append((id_to_word[id], ngram_word_probability[id]))
		context_word_probability[context] = word_probability_list
		# print(context, len(word_probability_list))
		if line_count % 1000 == 0:
			print(line_count)

	pickle.dump(context_word_probability, open(output_file_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

	# Z, Z_label, context, count = pickle.load(open(output_file_path, "rb"))
	# print(Z, Z_label, context, count)
	# print("successful")
	# break

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

	'''
	model_parser.add_argument("--nlm_cache_directory", dest="nlm_cache_directory", action='store', default=None,
	                          help="nlm cache directory [None]")
	model_parser.add_argument("--segment_size", dest="segment_size", type=int, action='store', default=100000,
	                          help="segment size [100K]")
	'''

	model_parser.add_argument("--min_candidates", dest="min_candidates", type=int, action='store',
	                          default=100, help="number of candidates [100]")
	model_parser.add_argument("--min_probability", dest="min_probability", type=float, action='store',
	                          default=1e-3, help="number of candidates [1e-3]")
	model_parser.add_argument("--output_file", dest="output_file", action='store', default=None,
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

	'''
	assert os.path.exists(arguments.nlm_cache_directory)
	assert arguments.segment_size > 0
	'''
	assert arguments.min_candidates > 0
	assert arguments.min_probability > 0

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
