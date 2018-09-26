# import logging
import datetime
import os
import timeit

import numpy
# import scipy
# import scipy.sparse
#import torch


# import porch

def generate_candidates(data_sequence, context_window_size, eos_id, outputs_cache=None, number_of_candidates=0):
	context_candidates = {}
	context_window = []
	for i in range(len(data_sequence)):
		assert len(context_window) <= context_window_size

		word_id = data_sequence[i]
		if len(context_window) == context_window_size:
			context_ids = tuple(context_window)
			if context_ids not in context_candidates:
				# print([id_to_word[id] for id in context_ids])
				context_candidates[context_ids] = set()

			context_candidates[context_ids].add(word_id)
			if outputs_cache is not None:
				if number_of_candidates > 0:
					for id in outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
						context_candidates[context_ids].add(id)
				'''
				elif number_of_candidates<0:
					for id in outputs_cache[i - 1].argsort()[::-1][number_of_candidates:]:
						context_candidates[context_ids].add(id)
				'''

		if word_id == eos_id:
			context_window.clear()
			context_window.append(word_id)
		else:
			if len(context_window) == context_window_size:
				context_window.pop(0)
			context_window.append(word_id)

		if (i + 1) % 100000 == 0:
			print("collected candidates for %d %d-grams..." % (len(context_candidates), context_window_size + 1))

	print("collected candidates for %d %d-grams..." % (len(context_candidates), context_window_size + 1))

	return context_candidates


'''
def generate_top_ranked_candidates(data_sequence, outputs_cache, context_window_size, number_of_candidates, eos_id):
	context_candidates = {}
	context_window = []
	for i in range(len(data_sequence)):
		assert len(context_window) <= context_window_size

		if len(context_window) == context_window_size:
			context_ids = tuple(context_window)
			if context_ids not in context_candidates:
				# print([id_to_word[id] for id in context_ids])
				context_candidates[context_ids] = set()

			for id in outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
				context_candidates[context_ids].add(id)

		word_id = data_sequence[i]
		if word_id == eos_id:
			context_window.clear()
			context_window.append(word_id)
		else:
			if len(context_window) == context_window_size:
				context_window.pop(0)
			context_window.append(word_id)

		if (i + 1) % 100000 == 0:
			print("collected candidates for %d %d-grams..." % (len(context_candidates), context_window_size + 1))

	print("collected candidates for %d %d-grams..." % (len(context_candidates), context_window_size + 1))

	return context_candidates
'''


def test(data_sequence, outputs_cache, context_window_size, context_candidates, id_to_word, eos_id):
	log_p_word_context = {}
	log_p_context = {}
	# log_normalizers = {}
	context_window = [data_sequence[1]]
	context_log_prob = numpy.log(outputs_cache[0][data_sequence[1]])
	for i in range(2, len(data_sequence)):
		assert len(context_window) <= context_window_size

		if len(context_window) == context_window_size:
			context_ids = tuple(context_window)
			assert context_ids in context_candidates, ([id_to_word[id] for id in context_ids])
			if context_ids not in log_p_context:
				log_p_context[context_ids] = -1e3
				log_p_word_context[context_ids] = {word_id: -1e3 for word_id in context_candidates[context_ids]}

			log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], context_log_prob)
			# log_normalizers[i - 1] = numpy.log(numpy.sum(outputs_cache[i - 1][list(context_candidates[context_ids])]))
			# print(log_normalizers[i - 1] )
			for candidate_id in context_candidates[context_ids]:
				log_p_word_context[context_ids][candidate_id] = numpy.logaddexp(
					log_p_word_context[context_ids][candidate_id],
					context_log_prob + numpy.log(outputs_cache[i - 1][candidate_id]))

		if data_sequence[i] == eos_id:
			context_window.clear()
			context_window.append(data_sequence[i])
			context_log_prob = numpy.log(outputs_cache[i - 1][data_sequence[i]])
		# if i - 1 in log_normalizers:
		# context_log_prob -= - log_normalizers[i - 1]
		else:
			if len(context_window) == context_window_size:
				context_window.pop(0)
				context_log_prob -= numpy.log(
					outputs_cache[i - context_window_size - 1][data_sequence[i - context_window_size]])
			# if i - context_window_size - 1 in log_normalizers:
			# context_log_prob += log_normalizers[i - context_window_size - 1]
			context_window.append(data_sequence[i])
			context_log_prob += numpy.log(outputs_cache[i - 1][data_sequence[i]])
		# if i - 1 in log_normalizers:
		# context_log_prob -= - log_normalizers[i - 1]

		if (i + 1) % 100000 == 0:
			print("processed %d %d-grams..." % (len(log_p_context), context_window_size + 1))

	print("processed %d %d-grams..." % (len(log_p_context), context_window_size + 1))

	return log_p_word_context, log_p_context


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
	#torch.manual_seed(settings.random_seed)

	#
	#
	#
	#
	#

	import porch.data
	word_to_id, id_to_word = porch.data.import_vocabulary(os.path.join(settings.data_directory, "type.info"))
	data_sequence = numpy.load(os.path.join(settings.data_directory, "train.npy"))

	subset = settings.subset
	if subset > 0:
		data_sequence = data_sequence[:subset]

	start_train = timeit.default_timer()

	from . import ngram_sos, ngram_eos, nlm_eos
	eos_word = nlm_eos
	eos_id = word_to_id[eos_word]
	number_of_candidates = settings.number_of_candidates

	#
	#
	#
	#
	#

	from .ComputeNNLMOutputs import import_output_cache
	outputs_cache = import_output_cache(settings.probability_cache_directory, cutoff=len(data_sequence))
	print("successfully load outputs cache...")

	#
	#
	#

	log_p_word = {word_id: -1e3 for word_id in id_to_word}
	for i in range(len(data_sequence)):
		for word_id in id_to_word:
			log_p_word[word_id] = numpy.logaddexp(log_p_word[word_id],
			                                      numpy.log(outputs_cache[i][word_id] / len(data_sequence)))

	ngram_file = os.path.join(settings.output_directory, "ngram=1.txt")
	ngram_stream = open(ngram_file, 'w')
	ngram_stream.write("\\1-grams:\n")
	ngram_stream.write("%g\t%s\n" % (-99, ngram_sos))
	for word_id in log_p_word:
		word = id_to_word[word_id] if word_id != eos_id else ngram_eos
		ngram_stream.write("%g\t%s\n" % (numpy.log10(numpy.exp(log_p_word[word_id])), word))

	#
	#
	#

	for context_window_size in range(1, 9):

		context_candidates = generate_candidates(data_sequence=data_sequence,
		                                         context_window_size=context_window_size,
		                                         eos_id=eos_id,
		                                         outputs_cache=outputs_cache,
		                                         number_of_candidates=number_of_candidates
		                                         )
		#
		#
		#
		log_p_word_context, log_p_context = test(data_sequence=data_sequence,
		                                         outputs_cache=outputs_cache,
		                                         context_window_size=context_window_size,
		                                         context_candidates=context_candidates,
		                                         id_to_word=id_to_word,
		                                         eos_id=eos_id)
		#
		#
		#

		ngram_file = os.path.join(settings.output_directory, "ngram=%d.txt" % (context_window_size + 1))
		ngram_stream = open(ngram_file, 'w')
		ngram_stream.write("\\%d-grams:\n" % (context_window_size + 1))
		for context_ids in log_p_word_context:
			context_words = [id_to_word[context_id] if context_id != eos_id else ngram_sos for context_id in
			                 context_ids]
			context = " ".join(context_words)
			for word_id in log_p_word_context[context_ids]:
				word = id_to_word[word_id] if word_id != eos_id else ngram_eos
				ngram_stream.write("%g\t%s\n" % (
					numpy.log10(numpy.exp(log_p_word_context[context_ids][word_id] - log_p_context[context_ids])),
					context + " " + word))

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument("--probability_cache_directory", dest="probability_cache_directory", action='store',
	                          default=None,
	                          help="probability cache directory [None]")
	model_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                          help="input directory [None]")

	model_parser.add_argument('--random_seed', dest="random_seed", type=int, default=-1,
	                          help='random seed (default: -1=time)')
	model_parser.add_argument('--subset', dest="subset", type=int, default=0,
	                          help='subset (default: 0=total)')
	model_parser.add_argument('--number_of_candidates', dest="number_of_candidates", type=int, default=0,
	                          help='number of candidates (default: -1=time)')

	return model_parser


def validate_options(arguments):
	#arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	#arguments.device = "cpu"
	#arguments.device = torch.device(arguments.device)

	assert os.path.exists(arguments.data_directory)
	assert os.path.exists(arguments.probability_cache_directory)
	if not os.path.exists(arguments.output_directory):
		os.mkdir(arguments.output_directory)

	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond
	assert arguments.subset >= 0
	# assert arguments.number_of_candidates >= 0
	# assert arguments.context_window > 0
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
