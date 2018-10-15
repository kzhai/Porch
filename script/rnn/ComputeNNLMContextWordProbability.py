# import logging
import datetime
import os
import sys
import timeit

import numpy
import scipy
import scipy.misc


# import scipy
# import scipy.sparse
# import torch


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

			if outputs_cache is not None:
				context_candidates[context_ids].add(word_id)
				if number_of_candidates > 0:
					for id in outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
						context_candidates[context_ids].add(id)

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

candidates_by_context = "context"
candidates_by_sample = "sample"
streaming_mode = "none"


def renormalize_ngrams(data_sequence, outputs_cache, context_window_size, id_to_word, eos_id,
                       temperatures,
                       streaming_mode=streaming_mode,
                       number_of_candidates=0):
	# assert normalize_mode == candidates_by_none or normalize_mode == candidates_by_sample or normalize_mode == candidates_by_context
	if streaming_mode:
		context_candidates = {}
	else:
		context_candidates = generate_candidates(data_sequence=data_sequence,
		                                         context_window_size=context_window_size,
		                                         eos_id=eos_id,
		                                         outputs_cache=outputs_cache,
		                                         number_of_candidates=number_of_candidates
		                                         )

	log_p_word_context = {}
	log_p_context = {}
	log_normalizers = {}
	# log_normalizers[0] = scipy.misc.logsumexp(numpy.log(outputs_cache[0]) / temperature)
	log_normalizers[0] = scipy.misc.logsumexp(outputs_cache[0] / temperatures[context_window_size])
	context_window = [data_sequence[1]]
	# context_log_prob = numpy.log(outputs_cache[0][data_sequence[1]]) - log_normalizers[0]
	context_log_prob = outputs_cache[0][data_sequence[1]] - log_normalizers[0]
	for i in range(2, len(data_sequence)):
		assert len(context_window) <= context_window_size

		# log_normalizers[i - 1] = scipy.misc.logsumexp(numpy.log(outputs_cache[i - 1]) / temperature)
		log_normalizers[i - 1] = scipy.misc.logsumexp(outputs_cache[i - 1] / temperatures[context_window_size])

		if len(context_window) == context_window_size:
			context_ids = tuple(context_window)
			if streaming_mode:
				if context_ids not in context_candidates:
					context_candidates[context_ids] = set()
				if context_ids not in log_p_context:
					log_p_context[context_ids] = -1e3
					log_p_word_context[context_ids] = {}

				log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], context_log_prob)

				word_candidates = set()
				word_candidates.add(data_sequence[i])
				if number_of_candidates > 0:
					for id in outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
						# outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
						word_candidates.add(id)

				# log_normalizers[i - 1] = numpy.log(numpy.sum(outputs_cache[i - 1][list(word_candidates)]) + spare_probs)

				for candidate_id in word_candidates:
					if candidate_id not in log_p_word_context[context_ids]:
						log_p_word_context[context_ids][candidate_id] = -1e3
					log_p_word_context[context_ids][candidate_id] = numpy.logaddexp(
						log_p_word_context[context_ids][candidate_id],
						context_log_prob + outputs_cache[i - 1][candidate_id] / temperatures[context_window_size] -
						log_normalizers[i - 1]
					)

			else:
				assert context_ids in context_candidates, ([id_to_word[id] for id in context_ids])
				if context_ids not in log_p_context:
					log_p_context[context_ids] = -1e3
					log_p_word_context[context_ids] = {word_id: -1e3 for word_id in context_candidates[context_ids]}

				log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], context_log_prob)
				for candidate_id in context_candidates[context_ids]:
					# log_p_word_context[context_ids][candidate_id] = numpy.logaddexp(
					# log_p_word_context[context_ids][candidate_id],
					# context_log_prob + numpy.log(outputs_cache[i - 1][candidate_id]) / temperature -
					# log_normalizers[i - 1])
					log_p_word_context[context_ids][candidate_id] = numpy.logaddexp(
						log_p_word_context[context_ids][candidate_id],
						context_log_prob + outputs_cache[i - 1][candidate_id] / temperatures[context_window_size] -
						log_normalizers[i - 1])

				# @TODO: change
				'''
				interpolation = False
				if interpolation:
					temp_context_log_prob = context_log_prob
					for temp_pos in range(len(context_window), 0, -1):
						temp_context_log_prob -= numpy.log(
							outputs_cache[i - temp_pos][data_sequence[i - temp_pos + 1]])
						# if (i - temp_pos) in log_normalizers:
						# context_log_prob += log_normalizers[i - context_window_size]

						log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], temp_context_log_prob)
						for candidate_id in context_candidates[context_ids]:
							log_p_word_context[context_ids][candidate_id] = numpy.logaddexp(
								log_p_word_context[context_ids][candidate_id],
								temp_context_log_prob + numpy.log(outputs_cache[i - 1][candidate_id]))
				'''

		#
		#
		#

		if data_sequence[i] == eos_id:
			context_window.clear()
			context_window.append(data_sequence[i])
			# context_log_prob = numpy.log(outputs_cache[i - 1][data_sequence[i]])
			# context_log_prob = numpy.log(outputs_cache[i - 1][data_sequence[i]]) / temperature - log_normalizers[i - 1]
			context_log_prob = outputs_cache[i - 1][data_sequence[i]] / temperatures[context_window_size] - \
			                   log_normalizers[i - 1]
		else:
			if len(context_window) == context_window_size:
				context_window.pop(0)
				# context_log_prob -= numpy.log(outputs_cache[i - context_window_size - 1][data_sequence[i - context_window_size]])
				# context_log_prob -= numpy.log(
				# outputs_cache[i - context_window_size - 1][data_sequence[i - context_window_size]]) / temperature - \
				# log_normalizers[i - context_window_size - 1]
				context_log_prob -= outputs_cache[i - context_window_size - 1][data_sequence[i - context_window_size]] \
				                    / temperatures[context_window_size] - log_normalizers[i - context_window_size - 1]
			context_window.append(data_sequence[i])
			# context_log_prob += numpy.log(outputs_cache[i - 1][data_sequence[i]])
			# context_log_prob += numpy.log(outputs_cache[i - 1][data_sequence[i]]) / temperature - log_normalizers[i - 1]
			context_log_prob += outputs_cache[i - 1][data_sequence[i]] / temperatures[context_window_size] - \
			                    log_normalizers[i - 1]
		assert context_log_prob < 0, (context_log_prob, context_window)

		if (i + 1) % 100000 == 0:
			print("processed %d %d-grams..." % (len(log_p_context), context_window_size + 1))

	print("processed %d %d-grams..." % (len(log_p_context), context_window_size + 1))

	return log_p_word_context, log_p_context


def assert_test(log_p_word_context, log_p_context, context_ids, id_to_word=None):
	temp_log_total = -1e3
	for word_id in log_p_word_context[context_ids]:
		assert log_p_word_context[context_ids][word_id] - log_p_context[context_ids] <= 0
		temp_log_total = numpy.logaddexp(temp_log_total, log_p_word_context[context_ids][word_id])
	assert temp_log_total <= log_p_context[context_ids], (
		" ".join([id_to_word[temp_id] for temp_id in context_ids]), numpy.exp(temp_log_total),
		numpy.exp(log_p_context[context_ids]))


def verify_ngrams(input_file):
	input_stream = open(input_file, 'r')
	context_prob = {}
	for line in input_stream:
		line = line.strip()
		fields = line.split("\t")
		if len(fields) <= 1:
			continue
		context = " ".join(fields[1].split()[:-1])
		prob = float(fields[0])
		if context not in context_prob:
			context_prob[context] = []
		context_prob[context].append(prob)

	for context in context_prob:
		probs = context_prob[context]
		probs = numpy.asarray(probs)
		prob_sum = numpy.sum(numpy.power(10, probs))
		if prob_sum <= 1:
			continue
		print("warning:", context, prob_sum)

	return


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
	# torch.manual_seed(settings.random_seed)

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
	streaming_mode = settings.streaming
	temperatures = settings.temperatures

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
	'''
	log_p_word = {word_id: -1e3 for word_id in id_to_word}
	for i in range(len(data_sequence)):
		for word_id in id_to_word:
			log_p_word[word_id] = numpy.logaddexp(log_p_word[word_id],
			                                      numpy.log(outputs_cache[i][word_id] / len(data_sequence)))
		if (i + 1) % 10000 == 0:
			print("processed %d 1-grams..." % (i + 1))

	ngram_file = os.path.join(settings.output_directory, "ngram=1.txt")
	ngram_stream = open(ngram_file, 'w')
	ngram_stream.write("\\1-grams:\n")
	ngram_stream.write("%g\t%s\n" % (-99, ngram_sos))
	for word_id in log_p_word:
		word = id_to_word[word_id] if word_id != eos_id else ngram_eos
		ngram_stream.write("%g\t%s\n" % (numpy.log10(numpy.exp(log_p_word[word_id])), word))
	'''
	#
	#
	#

	for context_window_size in range(1, 9):
		#
		#
		#
		log_p_word_context, log_p_context = renormalize_ngrams(data_sequence=data_sequence,
		                                                       outputs_cache=outputs_cache,
		                                                       context_window_size=context_window_size,
		                                                       id_to_word=id_to_word,
		                                                       eos_id=eos_id,
		                                                       temperatures=temperatures,
		                                                       streaming_mode=streaming_mode,
		                                                       number_of_candidates=number_of_candidates,
		                                                       )
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
				# log_prob = numpy.log10(numpy.exp(log_p_word_context[context_ids][word_id] - log_p_context[context_ids]))
				assert log_p_word_context[context_ids][word_id] - log_p_context[context_ids] <= 0
				log_prob = (log_p_word_context[context_ids][word_id] - log_p_context[context_ids]) / numpy.log(10)
				assert log_prob <= 0
				if log_prob > 0:
					sys.stdout.write("warning: %g\t%s\n" % (log_prob, context + " " + word))
				ngram_stream.write("%g\t%s\n" % (log_prob, context + " " + word))

		#
		#
		#
		# assert_test(log_p_word_context, log_p_context, context_ids, id_to_word)

		verify_ngrams(ngram_file)

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
	model_parser.add_argument('--subset', dest="subset", type=int, default=0, help='subset (default: 0=total)')
	model_parser.add_argument('--temperatures', dest="temperatures", default="1", help='temperatures (default: 1)')
	model_parser.add_argument('--number_of_candidates', dest="number_of_candidates", type=int, default=0,
	                          help='number of candidates (default: 0=observables)')
	model_parser.add_argument('--streaming', dest="streaming", action='store_true', default=False,
	                          help='streaming mode (default: False)')

	return model_parser


def validate_options(arguments):
	# arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	# arguments.device = "cpu"
	# arguments.device = torch.device(arguments.device)

	assert os.path.exists(arguments.data_directory)
	assert os.path.exists(arguments.probability_cache_directory)
	if not os.path.exists(arguments.output_directory):
		os.mkdir(arguments.output_directory)

	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond
	assert arguments.subset >= 0
	# assert arguments.streaming_mode in set([streaming_mode, candidates_by_sample, candidates_by_context])
	# assert arguments.number_of_candidates >= 0
	# assert arguments.context_window > 0
	temperatures = [float(temp) for temp in arguments.temperatures.split(",")]
	if len(temperatures) == 1:
		temperatures *= 9
	arguments.temperatures = temperatures
	assert len(arguments.temperatures) == 9
	for temperature in arguments.temperatures:
		assert temperature > 0

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


def backup_renormalize_ngrams(data_sequence, outputs_cache, context_window_size, id_to_word, eos_id,
                              normalize_mode=streaming_mode,
                              number_of_candidates=0):
	assert normalize_mode == streaming_mode or normalize_mode == candidates_by_sample or normalize_mode == candidates_by_context
	if normalize_mode == streaming_mode or normalize_mode == candidates_by_context:
		context_candidates = generate_candidates(data_sequence=data_sequence,
		                                         context_window_size=context_window_size,
		                                         eos_id=eos_id,
		                                         outputs_cache=outputs_cache,
		                                         number_of_candidates=number_of_candidates
		                                         )
	else:
		context_candidates = {}

	log_p_word_context = {}
	log_p_context = {}
	# if normalize_mode == candidates_by_sample or normalize_mode == candidates_by_context:
	log_normalizers = {}
	context_window = [data_sequence[1]]
	context_log_prob = numpy.log(outputs_cache[0][data_sequence[1]])
	for i in range(2, len(data_sequence)):
		assert len(context_window) <= context_window_size

		if len(context_window) == context_window_size:
			context_ids = tuple(context_window)
			if normalize_mode == streaming_mode or normalize_mode == candidates_by_context:
				assert context_ids in context_candidates, ([id_to_word[id] for id in context_ids])
				if context_ids not in log_p_context:
					log_p_context[context_ids] = -1e3
					log_p_word_context[context_ids] = {word_id: -1e3 for word_id in context_candidates[context_ids]}
			else:  # normalize == sample
				if context_ids not in context_candidates:
					context_candidates[context_ids] = set()
				if context_ids not in log_p_context:
					log_p_context[context_ids] = -1e3
					log_p_word_context[context_ids] = {}

			if normalize_mode == candidates_by_context:
				log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], context_log_prob)

				spare_probs = numpy.min(
					outputs_cache[i - 1][list(context_candidates[context_ids])]) / context_window_size
				spare_probs = 0
				log_normalizers[i - 1] = numpy.log(
					numpy.sum(outputs_cache[i - 1][list(context_candidates[context_ids])]) + spare_probs)

				for candidate_id in context_candidates[context_ids]:
					log_p_word_context[context_ids][candidate_id] = numpy.logaddexp(
						log_p_word_context[context_ids][candidate_id],
						context_log_prob + numpy.log(outputs_cache[i - 1][candidate_id]) - log_normalizers[i - 1])

				'''
				if " ".join([id_to_word[temp_id] for temp_id in context_ids]) == "study":
					total_temp1 = 0
					total_temp2 = 0
					normalizer = numpy.sum(outputs_cache[i - 1][list(context_candidates[context_ids])])
					for candidate_id in context_candidates[context_ids]:
						print(id_to_word[candidate_id], numpy.log(outputs_cache[i - 1][candidate_id]),
						      log_normalizers[i - 1])
						total_temp1 += numpy.exp(numpy.log(outputs_cache[i - 1][candidate_id]) - log_normalizers[i - 1])
						total_temp2 += outputs_cache[i - 1][candidate_id] / normalizer
					print("total", total_temp1, total_temp2,
					      numpy.sum(outputs_cache[i - 1][list(context_candidates[context_ids])]),
					      numpy.log(numpy.sum(outputs_cache[i - 1][list(context_candidates[context_ids])])))
				'''
			elif normalize_mode == candidates_by_sample:
				log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], context_log_prob)

				word_candidates = set()
				word_candidates.add(data_sequence[i])
				if number_of_candidates > 0:
					for id in outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
						# outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
						word_candidates.add(id)
				'''
				if argsorts[number_of_candidates] == data_sequence[i]:
					spare_probs = outputs_cache[i - 1][argsorts[number_of_candidates + 1]]
				else:
					spare_probs = outputs_cache[i - 1][argsorts[number_of_candidates]]
				'''
				spare_probs = outputs_cache[i - 1][data_sequence[i]]
				# negative_samples = [random.randrange(len(id_to_word)) for temp in range(100)]
				# spare_probs = numpy.sum(outputs_cache[i - 1][negative_samples])
				# print(negative_samples, spare_probs)

				log_normalizers[i - 1] = numpy.log(numpy.sum(outputs_cache[i - 1][list(word_candidates)]) + spare_probs)

				'''
				temp_log_total = -1e3
				for temp_word_candidate in word_candidates:
					assert numpy.exp(log_normalizers[i - 1]) > outputs_cache[i - 1][temp_word_candidate], (
						log_normalizers[i - 1], numpy.exp(log_normalizers[i - 1]), id_to_word[temp_word_candidate],
						outputs_cache[i - 1][temp_word_candidate], spare_probs)
					temp_log_total = numpy.logaddexp(temp_log_total,
					                                 numpy.log(outputs_cache[i - 1][temp_word_candidate]))
				assert temp_log_total <= numpy.exp(log_normalizers[i - 1]), (
					" ".join([id_to_word[temp_id] for temp_id in context_ids]), numpy.exp(temp_log_total),
					numpy.exp(log_p_context[context_ids]))
				'''

				for candidate_id in word_candidates:
					if candidate_id not in log_p_word_context[context_ids]:
						log_p_word_context[context_ids][candidate_id] = -1e3
					log_p_word_context[context_ids][candidate_id] = numpy.logaddexp(
						log_p_word_context[context_ids][candidate_id],
						context_log_prob + numpy.log(outputs_cache[i - 1][candidate_id]) - log_normalizers[i - 1])
					'''
					assert log_p_word_context[context_ids][candidate_id] < log_p_context[context_ids], (
						log_p_word_context[context_ids][candidate_id], log_p_context[context_ids],
						id_to_word[candidate_id], " ".join([id_to_word[temp_id] for temp_id in context_window]))

					print(" ".join([id_to_word[temp_id] for temp_id in context_ids]), spare_probs,
					      " ".join([id_to_word[temp_id] for temp_id in log_p_word_context[context_ids]]),
					      id_to_word[candidate_id], context_log_prob, numpy.log(outputs_cache[i - 1][candidate_id]),
					      log_normalizers[i - 1])
					assert_test(log_p_word_context, log_p_context, context_ids, id_to_word)
					'''
			else:  # normalize == none
				log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], context_log_prob)
				for candidate_id in context_candidates[context_ids]:
					log_p_word_context[context_ids][candidate_id] = numpy.logaddexp(
						log_p_word_context[context_ids][candidate_id],
						context_log_prob + numpy.log(outputs_cache[i - 1][candidate_id]))

				interpolation = True
				if interpolation:
					temp_context_log_prob = context_log_prob
					for temp_pos in range(len(context_window), 0, -1):
						temp_context_log_prob -= numpy.log(
							outputs_cache[i - temp_pos][data_sequence[i - temp_pos + 1]])
						if (i - temp_pos) in log_normalizers:
							context_log_prob += log_normalizers[i - context_window_size]

						log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], temp_context_log_prob)
						for candidate_id in context_candidates[context_ids]:
							log_p_word_context[context_ids][candidate_id] = numpy.logaddexp(
								log_p_word_context[context_ids][candidate_id],
								temp_context_log_prob + numpy.log(outputs_cache[i - 1][candidate_id]))

			#
			#
			#
			'''
			print(" ".join([id_to_word[temp_id] for temp_id in context_ids]), spare_probs,
			      " ".join([id_to_word[temp_id] for temp_id in log_p_word_context[context_ids]]),
			      )
			assert_test(log_p_word_context, log_p_context, context_ids, id_to_word)
			'''

		if data_sequence[i] == eos_id:
			context_window.clear()
			context_window.append(data_sequence[i])
			context_log_prob = numpy.log(outputs_cache[i - 1][data_sequence[i]])
			if normalize_mode != streaming_mode and ((i - 1) in log_normalizers):
				context_log_prob -= log_normalizers[i - 1]
		else:
			if len(context_window) == context_window_size:
				context_window.pop(0)
				context_log_prob -= numpy.log(
					outputs_cache[i - context_window_size - 1][data_sequence[i - context_window_size]])
				if normalize_mode != streaming_mode and ((i - context_window_size - 1) in log_normalizers):
					context_log_prob += log_normalizers[i - context_window_size - 1]
			context_window.append(data_sequence[i])
			context_log_prob += numpy.log(outputs_cache[i - 1][data_sequence[i]])
			if normalize_mode != streaming_mode and ((i - 1) in log_normalizers):
				context_log_prob -= log_normalizers[i - 1]
		assert context_log_prob < 0, (context_log_prob, context_window)

		if (i + 1) % 100000 == 0:
			print("processed %d %d-grams..." % (len(log_p_context), context_window_size + 1))

	print("processed %d %d-grams..." % (len(log_p_context), context_window_size + 1))

	return log_p_word_context, log_p_context


if __name__ == '__main__':
	main()
