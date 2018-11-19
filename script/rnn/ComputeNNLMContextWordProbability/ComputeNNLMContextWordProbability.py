# import logging
import os
import timeit

import numpy
import scipy
import scipy.misc

ngram_sos = "<s>"
ngram_eos = "</s>"
nlm_eos = "<eos>"


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
		if len(context_window) > context_window_size:
			context_window.pop(0)

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


def renormalize_unigrams(data_sequence, outputs_cache, id_to_word, temperatures, max_frequency=5,
                         reward_probability=0):
	log_s_table = {}
	count_word = {}
	log_E_c_word = {}
	for word_id in id_to_word:
		log_s_table[word_id] = numpy.zeros(max_frequency) - 1e3
		log_s_table[word_id][0] = 0
		count_word[word_id] = 0
		log_E_c_word[word_id] = -1e3

	# count_word = numpy.zeros(len(id_to_word), dtype=numpy.int)
	# log_p_word = {word_id: -1e3 for word_id in id_to_word}
	# log_E_c_word = numpy.zeros(len(id_to_word)) + -1e3
	for i in range(len(data_sequence)):
		word_id = data_sequence[i]
		count_word[word_id] += 1

		log_normalizers = scipy.misc.logsumexp(outputs_cache[i] / temperatures[0])
		# E_log_c_word = numpy.logaddexp(E_log_c_word, outputs_cache[i] / temperatures[0] - log_normalizers)
		log_E_c_word[word_id] = numpy.logaddexp(log_E_c_word[word_id],
		                                        outputs_cache[i][word_id] / temperatures[0] - log_normalizers)

		sample_log_p_word_context = outputs_cache[i][word_id] / temperatures[0] - log_normalizers

		for temp_r in range(min(max_frequency - 1, count_word[word_id]), -1, -1):
			# assert 0 < 1 - numpy.exp(sample_log_p_word_context) <= 1, (numpy.exp(sample_log_p_word_context), id_to_word[word_id])
			first_term = log_s_table[word_id][temp_r] + numpy.log(1 - numpy.exp(sample_log_p_word_context) + 1e-99)
			second_term = -1e3 if temp_r - 1 < 0 else log_s_table[word_id][temp_r - 1] + sample_log_p_word_context

			log_s_table[word_id][temp_r] = numpy.logaddexp(first_term, second_term)
		del temp_r

		# log_p_word[data_sequence[i]] = numpy.logaddexp(log_p_word[data_sequence[i]], outputs_cache[i][data_sequence[i]] / temperatures[0] - log_normalizers)
		# word_count[data_sequence[i]] += 1
		if (i + 1) % 100000 == 0:
			print("processed %d 1-grams..." % (i + 1))

	'''
	E_log_n_gt0_word = numpy.zeros(len(id_to_word)) + -1e3
	for word_id in id_to_word:
		# E_log_n_gt0_word[word_id] = max(numpy.log(1 - numpy.exp(log_s_table[word_id][0])), scipy.misc.logsumexp(log_s_table[word_id][1:]))
		E_log_n_gt0_word[word_id] = scipy.misc.logsumexp(log_s_table[word_id][1:])
	'''

	return log_E_c_word, count_word, log_s_table


def renormalize_ngrams(data_sequence, outputs_cache, context_window_size, id_to_word, eos_id,
                       temperatures,
                       streaming_mode=streaming_mode,
                       number_of_candidates=0,
                       max_frequency=5,
                       reward_probability=0,
                       log_hidden_probs=None,
                       interpolation_temperatures=None,
                       ):
	# assert normalize_mode == candidates_by_none or normalize_mode == candidates_by_sample or normalize_mode == candidates_by_context
	if streaming_mode:
		context_candidates = {}
	else:
		# candidates_generation_time = timeit.default_timer()
		context_candidates = generate_candidates(data_sequence=data_sequence,
		                                         context_window_size=context_window_size,
		                                         eos_id=eos_id,
		                                         outputs_cache=outputs_cache,
		                                         number_of_candidates=number_of_candidates
		                                         )
	# candidates_generation_time = timeit.default_timer() - candidates_generation_time
	# print("candidates generation time: %.2fs" % candidates_generation_time)

	log_s_table = {}

	E_log_c_context_word = {}
	count_context_word = {}
	# log_p_context = {}
	# count_context = {}

	log_normalizers = {}
	context_window = []
	# log_normalizers[0] = scipy.misc.logsumexp(numpy.log(outputs_cache[0]) / temperature)
	# log_normalizers[0] = scipy.misc.logsumexp(outputs_cache[0] / temperatures[context_window_size])
	# context_window = [data_sequence[1]]
	# context_log_prob = numpy.log(outputs_cache[0][data_sequence[1]]) - log_normalizers[0]
	context_log_prob = 0  # outputs_cache[0][data_sequence[1]] - log_normalizers[0]
	for i in range(1, len(data_sequence)):
		assert len(context_window) <= context_window_size

		# log_normalizers[i - 1] = scipy.misc.logsumexp(numpy.log(outputs_cache[i - 1]) / temperature)
		log_normalizers[i - 1] = scipy.misc.logsumexp(outputs_cache[i - 1] / temperatures[context_window_size])

		if len(context_window) == context_window_size:
			context_ids = tuple(context_window)
			if context_ids not in E_log_c_context_word:
				# log_p_context[context_ids] = -1e3
				E_log_c_context_word[context_ids] = {}
				count_context_word[context_ids] = {}
				# count_context[context_ids] = 0
				log_s_table[context_ids] = {}

			'''
			if data_sequence[i] not in count_word_context[context_ids]:
				count_word_context[context_ids][data_sequence[i]] = 0
			count_word_context[context_ids][data_sequence[i]] += 1
			'''

			if streaming_mode:
				if context_ids not in context_candidates:
					context_candidates[context_ids] = set()

				word_candidates = set()
				word_candidates.add(data_sequence[i])
				if number_of_candidates > 0:
					for id in outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
						# outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
						word_candidates.add(id)
					del id

				# log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], context_log_prob)
				for word_id in word_candidates:
					if word_id not in E_log_c_context_word[context_ids]:
						E_log_c_context_word[context_ids][word_id] = -1e3
				del word_id
			else:
				assert context_ids in context_candidates, ([id_to_word[id] for id in context_ids])
				word_candidates = context_candidates[context_ids]
				for word_id in word_candidates:
					if word_id not in E_log_c_context_word[context_ids]:
						E_log_c_context_word[context_ids][word_id] = -1e3
				del word_id
			# log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], context_log_prob)

			if log_hidden_probs is not None:
				log_hidden_probability = log_hidden_probs[i - context_window_size - 1]
			else:
				log_hidden_probability = 0

			# log_p_context[context_ids] = numpy.logaddexp(log_p_context[context_ids], context_log_prob + log_hidden_probability)

			for word_id in word_candidates:
				# print(id_to_word[word_id])
				if reward_probability > 0:
					if word_id == data_sequence[i]:
						sample_log_p_word_context = context_log_prob + log_hidden_probability + numpy.logaddexp(
							outputs_cache[i - 1][word_id] / temperatures[context_window_size],
							numpy.log(reward_probability)) - numpy.logaddexp(log_normalizers[i - 1],
						                                                     numpy.log(reward_probability))
					else:
						sample_log_p_word_context = context_log_prob + log_hidden_probability + outputs_cache[i - 1][
							word_id] / temperatures[context_window_size] - numpy.logaddexp(
							log_normalizers[i - 1],
							numpy.log(reward_probability))
				else:
					sample_log_p_word_context = context_log_prob + log_hidden_probability + outputs_cache[i - 1][
						word_id] / temperatures[context_window_size] - log_normalizers[i - 1]

				# print(id_to_word[word_id], sample_log_p_word_context)
				E_log_c_context_word[context_ids][word_id] = numpy.logaddexp(
					E_log_c_context_word[context_ids][word_id],
					sample_log_p_word_context
				)

				if word_id not in count_context_word[context_ids]:
					count_context_word[context_ids][word_id] = 0
				count_context_word[context_ids][word_id] += 1
				# count_context[context_ids] += 1

				if word_id not in log_s_table[context_ids]:
					log_s_table[context_ids][word_id] = numpy.zeros(max_frequency) - 1e3
					log_s_table[context_ids][word_id][0] = 0

				assert 0 < 1 - numpy.exp(sample_log_p_word_context) <= 1, numpy.exp(sample_log_p_word_context)
				for temp_r in range(min(max_frequency - 1, count_context_word[context_ids][word_id]), -1, -1):
					first_term = log_s_table[context_ids][word_id][temp_r] + numpy.log(
						1 - numpy.exp(sample_log_p_word_context) + 1e-99)
					second_term = -1e3 if temp_r - 1 < 0 else \
						log_s_table[context_ids][word_id][temp_r - 1] + sample_log_p_word_context

					log_s_table[context_ids][word_id][temp_r] = numpy.logaddexp(first_term, second_term)
				del temp_r
			del word_id

			if i % 100000 == 0:
				print("processed %d tokens..." % (i))
		#
		#
		#

		if data_sequence[i] == eos_id:
			context_window.clear()
			context_log_prob = 0

		context_window.append(data_sequence[i])
		assert outputs_cache[i - 1][data_sequence[i]] / temperatures[context_window_size] <= log_normalizers[i - 1], \
			(outputs_cache[i - 1][data_sequence[i]] / temperatures[context_window_size], log_normalizers[i - 1])
		# context_log_prob = numpy.log(outputs_cache[i - 1][data_sequence[i]])
		# context_log_prob = numpy.log(outputs_cache[i - 1][data_sequence[i]]) / temperature - log_normalizers[i - 1]
		context_log_prob += outputs_cache[i - 1][data_sequence[i]] / temperatures[context_window_size] \
		                    - log_normalizers[i - 1]

		if len(context_window) > context_window_size:
			context_window.pop(0)
			# context_log_prob -= numpy.log(outputs_cache[i - context_window_size - 1][data_sequence[i - context_window_size]])
			# context_log_prob -= numpy.log(
			# outputs_cache[i - context_window_size - 1][data_sequence[i - context_window_size]]) / temperature - \
			# log_normalizers[i - context_window_size - 1]
			context_log_prob -= outputs_cache[i - context_window_size - 1][data_sequence[i - context_window_size]] \
			                    / temperatures[context_window_size] - log_normalizers[i - context_window_size - 1]

		assert context_log_prob <= 0, (
			context_log_prob, context_window, [id_to_word[temp_context_id] for temp_context_id in context_window])

		if (i + 1) % 100000 == 0:
			print("processed %d %d-grams..." % (len(E_log_c_context_word), context_window_size + 1))

	'''
	E_log_n_gt0_context_word = {}
	for context_ids in count_context_word:
		E_log_n_gt0_context_word[context_ids] = {}
		for word_id in count_context_word[context_ids]:
			E_log_n_gt0_context_word[context_ids][word_id] = max(
				numpy.log(1 - numpy.exp(log_s_table[word_id][0])),
				scipy.misc.logsumexp(log_s_table[context_ids][word_id][1:]))
			#E_log_n_gt0_context_word[context_ids][word_id] = scipy.misc.logsumexp(log_s_table[context_ids][word_id][1:])
	'''

	print("processed %d %d-grams..." % (len(E_log_c_context_word), context_window_size + 1))

	return E_log_c_context_word, count_context_word, log_s_table


'''
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
'''


def import_vocabulary(path):
	stream = open(path, 'r')
	word_to_id = {}
	id_to_word = {}
	for line in stream:
		line = line.strip()
		tokens = line.split()
		word_to_id[tokens[0]] = len(word_to_id)
		id_to_word[len(id_to_word)] = tokens[0]

	return word_to_id, id_to_word


def import_output_cache(probability_directory, segment_size=100000, cutoff=0):
	outputs_cache = []
	i = 0
	while (True):
		j = i + segment_size
		input_file = os.path.join(probability_directory, "timestamp=%d-%d.npz" % (i, j - 1))
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

	# import porch.data
	# word_to_id, id_to_word = porch.data.import_vocabulary(os.path.join(settings.data_directory, "type.info"))
	word_to_id, id_to_word = import_vocabulary(os.path.join(settings.data_directory, "type.info"))
	data_sequence = numpy.load(os.path.join(settings.data_directory, "train.npy"))

	subset = settings.subset
	if subset > 0:
		data_sequence = data_sequence[:subset]
	else:
		data_sequence = data_sequence[:-1]

	start_train = timeit.default_timer()

	eos_word = nlm_eos
	eos_id = word_to_id[eos_word]
	number_of_candidates = settings.number_of_candidates
	streaming_mode = settings.stream
	temperatures = settings.temperatures
	output_directory = settings.output_directory
	max_frequency = settings.max_frequency
	backoff_to_uniform = settings.backoff_to_uniform
	# interpolating_mode = settings.interpolate
	# smoothing_mode = settings.smooth
	# interpolation_discount = settings.interpolation_discount
	# non_uniform_mode = settings.non_uniform
	reward_probability = settings.reward_probability
	# breakdown_directory = os.path.join(output_directory, "breakdown")
	# if not os.path.exists(breakdown_directory):
	# os.mkdir(breakdown_directory)

	#
	#
	#
	#
	#
	'''
	span_cache_file = settings.span_cache_file
	if span_cache_file is not None:
		span_cache = numpy.load(span_cache_file)
		print(span_cache.shape)
		span_cache = span_cache[:len(data_sequence) - 8 + 1, :]
		assert span_cache.shape == (len(data_sequence) - 8 + 1, 9), (span_cache.shape, len(data_sequence) - 8 + 1)
		print(span_cache.shape)
	'''

	# from .ComputeNNLMOutputs import import_output_cache
	outputs_cache = import_output_cache(settings.probability_cache_directory, cutoff=len(data_sequence))
	print("successfully load outputs cache...")

	'''
	if non_uniform_mode > 0:
		temp_outputs_cache = numpy.asarray(outputs_cache)
		assert temp_outputs_cache.shape == (len(outputs_cache), len(id_to_word))

		import sklearn
		import sklearn.covariance
		# from sklearn.covanriance import ShrunkCovariance
		cov = sklearn.covariance.ShrunkCovariance().fit(temp_outputs_cache[::non_uniform_mode])

		mean = cov.location_
		covariance = cov.covariance_

		covariance_inverse = numpy.linalg.pinv(covariance)
		print(covariance_inverse.shape)

		temp_diff = temp_outputs_cache - mean
		print(temp_diff.shape)
		log_hidden_probs = -0.5 * numpy.dot(numpy.dot(temp_diff, covariance_inverse), temp_diff.T)[0, :]
		print(log_hidden_probs.shape)
	else:
		log_hidden_probs = None
	'''

	#
	#
	#
	#
	#

	'''
	######## ######## ######## ######## ########
	# this function is optimized with empty context list
	######## ######## ######## ######## ########
	E_log_c_word, count_word, log_s_table_word = renormalize_unigrams(
		data_sequence,
		outputs_cache,
		id_to_word,
		temperatures,
		max_frequency=max_frequency)

	# log_E_n_ge1_word = - 1e3
	log_p_c_gt0_word = numpy.zeros(len(id_to_word)) - 1e3
	for word_id in id_to_word:
		log_p_c_gt0_word[word_id] = max(
			numpy.log(1 - numpy.exp(log_s_table_word[word_id][0]) + 1e-99),
			scipy.misc.logsumexp(log_s_table_word[word_id][1:]))
	# log_E_n_ge1_word = numpy.logaddexp(log_E_n_ge1_word, log_p_c_gt0_word[word_id])

	log_p_c_gt0_word -= scipy.misc.logsumexp(log_p_c_gt0_word)
	assert len(log_p_c_gt0_word) == len(id_to_word)
	assert numpy.all(log_p_c_gt0_word <= 0)
	ngram_file = os.path.join(output_directory, "ngram=1.txt")
	ngram_stream = open(ngram_file, 'w')
	ngram_stream.write("\\1-grams:\n")
	ngram_stream.write("%g\t%s\n" % (-99, ngram_sos))
	for word_id in id_to_word:
		word = id_to_word[word_id] if word_id != eos_id else ngram_eos
		# print(word_id, word)
		# print(log_p_word[word_id], word)
		ngram_stream.write("%g\t%s\n" % (log_p_c_gt0_word[word_id] / numpy.log(10), word))
	ngram_stream.close()

	previous_log_p_word_given_context = {(): log_p_c_gt0_word}
	# verify_ngrams(ngram_file)
	'''

	for context_window_size in range(0, 9):
		renorm_time = timeit.default_timer()
		if context_window_size == 0:
			######## ######## ######## ######## ########
			# this function is optimized with empty context list
			######## ######## ######## ######## ########
			E_log_c_word, count_word, log_s_table_word = renormalize_unigrams(
				data_sequence,
				outputs_cache,
				id_to_word,
				temperatures,
				max_frequency=max_frequency)
			# log_E_c_context_word = {(): {word_id: E_log_c_word[word_id] for word_id in range(len(E_log_c_word))}}
			# count_context_word = {(): {word_id: count_word[word_id] for word_id in range(len(count_word))}}
			log_E_c_context_word = {(): E_log_c_word}
			count_context_word = {(): count_word}
			log_s_table_context_word = {(): log_s_table_word}
		else:
			log_E_c_context_word, count_context_word, log_s_table_context_word = renormalize_ngrams(
				data_sequence=data_sequence,
				outputs_cache=outputs_cache,
				context_window_size=context_window_size,
				id_to_word=id_to_word,
				eos_id=eos_id,
				temperatures=temperatures,
				streaming_mode=streaming_mode,
				number_of_candidates=number_of_candidates,
				max_frequency=max_frequency,
				reward_probability=reward_probability,
				# log_hidden_probs=log_hidden_probs
			)
		renorm_time = timeit.default_timer() - renorm_time
		print("renorm %d-grams in %.2fs..." % (context_window_size + 1, renorm_time))

		'''
		if span_cache_file is not None:
			context_window = [data_sequence[temp_index] for temp_index in range(context_window_size)]
			for i in range(len(data_sequence) - 8):
				context_ids = tuple(context_window)
				word_id = data_sequence[i + context_window_size]

				if (context_ids in log_p_context) and (word_id in log_p_word_context[context_ids]):
					temp_log_prob_context = numpy.sum(span_cache[i, :context_window_size])
					log_p_context[context_ids] = numpy.logaddexp(
						log_p_context[context_ids],
						numpy.log(len(data_sequence) / count_word_context[context_ids][word_id]) +
						temp_log_prob_context)

					log_p_word_context[context_ids][word_id] = numpy.logaddexp(
						log_p_word_context[context_ids][word_id],
						numpy.log(len(data_sequence) / count_word_context[context_ids][word_id]) +
						temp_log_prob_context + span_cache[i, context_window_size])

				context_window.pop(0)
				context_window.append(data_sequence[i + context_window_size])
		'''

		post_renorm_time = timeit.default_timer()

		log_E_n_ge1_context = {}
		log_p_c_gt0_context_word = {}
		log_p_c_eq0_context_word = {}
		for context_ids in log_s_table_context_word:
			log_E_n_ge1_context[context_ids] = -1e3
			log_p_c_gt0_context_word[context_ids] = {}
			log_p_c_eq0_context_word[context_ids] = {}
			# for word_id in id_to_word:
			for word_id in log_s_table_context_word[context_ids]:
				'''
				if word_id in log_s_table_context_word[context_ids]:
					log_p_c_gt0_context_word[context_ids][word_id] = max(
						numpy.log(1 - numpy.exp(log_s_table_context_word[context_ids][word_id][0]) + 1e-99),
						scipy.misc.logsumexp(log_s_table_context_word[context_ids][word_id][1:]))
					assert (not numpy.isnan(log_p_c_gt0_context_word[context_ids][word_id]))
				else:
					log_p_c_gt0_context_word[context_ids][word_id] = -1e3
				'''

				log_p_c_gt0_context_word[context_ids][word_id] = max(
					numpy.log(1 - numpy.exp(log_s_table_context_word[context_ids][word_id][0]) + 1e-99),
					scipy.misc.logsumexp(log_s_table_context_word[context_ids][word_id][1:]))
				assert (not numpy.isnan(log_p_c_gt0_context_word[context_ids][word_id]))

				log_p_c_eq0_context_word[context_ids][word_id] = log_s_table_context_word[context_ids][word_id][0]

				log_E_n_ge1_context[context_ids] = numpy.logaddexp(
					log_E_n_ge1_context[context_ids],
					log_p_c_gt0_context_word[context_ids][word_id]
					# scipy.misc.logsumexp(log_s_table_context_word[context_ids][word_id][1:])
				)
				assert (not numpy.isnan(log_E_n_ge1_context[context_ids]))

		if context_window_size == 0 and (not backoff_to_uniform):
			for context_ids in log_p_c_gt0_context_word:
				for word_id in log_p_c_gt0_context_word[context_ids]:
					log_p_c_gt0_context_word[context_ids][word_id] -= log_E_n_ge1_context[context_ids]
			log_p_word_given_context = log_p_c_gt0_context_word
		else:
			E_n_1 = 1e-99
			E_n_2 = 1e-99
			for context_ids in log_s_table_context_word:
				for word_id in log_s_table_context_word[context_ids]:
					E_n_1 += numpy.exp(log_s_table_context_word[context_ids][word_id][1])
					E_n_2 += numpy.exp(log_s_table_context_word[context_ids][word_id][2])
			interpolation_discount = E_n_1 / (E_n_1 + 2 * E_n_2)
			# print(E_n_1, E_n_2)
			print("%d-gram interpolation discount: %g" % (context_window_size + 1, interpolation_discount))

			log_E_c_context = {}
			# log_E_c_context_word_tilda = {}#[context_ids][word_id]
			for context_ids in log_s_table_context_word:
				temp_context_ids = context_ids[1:]
				log_E_c_context[context_ids] = -1e3
				# log_E_c_context_word_tilda[context_ids] = {}
				# for word_id in log_E_c_context_word[context_ids]:
				for word_id in id_to_word:
					if context_window_size == 0 and backoff_to_uniform:
						log_p_word_given_temp_context = -numpy.log(len(id_to_word))
					else:
						log_p_word_given_temp_context = log_p_word_given_context[temp_context_ids][word_id]
					assert (not numpy.isnan(log_p_word_given_temp_context))

					if word_id in log_s_table_context_word[context_ids]:
						log_E_c_context_word[context_ids][word_id] = numpy.log(
							max(0, numpy.exp(log_E_c_context_word[context_ids][word_id])
							    - interpolation_discount * numpy.exp(log_p_c_gt0_context_word[context_ids][word_id]))
							+ interpolation_discount * numpy.exp(
								log_E_n_ge1_context[context_ids] + log_p_word_given_temp_context)
							#+ interpolation_discount * numpy.exp(
								#log_p_c_eq0_context_word[context_ids][word_id] + log_p_word_given_temp_context)
						)
					else:
						log_E_c_context_word[context_ids][word_id] = numpy.log(
							+ interpolation_discount * numpy.exp(
								log_E_n_ge1_context[context_ids] + log_p_word_given_temp_context)
							#+ interpolation_discount * numpy.exp(log_p_word_given_temp_context)
						)

					log_E_c_context[context_ids] = numpy.logaddexp(
						log_E_c_context[context_ids],
						log_E_c_context_word[context_ids][word_id]
					)

					assert (not numpy.isnan(log_E_c_context_word[context_ids][word_id]))
					assert (not numpy.isnan(log_E_c_context[context_ids]))

					'''
					log_p_word_context[context_ids][word_id] = numpy.log(
						numpy.exp(log_p_word_context[context_ids][word_id])
						- (1 - numpy.exp(log_s_table[context_ids][word_id][0])) * interpolation_discount
						+ interpolation_discount
					)
					
					assert log_p_word_context[context_ids][word_id] <= log_p_context[context_ids], (
						log_p_word_context[context_ids][word_id], log_p_context[context_ids],
						" ".join([id_to_word[context_id] if context_id != eos_id else ngram_sos for context_id in
						          context_ids]),
						id_to_word[word_id] if word_id != eos_id else ngram_eos
					)
					'''

			for context_ids in log_s_table_context_word:
				for word_id in id_to_word:
					log_E_c_context_word[context_ids][word_id] -= log_E_c_context[context_ids]
			log_p_word_given_context = log_E_c_context_word

		ngram_file = os.path.join(output_directory, "ngram=%d.txt" % (context_window_size + 1))
		ngram_stream = open(ngram_file, 'w')
		ngram_stream.write("\\%d-grams:\n" % (context_window_size + 1))
		if context_window_size == 0:
			ngram_stream.write("%g\t%s\n" % (-99, ngram_sos))

		for context_ids in log_p_word_given_context:
			context_words = [id_to_word[context_id] if context_id != eos_id else ngram_sos for context_id in
			                 context_ids]
			for word_id in id_to_word:
				if word_id not in log_s_table_context_word[context_ids]:
					continue
				word = [id_to_word[word_id] if word_id != eos_id else ngram_eos]
				ngram_stream.write("%g\t%s\n" % (
					log_p_word_given_context[context_ids][word_id] / numpy.log(10), " ".join(context_words + word)))
		ngram_stream.close()

		post_renorm_time = timeit.default_timer() - post_renorm_time
		print("post renorm %d-grams in %.2fs..." % (context_window_size + 1, post_renorm_time))

	# verify_ngrams(ngram_file)

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument("--probability_cache_directory", dest="probability_cache_directory", action='store',
	                          default=None,
	                          help="probability cache directory [None]")
	# model_parser.add_argument("--span_cache_file", dest="span_cache_file", action='store', default=None, help="ngram count directory [None]")
	model_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                          help="input directory [None]")

	model_parser.add_argument('--subset', dest="subset", action='store', type=int, default=0,
	                          help='subset (default: 0=total)')
	model_parser.add_argument('--temperatures', dest="temperatures", action='store', default="1",
	                          help='temperatures (default: 1, higher value --> more flat distribution)')
	model_parser.add_argument('--number_of_candidates', dest="number_of_candidates", type=int, default=0,
	                          help='number of candidates (default: 0=observables)')
	model_parser.add_argument('--max_frequency', dest="max_frequency", action='store', type=int, default=5,
	                          help='max_frequency (default: 5)')

	model_parser.add_argument('--stream', dest="stream", action='store', default="False",
	                          help='streaming mode (default: False)')
	model_parser.add_argument('--backoff_to_uniform', dest="backoff_to_uniform", action='store', default="False",
	                          help='backoff to uniform mode (default: False)')
	'''
	model_parser.add_argument('--interpolate', dest="interpolate", action='store', default="False",
	                          help='interpolating mode (default: False)')
	model_parser.add_argument('--smooth', dest="smooth", action='store', default="False",
	                          help='background unigram smooth (default: False)')
	model_parser.add_argument('--non_uniform', dest="non_uniform", action='store', type=int, default=0,
	                          help='non-uniform (default: 0)')

	model_parser.add_argument('--interpolation_discount', dest="interpolation_discount", action='store', type=float,
	                          default=0,
	                          help='interpolation discount (default: 0)')
	'''
	model_parser.add_argument('--reward_probability', dest="reward_probability", action='store', type=float, default=0,
	                          help='reward probability for inverse smoothing (default: 0)')

	return model_parser


def validate_options(arguments):
	# arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	# arguments.device = "cpu"
	# arguments.device = torch.device(arguments.device)

	assert os.path.exists(arguments.data_directory)
	assert os.path.exists(arguments.probability_cache_directory)
	# assert (arguments.span_cache_file is None) or os.path.exists(arguments.span_cache_file)
	if not os.path.exists(arguments.output_directory):
		os.mkdir(arguments.output_directory)

	assert arguments.subset >= 0
	# assert arguments.streaming_mode in set([streaming_mode, candidates_by_sample, candidates_by_context])
	assert arguments.number_of_candidates >= 0
	assert arguments.max_frequency >= 3
	# assert arguments.context_window > 0
	temperatures = [float(temp) for temp in arguments.temperatures.split("-")]
	if len(temperatures) == 1:
		temperatures *= 9
	arguments.temperatures = temperatures
	assert len(arguments.temperatures) == 9
	for temperature in arguments.temperatures:
		assert temperature > 0

	if arguments.stream.lower() == "false":
		arguments.stream = False
	elif arguments.stream.lower() == "true":
		arguments.stream = True
	else:
		raise ValueError()

	if arguments.backoff_to_uniform.lower() == "false":
		arguments.backoff_to_uniform = False
	elif arguments.backoff_to_uniform.lower() == "true":
		arguments.backoff_to_uniform = True
	else:
		raise ValueError()

	'''
	if arguments.smooth.lower() == "false":
		arguments.smooth = False
	elif arguments.smooth.lower() == "true":
		arguments.smooth = True
	else:
		raise ValueError()

	arguments.non_uniform >= 0
	assert arguments.interpolation_discount >= 0
	'''

	assert arguments.reward_probability >= 0

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
