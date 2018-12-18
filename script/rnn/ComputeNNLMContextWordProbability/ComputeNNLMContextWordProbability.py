"""
Add a trie structure to store the backtracking probabilities.
Add a misc column in s-table for faster computing.
"""

import linecache
import multiprocessing
# import logging
import os
import queue
import sys
import timeit
import tracemalloc

import numpy
import scipy
import scipy.misc

ngram_sos = "<s>"
ngram_eos = "</s>"
nlm_eos = "<eos>"


def display_top(snapshot, key_type='lineno', limit=3):
	snapshot = snapshot.filter_traces((
		tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
		tracemalloc.Filter(False, "<unknown>"),
	))
	top_stats = snapshot.statistics(key_type)

	print("Top %s lines" % limit)
	for index, stat in enumerate(top_stats[:limit], 1):
		frame = stat.traceback[0]
		# replace "/path/to/module/file.py" with "module/file.py"
		filename = os.sep.join(frame.filename.split(os.sep)[-2:])
		print("#%s: %s:%s: %.1f KiB"
		      % (index, filename, frame.lineno, stat.size / 1024))
		line = linecache.getline(frame.filename, frame.lineno).strip()
		if line:
			print('    %s' % line)

	other = top_stats[limit:]
	if other:
		size = sum(stat.size for stat in other)
		print("%s other: %.1f KiB" % (len(other), size / 1024))
	total = sum(stat.size for stat in top_stats)
	print("Total allocated size: %.1f KiB" % (total / 1024))
	sys.stdout.flush()


class TrieNode():
	def __init__(self,
	             max_frequency=5
	             ):
		self._children = {}
		# self._backoff_node = None

		self._count = 0

		self._log_s_table = numpy.zeros(1 + max_frequency) - 1e3
		self._log_s_table[1] = 0

		self._log_E_c = -1e3

		# self._log_sum_E_n_ge1 = None
		# self._log_p_c_eq0 = Nonee
		# self._log_p_c_gt0 = None

		# self._log_E_c_tilda = None
		# self._log_sum_E_c_tilda = None

		self._log_p_word = {}


def find(node, context_ids):
	for context_id in context_ids:
		if context_id not in node._children:
			return None
		node = node._children[context_id]
	return node


def dfs(root, context_ids=[], display_type=3):
	'''
	print(", ".join(["%d" % context_id for context_id in context_ids]),
	      ", ".join(["%d:%f" % (temp_word_id, root._log_p_word[temp_word_id]) for temp_word_id in
	                 root._log_p_word]))
	'''
	if display_type == 3:
		print(", ".join(["%d" % context_id for context_id in context_ids]), root._log_sum_E_n_ge1)
	if display_type == 5:
		print(", ".join(["%d" % context_id for context_id in context_ids]), root._log_sum_E_c_tilda)
	for temp_word_id in root._children:
		child = root._children[temp_word_id]
		if display_type == 1:
			print(", ".join(["%d" % context_id for context_id in context_ids]), temp_word_id, child._count,
			      child._log_E_c, child._log_s_table)
		elif display_type == 2:
			print(", ".join(["%d" % context_id for context_id in context_ids]), temp_word_id, child._log_p_c_eq0,
			      child._log_p_c_gt0)
		elif display_type == 4:
			print(", ".join(["%d" % context_id for context_id in context_ids]), temp_word_id, child._log_E_c_tilda)
		dfs(child, context_ids + [temp_word_id], display_type=display_type)


'''
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
			sys.stdout.flush()

	print("collected candidates for %d %d-grams..." % (len(context_candidates), context_window_size + 1))
	sys.stdout.flush()

	return context_candidates
'''

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


def ngram_to_trie(data_sequence,
                  outputs_cache,
                  root,
                  temperatures,
                  id_to_word,
                  number_of_candidates=0,
                  max_frequency=5,
                  min_ngram_order=1,
                  max_ngram_order=9,
                  eos_id=None,
                  ):
	context_buffer_ids = []
	context_buffer_logprobs = []
	word_candidates = set()
	log_p_word = numpy.zeros(len(id_to_word)) + -1e3
	for i in range(0, len(data_sequence)):
		assert len(context_buffer_ids) < max_ngram_order

		word_candidates.clear()
		word_candidates.add(data_sequence[i])
		if number_of_candidates > 0:
			for temp_word_id in outputs_cache[i - 1].argsort()[::-1][:number_of_candidates]:
				word_candidates.add(temp_word_id)

		log_normalizers = scipy.misc.logsumexp(outputs_cache[i - 1] / temperatures[max_ngram_order - 1])
		log_p_word = numpy.logaddexp(log_p_word,
		                             outputs_cache[i - 1] / temperatures[max_ngram_order - 1] - log_normalizers)
		# for j in range(len(context_buffer_ids)):
		for j in range(len(context_buffer_ids) + 2 - min_ngram_order):
			parent = find(root, context_buffer_ids[j:])

			# print(parent, context_buffer_ids, j)
			for temp_word_id in word_candidates:
				if temp_word_id not in parent._children:
					parent._children[temp_word_id] = TrieNode(max_frequency=max_frequency)

				sample_log_p_word_context = outputs_cache[i - 1][temp_word_id] / temperatures[
					max_ngram_order - 1] - log_normalizers
				if j < len(context_buffer_ids):
					sample_log_p_word_context += numpy.sum(context_buffer_logprobs[j:])
				if sample_log_p_word_context > 0:
					print(numpy.exp(sample_log_p_word_context), context_buffer_ids, context_buffer_logprobs)
					sample_log_p_word_context = 0
				assert 0 <= numpy.exp(sample_log_p_word_context) <= 1, numpy.exp(sample_log_p_word_context)

				child = parent._children[temp_word_id]
				child._count += 1
				child._log_E_c = numpy.logaddexp(
					child._log_E_c,
					sample_log_p_word_context
				)
				child._log_s_table[1:] = numpy.logaddexp(
					child._log_s_table[1:] + numpy.log(1 - numpy.exp(sample_log_p_word_context) + 1e-99),
					child._log_s_table[:-1] + sample_log_p_word_context
				)

		if (eos_id is not None) and data_sequence[i] == eos_id:
			context_buffer_ids.clear()
			context_buffer_logprobs.clear()

		assert outputs_cache[i - 1][data_sequence[i]] / temperatures[max_ngram_order - 1] <= log_normalizers, \
			(outputs_cache[i - 1][data_sequence[i]] / temperatures[max_ngram_order - 1], log_normalizers)
		context_buffer_ids.append(data_sequence[i])
		context_buffer_logprobs.append(
			outputs_cache[i - 1][data_sequence[i]] / temperatures[max_ngram_order - 1] - log_normalizers)

		if len(context_buffer_ids) == max_ngram_order:
			context_buffer_ids.pop(0)
			context_buffer_logprobs.pop(0)

		if i % 100000 == 0:
			print("processed %d tokens..." % i)
			sys.stdout.flush()

	return root, log_p_word


def compute_E_n_x(current, ngram_order):
	if ngram_order == 0:
		return numpy.exp(current._log_s_table[2]), numpy.exp(current._log_s_table[3])

	E_n_1 = 1e-99
	E_n_2 = 1e-99
	for temp_word_id in current._children:
		child = current._children[temp_word_id]
		temp_E_n_1, temp_E_n_2 = compute_E_n_x(child, ngram_order - 1)
		E_n_1 += temp_E_n_1
		E_n_2 += temp_E_n_2
	return E_n_1, E_n_2


def compute_log_E_n_ge1(parent):
	log_sum_E_n_ge1 = -1e3
	for temp_word_id in parent._children:
		child = parent._children[temp_word_id]

		log_sum_E_n_ge1 = numpy.logaddexp(log_sum_E_n_ge1, child._log_p_c_gt0)
		assert (not numpy.isnan(log_sum_E_n_ge1))
	parent._log_sum_E_n_ge1 = log_sum_E_n_ge1

	for temp_word_id in parent._children:
		child = parent._children[temp_word_id]
		compute_log_E_n_ge1(child)


def compute_log_p_c_gt0(parent):
	for temp_word_id in parent._children:
		child = parent._children[temp_word_id]

		log_p_c_gt0 = max(
			numpy.log(1 - numpy.exp(child._log_s_table[1]) + 1e-99),
			scipy.misc.logsumexp(child._log_s_table[2:]))
		assert (not numpy.isnan(log_p_c_gt0))

		child._log_p_c_gt0 = log_p_c_gt0
		compute_log_p_c_gt0(child)


def compute_log_p_c_eq0(parent):
	for temp_word_id in parent._children:
		child = parent._children[temp_word_id]
		child._log_p_c_eq0 = child._log_s_table[1]
		compute_log_p_c_eq0(child)


def connect_backoff_node(root):
	root._backoff_node = root

	nodes = []
	for temp_word_id in root._children:
		child = root._children[temp_word_id]
		nodes.append((child, [temp_word_id]))

	while (len(nodes) > 0):
		current, context_ids = nodes.pop(0)
		backoff_node = find(root, context_ids[1:])
		current._backoff_node = backoff_node

		for temp_word_id in current._children:
			child = current._children[temp_word_id]
			nodes.append((child, context_ids + [temp_word_id]))


def compute_log_p_word(root, log_discounts, id_to_word, log_sum_p_word=None, backoff_to_uniform=False):
	nodes = []

	current_order = 0
	renorm_time = timeit.default_timer()
	number_of_ngrams = 0

	#
	#
	#
	log_sum_E_c_tilda = -1e3
	for temp_word_id in root._children:
		child = root._children[temp_word_id]
		child._log_E_c_tilda = child._log_p_c_gt0 if (log_sum_p_word is None) else log_sum_p_word[temp_word_id]
		log_sum_E_c_tilda = numpy.logaddexp(
			log_sum_E_c_tilda,
			child._log_E_c_tilda
		)
		assert (not numpy.isnan(log_sum_E_c_tilda))
		nodes.append((child, [temp_word_id]))
	root._log_sum_E_c_tilda = log_sum_E_c_tilda if (log_sum_p_word is None) else scipy.misc.logsumexp(log_sum_p_word)

	for temp_word_id in root._children:
		child = root._children[temp_word_id]
		root._log_p_word[temp_word_id] = child._log_E_c_tilda - root._log_sum_E_c_tilda
		number_of_ngrams += 1
	#
	#
	#

	current_order += 1
	renorm_time = timeit.default_timer() - renorm_time
	print("renorm %d %d-gram took %.2fs..." % (number_of_ngrams, current_order, renorm_time))
	sys.stdout.flush()
	renorm_time = timeit.default_timer()
	number_of_ngrams = 0

	while (len(nodes) > 0):
		parent, context_ids = nodes.pop(0)
		if len(context_ids) > current_order:
			current_order += 1
			renorm_time = timeit.default_timer() - renorm_time
			print("renorm %d %d-gram took %.2fs..." % (number_of_ngrams, current_order, renorm_time))
			sys.stdout.flush()
			renorm_time = timeit.default_timer()
			number_of_ngrams = 0

		compute_log_E_c_tilda(parent, context_ids, log_discounts, id_to_word, backoff_to_uniform)

		for temp_word_id in parent._children:
			child = parent._children[temp_word_id]
			parent._log_p_word[temp_word_id] = child._log_E_c_tilda - parent._log_sum_E_c_tilda
			number_of_ngrams += 1

			nodes.append((child, context_ids + [temp_word_id]))


def compute_log_E_c_tilda(parent, context_ids, log_discounts, id_to_word, backoff_to_uniform=False):
	if len(parent._children) == 0:
		return

	log_sum_E_c_tilda = -1e3
	context_window_size = len(context_ids)
	# for word_id in log_E_c_context_word[context_ids]:
	for temp_word_id in id_to_word:
		if temp_word_id in parent._children:
			child = parent._children[temp_word_id]
			# log_s_table_context_word[context_ids]:

			if context_window_size == 0 and backoff_to_uniform:
				log_p_word_given_temp_context = -numpy.log(len(id_to_word))
			else:
				# temp_node = find(root, context_ids[1:])
				temp_node = parent._backoff_node
				log_p_word_given_temp_context = temp_node._log_p_word[temp_word_id]
			assert (not numpy.isnan(log_p_word_given_temp_context))

			temp = numpy.log(
				max(0, numpy.exp(child._log_E_c) -
				    numpy.exp(log_discounts[context_window_size] + child._log_p_c_gt0))
				+ numpy.exp(
					log_discounts[context_window_size] + parent._log_sum_E_n_ge1 + log_p_word_given_temp_context)
				# + interpolation_discount * numpy.exp(
				# log_p_c_eq0_context_word[context_ids][word_id] + log_p_word_given_temp_context)
			)
			child._log_E_c_tilda = temp
		else:
			temp = log_discounts[context_window_size] + parent._log_sum_E_n_ge1
			temp_node = parent
			for j in range(len(context_ids)):
				# temp_node = find(root, context_ids[j + 1:])
				temp_node = temp_node._backoff_node

				'''
				if context_ids == [3, 4]:
					if temp_word_id in temp_node._log_p_word:
						print("found", temp_word_id, temp, context_ids[j + 1:], temp_node._log_p_word[temp_word_id])
					else:
						print("not found", temp_word_id, temp, context_ids[j + 1:],
						      log_discounts[context_window_size - j - 1],
						      temp_node._log_sum_E_n_ge1, temp_node._log_sum_E_c_tilda)
				'''
				if temp_word_id in temp_node._log_p_word:
					temp += temp_node._log_p_word[temp_word_id]
					break
				if context_window_size - j - 1 not in log_discounts:
					break
				temp += log_discounts[context_window_size - j - 1]
				temp += temp_node._log_sum_E_n_ge1 - temp_node._log_sum_E_c_tilda

		assert (not numpy.isnan(temp))

		log_sum_E_c_tilda = numpy.logaddexp(log_sum_E_c_tilda, temp)
		assert (not numpy.isnan(log_sum_E_c_tilda))

	parent._log_sum_E_c_tilda = log_sum_E_c_tilda


def output_ngrams(root, id_to_word, eos_id, output_directory):
	nodes = []
	nodes.append((None, 1))
	nodes.append((root, ""))
	while (len(nodes) > 1):
		parent, context = nodes.pop(0)
		if parent is None:
			ngram_order = context
			ngram_file = os.path.join(output_directory, "ngram=%d.txt" % (ngram_order))
			ngram_stream = open(ngram_file, 'w')
			ngram_stream.write("\\%d-grams:\n" % (ngram_order))
			if ngram_order == 1:
				ngram_stream.write("%g\t%s\n" % (-99, ngram_sos))
			nodes.append((None, ngram_order + 1))
		else:
			for temp_word_id in parent._children:
				child = parent._children[temp_word_id]
				word = id_to_word[temp_word_id] if temp_word_id != eos_id else ngram_eos
				if len(context) == 0:
					context_word = word
				else:
					context_word = context + " " + word
				ngram_stream.write("%g\t%s\n" % (parent._log_p_word[temp_word_id] / numpy.log(10), context_word))
				if len(child._children) > 0:
					nodes.append((child, context_word))


#
#
#
#
#

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
	sys.stdout.flush()
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
	# streaming_mode = settings.stream
	temperatures = settings.temperatures
	output_directory = settings.output_directory
	max_frequency = settings.max_frequency
	max_ngram_order = settings.max_ngram_order
	discount_unigram = settings.discount_unigram
	memory_footprint = settings.memory_footprint

	#
	#
	#
	#
	#

	if memory_footprint:
		tracemalloc.start()

	# from .ComputeNNLMOutputs import import_output_cache
	outputs_cache = import_output_cache(settings.probability_cache_directory, cutoff=len(data_sequence))
	print("successfully load outputs cache...")
	sys.stdout.flush()

	# root = TrieNode()
	root = TrieNode(max_frequency=max_frequency)

	ngram_to_trie_time = timeit.default_timer()
	root, log_sum_p_word = ngram_to_trie(data_sequence=data_sequence,
	                                     outputs_cache=outputs_cache,
	                                     root=root,
	                                     eos_id=eos_id,
	                                     temperatures=temperatures,
	                                     id_to_word=id_to_word,
	                                     number_of_candidates=number_of_candidates,
	                                     max_frequency=max_frequency,
	                                     min_ngram_order=1,
	                                     max_ngram_order=max_ngram_order,
	                                     )
	ngram_to_trie_time = timeit.default_timer() - ngram_to_trie_time
	print("ngram to trie took %.2fs..." % (ngram_to_trie_time))
	sys.stdout.flush()
	if memory_footprint:
		snapshot = tracemalloc.take_snapshot()
		display_top(snapshot)
	#
	#
	# dfs(root, [], display_type=1)
	#
	#
	connect_backoff_node_time = timeit.default_timer()
	connect_backoff_node(root)
	connect_backoff_node_time = timeit.default_timer() - connect_backoff_node_time
	print("connect backoff node took %.2fs..." % (connect_backoff_node_time))
	sys.stdout.flush()
	if memory_footprint:
		snapshot = tracemalloc.take_snapshot()
		display_top(snapshot)
	#
	#
	#
	compute_log_p_c_gt0_time = timeit.default_timer()
	compute_log_p_c_gt0(root)
	compute_log_p_c_gt0_time = timeit.default_timer() - compute_log_p_c_gt0_time
	print("compute log_p_c_gt0 took %.2fs..." % (compute_log_p_c_gt0_time))
	sys.stdout.flush()
	if memory_footprint:
		snapshot = tracemalloc.take_snapshot()
		display_top(snapshot)
	#
	#
	#
	compute_log_p_c_eq0_time = timeit.default_timer()
	compute_log_p_c_eq0(root)
	compute_log_p_c_eq0_time = timeit.default_timer() - compute_log_p_c_eq0_time
	print("compute log_p_c_eq0 took %.2fs..." % (compute_log_p_c_eq0_time))
	sys.stdout.flush()
	if memory_footprint:
		snapshot = tracemalloc.take_snapshot()
		display_top(snapshot)
	#
	#
	# dfs(root, [], display_type=2)
	#
	#
	compute_log_E_n_ge1_time = timeit.default_timer()
	compute_log_E_n_ge1(root)
	compute_log_E_n_ge1_time = timeit.default_timer() - compute_log_E_n_ge1_time
	print("compute log_E_n_ge1 took %.2fs..." % (compute_log_E_n_ge1_time))
	sys.stdout.flush()
	if memory_footprint:
		snapshot = tracemalloc.take_snapshot()
		display_top(snapshot)
	#
	#
	# dfs(root, [], display_type=3)
	#
	#
	compute_log_discounts_time = timeit.default_timer()
	log_discounts = {}
	for context_length in range(1, max_ngram_order):
		# if (context_window_size > 0) or backoff_to_uniform:
		E_n_1, E_n_2 = compute_E_n_x(current=root, ngram_order=context_length + 1)
		log_discounts[context_length] = numpy.log(E_n_1) - numpy.log(E_n_1 + 2 * E_n_2)
		# print(ngram_order, E_n_1, E_n_2, log_discounts[ngram_order])
		print("%d-gram discount: %g" % (context_length + 1, numpy.exp(log_discounts[context_length])))
		sys.stdout.flush()
	# print(len(log_discounts), log_discounts)
	compute_log_discounts_time = timeit.default_timer() - compute_log_discounts_time
	print("compute log_discounts took %.2fs..." % (compute_log_discounts_time))
	sys.stdout.flush()
	if memory_footprint:
		snapshot = tracemalloc.take_snapshot()
		display_top(snapshot)
	#
	#
	#
	compute_log_p_word_time = timeit.default_timer()
	compute_log_p_word(root=root,
	                   log_discounts=log_discounts,
	                   id_to_word=id_to_word,
	                   log_sum_p_word=(None if discount_unigram else log_sum_p_word)
	                   )
	'''
	compute_log_p_word_multiprocessing(root=root,
	                                   log_discounts=log_discounts,
	                                   id_to_word=id_to_word,
	                                   number_of_processes=4
	                                   )
	'''
	compute_log_p_word_time = timeit.default_timer() - compute_log_p_word_time
	print("compute log_p_word took %.2fs..." % (compute_log_p_word_time))
	sys.stdout.flush()
	if memory_footprint:
		snapshot = tracemalloc.take_snapshot()
		display_top(snapshot)
	#
	#
	#
	# dfs(root=root, context_ids=[], display_type=4)
	#
	#
	#
	output_ngrams_time = timeit.default_timer()
	output_ngrams(root=root,
	              id_to_word=id_to_word,
	              eos_id=eos_id,
	              output_directory=output_directory)
	output_ngrams_time = timeit.default_timer() - output_ngrams_time
	print("output ngrams took %.2fs..." % (output_ngrams_time))
	sys.stdout.flush()
	if memory_footprint:
		snapshot = tracemalloc.take_snapshot()
		display_top(snapshot)
	#
	#
	#
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

	model_parser.add_argument('--subset', dest="subset", action='store', type=int, default=0,
	                          help='subset (default: 0=total)')
	model_parser.add_argument('--temperatures', dest="temperatures", action='store', default="1",
	                          help='temperatures (default: 1, higher value --> more flat distribution)')
	model_parser.add_argument('--number_of_candidates', dest="number_of_candidates", type=int, default=0,
	                          help='number of candidates (default: 0=observables)')

	model_parser.add_argument('--max_frequency', dest="max_frequency", action='store', type=int, default=5,
	                          help='max_frequency (default: 5)')
	model_parser.add_argument('--max_ngram_order', dest="max_ngram_order", action='store', type=int, default=5,
	                          help='max ngram order (default: 5)')

	model_parser.add_argument('--discount_unigram', dest="discount_unigram", action='store', default="False",
	                          help='discount unigram (default: False)')

	# model_parser.add_argument('--stream', dest="stream", action='store', default="False",
	# help='streaming mode (default: False)')
	# model_parser.add_argument('--backoff_to_uniform', dest="backoff_to_uniform", action='store', default="False",
	# help='backoff to uniform mode (default: False)')

	model_parser.add_argument('--memory_footprint', dest="memory_footprint", action='store', default="False",
	                          help='trace memory allocation and footprint (default: False)')

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
	
	model_parser.add_argument('--reward_probability', dest="reward_probability", action='store', type=float, default=0,
	                          help='reward probability for inverse smoothing (default: 0)')
	'''

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
	assert arguments.max_ngram_order > 1
	# assert arguments.context_window > 0
	temperatures = [float(temp) for temp in arguments.temperatures.split("-")]
	if len(temperatures) == 1:
		temperatures *= 9
	arguments.temperatures = temperatures
	assert len(arguments.temperatures) == 9
	for temperature in arguments.temperatures:
		assert temperature > 0

	'''
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

	if arguments.memory_footprint.lower() == "false":
		arguments.memory_footprint = False
	elif arguments.memory_footprint.lower() == "true":
		arguments.memory_footprint = True
	else:
		raise ValueError()

	if arguments.discount_unigram.lower() == "false":
		arguments.discount_unigram = False
	elif arguments.discount_unigram.lower() == "true":
		arguments.discount_unigram = True
	else:
		raise ValueError()

	'''
	arguments.non_uniform >= 0
	assert arguments.interpolation_discount >= 0
	'''

	# assert arguments.reward_probability >= 0

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


def compute_log_p_word_multiprocessing(root, log_discounts, id_to_word, number_of_processes=2,
                                       backoff_to_uniform=False):
	queue_of_nodes = multiprocessing.Queue()
	# nodes_of_current_level = []
	# nodes_of_next_level = []

	context_node_mapping = {}

	current_order = 0
	renorm_time = timeit.default_timer()
	number_of_ngrams = 0
	log_sum_E_c_tilda = -1e3
	for temp_word_id in root._children:
		child = root._children[temp_word_id]
		child._log_E_c_tilda = child._log_p_c_gt0

		log_sum_E_c_tilda = numpy.logaddexp(
			log_sum_E_c_tilda,
			child._log_p_c_gt0
		)
		assert (not numpy.isnan(log_sum_E_c_tilda))
		context_node_mapping[(temp_word_id,)] = child
		queue_of_nodes.put((temp_word_id,))
	# nodes_of_current_level.put((child, [temp_word_id]))
	# nodes_of_current_level.append((child, [temp_word_id]))
	root._log_sum_E_c_tilda = log_sum_E_c_tilda

	for temp_word_id in root._children:
		child = root._children[temp_word_id]
		root._log_p_word[temp_word_id] = child._log_E_c_tilda - root._log_sum_E_c_tilda
		number_of_ngrams += 1

	current_order += 1
	renorm_time = timeit.default_timer() - renorm_time
	print("renorm %d %d-gram took %.2fs..." % (number_of_ngrams, current_order, renorm_time))
	sys.stdout.flush()
	renorm_time = timeit.default_timer()
	# print("checkpoint", root._log_p_word[0])

	# while (len(nodes_of_current_level)>0):
	while (not queue_of_nodes.empty()):
		#
		#
		#

		processes = []

		# creating processes
		for w in range(number_of_processes):
			p = multiprocessing.Process(target=compute_log_E_c_tilda_multiprocessing,
			                            args=(queue_of_nodes,
			                                  # nodes_of_next_level,
			                                  context_node_mapping, id_to_word, log_discounts, backoff_to_uniform))
			processes.append(p)
		print("creating %d processes to renormalize %d-gram..." % (number_of_processes, current_order + 1))

		for p in processes:
			p.start()

		for p in processes:
			p.join()

		renorm_time = timeit.default_timer() - renorm_time
		print("renorm %d %d-gram took %.2fs..." % (len(context_node_mapping), current_order, renorm_time))
		sys.stdout.flush()
		renorm_time = timeit.default_timer()

		current_order += 1
		temp_context_node_mapping = {}
		for context_ids in context_node_mapping:
			parent = context_node_mapping[context_ids]
			for temp_word_id in parent._children:
				child = parent._children[temp_word_id]
				# parent._log_p_word[temp_word_id] = child._log_E_c_tilda - parent._log_sum_E_c_tilda
				temp_context_node_mapping[context_ids + (temp_word_id,)] = child
				queue_of_nodes.put(context_ids + (temp_word_id,))
		# nodes_of_next_level.put((child, context_ids + [temp_word_id]))
		# nodes_of_next_level.append((child, context_ids + [temp_word_id]))
		context_node_mapping = temp_context_node_mapping
		print("checkpoint2", queue_of_nodes.qsize(), len(context_node_mapping))
		for x in context_node_mapping:
			print(x)

		'''
		parent, context_ids = nodes_of_current_level.pop(0)
		if len(context_ids) > current_order:
			current_order += 1
			renorm_time = timeit.default_timer() - renorm_time
			print("renorm %d %d-gram took %.2fs..." % (number_of_ngrams, current_order, renorm_time))
			sys.stdout.flush()
			renorm_time = timeit.default_timer()
			number_of_ngrams = 0

		compute_log_E_c_tilda(parent, context_ids, log_discounts, id_to_word, backoff_to_uniform)
		# if len(parent._children)>0:
		# print(context_ids, parent._log_sum_E_c_tilda)
		'''


def compute_log_E_c_tilda_multiprocessing(nodes_of_current_level,
                                          # nodes_of_next_level,
                                          context_node_mapping, id_to_word, log_discounts, backoff_to_uniform=False):
	while True:
		try:
			'''
				try to get task from the queue. get_nowait() function will 
				raise queue.Empty exception if the queue is empty. 
				queue(False) function would do the same task also.
			'''
			# task = tasks_to_accomplish.get_nowait()
			# parent, context_ids = nodes_of_current_level.pop(0)
			# parent, context_ids = nodes_of_current_level.get_nowait()
			context_ids = nodes_of_current_level.get(False)
		except queue.Empty:
			break
		except IndexError:
			break
		else:
			'''
				if no exception has been raised, add the task completion 
				message to task_that_are_done queue
			'''

			# print(context_ids)
			print(str(context_ids) + ' is done by ' + multiprocessing.current_process().name)
			# print(task)
			# tasks_that_are_done.put(task + ' is done by ' + current_process().name)
			# time.sleep(numpy.random.random())

			#
			#
			#
			#
			#

			parent = context_node_mapping[context_ids]
			# (parent, context_ids, log_discounts, id_to_word, backoff_to_uniform=False):
			if len(parent._children) == 0:
				continue

			log_sum_E_c_tilda = -1e3
			context_window_size = len(context_ids)
			# for word_id in log_E_c_context_word[context_ids]:
			for temp_word_id in id_to_word:
				if temp_word_id in parent._children:
					child = parent._children[temp_word_id]
					# log_s_table_context_word[context_ids]:

					if context_window_size == 0 and backoff_to_uniform:
						log_p_word_given_temp_context = -numpy.log(len(id_to_word))
					else:
						# temp_node = find(root, context_ids[1:])
						temp_node = parent._backoff_node
						log_p_word_given_temp_context = temp_node._log_p_word[temp_word_id]
					assert (not numpy.isnan(log_p_word_given_temp_context))

					# print(context_ids, temp_word_id, child._log_E_c, log_discounts[context_window_size], child._log_p_c_gt0,
					# parent._log_sum_E_n_ge1, log_p_word_given_temp_context)
					# sys.stdout.flush()

					temp = numpy.log(
						max(0, numpy.exp(child._log_E_c) -
						    numpy.exp(log_discounts[context_window_size] + child._log_p_c_gt0))
						+ numpy.exp(
							log_discounts[
								context_window_size] + parent._log_sum_E_n_ge1 + log_p_word_given_temp_context)
						# + interpolation_discount * numpy.exp(
						# log_p_c_eq0_context_word[context_ids][word_id] + log_p_word_given_temp_context)
					)
					child._log_E_c_tilda = temp
				else:
					temp = log_discounts[context_window_size] + parent._log_sum_E_n_ge1
					temp_node = parent
					for j in range(len(context_ids)):
						# temp_node = find(root, context_ids[j + 1:])
						temp_node = temp_node._backoff_node

						'''
						if context_ids == [3, 4]:
							if temp_word_id in temp_node._log_p_word:
								print("found", temp_word_id, temp, context_ids[j + 1:], temp_node._log_p_word[temp_word_id])
							else:
								print("not found", temp_word_id, temp, context_ids[j + 1:],
								      log_discounts[context_window_size - j - 1],
								      temp_node._log_sum_E_n_ge1, temp_node._log_sum_E_c_tilda)
						'''
						if temp_word_id in temp_node._log_p_word:
							temp += temp_node._log_p_word[temp_word_id]
							break
						if context_window_size - j - 1 not in log_discounts:
							break
						temp += log_discounts[context_window_size - j - 1]
						temp += temp_node._log_sum_E_n_ge1 - temp_node._log_sum_E_c_tilda

				assert (not numpy.isnan(temp))

				# if context_ids == [3, 4]:
				# print(temp_word_id, (temp_word_id in parent._children), log_sum_E_c_tilda, temp)

				log_sum_E_c_tilda = numpy.logaddexp(log_sum_E_c_tilda, temp)
				assert (not numpy.isnan(log_sum_E_c_tilda))

			parent._log_sum_E_c_tilda = log_sum_E_c_tilda

			#
			#
			#
			#
			#
			# '''
			for temp_word_id in parent._children:
				child = parent._children[temp_word_id]
				parent._log_p_word[temp_word_id] = child._log_E_c_tilda - parent._log_sum_E_c_tilda
	# context_node_mapping[context_ids + (temp_word_id,)] = child
	# nodes_of_next_level.put(context_ids + (temp_word_id,), False)
	# print(nodes_of_next_level.qsize())
	# nodes_of_next_level.put((child, context_ids + [temp_word_id]))
	# nodes_of_next_level.append((child, context_ids + [temp_word_id]))
	# '''

	# nodes_of_current_level.close()
	# print("checkpoint1", nodes_of_next_level.qsize())
	return True
