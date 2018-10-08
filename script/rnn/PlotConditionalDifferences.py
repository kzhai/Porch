import os
import re

import numpy

from . import ngram_eos, ngram_sos, nlm_eos
from . import word_context_probability_pattern, ngram_conditionals_pattern, nlm_conditionals_pattern




def parse_ngram_conditional_probabilities(context_word_prob_file, unified_eos=nlm_eos):
	context_word_prob = {}
	context_word_prob_stream = open(context_word_prob_file, "r")
	for line in context_word_prob_stream:
		line = line.strip()
		if len(line) == 0:
			continue

		matcher = re.match(word_context_probability_pattern, line)
		if matcher is None:
			# print("unmatched line: %s" % line)
			continue

		word = matcher.group("word")
		context = " ".join(matcher.group("context").split(" ")[::-1])
		ngram = matcher.group("ngram")
		probability = float(matcher.group("probability"))
		log_prob_info = matcher.group("logprobinfo").split("*")

		if unified_eos is not None:
			word = word.replace(ngram_sos, unified_eos)
			word = word.replace(ngram_eos, unified_eos)
			context = context.replace(ngram_sos, unified_eos)
			context = context.replace(ngram_eos, unified_eos)

		context_word_prob[(context, word)] = probability

	return context_word_prob


def parse_nlm_conditional_probabilities(context_word_prob_file, unified_eos=None):
	context_word_prob = {}
	# print(context_word_prob_file)
	context_word_prob_stream = open(context_word_prob_file, "r")
	for line in context_word_prob_stream:
		# line = line.strip()
		if len(line) == 0:
			continue

		fields = line.split("\t")
		assert len(fields) == 3
		context = fields[0]
		word = fields[1]
		probabilities = [float(probability) for probability in fields[2].split()]

		context_word_prob[(context, word)] = probabilities

	return context_word_prob


def main():
	import argparse

	model_parser = argparse.ArgumentParser()
	model_parser.add_argument("--ngram_directory", dest="ngram_directory", action='store', default=None,
	                          help="input ngram conditional probabilities file [None]")
	model_parser.add_argument("--nlm_directory", dest="nlm_directory", action='store', default=None,
	                          help="input nlm conditional probabilities file [None]")
	model_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                          help="output histogram file [None]")

	settings, additionals = model_parser.parse_known_args()
	assert (len(additionals) == 0)

	print("========== ========== ========== ========== ==========")
	for key, value in vars(settings).items():
		print("%s=%s" % (key, value))
	print("========== ========== ========== ========== ==========")

	ngram_directory = settings.ngram_directory
	nlm_directory = settings.nlm_directory
	output_directory = settings.output_directory

	#
	#
	#

	# input_ngram = os.path.join()
	for ngram_conditionals_file_name in os.listdir(ngram_directory):
		matcher = re.match(ngram_conditionals_pattern, ngram_conditionals_file_name)
		if matcher is None:
			continue

		ngram_context = int(matcher.group("context"))
		ngram_order = int(matcher.group("order"))

		input_ngram = os.path.join(ngram_directory, ngram_conditionals_file_name)
		ngram_context_word_prob = parse_ngram_conditional_probabilities(input_ngram)

		for nlm_conditionals_file_name in os.listdir(nlm_directory):
			matcher = re.match(nlm_conditionals_pattern, nlm_conditionals_file_name)
			if matcher is None:
				continue

			nlm_context = int(matcher.group("context"))
			if nlm_context != ngram_context:
				continue

			input_nlm = os.path.join(nlm_directory, nlm_conditionals_file_name)
			nlm_context_word_prob = parse_nlm_conditional_probabilities(input_nlm)

			sample_differences = []
			mean_differences = []
			for context, word in nlm_context_word_prob:
				if (context, word) not in ngram_context_word_prob:
					# print("no match found for p( %s | %s )" % (word, context))
					continue
				ngram_probability = ngram_context_word_prob[(context, word)]
				for nlm_probability in nlm_context_word_prob[(context, word)]:
					sample_differences.append(nlm_probability - ngram_probability)
				mean_differences.append(numpy.mean(nlm_probability) - ngram_probability)

			if len(sample_differences) == 0:
				continue

			sample_differences = numpy.asarray(sample_differences)
			print("\t".join([
				"context=%d" % ngram_context,
				"ngram=%d" % ngram_order,
				"sample conditionals=%d" % len(sample_differences),
				"sample mean=%g" % numpy.mean(sample_differences),
				"sample min=%g" % numpy.min(sample_differences),
				"sample max=%g" % numpy.max(sample_differences),
				"mean conditionals=%d" % len(mean_differences),
				"mean mean=%g" % numpy.mean(mean_differences),
				"mean min=%g" % numpy.min(mean_differences),
				"mean max=%g" % numpy.max(mean_differences)
			]))

			title = 'p(word|%d context; NLM) - p(word|%d context; %d-Gram)' % (
				nlm_context, ngram_context, ngram_order)
			assert ngram_context == nlm_context

			output_file_path = None if output_directory is None else \
				os.path.join(output_directory, "context=%d,ngram_order=%d,nlm=sample.png" % (
					nlm_context, ngram_order))
			histogram(sample_differences, title=title, output_file_path=output_file_path)

			output_file_path = None if output_directory is None else \
				os.path.join(output_directory, "context=%d,ngram_order=%d,nlm=mean.png" % (
					nlm_context, ngram_order))
			histogram(mean_differences, title=title, output_file_path=output_file_path)


if __name__ == '__main__':
	main()
