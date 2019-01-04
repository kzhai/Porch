import os

import numpy


def barchart(X, Y, labels, title=None, output_file_path=None):
	import matplotlib.pyplot as plt

	# fig, ax = plt.subplots()
	fig, ax = plt.subplots(figsize=(len(labels) / 5, len(labels) / 25))

	index = numpy.arange(len(labels))  # the x locations for the groups
	width = 0.35  # the width of the bars

	rects1 = ax.bar(index - width / 2, X, width,  # yerr=men_std,
	                color='g', label='ngram')
	rects2 = ax.bar(index + width / 2, Y, width,  # yerr=women_std,
	                color='r', label='nnlm')

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel('probability')
	# ax.set_title('Scores by group and gender')
	ax.set_xticks(index)
	ax.set_xticklabels(labels, rotation='vertical')
	ax.legend()

	def autolabel(rects, xpos='center'):
		"""
		Attach a text label above each bar in *rects*, displaying its height.

		*xpos* indicates which side to place the text w.r.t. the center of
		the bar. It can be one of the following {'center', 'right', 'left'}.
		"""

		xpos = xpos.lower()  # normalize the case of the parameter
		ha = {'center': 'center', 'right': 'left', 'left': 'right'}
		offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
			        '{}'.format(height), ha=ha[xpos], va='bottom')

	# autolabel(rects1, "left")
	# autolabel(rects2, "right")

	# Tweak spacing to prevent clipping of ylabel
	fig.tight_layout()

	if title is not None:
		plt.title(title)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


def load_context_word_prob(context_word_prob_file):
	context_word_prob = {}

	context_word_prob_stream = open(context_word_prob_file, 'r')
	line = context_word_prob_stream.readline()
	for line in context_word_prob_stream:
		line = line.strip()
		if len(line) == 0:
			continue
		fields = line.split("\t")
		tokens = fields[1].split(" ")
		word = tokens[-1]
		context = " ".join(tokens[:-1])
		if context not in context_word_prob:
			context_word_prob[context] = {}
		context_word_prob[context][word] = numpy.power(10., float(fields[0]))

	return context_word_prob


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

	srilm_breakdown_directory = settings.srilm_breakdown_directory
	nnlm_breakdown_directory = settings.nnlm_breakdown_directory
	output_plot_directory = settings.output_directory
	max_number_of_candidates = settings.number_of_candidates

	for order in range(1, 10):
		srilm_breakdown_file = os.path.join(srilm_breakdown_directory, "ngram=%d.txt" % order)
		srilm_context_word_prob = load_context_word_prob(srilm_breakdown_file)

		nnlm_breakdown_file = os.path.join(nnlm_breakdown_directory, "ngram=%d.txt" % order)
		nnlm_context_word_prob = load_context_word_prob(nnlm_breakdown_file)

		number_of_candidates = max_number_of_candidates // order

		context_id = 1
		for context in srilm_context_word_prob:
			if context not in nnlm_context_word_prob:
				continue
			if len(srilm_context_word_prob[context]) <= number_of_candidates:
				continue
			if len(nnlm_context_word_prob[context]) <= number_of_candidates:
				continue

			if len(context) > 0 and context != "black":
				continue

			srilm_word_candidates = sorted(srilm_context_word_prob[context],
			                               key=srilm_context_word_prob[context].__getitem__,
			                               reverse=True)[:number_of_candidates]
			nnlm_word_candidates = sorted(nnlm_context_word_prob[context],
			                              key=nnlm_context_word_prob[context].__getitem__,
			                              reverse=True)[:number_of_candidates]
			all_candidates = list(set(srilm_word_candidates + nnlm_word_candidates))
			all_candidates.sort(key=lambda x: srilm_context_word_prob[context][x], reverse=True)

			title = "p(w)" if len(context) == 0 else "p(w | %s)" % (context)
			if len(context) == 0:
				output_file_path = os.path.join(output_plot_directory, "order=%d.png" % (order))
			else:
				output_file_path = os.path.join(output_plot_directory, "order=%d,index=%d.png" % (order, context_id))

			X = numpy.asarray([srilm_context_word_prob[context][candidate] for candidate in all_candidates])
			Y = numpy.asarray([nnlm_context_word_prob[context][candidate] for candidate in all_candidates])
			barchart(X, Y, all_candidates, title, output_file_path)

			if context_id % 1000 == 0:
				print("processed %d contexts of order %d" % (context_id, order))

			context_id += 1

		print("processed order %d" % (order))

	return


def add_options(model_parser):
	model_parser.add_argument("--srilm_breakdown_directory", dest="srilm_breakdown_directory", action='store',
	                          default=None,
	                          help="srilm breakdown directory [None]")
	model_parser.add_argument("--nnlm_breakdown_directory", dest="nnlm_breakdown_directory", action='store',
	                          default=None,
	                          help="nnlm breakdown directory [None]")
	model_parser.add_argument("--output_directory", dest="output_directory", action='store', default=None,
	                          help="output directory [None]")

	model_parser.add_argument("--number_of_candidates", dest="number_of_candidates", type=int, action='store',
	                          default=100,
	                          help="number of candidates [100]")

	return model_parser


def validate_options(arguments):
	assert os.path.exists(arguments.srilm_breakdown_directory)
	assert os.path.exists(arguments.nnlm_breakdown_directory)
	if not os.path.exists(arguments.output_directory):
		os.mkdir(arguments.output_directory)

	assert arguments.number_of_candidates > 0

	return arguments


if __name__ == '__main__':
	main()
