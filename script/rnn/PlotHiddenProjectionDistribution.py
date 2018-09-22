# import logging
import collections
import datetime
import os
import timeit

import numpy
import torch

import porch


# logger = logging.getLogger(__name__)

def pca(X, number_of_components=2, Y=None):
	from sklearn.decomposition import PCA
	pca = PCA(n_components=number_of_components)
	pca.fit(X)
	if Y is None:
		return pca.transform(X), None
	else:
		return pca.transform(X), pca.transform(Y)


def scatter_plot_2D_histogram(X, output_file_path=None, title=None):
	import matplotlib.pyplot as plt
	from matplotlib.ticker import NullFormatter

	# fig, ax = plt.subplots(figsize=(12, 9))
	# ax.plot(X[:, 0], X[:, 1], 'ro')

	#
	#
	#

	nullfmt = NullFormatter()  # no labels

	# definitions for the axes
	left, width = 0.1, 0.65
	bottom, height = 0.1, 0.65
	bottom_h = left_h = left + width + 0.02

	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom_h, width, 0.2]
	rect_histy = [left_h, bottom, 0.2, height]

	# start with a rectangular Figure
	plt.figure(1, figsize=(25, 25))

	axScatter = plt.axes(rect_scatter)
	axHistx = plt.axes(rect_histx)
	axHisty = plt.axes(rect_histy)

	# no labels
	axHistx.xaxis.set_major_formatter(nullfmt)
	axHisty.yaxis.set_major_formatter(nullfmt)

	# the scatter plot:
	axScatter.scatter(X[:, 0], X[:, 1], alpha=0.5)

	# now determine nice limits by hand:
	binwidth = 0.25
	xmax, xmin = numpy.max(X[:, 0]), numpy.min(X[:, 0])
	ymax, ymin = numpy.max(X[:, 1]), numpy.min(X[:, 1])
	# ylim = max(numpy.max(numpy.abs(X[:, 0])), numpy.max(numpy.abs(X[:, 1])))
	# lim = (int(xlim / binwidth) + 1) * binwidth

	axScatter.set_xlim((xmin, xmax))
	axScatter.set_ylim((ymin, ymax))

	xbins = numpy.arange(xmin, xmax + binwidth, binwidth)
	axHistx.hist(X[:, 0], bins=xbins)
	ybins = numpy.arange(ymin, ymax + binwidth, binwidth)
	axHisty.hist(X[:, 1], bins=ybins, orientation='horizontal')

	axHistx.set_xlim(axScatter.get_xlim())
	axHisty.set_ylim(axScatter.get_ylim())

	if title is not None:
		plt.title(title)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


def scatter_plot_3D(X, output_file_path=None, title=None):
	import matplotlib.pyplot as plt
	#from matplotlib.patches import FancyArrowPatch
	#from mpl_toolkits.mplot3d import proj3d

	# fig, ax = plt.subplots(figsize=(50, 40))
	fig = plt.figure(figsize=(25, 25))
	ax = fig.add_subplot(111, projection='3d')

	# For each set of style and range settings, plot n random points in the box
	# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
	# ax.scatter(X[::2, 0], X[::2, 1], X[::2, 2], color='r', marker='o')
	# ax.scatter(X[1::2, 0], X[1::2, 1], X[1::2, 2], color='b', marker='^')
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='r', marker='o', alpha=0.5)

	if title is not None:
		plt.title(title)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


def get_hidden_states_for_sequences(network,
                                    sequences):
	network.eval()

	cache = []
	with torch.no_grad():
		# assert tokens.shape[0] == 1
		hiddens = porch.models.rnn.initialize_hidden_states(network, sequences[0].shape[1])
		hiddens_temp = porch.base.detach(hiddens)
		kwargs = {"hiddens": hiddens}
		cache.append(hiddens_temp)
		for tokens in sequences:
			# tokens = tokens.to(device)
			# print("tokens.shape", tokens.shape)
			output, hiddens = network(tokens.t(), **kwargs)
			hiddens_temp = porch.base.detach(hiddens)
			kwargs["hiddens"] = hiddens
			cache.append(hiddens_temp)

			#if numpy.random.random()<0.01:
				#break

	return cache


def reformat_hidden_states(cache):
	hiddens_sequence = {}
	# print("cache", len(cache))
	for token_index in range(len(cache)):
		hiddens_token = cache[token_index]
		# print("hiddens_token", len(hiddens_token))
		for lstm_group_index in range(len(hiddens_token)):
			hiddens_token_group = hiddens_token[lstm_group_index]
			# print("hiddens_token_group", len(hiddens_token_group), type(hiddens_token_group))
			if type(hiddens_token_group) == tuple:
				# if it is lstm, then we only take the hidden states
				hiddens_token_group = hiddens_token_group[0]

			# print(hiddens_token_group.shape)
			for lstm_layer_index in range(hiddens_token_group.shape[0]):
				# assert (hiddens_token_group.shape[1] == 1)
				hiddens_token_group_layer = hiddens_token_group[lstm_layer_index]

				for batch_index in range(hiddens_token_group.shape[0]):
					hiddens_token_group_layer_batch = hiddens_token_group_layer[batch_index, :]
					# print("hiddens_token_group_layer_batch.shape", hiddens_token_group_layer_batch.shape)

					if (lstm_group_index, lstm_layer_index, batch_index) not in hiddens_sequence:
						hiddens_sequence[(lstm_group_index, lstm_layer_index, batch_index)] = numpy.zeros(
							(len(cache), len(hiddens_token_group_layer_batch)))
					hiddens_sequence[(lstm_group_index, lstm_layer_index, batch_index)][token_index, :] = \
						hiddens_token_group_layer_batch.numpy()

	return hiddens_sequence


def tokenize(sequences):
	temp = []
	for tokens in sequences.t():
		temp.append(torch.unsqueeze(tokens, 0))
	return temp


def indexify(phrase_file, word_to_id):
	phrase_stream = open(phrase_file, "r")
	sequence = []
	for line in phrase_stream:
		line = line.strip()
		tokens = line.split()

		for i, token in enumerate(tokens):
			sequence.append(torch.tensor([[word_to_id[token]]]))

	#
	#
	#

	phrase_to_sequence = collections.OrderedDict()
	for line in phrase_stream:
		line = line.strip()
		tokens = line.split()
		sequence = numpy.zeros((1, len(tokens)))
		for i, token in enumerate(tokens):
			sequence[0, i] = word_to_id[token]
		phrase_to_sequence[" ".join(tokens)] = torch.Tensor(sequence)
	return phrase_to_sequence


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

	import porch.data
	dataset = porch.data.load_datasets(
		input_directory=settings.data_directory,
		data_mode="train",
		function_parameter_mapping=settings.data
	)
	minibatch_x, minibatch_y = dataset
	# print(minibatch_x.shape)
	# print(minibatch_x[:5, :])
	minibatch_x = tokenize(minibatch_x)
	# print(len(minibatch_x))

	# import porch.data
	# word_to_id, id_to_word = porch.data.import_vocabulary(os.path.join(settings.data_directory, "type.info"))

	model = settings.model(**settings.model_kwargs).to(settings.device)
	model_file = os.path.join(settings.model_directory, "model.pth")
	model.load_state_dict(torch.load(model_file))
	print('Successfully load model state from {}'.format(model_file))

	torch.manual_seed(settings.random_seed)

	start_train = timeit.default_timer()

	hiddens_sequence = {}
	cache = get_hidden_states_for_sequences(network=model, sequences=minibatch_x)
	temp_hiddens_sequence = reformat_hidden_states(cache)
	for lstm_group_index, lstm_layer_index, batch_index in temp_hiddens_sequence:
		print(temp_hiddens_sequence[(lstm_group_index, lstm_layer_index, batch_index)].shape)
		if (lstm_group_index, lstm_layer_index) not in hiddens_sequence:
			hiddens_sequence[(lstm_group_index, lstm_layer_index)] = \
				temp_hiddens_sequence[(lstm_group_index, lstm_layer_index, batch_index)][settings.skip_steps:, :]
		else:
			hiddens_sequence[(lstm_group_index, lstm_layer_index)] = \
				numpy.vstack((hiddens_sequence[(lstm_group_index, lstm_layer_index)],
				              temp_hiddens_sequence[(lstm_group_index, lstm_layer_index, batch_index)][
				              settings.skip_steps:, :]))

	for lstm_group_index, lstm_layer_index in hiddens_sequence:
		print(lstm_group_index, lstm_layer_index)
		# print(before_after_mapping[(layer_index, sub_layer_index, sequence_index)])
		hiddens = hiddens_sequence[(lstm_group_index, lstm_layer_index)]
		# print(numpy.max((before - after), axis=1))
		# print(numpy.min((before - after), axis=1))

		# X, Y = pca(before, number_of_components=2, Y=after)
		# Y, X = pca(after, before, number_of_components=2)
		# scatter_plot_2D(X, Y)

		from script.rnn.Backup.ProjectHiddensFromRandom import pca
		hiddens_project, dummy = pca(hiddens, number_of_components=2)
		# print(hiddens_project.shape)
		# X = hiddens_project[:-1, :]
		# Y = hiddens_project[1:, :]
		# print(X.shape, Y.shape)
		scatter_plot_2D_histogram(
			hiddens_project,
			output_file_path=os.path.join(settings.phrase_directory,
			                              "projection=2D,lstm_group=%d,lstm_layer=%d,skip_steps=%d.png" % (
				                              lstm_group_index, lstm_layer_index, settings.skip_steps))
		)

		hiddens_project, dummy = pca(hiddens, number_of_components=3)
		# before_after, dummy = pca(numpy.vstack((before, after)), number_of_components=3)
		# X = before_after[:(len(before_after) // 2), :]
		# Y = before_after[(len(before_after) // 2):, :]
		scatter_plot_3D(
			hiddens_project,
			output_file_path=os.path.join(settings.phrase_directory,
			                              "projection=3D,lstm_group=%d,lstm_layer=%d,skip_steps=%d.png" % (
				                              lstm_group_index, lstm_layer_index, settings.skip_steps))
		)

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	# generic argument set 1
	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument("--phrase_directory", dest="phrase_directory", action='store', default=None,
	                          help="phrase directory [None]")
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')

	# generic argument set 3
	model_parser.add_argument("--number_of_samples", dest="number_of_samples", type=int, action='store', default=-1,
	                          help="number of samples [-1]")
	model_parser.add_argument("--sampling_function", dest="sampling_function", action='store', default="rand",
	                          help="sampling function to initialize hidden state [rand] defined for torch.Tensor")
	model_parser.add_argument("--sampling_scale", dest="sampling_scale", type=float, action='store', default=2.,
	                          help="sampling scale to initialize hidden state [2.]")
	model_parser.add_argument("--sampling_offset", dest="sampling_offset", type=float, action='store', default=1.,
	                          help="sampling offset to initialize hidden state [1.]")
	# model_parser.add_argument("--max_prefix_padding", dest="max_prefix_padding", type=int, action='store', default=-1,
	# help="number of samples [-1]")

	# generic argument set 4
	model_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                          help="model directory [None, resume mode if specified]")
	model_parser.add_argument("--model", dest="model", action='store', default="porch.models.mlp.GenericMLP",
	                          help="neural network model [porch.mnist.MLP]")
	model_parser.add_argument("--model_kwargs", dest="model_kwargs", action='store', default="",
	                          help="model kwargs specified for neural network model [None]")

	model_parser.add_argument("--data", dest='data', action='append', default=[],
	                          help="data preprocess function [None] defined in porch.data")
	model_parser.add_argument("--skip_steps", dest="skip_steps", type=int, action='store', default=0,
	                          help="skip the first n steps for each minibatch [0] set to larger positive to warm start]")

	return model_parser


def validate_options(arguments):
	from porch.argument import param_deliminator, specs_deliminator

	# use_cuda = arguments.device.lower() == "cuda" and torch.cuda.is_available()
	arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	arguments.device = torch.device(arguments.device)

	# generic argument set snapshots
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond

	# generic argument set 5
	data = collections.OrderedDict()
	for data_function_params_mapping in arguments.data:
		fields = data_function_params_mapping.split(param_deliminator)
		data_function = getattr(porch.data, fields[0])
		data_function_params = {}
		for param_value in fields[1:]:
			param_value_fields = param_value.split(specs_deliminator)
			assert (len(param_value_fields) == 2)
			data_function_params[param_value_fields[0]] = param_value_fields[1]
		data[data_function] = data_function_params
	arguments.data = data

	assert arguments.skip_steps >= 0

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

	# generic argument set 1
	assert os.path.exists(arguments.data_directory)
	assert os.path.exists(arguments.phrase_directory)

	return arguments


if __name__ == '__main__':
	main()
