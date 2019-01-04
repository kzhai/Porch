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


def scatter_plot_2D(X, Y, output_file_path=None, title=None):
	import matplotlib.pyplot as plt

	assert (X.shape == Y.shape)
	fig, ax = plt.subplots(figsize=(12, 9))
	ax.plot(X[:, 0], X[:, 1], 'ro')
	ax.plot(Y[:, 0], Y[:, 1], 'b^')



	for x in range(len(X)):
		# print(X[x, 0], X[x, 1], X[x + 1, 0], X[x + 1, 1])
		# head_width=.05, head_length=.05, fc='k', ec='k'
		ax.arrow(X[x, 0], X[x, 1], Y[x, 0] - X[x, 0], Y[x, 1] - X[x, 1], length_includes_head=True, color='g',
		         alpha=0.75)

	if title is not None:
		plt.title(title)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


def scatter_plot_3D(X, Y, output_file_path=None, title=None):
	import matplotlib.pyplot as plt
	from matplotlib.patches import FancyArrowPatch
	from mpl_toolkits.mplot3d import proj3d

	class Arrow3D(FancyArrowPatch):
		def __init__(self, xs, ys, zs, *args, **kwargs):
			FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
			self._verts3d = xs, ys, zs

		def draw(self, renderer):
			xs3d, ys3d, zs3d = self._verts3d
			xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
			self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
			FancyArrowPatch.draw(self, renderer)

	assert (X.shape == Y.shape)
	# fig, ax = plt.subplots(figsize=(50, 40))
	fig = plt.figure(figsize=(16, 12))
	ax = fig.add_subplot(111, projection='3d')

	# For each set of style and range settings, plot n random points in the box
	# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
	# ax.scatter(X[::2, 0], X[::2, 1], X[::2, 2], color='r', marker='o')
	# ax.scatter(X[1::2, 0], X[1::2, 1], X[1::2, 2], color='b', marker='^')
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='r', marker='o')
	ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='b', marker='^')

	for x in range(len(X)):
		a = Arrow3D([X[x, 0], Y[x, 0]],
		            [X[x, 1], Y[x, 1]],
		            [X[x, 2], Y[x, 2]],
		            mutation_scale=1,
		            lw=1, arrowstyle="->",
		            color='g', alpha=0.75)
		# color = "g"
		ax.add_artist(a)

	if title is not None:
		plt.title(title)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


def recur_to_random_hiddens_by_sequence(device,
                                        network,
                                        sequence,
                                        #
                                        number_of_samples=1000,
                                        sampling_function=torch.rand,
                                        sampling_scale=1.,
                                        sampling_offset=0.,
                                        #
                                        *args,
                                        **kwargs
                                        ):
	network.eval()
	sequence = sequence.to(device)

	cache = []
	with torch.no_grad():
		# automatically handles the left-over data
		hiddens = porch.models.rnn.initialize_hidden_states(network, sequence.shape[0])
		hiddens_before = porch.base.detach(hiddens)
		# kwargs["hiddens"] = hiddens_before
		# output = network(sequence, ** kwargs)
		output = network(sequence, **{"hiddens": hiddens})
		if isinstance(output, tuple):
			output, hiddens = output
		hiddens_after = porch.base.detach(hiddens)
		cache.append((hiddens_before, hiddens_after))

		for x in range(number_of_samples - 1):
			# automatically handles the left-over data
			hiddens = porch.models.rnn.initialize_hidden_states(network, sequence.shape[0], sampling_function,
			                                                    sampling_scale, sampling_offset)
			hiddens_before = porch.base.detach(hiddens)
			# kwargs["hiddens"] = hiddens_before
			# output = network(sequence, ** kwargs)
			output = network(sequence, **{"hiddens": hiddens})
			if isinstance(output, tuple):
				output, hiddens = output
			hiddens_after = porch.base.detach(hiddens)

			cache.append((hiddens_before, hiddens_after))

	return cache


def load_dictionary(type_info_file):
	type_info_stream = open(type_info_file, 'r')
	word_to_id = {}
	id_to_word = {}
	for line in type_info_stream:
		line = line.strip()
		tokens = line.split()
		word_to_id[tokens[0]] = len(word_to_id)
		id_to_word[len(id_to_word)] = tokens[0]

	return word_to_id, id_to_word


def indexify(phrase_file, word_to_id):
	phrase_stream = open(phrase_file, "r")
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

	word_to_id, id_to_word = load_dictionary(os.path.join(settings.data_directory, "type.info"))
	phrase_to_sequence = indexify(os.path.join(settings.phrase_directory, "phrases.txt"), word_to_id)

	model = settings.model(**settings.model_kwargs).to(settings.device)
	model_file = os.path.join(settings.model_directory, "model.pth")
	model.load_state_dict(torch.load(model_file))
	print('Successfully load model state from {}'.format(model_file))

	torch.manual_seed(settings.random_seed)

	start_train = timeit.default_timer()
	phrase_index = 0
	for phrase in phrase_to_sequence:
		phrase_index += 1
		cache = recur_to_random_hiddens_by_sequence(device=settings.device,
		                                            network=model,
		                                            sequence=phrase_to_sequence[phrase],
		                                            number_of_samples=settings.number_of_samples,
		                                            sampling_function=settings.sampling_function,
		                                            sampling_scale=settings.sampling_scale,
		                                            sampling_offset=settings.sampling_offset,
		                                            )

		assert (len(cache) == settings.number_of_samples)
		before_after_mapping = {}
		for sample_index in range(len(cache)):
			hiddens_before, hiddens_after = cache[sample_index]
			# print(type(hiddens_before), type(hiddens_after))
			assert len(hiddens_before) == len(hiddens_after)

			for lstm_group_index in range(len(hiddens_before)):
				hiddens_before_layer = hiddens_before[lstm_group_index]
				hiddens_after_layer = hiddens_after[lstm_group_index]

				assert type(hiddens_before_layer) == type(hiddens_after_layer)
				if type(hiddens_before_layer) == tuple:
					hiddens_before_layer = hiddens_before_layer[0]
					hiddens_after_layer = hiddens_after_layer[0]
				assert hiddens_before_layer.shape == hiddens_after_layer.shape

				for lstm_layer_index in range(hiddens_before_layer.shape[0]):
					assert (hiddens_before_layer.shape[1] == 1)
					for sequence_index in range(hiddens_before_layer.shape[1]):
						if (lstm_group_index, lstm_layer_index, sequence_index) not in before_after_mapping:
							before_after_mapping[(lstm_group_index, lstm_layer_index, sequence_index)] = (numpy.zeros(
								(settings.number_of_samples, hiddens_before_layer.shape[2])), numpy.zeros(
								(settings.number_of_samples, hiddens_after_layer.shape[2])))
						before_after_mapping[(lstm_group_index, lstm_layer_index, sequence_index)][0][sample_index, :] \
							= hiddens_before_layer[lstm_layer_index, sequence_index, :].numpy()
						before_after_mapping[(lstm_group_index, lstm_layer_index, sequence_index)][1][sample_index, :] \
							= hiddens_after_layer[lstm_layer_index, sequence_index, :].numpy()

		for lstm_group_index, lstm_layer_index, sequence_index in before_after_mapping:
			print(lstm_group_index, lstm_layer_index, phrase_index, phrase)
			# print(before_after_mapping[(layer_index, sub_layer_index, sequence_index)])
			before, after = before_after_mapping[(lstm_group_index, lstm_layer_index, sequence_index)]
			# print(numpy.max((before - after), axis=1))
			# print(numpy.min((before - after), axis=1))

			# X, Y = pca(before, number_of_components=2, Y=after)
			# Y, X = pca(after, before, number_of_components=2)
			# scatter_plot_2D(X, Y)

			before_after, dummy = pca(numpy.vstack((before, after)), number_of_components=2)
			X = before_after[:(len(before_after) // 2), :]
			Y = before_after[(len(before_after) // 2):, :]
			scatter_plot_2D(X, Y,
			                output_file_path=os.path.join(settings.phrase_directory,
			                                              "projection=2D,lstm_group=%d,lstm_layer=%d,phrase_index=%d.png" % (
				                                              lstm_group_index, lstm_layer_index, phrase_index)),
			                title=phrase
			                )

			before_after, dummy = pca(numpy.vstack((before, after)), number_of_components=3)
			X = before_after[:(len(before_after) // 2), :]
			Y = before_after[(len(before_after) // 2):, :]
			scatter_plot_3D(X, Y,
			                output_file_path=os.path.join(settings.phrase_directory,
			                                              "projection=3D,lstm_group=%d,lstm_layer=%d,phrase_index=%d.png" % (
				                                              lstm_group_index, lstm_layer_index, phrase_index)),
			                title=phrase
			                )

			# X, Y = pca(before, after, number_of_components=3)
			# Y, X = pca(after, before, number_of_components=3)
			# scatter_plot_3D(X, Y)

			'''
			embeddings = tSNE(before_after_mapping[(layer_index, sub_layer_index, sequence_index)],
			                  number_of_components=2)
			scatter_plot_2D(embeddings)
			
			embeddings = tSNE(before_after_mapping[(layer_index, sub_layer_index, sequence_index)],
			                  number_of_components=3)
			scatter_plot_3D(embeddings)
			'''
	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	# generic argument set 1
	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument("--phrase_directory", dest="phrase_directory", action='store', default=None,
	                          help="phrase directory [None]")
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')

	'''
	# generic argument set 2
	model_parser.add_argument("--loss", dest="loss", action='append', default=[],
	                          help="loss function [None] defined in porch.loss")
	model_parser.add_argument("--regularizer", dest="regularizer", action='append', default=[],
	                          help="regularizer function [None] defined in porch.regularizer")
	model_parser.add_argument("--information", dest='information', action='append', default=[],
	                          help="information function [None] defined in porch.loss")
	'''
	# generic argument set 3
	model_parser.add_argument("--number_of_samples", dest="number_of_samples", type=int, action='store', default=-1,
	                          help="number of samples [-1]")
	model_parser.add_argument("--sampling_function", dest="sampling_function", action='store', default="rand",
	                          help="sampling function to initialize hidden state [rand] defined for torch.Tensor")
	model_parser.add_argument("--sampling_scale", dest="sampling_scale", type=float, action='store', default=2.,
	                          help="sampling scale to initialize hidden state [2.]")
	model_parser.add_argument("--sampling_offset", dest="sampling_offset", type=float, action='store', default=1.,
	                          help="sampling offset to initialize hidden state [1.]")
	#model_parser.add_argument("--max_prefix_padding", dest="max_prefix_padding", type=int, action='store', default=-1,
	                          #help="number of samples [-1]")

	# generic argument set 4
	model_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                          help="model directory [None, resume mode if specified]")
	model_parser.add_argument("--model", dest="model", action='store', default="porch.models.mlp.GenericMLP",
	                          help="neural network model [porch.mnist.MLP]")
	model_parser.add_argument("--model_kwargs", dest="model_kwargs", action='store', default="",
	                          help="model kwargs specified for neural network model [None]")

	'''
	model_parser.add_argument("--optimizer", dest="optimizer", action='store', default="SGD",
	                          help="optimizer algorithm [SGD] defined in torch.optim or porch.optim")
	model_parser.add_argument("--optimizer_kwargs", dest='optimizer_kwargs', action='store',
	                          default="lr{}1e-3{}momentum{}0.9".format(specs_deliminator, param_deliminator,
	                                                                   specs_deliminator),
	                          help="optimizer kwargs specified for optimization algorithm [lr:1e-3,momentum:0.9], consult the api for more info")
	
	# generic argument set 5
	model_parser.add_argument("--data", dest='data', action='append', default=[],
	                          help="data preprocess function [None] defined in porch.data")
	model_parser.add_argument("--train_kwargs", dest='train_kwargs', action='store',
	                          default="", help="kwargs specified for model training")
	model_parser.add_argument("--test_kwargs", dest='test_kwargs', action='store',
	                          default="", help="kwargs specified for model testing")
	'''

	return model_parser


def validate_options(arguments):
	from porch.argument import param_deliminator, specs_deliminator

	# use_cuda = arguments.device.lower() == "cuda" and torch.cuda.is_available()
	arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	arguments.device = torch.device(arguments.device)

	# generic argument set snapshots
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond

	'''
	snapshots = {}
	for snapshot_interval_mapping in arguments.snapshot:
		fields = snapshot_interval_mapping.split(specs_deliminator)
		snapshot_function = getattr(porch.debug, fields[0])
		if len(fields) == 1:
			interval = 1
		elif len(fields) == 2:
			interval = int(fields[1])
		else:
			logger.error("unrecognized snapshot function setting %s..." % (snapshot_interval_mapping))
		snapshots[snapshot_function] = interval
	arguments.snapshot = snapshots

	debugs = {}
	for debug_interval_mapping in arguments.debug:
		fields = debug_interval_mapping.split(specs_deliminator)
		debug_function = getattr(porch.debug, fields[0])
		if len(fields) == 1:
			interval = 1
		elif len(fields) == 2:
			interval = int(fields[1])
		else:
			logger.error("unrecognized debug function setting %s..." % (debug_interval_mapping))
		debugs[debug_function] = interval
	arguments.debug = debugs

	
	# assert arguments.snapshot_interval >= 0

	# generic argument set 2
	losses = {}
	for loss_weight_mapping in arguments.loss:
		fields = loss_weight_mapping.split(specs_deliminator)
		loss_function = getattr(porch.loss, fields[0])
		if len(fields) == 1:
			losses[loss_function] = 1.0
		elif len(fields) == 2:
			losses[loss_function] = float(fields[1])
		else:
			logger.error("unrecognized loss function setting %s..." % (loss_weight_mapping))
	arguments.loss = losses
	arguments.loss_kwargs = {}

	regularizers = {}
	for regularizer_weight_mapping in arguments.regularizer:
		fields = regularizer_weight_mapping.split(specs_deliminator)
		regularizer_function = getattr(porch.regularizer, fields[0])
		if len(fields) == 1:
			regularizers[regularizer_function] = 1.0
		elif len(fields) == 2:
			regularizers[regularizer_function] = float(fields[1])
		else:
			logger.error("unrecognized regularizer function setting %s..." % (regularizer_weight_mapping))
	arguments.regularizer = regularizers
	arguments.regularizer_kwargs = {}

	informations = {}
	for information_weight_mapping in arguments.information:
		fields = information_weight_mapping.split(specs_deliminator)
		information_function = getattr(porch.loss, fields[0])
		if len(fields) == 1:
			informations[information_function] = 1.0
		elif len(fields) == 2:
			informations[information_function] = float(fields[1])
		else:
			logger.error("unrecognized information function setting %s..." % (information_weight_mapping))
	arguments.information = informations
	arguments.information_kwargs = {}

	arguments.optimizer = getattr(torch.optim, arguments.optimizer)
	# arguments.optimizer = getattr(porch.optim, arguments.optimizer)
	optimizer_kwargs = {}
	optimizer_kwargs_tokens = arguments.optimizer_kwargs.split(param_deliminator)
	for optimizer_kwargs_token in optimizer_kwargs_tokens:
		key_value_pair = optimizer_kwargs_token.split(specs_deliminator)
		assert len(key_value_pair) == 2
		optimizer_kwargs[key_value_pair[0]] = float(key_value_pair[1])
	arguments.optimizer_kwargs = optimizer_kwargs

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

	train_kwargs = {}
	train_kwargs_tokens = arguments.train_kwargs.split(param_deliminator)
	for train_kwargs_token in train_kwargs_tokens:
		if len(train_kwargs_token) == 0:
			continue
		key_value_pair = train_kwargs_token.split(specs_deliminator)
		assert len(key_value_pair) == 2
		train_kwargs[key_value_pair[0]] = key_value_pair[1]
	arguments.train_kwargs = train_kwargs

	test_kwargs = {}
	test_kwargs_tokens = arguments.test_kwargs.split(param_deliminator)
	for test_kwargs_token in test_kwargs_tokens:
		if len(test_kwargs_token) == 0:
			continue
		key_value_pair = test_kwargs_token.split(specs_deliminator)
		assert len(key_value_pair) == 2
		test_kwargs[key_value_pair[0]] = key_value_pair[1]
	arguments.test_kwargs = test_kwargs
	'''

	# generic argument set 3
	assert arguments.number_of_samples > 0
	arguments.sampling_function = getattr(torch, arguments.sampling_function)
	assert arguments.sampling_scale > 0
	assert arguments.sampling_offset >= 0

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
