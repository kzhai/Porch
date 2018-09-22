# import logging
import datetime
import os
import timeit

import numpy
import torch

# logger = logging.getLogger(__name__)
'''
def pca(X, number_of_components=2, Y=None):
	from sklearn.decomposition import PCA
	pca = PCA(n_components=number_of_components)
	pca.fit(X)
	if Y is None:
		return pca.transform(X), None
	else:
		return pca.transform(X), pca.transform(Y)
'''

from . import nlm_eos, ngram_eos, ngram_sos


def scatter_plot_2D(X, tokens=None, output_file_path=None, title=None):
	import matplotlib.pyplot as plt

	# assert (X.shape == Y.shape)
	fig, ax = plt.subplots(figsize=(20, 20))
	#ax.plot(X[:, 0], X[:, 1], 'ro', markersize=1)
	# ax.plot(Y[:, 0], Y[:, 1], 'b^')

	from matplotlib.colors import ColorConverter
	colors = list(ColorConverter.colors)[::5]
	# random.shuffle(colors)
	print(len(colors))

	if tokens is not None:
		assert (len(tokens) == X.shape[0])
		color_index = -1
		for i in range(len(X)):
			# print(X[x, 0], X[x, 1], X[x + 1, 0], X[x + 1, 1])
			# head_width=.05, head_length=.05, fc='k', ec='k'
			# color = 'g', alpha = 0.75,

			if tokens[i] is None:
				ax.plot(X[i, 0], X[i, 1], 'b*', markersize=5)
				if i > 0:
					ax.plot(X[i - 1, 0], X[i - 1, 1], 'ro', markersize=5)
					#ground_truth = False
					#linewidth = 1
				#else:
					#ground_truth = True
					#linewidth = 10
				color_index += 1
			else:
				ax.arrow(X[i - 1, 0], X[i - 1, 1], X[i, 0] - X[i - 1, 0], X[i, 1] - X[i - 1, 1], linewidth=1,
				         head_width=0.01, head_length=0.01, color=colors[color_index % len(colors)], alpha=0.5,
				         length_includes_head=True)

				ax.text((X[i, 0] + X[i - 1, 0]) / 2, (X[i, 1] + X[i - 1, 1]) / 2, tokens[i],
				        size=10, zorder=1, color=colors[color_index % len(colors)], alpha=0.5)

			'''
			ax.annotate("",
			            (X[i - 1, 0], X[i - 1, 1]),
			            (X[i, 0], X[i, 1]),
			            # xycoords="figure fraction", textcoords="figure fraction",
			            ha="right", va="center",
			            size=1,
			            arrowprops=dict(arrowstyle="simple",
			                            # headwidth=0.01,
			                            # headlength=0.01,
			                            # shrinkA=5,
			                            # shrinkB=5,
			                            # fc="g", ec="g",
			                            fc=colors[color_index % len(colors)],
			                            ec=colors[color_index % len(colors)],
			                            connectionstyle="arc3",
			                            ),
			            # bbox=dict(boxstyle="square", fc="w"),
			            alpha=0.75
			            )
			            
			ax.annotate(tokens[i],
			            xy=((X[i - 1, 0] + X[i, 0]) / 2, (X[i - 1, 1] + X[i, 1]) / 2),
			            xycoords='data',
			            xytext=((X[i - 1, 0] + X[i, 0]) / 2 + 0.1, (X[i - 1, 1] + X[i, 1]) / 2 + 0.1),
			            # textcoords='offset points',
			            textcoords='data',
			            # bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec="none"),
			            bbox=dict(boxstyle="round", fc="w", ec=colors[color_index % len(colors)]),
			            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", alpha=0.5),
			            color=colors[color_index % len(colors)],
			            rotation=15, alpha=0.75, weight="medium",
			            )
			'''
		ax.plot(X[- 1, 0], X[- 1, 1], 'ro', markersize=5)

	if title is not None:
		plt.title(title)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


def scatter_plot_3D(X, tokens=None, output_file_path=None, title=None):
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

	from matplotlib.colors import ColorConverter
	colors = list(ColorConverter.colors)[::5]
	# random.shuffle(colors)
	print(len(colors))

	# assert (X.shape == tokens.shape)
	# fig, ax = plt.subplots(figsize=(50, 40))
	fig = plt.figure(figsize=(16, 12))
	ax = fig.add_subplot(111, projection='3d')

	# For each set of style and range settings, plot n random points in the box
	# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
	# ax.scatter(X[::2, 0], X[::2, 1], X[::2, 2], color='r', marker='o')
	# ax.scatter(X[1::2, 0], X[1::2, 1], X[1::2, 2], color='b', marker='^')
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='r', marker='o')
	# ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='b', marker='^')

	if tokens is not None:
		assert (len(tokens) == X.shape[0])

		color_index = -1
		for i in range(len(X)):
			# print(X[x, 0], X[x, 1], X[x + 1, 0], X[x + 1, 1])
			# head_width=.05, head_length=.05, fc='k', ec='k'
			# color = 'g', alpha = 0.75, length_includes_head=True
			# ax.arrow(X[i, 0], X[i, 1], X[i + 1, 0] - X[i, 0], X[i + 1, 1] - X[i, 1], head_width=0.01, head_length=0.01,
			# color='g', alpha=0.75, )

			if tokens[i] is None:
				ax.scatter(X[i, 0], X[i, 1], X[i, 2], color='b', marker='*')
				if i > 0:
					ax.scatter(X[i - 1, 0], X[i - 1, 1], X[i - 1, 2], color='b', marker='*')

				color_index += 1
				continue

			a = Arrow3D([X[i - 1, 0], X[i, 0]],
			            [X[i - 1, 1], X[i, 1]],
			            [X[i - 1, 2], X[i, 2]],
			            mutation_scale=1,
			            lw=1, arrowstyle="->",
			            color=colors[color_index % len(colors)], alpha=0.75)
			ax.add_artist(a)

			ax.text((X[i - 1, 0] + X[i, 0]) / 2, (X[i - 1, 1] + X[i, 1]) / 2, (X[i - 1, 2] + X[i, 2]) / 2, tokens[i],
			        size=10, zorder=1, color=colors[color_index % len(colors)])
		ax.scatter(X[- 1, 0], X[- 1, 1], X[- 1, 2], color='b', marker='*')
	#
	#
	#

	if title is not None:
		plt.title(title)

	if output_file_path is None:
		plt.show()
	else:
		plt.savefig(output_file_path, bbox_inches='tight')
		plt.close()


def a(tokens, model, word_to_id, id_to_word, perturb_history=1, perturb_sample=9):
	import porch
	from .ComputeNNLMHiddens import reformat_hidden_states

	hidden_sequence_mapping = {}

	sequence = []
	for i, token in enumerate(tokens):
		sequence.append(torch.tensor([[word_to_id[token]]]))
	hiddens_cache, outputs_cache = porch.models.rnn.unfold_internal_states(network=model, sequence=sequence)
	#
	temp_hiddens_sequence = reformat_hidden_states(hiddens_cache)
	for lstm_group_index, lstm_layer_index in temp_hiddens_sequence:
		# print(temp_hiddens_sequence[(lstm_group_index, lstm_layer_index)].shape)
		assert (numpy.all(temp_hiddens_sequence[(lstm_group_index, lstm_layer_index)][0, :] == 0)), \
			temp_hiddens_sequence[
				(lstm_group_index, lstm_layer_index)]
		hidden_sequence_mapping[(lstm_group_index, lstm_layer_index)] = \
			(temp_hiddens_sequence[(lstm_group_index, lstm_layer_index)], [None] + tokens)

	for sample_index in range(perturb_sample):
		sequence = []
		perturbations = [id_to_word[numpy.random.randint(0, len(id_to_word))] for i in range(perturb_history)]
		perturbations += tokens[perturb_history:]
		for i, token in enumerate(perturbations):
			sequence.append(torch.tensor([[word_to_id[token]]]))
		hiddens_cache, outputs_cache = porch.models.rnn.unfold_internal_states(network=model, sequence=sequence)
		#
		temp_hiddens_sequence = reformat_hidden_states(hiddens_cache)
		for lstm_group_index, lstm_layer_index in hidden_sequence_mapping:
			hiddens_temp = hidden_sequence_mapping[(lstm_group_index, lstm_layer_index)][0]
			sequence_temp = hidden_sequence_mapping[(lstm_group_index, lstm_layer_index)][1]

			hidden_sequence_mapping[(lstm_group_index, lstm_layer_index)] = \
				(numpy.vstack((hiddens_temp, temp_hiddens_sequence[(lstm_group_index, lstm_layer_index)])),
				 sequence_temp + [None] + perturbations)

	return hidden_sequence_mapping


def b(tokens, data_sequence, hiddens_cache, word_to_id, id_to_word, perturb_history=1):
	token_ids = []
	for token in tokens:
		token_ids.append(word_to_id[nlm_eos] if token == ngram_eos or token == ngram_sos else word_to_id[token])

	hidden_sequence_mapping = {}
	for i in range(len(data_sequence) - perturb_history - len(token_ids)):
		found = True
		for j in range(len(token_ids)):
			if data_sequence[i + perturb_history + j] != token_ids[j]:
				found = False
				break

		if not found:
			continue

		sequence = [id_to_word[k] for k in data_sequence[i: i + perturb_history + len(token_ids)]]

		if len(hidden_sequence_mapping) == 0:
			for lstm_group_index, lstm_layer_index in hiddens_cache:
				hidden_sequence_mapping[(lstm_group_index, lstm_layer_index)] = \
					(hiddens_cache[(lstm_group_index, lstm_layer_index)][i:i + perturb_history + len(token_ids) + 1],
					 [None] + sequence)
		else:
			for lstm_group_index, lstm_layer_index in hiddens_cache:
				hiddens_temp = hidden_sequence_mapping[(lstm_group_index, lstm_layer_index)][0]
				sequence_temp = hidden_sequence_mapping[(lstm_group_index, lstm_layer_index)][1]

				hidden_sequence_mapping[(lstm_group_index, lstm_layer_index)] = \
					(numpy.vstack((hiddens_temp, hiddens_cache[(lstm_group_index, lstm_layer_index)][
					                             i:i + perturb_history + len(token_ids) + 1])),
					 sequence_temp + [None] + sequence)

	return hidden_sequence_mapping


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

	data_directory = settings.data_directory
	plot_directory = settings.plot_directory

	import porch.data
	word_to_id, id_to_word = porch.data.import_vocabulary(os.path.join(data_directory, "type.info"))

	'''
	model = settings.model(**settings.model_kwargs).to(settings.device)
	model_file = os.path.join(settings.model_directory, "model.pth")
	model.load_state_dict(torch.load(model_file))
	print('Successfully load model state from {}'.format(model_file))
	'''

	if settings.hidden_cache_directory is not None:
		assert settings.model_directory is None
		model = None

		from .ComputeNNLMHiddens import import_hidden_cache
		data_sequence = numpy.load(os.path.join(data_directory, "train.npy"))
		#data_sequence = data_sequence[:100000]
		hiddens_cache = import_hidden_cache(settings.hidden_cache_directory, cutoff=len(data_sequence))
		print('Successfully load hiddens cache from {}'.format(settings.hidden_cache_directory))
	else:
		assert settings.hidden_cache_directory is None
		hiddens_cache = None

		model = settings.model(**settings.model_kwargs).to(settings.device)
		model_file = os.path.join(settings.model_directory, "model.pth")
		model.load_state_dict(torch.load(model_file))
		print('Successfully load model state from {}'.format(settings.model_directory))

	perturb_sample = settings.number_of_samples
	perturb_token = settings.perturb_history
	minimum_count = settings.minimum_count

	torch.manual_seed(settings.random_seed)

	start_train = timeit.default_timer()

	#
	#
	#

	phrase_index = 0
	# phrase_stream = open(os.path.join(settings.phrase_directory, "phrases.txt"), "r")
	phrase_stream = open(settings.ngram_file, "r")
	for line in phrase_stream:
		line = line.strip()
		fields = line.split("\t")
		tokens = fields[0].split()
		frequency = float(fields[1]) if len(fields) >= 2 else 1.
		if minimum_count > 0 and frequency < minimum_count:
			break

		phrase_index += 1

		'''
		if numpy.random.random() > numpy.power(.9, phrase_index - 1):
			print("skip phrase %d: %s" % (phrase_index, fields[0]))
			continue
		'''

		#
		#
		#

		if hiddens_cache is None:
			assert data_sequence is None
			hidden_sequence_mapping = a(tokens, model,
			                            word_to_id=word_to_id,
			                            id_to_word=id_to_word,
			                            perturb_history=perturb_token,
			                            perturb_sample=perturb_sample
			                            )
		else:
			assert model is None
			hidden_sequence_mapping = b(tokens, data_sequence,
			                            hiddens_cache=hiddens_cache,
			                            word_to_id=word_to_id,
			                            id_to_word=id_to_word,
			                            perturb_history=perturb_token)

		# print(hiddens_sequence[(lstm_group_index, lstm_layer_index)])
		for lstm_group_index, lstm_layer_index in hidden_sequence_mapping:
			print(lstm_group_index, lstm_layer_index, phrase_index, " ".join(tokens))
			# print(before_after_mapping[(layer_index, sub_layer_index, sequence_index)])
			hiddens, sequence = hidden_sequence_mapping[(lstm_group_index, lstm_layer_index)]
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
			'''
			output_file_path = os.path.join(settings.phrase_directory,
			                                "projection=2D,lstm_group=%d,lstm_layer=%d,phrase_index=%d,perturb_sample=%d,perturb_token=%d.png" % (
			                                lstm_group_index, lstm_layer_index, perturb_sample, perturb_token,
			                                phrase_index)),
			'''
			output_file_path = os.path.join(plot_directory,
			                                "projection=2D,lstm_group=%d,lstm_layer=%d,phrase_index=%d.png" % (
				                                lstm_group_index, lstm_layer_index,
				                                phrase_index)) if plot_directory is not None else None
			scatter_plot_2D(hiddens_project, sequence,
			                output_file_path=output_file_path,
			                title=" ".join(tokens).replace("$", "\$")
			                )
			'''
			hiddens_project, dummy = pca(hiddens, number_of_components=3)
			# before_after, dummy = pca(numpy.vstack((before, after)), number_of_components=3)
			# X = before_after[:(len(before_after) // 2), :]
			# Y = before_after[(len(before_after) // 2):, :]

			output_file_path = os.path.join(settings.phrase_directory,
			                                "projection=3D,lstm_group=%d,lstm_layer=%d,phrase_index=%d.png" % (
				                                lstm_group_index, lstm_layer_index, phrase_index))
			scatter_plot_3D(hiddens_project, sequence,
			                output_file_path=output_file_path,
			                title=" ".join(tokens).replace("$", "\$")
			                )
			'''
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
	                          help="data directory [None]")
	model_parser.add_argument("--plot_directory", dest="plot_directory", action='store', default=None,
	                          help="plot directory [None]")
	model_parser.add_argument("--ngram_file", dest="ngram_file", action='store', default=None,
	                          help="ngram file directory [None]")
	model_parser.add_argument("--hidden_cache_directory", dest="hidden_cache_directory", action='store', default=None,
	                          help="hidden cache directory [None]")

	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')

	# generic argument set 3
	model_parser.add_argument("--number_of_samples", dest="number_of_samples", type=int, action='store', default=0,
	                          help="number of samples [0]")
	model_parser.add_argument("--perturb_history", dest="perturb_history", type=int, action='store',
	                          default=-1, help="number of perturbation tokens on front")
	model_parser.add_argument("--minimum_count", dest="minimum_count", type=int, action='store', default=-1,
	                          help="minimum count to plot")

	# generic argument set 4
	model_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                          help="model directory [None, resume mode if specified]")
	model_parser.add_argument("--model", dest="model", action='store', default="porch.models.mlp.GenericMLP",
	                          help="neural network model [porch.mnist.MLP]")
	model_parser.add_argument("--model_kwargs", dest="model_kwargs", action='store', default="",
	                          help="model kwargs specified for neural network model [None]")

	return model_parser


def validate_options(arguments):
	from porch.argument import param_deliminator, specs_deliminator

	# use_cuda = arguments.device.lower() == "cuda" and torch.cuda.is_available()
	arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	arguments.device = torch.device(arguments.device)

	# generic argument set snapshots
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond

	# generic argument set 3
	assert arguments.number_of_samples >= 0
	assert arguments.perturb_history >= 0

	# generic argument set 4
	if arguments.model_directory is not None:
		assert arguments.hidden_cache_directory is None
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
	else:
		assert arguments.hidden_cache_directory is not None
		assert os.path.exists(arguments.hidden_cache_directory)

	# generic argument set 1
	assert os.path.exists(arguments.data_directory)
	assert os.path.exists(arguments.ngram_file)
	if arguments.plot_directory is not None and not os.path.exists(arguments.plot_directory):
		os.mkdir(arguments.plot_directory)

	return arguments


if __name__ == '__main__':
	main()
