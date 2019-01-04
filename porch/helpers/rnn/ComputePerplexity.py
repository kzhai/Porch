# import logging
import datetime
import os
import timeit

import numpy
# import scipy
# import scipy.sparse
import torch

import porch
from porch.argument import param_deliminator, specs_deliminator


def get_output_probability(network,
                           sequence,
                           directory=None,
                           hiddens=None):
	network.eval()

	# hiddens_cache = []
	if directory is None:
		outputs_cache = []
	with torch.no_grad():
		# assert tokens.shape[0] == 1
		if hiddens is None:
			hiddens = porch.models.rnn.initialize_hidden_states(network, 1)
		# hiddens_temp = porch.base.detach(hiddens)
		kwargs = {"hiddens": hiddens}
		# hiddens_cache.append(hiddens_temp)
		for i, token in enumerate(sequence):
			assert (token.shape == (1, 1))
			output, hiddens = network(token.t(), **kwargs)
			# hiddens_temp = porch.base.detach(hiddens)
			kwargs["hiddens"] = hiddens
			# hiddens_cache.append(hiddens_temp)

			distribution = torch.nn.functional.softmax(output, dim=1)
			# distribution = porch.base.detach(distribution)[0, :]
			distribution = distribution.numpy()[0, :]
			if directory is None:
				outputs_cache.append(distribution)
			else:
				distribution_file = os.path.join(directory, "output=%d.npy" % (i))
				numpy.save(distribution_file, distribution)

			if i % 1000 == 0:
				print("progress: %d / %d" % (i, len(sequence)))

	if directory is None:
		# assert (len(hiddens_cache) == len(sequence) + 1)
		assert (len(outputs_cache) == len(sequence))

		return outputs_cache
	else:
		return directory


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
	torch.manual_seed(settings.random_seed)

	#
	#
	#
	#
	#

	import porch.data
	word_to_id, id_to_word = porch.data.import_vocabulary(os.path.join(settings.data_directory, "type.info"))
	data_file = os.path.join(settings.data_directory, "test.npy")
	data_sequence = numpy.load(data_file)
	# data_sequence = data_sequence[:100]
	assert settings.eos_token in word_to_id
	eos_index = word_to_id[settings.eos_token]

	output_file = settings.perplexity_file

	model = settings.model(**settings.model_kwargs).to(settings.device)
	model_file = os.path.join(settings.model_directory, "model.pth")
	model.load_state_dict(torch.load(model_file))
	print('Successfully load model state from {}'.format(model_file))

	sequence = []
	for word_id in data_sequence:
		sequence.append(torch.tensor([[word_id]], dtype=torch.int))
	assert (len(sequence) == len(data_sequence))

	start_train = timeit.default_timer()

	#
	#
	#
	#
	#

	model.eval()
	hiddens = porch.models.rnn.initialize_hidden_states(model, 1)
	current_token = torch.tensor([[eos_index]], dtype=torch.long)
	kwargs = {"hiddens": hiddens}
	output, hiddens = model(current_token, **kwargs)
	log_probs = torch.nn.functional.log_softmax(output, dim=1)[0, :].detach().numpy()
	# log_probs = output.squeeze().log_softmax()

	word_list = []
	word_list.append("<s>")
	ln_prob_list = []

	total_sentences = 0
	total_words = 0
	total_log10_prob = 0
	total_log2_prob = 0

	output_stream = open(output_file, 'w')
	with torch.no_grad():
		for i in range(len(data_sequence)):
			word_list.append(id_to_word[data_sequence[i]])
			ln_prob_list.append(log_probs[data_sequence[i]])

			if data_sequence[i] == eos_index:
				word_list[-1] = "</s>"
				log_prob_list = numpy.asarray(ln_prob_list)
				output_stream.write("%s\n" % " ".join(word_list[1:-1]))
				for j in range(0, len(word_list) - 1):
					output_stream.write("\tp( %s | %s ...)\t= [nnlm] %g [ %g ]\n" % (
						word_list[j + 1], word_list[j], numpy.exp(log_prob_list[j]),
						log_prob_list[j] / numpy.log(10)))
				output_stream.write("1 sentences, %d words, 0 OOVs\n" % (len(word_list) - 2))
				output_stream.write("0 zeroprobs, logprob= %g ppl= %g\n" % (
					numpy.sum(log_prob_list / numpy.log(10)),
					numpy.power(2, -numpy.mean(log_prob_list / numpy.log(2)))
				))
				output_stream.write("\n")

				total_sentences += 1
				total_words += len(word_list) - 2
				total_log10_prob += numpy.sum(log_prob_list / numpy.log(10))
				total_log2_prob += numpy.sum(log_prob_list / numpy.log(2))

				word_list.clear()
				word_list.append("<s>")
				ln_prob_list.clear()

			current_token = torch.tensor([[data_sequence[i]]], dtype=torch.long)
			# current_token.fill_(data_sequence[i], dtype=torch.long)
			kwargs = {"hiddens": hiddens}
			output, hiddens = model(current_token, **kwargs)
			# log_probs = output.squeeze().log_softmax()
			log_probs = torch.nn.functional.log_softmax(output, dim=1)[0, :].detach().numpy()

	# next_token_id = torch.multinomial(word_probs, 1)[0]
	# current_token.fill_(next_token_id)

	# prob = torch.nn.functional.softmax(output, dim=1)
	# next_token_id = torch.distributions.categorical.Categorical(prob).sample()
	output_stream.write("file %s: %d sentences, %d words, 0 OOVs\n" % (data_file, total_sentences, total_words))
	output_stream.write("0 zeroprobs, logprob= %g ppl= %g\n" % (
		total_log10_prob, numpy.power(2, -total_log2_prob / (total_words + total_sentences))))

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	model_parser.add_argument("--data_directory", dest="data_directory", action='store', default=None,
	                          help="input directory [None]")
	model_parser.add_argument('--random_seed', dest='random_seed', type=int, default=-1,
	                          help='random seed (default: -1=time)')
	model_parser.add_argument("--eos_token", dest="eos_token", action='store', default="<eos>", help="eos token")
	model_parser.add_argument("--perplexity_file", dest="perplexity_file", action='store', default=None,
	                          help="perplexity output file directory [None]")
	# model_parser.add_argument("--context_window", dest="context_window", type=int, action='store', default=9,
	# help="context window [9]")

	# model_parser.add_argument("--segment_size", dest="segment_size", type=int, action='store', default=100000, help="segment size [100K]")

	model_parser.add_argument("--model_directory", dest="model_directory", action='store', default=None,
	                          help="model directory [None, resume mode if specified]")
	model_parser.add_argument("--model", dest="model", action='store', default="porch.models.mlp.GenericMLP",
	                          help="neural network model [porch.mnist.MLP]")
	model_parser.add_argument("--model_kwargs", dest="model_kwargs", action='store', default="",
	                          help="model kwargs specified for neural network model [None]")

	return model_parser


def validate_options(arguments):
	#arguments.device = "cuda" if torch.cuda.is_available() else "cpu"
	arguments.device = "cpu"
	arguments.device = torch.device(arguments.device)

	assert os.path.exists(arguments.data_directory)
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond
	# assert arguments.context_window > 0
	# assert arguments.segment_size > 0

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

	return arguments


if __name__ == '__main__':
	main()
