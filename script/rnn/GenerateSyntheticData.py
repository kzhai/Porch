# import logging
import datetime
import os
import timeit

import numpy
import torch

import porch


def generate_sequence(network,
                      eos_index,
                      # word_to_id,
                      # id_to_word,
                      number_of_samples,
                      # eos="<eos>",
                      reset_probability=0.01,
                      hidden_init_method=torch.rand,
                      ):
	network.eval()
	current_token = torch.tensor([[eos_index]])
	generated_sequence = []

	with torch.no_grad():
		hiddens = porch.models.rnn.initialize_hidden_states(network, 1)
		while True:
			# assert tokens.shape[0] == 1
			# hiddens_temp = porch.base.detach(hiddens)
			kwargs = {"hiddens": hiddens}
			# cache.append(hiddens_temp)
			output, hiddens = network(current_token, **kwargs)
			prob = torch.nn.functional.softmax(output, dim=1)
			next_token = torch.distributions.categorical.Categorical(prob).sample()
			generated_sequence.append(next_token.item())
			current_token = torch.unsqueeze(next_token, dim=1)

			if generated_sequence[-1] == eos_index:
				if len(generated_sequence) > number_of_samples:
					return generated_sequence
				if reset_probability > 0 and numpy.random.random() < reset_probability:
					hiddens = porch.models.rnn.initialize_hidden_states(network, 1, method=hidden_init_method)


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

	# from .ProjectHiddensFromRandom import load_dictionary
	import porch.data
	word_to_id, id_to_word = porch.data.import_vocabulary(settings.input_type)
	# phrase_to_sequence = indexify(os.path.join(settings.phrase_directory, "phrases.txt"), word_to_id)
	assert settings.eos_token in word_to_id
	eos_index = word_to_id[settings.eos_token]

	model = settings.model(**settings.model_kwargs).to(settings.device)
	model_file = os.path.join(settings.model_directory, "model.pth")
	model.load_state_dict(torch.load(model_file))
	print('Successfully load model state from {}'.format(model_file))

	torch.manual_seed(settings.random_seed)

	start_train = timeit.default_timer()

	output_stream = open(settings.output_sequence, 'w')

	model.eval()
	current_token = torch.tensor([[eos_index]])

	with torch.no_grad():
		hiddens = porch.models.rnn.initialize_hidden_states(model, 1)
		count = 0
		while True:
			kwargs = {"hiddens": hiddens}

			output, hiddens = model(current_token, **kwargs)
			prob = torch.nn.functional.softmax(output, dim=1)
			next_token = torch.distributions.categorical.Categorical(prob).sample()

			if next_token.item() == eos_index:
				output_stream.write("\n")
			else:
				output_stream.write(id_to_word[next_token.item()])
				output_stream.write(" ")

			if count > settings.number_of_samples and next_token.item() == eos_index:
				break

			count += 1
			current_token = torch.unsqueeze(next_token, dim=1)

	end_train = timeit.default_timer()

	print('The code for file {} ran for {:.2f}m'.format(os.path.split(__file__)[1], (end_train - start_train) / 60.))


def add_options(model_parser):
	# generic argument set 1
	model_parser.add_argument("--input_type", dest="input_type", action='store', default=None,
	                          help="input type mapping file [None]")
	model_parser.add_argument("--output_sequence", dest="output_sequence", action='store', default=None,
	                          help="output sequence file [None]")
	model_parser.add_argument('--random_seed', type=int, default=-1, help='random seed (default: -1=time)')

	# generic argument set 3
	model_parser.add_argument("--number_of_samples", dest="number_of_samples", type=int, action='store', default=-1,
	                          help="number of samples [-1]")
	model_parser.add_argument("--eos_token", dest="eos_token", action='store', default="<eos>", help="eos token")

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
	arguments.device = "cpu"
	arguments.device = torch.device(arguments.device)

	# generic argument set snapshots
	if arguments.random_seed < 0:
		arguments.random_seed = datetime.datetime.now().microsecond

	# generic argument set 3
	assert arguments.number_of_samples > 0
	# assert arguments.perturbation_tokens > 0

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
	assert os.path.exists(arguments.input_type)
	# assert os.path.exists(arguments.phrase_directory)

	return arguments


if __name__ == '__main__':
	main()
