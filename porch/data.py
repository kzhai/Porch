import collections
import logging
import os
import sys

import numpy
import torch

logger = logging.getLogger(__name__)

__all__ = [
	"load_datasets",
	#
	"loadFeatureAndLabel",
	"loadSequence",
	"batchify",
	"sequencify",
	# "toSequenceMinibatch",
	# "convert_to_sequence_minibatch",
	# "batchify_and_sequencify",
	"export_vocabulary",
	"import_vocabulary"
]


def loadSequence(input_directory, data_mode="test"):
	dataset = numpy.load(os.path.join(input_directory, "%s.npy" % data_mode))
	dataset = torch.from_numpy(dataset)
	logger.info("Successfully load %d %s data from %s..." % (len(dataset), data_mode, input_directory))
	return dataset


def loadFeatureAndLabel(input_directory, data_mode="test"):
	dataset_x = numpy.load(os.path.join(input_directory, "%s.feature.npy" % data_mode))
	dataset_y = numpy.load(os.path.join(input_directory, "%s.label.npy" % data_mode))
	dataset_x = torch.from_numpy(dataset_x)
	dataset_y = torch.from_numpy(dataset_y).to(torch.int64)
	assert len(dataset_x) == len(dataset_y)
	logger.info("Successfully load %d %s data from %s..." % (len(dataset_x), data_mode, input_directory))
	return (dataset_x, dataset_y)


'''
def convert_to_sequence_minibatch(dataset, **kwargs):
	print(kwargs)
	assert ("minibatch_size" in kwargs)
	minibatch_size = kwargs["minibatch_size"]
	assert ("sequence_length" in kwargs)
	sequence_length = kwargs["sequence_length"]
	if type(minibatch_size) is not int:
		minibatch_size = int(minibatch_size)
	if type(sequence_length) is not int:
		sequence_length = int(sequence_length)

	train_dataset, validate_dataset, test_dataset = dataset
	train_dataset = batchify_and_sequencify(train_dataset, minibatch_size=minibatch_size,
	                                        sequence_length=sequence_length)
	validate_dataset = batchify_and_sequencify(validate_dataset, minibatch_size=minibatch_size,
	                                           sequence_length=sequence_length)
	test_dataset = batchify_and_sequencify(test_dataset, minibatch_size=minibatch_size, sequence_length=sequence_length)
	return train_dataset, validate_dataset, test_dataset
'''


# @TODO: deprecated
def batchify_and_sequencify(dataset, **kwargs):
	assert ("minibatch_size" in kwargs)
	minibatch_size = kwargs.get("minibatch_size")
	if type(minibatch_size) is not int:
		minibatch_size = int(minibatch_size)

	assert ("sequence_length" in kwargs)
	sequence_length = kwargs.get("sequence_length")
	if type(sequence_length) is not int:
		sequence_length = int(sequence_length)

	# Work out how cleanly we can divide the dataset into bsz parts.
	temp_sequence_length = dataset.size(0) // minibatch_size
	# Trim off any extra elements that wouldn't cleanly fit (remainders).
	dataset = dataset.narrow(0, 0, temp_sequence_length * minibatch_size)
	# Evenly divide the data across the bsz batches.
	dataset = dataset.view(minibatch_size, -1).contiguous()

	number_of_sequences = (temp_sequence_length - 1) // sequence_length
	dataset_x = numpy.zeros((minibatch_size * number_of_sequences, sequence_length), dtype=numpy.int)
	dataset_y = numpy.zeros((minibatch_size * number_of_sequences, sequence_length), dtype=numpy.int)

	for i in range(number_of_sequences):
		dataset_x[i * minibatch_size:(i + 1) * minibatch_size, :] = \
			dataset[:, i * sequence_length:(i + 1) * sequence_length]
		# print("dataset_y", dataset_y[i * minibatch_size:(i + 1) * minibatch_size, :].shape, i * minibatch_size, (i + 1) * minibatch_size)
		# print("dataset", dataset[:, i * sequence_length + 1:(i + 1) * sequence_length + 1].shape, i * sequence_length + 1, (i + 1) * sequence_length + 1)
		dataset_y[i * minibatch_size:(i + 1) * minibatch_size, :] = \
			dataset[:, i * sequence_length + 1:(i + 1) * sequence_length + 1]

	return torch.tensor(dataset_x), torch.tensor(dataset_y)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(sequence, **kwargs):
	# assert ("batch_size" in kwargs)
	minibatch_size = kwargs.get("batch_size", 1)
	if type(minibatch_size) is not int:
		minibatch_size = int(minibatch_size)

	# Work out how cleanly we can divide the dataset into bsz parts.
	temp_sequence_length = sequence.size(0) // minibatch_size
	# Trim off any extra elements that wouldn't cleanly fit (remainders).
	sequences = sequence.narrow(0, 0, temp_sequence_length * minibatch_size)
	# Evenly divide the data across the bsz batches.
	sequences = sequences.view(minibatch_size, -1).contiguous()

	'''
	# Work out how cleanly we can divide the dataset into bsz parts.
	temp_sequence_length = dataset.size(0) // minibatch_size
	# Trim off any extra elements that wouldn't cleanly fit (remainders).
	dataset = dataset.narrow(0, 0, temp_sequence_length * minibatch_size)
	# Evenly divide the data across the bsz batches.
	dataset = dataset.view(minibatch_size, -1).t().contiguous()
	'''

	return sequences


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def sequencify(sequences, **kwargs):
	# assert ("sequence_length" in kwargs)
	sequence_length = kwargs.get("sequence_length", 0)
	if type(sequence_length) is not int:
		sequence_length = int(sequence_length)
	if sequence_length <= 0:
		sequence_length = sequences.shape[1] - 1

	minibatch_size = sequences.shape[0]
	temp_sequence_length = sequences.shape[1]
	number_of_sequences = (temp_sequence_length - 1) // sequence_length
	dataset_x = numpy.zeros((minibatch_size * number_of_sequences, sequence_length), dtype=numpy.int)
	dataset_y = numpy.zeros((minibatch_size * number_of_sequences, sequence_length), dtype=numpy.int)

	for i in range(number_of_sequences):
		# print(dataset_x[i * minibatch_size:(i + 1) * minibatch_size, :].shape)
		# print(sequences[:, i * sequence_length:(i + 1) * sequence_length].shape)
		dataset_x[i * minibatch_size:(i + 1) * minibatch_size, :] = \
			sequences[:, i * sequence_length:(i + 1) * sequence_length]
		# print("dataset_y", dataset_y[i * minibatch_size:(i + 1) * minibatch_size, :].shape, i * minibatch_size, (i + 1) * minibatch_size)
		# print("dataset", dataset[:, i * sequence_length + 1:(i + 1) * sequence_length + 1].shape, i * sequence_length + 1, (i + 1) * sequence_length + 1)
		dataset_y[i * minibatch_size:(i + 1) * minibatch_size, :] = \
			sequences[:, i * sequence_length + 1:(i + 1) * sequence_length + 1]

	dataset_x = torch.tensor(dataset_x)
	dataset_y = torch.tensor(dataset_y)

	return dataset_x, dataset_y


# @TODO: deprecated
def backup_batchify(data, i, sequence_length=35):
	seq_len = min(sequence_length, len(data) - 1 - i)
	minibatch_x = data[i:i + seq_len]
	minibatch_y = data[i + 1:i + 1 + seq_len].view(-1)
	return minibatch_x, minibatch_y


def load_datasets(input_directory, data_mode="test",
                  function_parameter_mapping=collections.OrderedDict({loadSequence: {}})):
	dataset = None
	for data_function in function_parameter_mapping:
		if dataset is None:
			dataset = data_function(input_directory, data_mode)
		else:
			data_parameter = function_parameter_mapping[data_function]
			dataset = data_function(dataset, **data_parameter)

	return dataset


def load_corpus(path):
	word_to_id = {}
	id_to_word = {}
	with open(path, 'r') as f:
		for line in f:
			words = line.strip().split()
			for word in words:
				if word in word_to_id:
					continue
				word_to_id[word] = len(word_to_id)
				id_to_word[len(id_to_word)] = word

	return word_to_id, id_to_word


def export_vocabulary(path, id_to_word):
	stream = open(path, 'w')
	for i in range(len(id_to_word)):
		stream.write("%s\n" % id_to_word[i])

	return


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


def tokenize(path, word_to_id):
	with open(path, 'r') as f:
		ids = []
		for line in f:
			words = line.strip().split()  # + ['<eos>']
			for word in words:
				ids.append(word_to_id[word])
		ids = numpy.asarray(ids)
	# ids = torch.LongTensor(ids)

	return ids


if __name__ == '__main__':

	input_directory = sys.argv[1]
	output_directory = sys.argv[2]

	all_data = os.path.join(input_directory, 'all.txt')
	word_to_id, id_to_word = load_corpus(all_data)

	type_info = os.path.join(output_directory, 'type.info')
	export_vocabulary(type_info, id_to_word)
	for file_type in ["train", "test", "validate"]:
		input_file_path = os.path.join(input_directory, "%s.txt" % file_type)
		output_file_path = os.path.join(output_directory, "%s.npy" % file_type)
		data_matrix = tokenize(input_file_path, word_to_id)
		numpy.save(output_file_path, data_matrix)
