# Porch

This is a wrapper of [PyTorch](https://github.com/pytorch/pytorch) with a customizable and user-friendly API.

The package includes a pre-processed copy of [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for experimental purpose.
Each image is reshaped to a 784 dimension float vector, and all pixels are unitized to a float value between 0 and 1.

Please report any bugs or problems to [Ke Zhai](http://kzhai.github.io/).

## Prepare the Environment

Assume the Porch package is downloaded to directory `$PROJECT_SPACE/`, i.e., 

```bash
$PROJECT_SPACE/Porch/
```

To prepare the example MNIST dataset,

```bash
cd $PROJECT_SPACE/Porch/
tar zxvf data/mnist_784.tar.gz
```
	
You should see the data directory called `mnist_784/` containing the `[test|train].[feature|label].npy` files.

## Multi-Layer Perceptron

To start a generic multi-layer perceptron with one hidden layer of 100 ReLU neurons,

```bash
python -um porch.base \
--model=porch.models.mlp.GenericMLP \
--model_kwargs=input_shape:784,dimensions:1000*10,activations:ReLU*None,drop_modes:Dropout,drop_rates:0.2*0.5 \
--loss=cross_entropy \
--data=loadFeatureAndLabel \
--input_directory=./data/mnist_784/ \
--output_directory=./data/mnist_784/ \
--minibatch_size=64 \
--number_of_epochs=10 \
--optimizer_kwargs=lr:1e-3,momentum:0.9 \
--information=accuracy
```

The `--model` option specifies the exact model class you want to train, where `--model_kwargs` specifies the corresponding arguments for that class.
The `--loss` function should be consistant with the output layer activation function.
The `--information` option specifies the information metric you want to track after each epoch.
You may stack as many information functions into the training framework.

Alternatively, you could extend or implement a class with some pre-defined settings.
For example, `porch.models.mlp.MLP_test` extends `porch.models.mlp.GenericMLP` with 1 hidden layer of 1024 `ReLU` units, 0.2 dropout rate on input layer, 0.5 dropout on hidden layer, and `cross_entropy` as the loss of the output layer.

```bash
python -um porch.base \
--model=porch.models.mlp.MLP_test \
--model_kwargs=input_shape:784,output_shape:10 \
--loss=cross_entropy \
--data=loadFeatureAndLabel \
--input_directory=./data/mnist_784/ \
--output_directory=./data/mnist_784/ \
--minibatch_size=64 \
--number_of_epochs=10 \
--optimizer_kwargs=lr:1e-3,momentum:0.9 \
--information=accuracy
```

Similarly for `porch.models.mlp.MLP_GaussianDropout_test` and `porch.models.mlp.MLP_VariationalGaussianDropout_test`, with 1 hidden layer of 1024 `ReLU` units, 0.2 dropout rate on input layer, 0.5 dropout on hidden layer, and `cross_entropy` as the loss of the output layer.

```bash
python -um porch.base \
--model=porch.models.mlp.MLP_GaussianDropout_test \
--model_kwargs=input_shape:784,output_shape:10 \
--loss=cross_entropy \
--data=loadFeatureAndLabel \
--input_directory=./data/mnist_784/ \
--output_directory=./data/mnist_784/ \
--minibatch_size=64 \
--number_of_epochs=10 \
--optimizer_kwargs=lr:1e-3,momentum:0.9 \
--information=accuracy
```

```bash
python -um porch.base \
--model=porch.models.mlp.MLP_VariationalGaussianDropout_test \
--model_kwargs=input_shape:784,output_shape:10 \
--loss=cross_entropy \
--data=loadFeatureAndLabel \
--input_directory=./data/mnist_784/ \
--output_directory=./data/mnist_784/ \
--minibatch_size=64 \
--number_of_epochs=10 \
--optimizer_kwargs=lr:1e-3,momentum:0.9 \
--information=accuracy
```

Under any circumstances, you may also get help information and usage hints by adding `-h` or `--help` option.

## Convolutional Neural Network 2D

Similar to multi-layer perceptron, you can launch a convolutional neural network using the same command, simply refer to `porch.models.cnn` for more information.
One testing example, `porch.models.cnn.CNN_test` extends `porch.models.cnn.Generic2DCNN`, simply specify the input and output parameters.

```bash
python -um porch.base \
--model=porch.models.cnn.CNN_test \
--model_kwargs=input_shape:28*28,input_channel:1,output_shape:10 \
--loss=cross_entropy \
--data=loadFeatureAndLabel \
--input_directory=./data/mnist_1x28x28/ \
--output_directory=./data/mnist_1x28x28/ \
--minibatch_size=64 \
--number_of_epochs=5 \
--optimizer_kwargs=lr:1e-3,momentum:0.9 \
--information=accuracy \
--debug=subsample_dataset
```

## Recurrent Neural Network

```bash
python -um porch.base \
--model=porch.models.rnn.RNN_WordLanguageModel_test \
--model_kwargs=input_shape:33278,embedding_dimension:50,recurrent_dimension:50,drop_rate:0.5,output_shape:33278 \
--loss=cross_entropy \
--data=loadSequence \
--data=toSequenceMinibatch,minibatch_size:20,sequence_length:35 \
--input_directory=./data/wikitext-2/ \
--output_directory=./data/wikitext-2/ \
--minibatch_size=20 \
--number_of_epochs=4 \
--optimizer_kwargs=lr:20 \
--train_kwargs=clip_grad_norm:0.25 \
--information=accuracy
```