# Porch

Run sparse variational dropout

	python3.5 -um porch.mnist \
	--input_directory=../PyANN/input/mnist_784_unitized/ \
	--output_directory=../PyANN/output/mnist_784_unitized/ \
	--loss=nll_loss \
	--regularizer=vardrop_kld_approximation \
	--information=accuracy \
	--minibatch_size=64 \
	--number_of_epochs=30 \
	--optimization_kwargs=lr:1e-2,momentum:0.9

