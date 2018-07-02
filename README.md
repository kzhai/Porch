# Porch

Run sparse variational dropout

	python3.5 -um porch.mnist \
	--input_directory=../PyANN/input/mnist_784_unitized/ \
	--output_directory=../PyANN/output/mnist_784_unitized/ \
	--loss=nll_loss \
	--information=accuracy \
	--minibatch_size=64 \
	--number_of_epochs=30 \
	--optimization_kwargs=lr:1e-2,momentum:0.9 \
	--snapshot=snapshot_dropout

	python3.5 -um porch.mnist \
	--input_directory=../PyANN/input/mnist_784_unitized/ \
	--output_directory=../PyANN/output/mnist_784_unitized/ \
	--loss=nll_loss \
	--regularizer=vardrop_kld_approximation \
	--information=accuracy \
	--minibatch_size=64 \
	--number_of_epochs=30 \
	--optimization_kwargs=lr:1e-2,momentum:0.9 \
	--snapshot=snapshot_dropout
	
	

python3 -um porch.base --model=porch.models.mnist.MLPGeneric --model_kwargs=dimensions:784*1000*10,activations:ReLU*Softmax --loss=cross_entropy --input_directory=../PyANN/input/mnist_784_unitized/ --output_directory=../PyANN/output/mnist_784_unitized/ --minibatch_size=64 --number_of_epochs=100 --optimizer_kwargs=lr:1e-3,momentum:0.9 --information=accuracy

python3.5 -um porch.mnist --input_directory=../PyANN/input/mnist_784_unitized/ --output_directory=../PyANN/output/mnist_784_unitized/ --loss=nll_loss --minibatch_size=64 --number_of_epochs=10 --optimizer_kwargs=lr:1e-2,momentum:0.9,drop_rate:0.2 --information=accuracy --snapshot=snapshot_dropout


python -um porch.base \
--model=porch.models.cnn.CNN_3x32x32_11pts \
--model_kwargs=num_classes:10 \
--loss=nll_loss \
--input_directory=../PyANN/input/cifar10_3x32x32_rgb.gcn+zca/ \
--output_directory=../PyANN/output/cifar10_3x32x32_rgb.gcn+zca/ \
--minibatch_size=64 \
--number_of_epochs=10 \
--optimizer_kwargs=lr:1e-3,momentum:0.9 \
--information=accuracy \
--debug=subsample_dataset