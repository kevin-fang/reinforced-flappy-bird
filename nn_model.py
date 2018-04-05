import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

learning_rate = 0.01
dropout_rate = 0.8

def neural_network_model():

	# the +1 accounts for last jump time - for now, don't use conv net
	network = input_data(shape=[None, 400 * 708 * 3 + 1], name='input')
	network = fully_connected(network, 512, activation="relu", name="fc1")
	network = dropout(network, dropout_rate)
	network = fully_connected(network, 256, activation="relu", name="fc2")
	network = dropout(network, dropout_rate)
	network = fully_connected(network, 2, activation="softmax", name="fc3")
	network = regression(network, optimizer="adam", learning_rate = learning_rate,
							loss = "categorical_crossentropy", name="target")

	model = tflearn.DNN(network, tensorboard_dir="log")
	return model