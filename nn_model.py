import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

learning_rate = 0.0001
dropout_rate = 0.8

def neural_network_model(input_size):

	# the +1 accounts for last jump time - for now, don't use conv net
	network = input_data(shape=[None, input_size], name='input')
	network = fully_connected(network, 128, activation="relu", name="fc1")
	network = fully_connected(network, 64, activation="relu", name="fc2")
	network = fully_connected(network, 1, activation="sigmoid", name="final")
	network = regression(network, optimizer="adam", learning_rate = learning_rate,
							loss = "categorical_crossentropy", name="target")

	model = tflearn.DNN(network, tensorboard_dir="log")
	return model

	# want to backpropogate it with loss as negative score
	# use pure tensorflow so can see weights
	# difference frames?
	# karpathy.github.io/2016/05/31/rl

	# initialize weights randomly, let it play a game
	# hopefully, with random weights shouldn't get the same answers
	# when play game, change game playing code so when it calls neural network, takes probability and does a coin flip
	# shouldn't have two output neurons, just use sigmoid
	# play game w/random weights. Save the screnshot, move made, and the game number. Don't delete if it didn't work, this time
	# Screenshot is X, move made is Y, also need to save whether game worked or not (score)
	# Want under threshold frames make it negative, above positive
	# Scale frames - -.3 would be living longer than -.7 --> advantages
	# save this for each game
	# retroactively apply screneshot
	# Don't use built in loss, use own loss function
	# loss function is negative sum on karpathy update -Sum(Ai log p(yi,| Xi))
	# Train for a while - play 100 games w/200 frames each (for pong)
	# train on all of X, y with loss + repeat process
	# Won't learn very well after just one round. Have to do many times.
	# Don't give the same A_i to decision - make it so decisions closer to a pipe are more important
	# Each frame, get some reward. Certian reward for staying alive, big negative for dying, even higher reward for making it through pipe (increasing score)
	# value of an action is the sum of decision factor * 0.01 (staying alive) + 0.99^2 * .01 (stayed alive for next frame) + .99^3 * .01 (stayed alive for next frame) + .99^4 * - .001 (died)
	# sum 0.99 ^k * reward for every reward after T --> replace Ai
	# one gradient update per dataset (a bunch of episodes - each episode is a few dozen games)
	# three gradient steps - not optimizing, not minimizing the loss all the way. Updating couple of gradient steps and playing again.	
