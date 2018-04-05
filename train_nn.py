import random, os
import numpy as np

from nn_model import neural_network_model
import tflearn

def train_model(training_data, model=False):
	X = training_data[0]
	y = training_data[1]

	if not model:
		model = neural_network_model()

	model.fit({'input': X}, {'targets', y}, n_epoch = 5, snapshot_step = 500, show_metric = True, run_id="flappy_learning")

	return model

def run_train():
	training_data = get_training_data('./screenshots')
	model = train_model(training_data)

    if not os.path.exists('./models'):
        os.makedirs(directory)
	model.save('./models/trained_flappy.model')

def get_training_data(directory):
	# todo: implement script that recursively reads from all the folders in the directory containing screenshots.
	# once have all these, collect data about last jump and whether jumped from file name
	# then, split into X and y training data.
	pass