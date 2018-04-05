import random, os
import numpy as np
from config import *

from nn_model import neural_network_model
import tflearn

def train_model(training_images, actions, last_jumps, model=False):
	X = training_data[0]
	y = training_data[1]

	if not model:
		model = neural_network_model()

	model.fit({'input': X}, {'targets', y}, n_epoch = 5, snapshot_step = 500, show_metric = True, run_id="flappy_learning")

	return model

def run_train():
	training_images = np.load(os.path.join(DATA_DIR, "images.npy"))
	actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
	last_jumps = np.load(os.path.join(DATA_DIR, "last_jumps.npy"))
	X_data = add_jumps_to_training(training_images = training_images, last_jumps = last_jumps)
	model = train_model(training_images = training_images, actions = actions, last_jumps = last_jumps)

	if not os.path.exists(MODEL_DIR):
		os.makedirs(directory)
	model.save(os.path.join(MODEL_DIR, "trained_flappy.model"))

def add_jumps_to_training(training_images, last_jumps):
	flattened_training = training_images.ravel().reshape([2420, 708 * 400 * 3])
	X_data = np.concatenate((flattened_training, np.array([last_jumps]).T), axis=1)
	return X_data

if __name__ == "__main__":
	run_train()