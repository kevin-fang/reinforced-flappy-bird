import random, os
import numpy as np
from config import *

from nn_model import neural_network_model
import tensorflow as tf
import tflearn

def train_model(X_data, actions, last_jumps, model=False):
	if not model:
		print(X_data.shape)
		model = neural_network_model(X_data.shape[1])

	model.fit({'input': X_data}, {'target': actions}, validation_set = 0.1, batch_size=100, n_epoch = 5, snapshot_step = 500, show_metric = True, run_id="flappy_learning")

	return model

def run_train():
	print("Loading data...")
	training_images = np.load(os.path.join(DATA_DIR, "images.npy"))
	actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
	last_jumps = np.load(os.path.join(DATA_DIR, "last_jumps.npy"))
	X_data = add_jumps_to_training(training_images = training_images, last_jumps = last_jumps)
	print("Training model...")
	model = train_model(X_data, actions, last_jumps)

	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	model.save(os.path.join(MODEL_DIR, "trained_flappy.model"))

def add_jumps_to_training(training_images, last_jumps):
	print("Parsing data...")
	flattened_training = training_images.ravel().reshape([training_images.shape[0], training_images.shape[1] * training_images.shape[2]])
	X_data = np.concatenate((flattened_training, np.array([last_jumps]).T), axis=1)
	return X_data

if __name__ == "__main__":
	run_train()