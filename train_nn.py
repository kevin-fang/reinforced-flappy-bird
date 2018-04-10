import random, os
import numpy as np
from config import *
import tensorflow as tf
from tf_graph import FlappyGraph

def train_model(X_data, actions, last_jumps, model=False):
	flappy_graph = FlappyGraph(11361)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		_, train_loss, acc = sess.run([flappy_graph.train_step, flappy_graph.loss, flappy_graph.accuracy], 
										feed_dict={flappy_graph.inputs: X_data, flappy_graph.actions: actions, flappy_graph.lr: 0.0001})
		saver = tf.train.Saver()

		if not os.path.exists(MODEL_DIR):
			os.makedirs(MODEL_DIR)
			saver.save(sess, os.path.join(MODEL_DIR, "trained_flappy"))

def run_train():
	print("Loading data...")
	training_images = np.load(os.path.join(DATA_DIR, "images.npy"))
	actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
	last_jumps = np.load(os.path.join(DATA_DIR, "last_jumps.npy"))
	X_data = add_jumps_to_training(training_images = training_images, last_jumps = last_jumps)
	print("Training model...")
	
	train_model(X_data, actions, last_jumps)


def add_jumps_to_training(training_images, last_jumps):
	print("Parsing data...")
	flattened_training = training_images.ravel().reshape([training_images.shape[0], training_images.shape[1] * training_images.shape[2]])
	X_data = np.concatenate((flattened_training, np.array([last_jumps]).T), axis=1)
	return X_data

if __name__ == "__main__":
	run_train()