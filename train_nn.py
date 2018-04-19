import random, os
import numpy as np
from config import *
import tensorflow as tf
from tf_graph import FlappyGraph

def train_model(X_data, actions, last_jumps, rewards, model=False):
	# create a Flappy tensorflow graph
	flappy_graph = FlappyGraph(int((CANVAS_WIDTH * IMG_SCALE_FACTOR) * round(CANVAS_HEIGHT * IMG_SCALE_FACTOR)) + 1)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		# run a single train step
		_, train_loss, acc = sess.run([flappy_graph.train_step, flappy_graph.loss, flappy_graph.accuracy], 
										feed_dict={flappy_graph.inputs: X_data, flappy_graph.actions: actions, flappy_graph.rewards: rewards, flappy_graph.lr: 0.0001})
		
		# save the model
		saver = tf.train.Saver()
		if not os.path.exists(MODEL_DIR):
			os.makedirs(MODEL_DIR)
			saver.save(sess, os.path.join(MODEL_DIR, "trained_flappy"))

# load data from file
def run_train(model = False):
	print("Loading data...")
	training_images = np.load(os.path.join(DATA_DIR, "images.npy"))
	actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
	rewards = np.load(os.path.join(DATA_DIR, "adjusted_rewards.npy"))
	last_jumps = np.load(os.path.join(DATA_DIR, "last_jumps.npy"))
	X_data = add_jumps_to_training(training_images = training_images, last_jumps = last_jumps)
	print("Training model...")
	
	train_model(X_data[0], actions, last_jumps, rewards, model)

# add frames since last jump as a feature to the image
def add_jumps_to_training(training_images, last_jumps):
	print("Parsing data...")
	iter_counter = 0
	X_data = []
	# (game frames, height, width)
	for i, game in enumerate(training_images):
		X_data.append([])
		for j, image in enumerate(game):
			X_data[iter_counter].append(np.append(image.ravel(), last_jumps[i][j]))
		X_data[iter_counter] = np.array(X_data[iter_counter], dtype=np.float32)
		iter_counter += 1

	return np.array(X_data)

if __name__ == "__main__":
	if len(sys.argv) == 1:
		run_train(model = False)
    elif len(sys.argv) == 2:
    	run_train(model = sys.argv[1])