import random
import cv2
import numpy as np
from config import *
import tensorflow as tf
from tf_graph import FlappyGraph
import os
from sklearn import preprocessing

graph = FlappyGraph(NUM_NEURAL_DIMS)

global initialized
initialized = False
sess = tf.Session()

def initialize_network(model = False):
	global initialized
	if not model:
		init = tf.global_variables_initializer()
		sess.run(init)
		initialized = True
	elif model:
		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
		initialized = True
	else:
		initialized = False

'''
- Want frames alive to be the reward
- Feed in Flappy Bird image + last jumps to convolutional (?) neural net, have it predict whether to jump or stay
'''

# image takes input of the screenshot, last_jump is the number of frames since the last jump
def get_jump(data_arr, last_jump):
	if not initialized:
		print("Neural network not initialized. Please run initialize_session() to create a new model.")
		import sys
		sys.exit(1)
	#image = bw(shrink(decode_image_buffer(buf)))

	X_data = np.append(data_arr, last_jump)
	#print(X_data)
	logits, prob = sess.run([graph.y_logits, graph.sigmoid], feed_dict={graph.inputs: np.array([X_data])})
	#print(logits, prob)
	result = np.random.choice(2, 1, p=[1-prob[0][0], prob[0][0]])
	if result == 1:
		print(logits, prob)
	return result