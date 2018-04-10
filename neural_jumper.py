import random
import tflearn
import cv2
import numpy as np
import preprocess
import tensorflow as tf
from tf_graph import FlappyGraph

graph = FlappyGraph(11361)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

'''
- Want frames alive to be the reward
- Feed in Flappy Bird image + last jumps to convolutional (?) neural net, have it predict whether to jump or stay
'''

# image takes input of the screenshot, last_jump is the number of frames since the last jump
def get_jump(img, last_jump):
	image = preprocess.bw_shrink(img)
	flattened_img = image.ravel().reshape([1, image.shape[0] * image.shape[1]])
	X_data = np.append(flattened_img, last_jump)
	logits = sess.run(graph.y_logits, feed_dict={graph.inputs: np.array([X_data])})
	#print(logits)
	return logits