import random
import cv2
import numpy as np
from preprocess import *
from config import *
import tensorflow as tf
from tf_graph import FlappyGraph
import os

graph = FlappyGraph(int((CANVAS_WIDTH * IMG_SCALE_FACTOR) * round(CANVAS_HEIGHT * IMG_SCALE_FACTOR)) + 1, None)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, os.path.join(MODEL_DIR, 'trained_flappy'))

'''
- Want frames alive to be the reward
- Feed in Flappy Bird image + last jumps to convolutional (?) neural net, have it predict whether to jump or stay
'''

# image takes input of the screenshot, last_jump is the number of frames since the last jump
def get_jump(buf, last_jump):
	image = bw(shrink(decode_image_buffer(buf)))
	flattened_img = image.ravel().reshape([1, image.shape[0] * image.shape[1]])
	X_data = np.append(flattened_img, last_jump)
	logits = sess.run(graph.y_logits, feed_dict={graph.inputs: np.array([X_data])})
	#print(logits)
	return logits