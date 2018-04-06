import random
import tflearn
import cv2
import numpy as np
import preprocess
from nn_model import neural_network_model
'''
- Want frames alive to be the reward
- Feed in Flappy Bird image + last jumps to convolutional (?) neural net, have it predict whether to jump or stay
'''

model = neural_network_model(11361)
model.load("./models/trained_flappy.model")

# image takes input of the screenshot, last_jump is the number of frames since the last jump
def get_jump(img, last_jump):
	image = preprocess.bw_shrink(img)
	flattened_img = image.ravel().reshape([1, image.shape[0] * image.shape[1]])
	X_data = np.append(flattened_img, last_jump)
	result = model.predict(np.array([X_data]))
	print(result)
	return np.argmax(result)