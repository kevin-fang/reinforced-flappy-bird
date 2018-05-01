import cv2
import numpy as np
from config import *

# read an image from a buffer created by pygame
def decode_image_buffer(buf):
	nparr = np.fromstring(buf, dtype=np.uint8)
	#print(nparr.shape)
	reshaped = nparr.reshape([CANVAS_HEIGHT, CANVAS_WIDTH, 3])
	# change the colors - because pygame uses different colors
	recolored = cv2.cvtColor(reshaped, cv2.COLOR_BGR2RGB)
	return recolored

# return a black and white version of input image
def bw(img):
	bw_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	return bw_img

# downscale the image to whatever IMG_SCALE_FACTOR is
def shrink(img):
	resized = cv2.resize(img, (0, 0), fx=IMG_SCALE_FACTOR , fy=IMG_SCALE_FACTOR)
	return resized