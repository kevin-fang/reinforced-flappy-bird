import cv2
import numpy as np
from config import *

def decode_image_buffer(buf):
	nparr = np.fromstring(buf, dtype=np.uint8)
	#print(nparr.shape)
	reshaped = nparr.reshape([CANVAS_HEIGHT, CANVAS_WIDTH, 3])
	recolored = cv2.cvtColor(reshaped, cv2.COLOR_BGR2RGB)
	return recolored

def bw(img):
	bw_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	return bw_img

def shrink(img):
	resized = cv2.resize(img, (0, 0), fx=IMG_SCALE_FACTOR , fy=IMG_SCALE_FACTOR)
	return resized