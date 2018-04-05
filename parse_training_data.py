import os
import numpy as np
import cv2
from config import *

def get_training_data(directory):
	# todo: implement script that recursively reads from all the folders in the directory containing screenshots.
	# once have all these, collect data about last jump and whether jumped from file name
	# then, split into X and y training data.
	actions = []
	last_jumps = []
	images = []
	for root, dirs, files in os.walk(directory):
		path = root.split(os.sep)
		for file in files:
			if "screenshot" in file:
				image = cv2.imread(os.path.join(*path, file))
				info = file.split("_")
				actions.append(info[0])
				last_jumps.append(info[1])
				images.append(image)
				print("Finished file: {}".format(file))
			else:
				print("Skipped file: {}".format(file))

	return actions, last_jumps, images

if __name__ == "__main__":
	actions, last_jumps, images = get_training_data('./screenshots')
	print("Converting training data...")
	actions = np.array(actions)
	last_jumps = np.array(last_jumps, dtype=np.uint8)
	images = np.array(images)

	if not os.path.exists(DATA_DIR):
	   os.makedirs(DATA_DIR)
	print("Saving training data...")
	np.save(os.path.join(DATA_DIR, "actions.npy"), actions)
	np.save(os.path.join(DATA_DIR, "last_jumps.npy"), last_jumps)
	np.save(os.path.join(DATA_DIR, "images.npy"), images)