import os
import numpy as np
import cv2
from config import *
import preprocess


def get_training_data(directory):
	num_saved = 0
	num_jumps = 0
	num_skips = 0
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
				image = preprocess.bw_shrink(os.path.join(*path, file))
				info = file.split("_")
				if info[0] == "j":
					num_jumps += 1
					actions.append([0, 1])
				elif num_skips < num_jumps:
					actions.append([1, 0])
					num_skips += 1
				else:
					continue

				num_saved += 1
				actions.append([0, 1] if info[0] == "j" else [1, 0])
				last_jumps.append(info[1])
				images.append(image)
				print("Parsed file: {}".format(file))
			else:
				print("Skipped file: {}".format(file))

	print("{} images saved.".format(num_saved))
	return actions, last_jumps, images

if __name__ == "__main__":
	actions, last_jumps, images = get_training_data('./screenshots')
	print("Converting training data...")
	actions = np.array(actions, dtype=np.float32)
	last_jumps = np.array(last_jumps, dtype=np.float32)
	images = np.array(images)

	if not os.path.exists(DATA_DIR):
	   os.makedirs(DATA_DIR)
	print("Saving training data...")
	np.save(os.path.join(DATA_DIR, "actions.npy"), actions)
	np.save(os.path.join(DATA_DIR, "last_jumps.npy"), last_jumps)
	np.save(os.path.join(DATA_DIR, "images.npy"), images)
	