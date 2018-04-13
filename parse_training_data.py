import os
import numpy as np
import cv2
from config import *
from preprocess import *

def get_training_data(directory):
	num_saved = 0
	num_jumps = 0
	num_skips = 0

	actions = []
	last_jumps = []
	images = []
	rewards = []
	iter_counter = 0
	# "{img_num}_{jumped}_{reward}_{last_jump}_capture.jpg"
	for root, dirs, files in os.walk(directory):
		path = root.split(os.sep)
		files = list(filter(lambda filename: True if "capture" in filename else False, files))
		files = sorted(files, key=lambda name: int(name.split("_")[0]))

		if len(files) > 0:
			rewards.append([])
			images.append([])


			for file in files:
				image = cv2.imread(os.path.join(*path, file), cv2.IMREAD_GRAYSCALE)
				img_num, jumped, reward, frames_since_jump, _ = file.split("_")

				actions.append([0, 1] if jumped == "j" else [1, 0])
				rewards[iter_counter].append(reward)
				images[iter_counter].append(image)
				last_jumps.append(frames_since_jump)
				num_saved += 1
				print("Parsed file: {}".format(file))
			rewards[iter_counter] = np.array(rewards[iter_counter], np.float32)
			images[iter_counter] = np.array(images[iter_counter], np.float32)
			iter_counter += 1


	print("{} images saved.".format(num_saved))
	return actions, last_jumps, images, rewards

if __name__ == "__main__":
	actions, last_jumps, images, rewards = get_training_data('./screenshots')
	print("Process training data...")
	actions = np.array(actions, dtype=np.float32)
	last_jumps = np.array(last_jumps, dtype=np.float32)
	images = np.array(images)
	rewards = np.array(rewards)

	if not os.path.exists(DATA_DIR):
	   os.makedirs(DATA_DIR)
	print("Saving data...")
	np.save(os.path.join(DATA_DIR, "actions.npy"), actions)
	np.save(os.path.join(DATA_DIR, "last_jumps.npy"), last_jumps)
	np.save(os.path.join(DATA_DIR, "images.npy"), images)
	np.save(os.path.join(DATA_DIR, "rewards.npy"), rewards)
	