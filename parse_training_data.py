import os
import numpy as np
from config import *

def get_training_data(directory):
	# counter for number of saved games
	num_saved = 0

	actions = []
	last_jumps = []
	images = []
	rewards = []
	iter_counter = 0
	# iterate through the training image directory
	for root, dirs, files in os.walk(directory):
		path = root.split(os.sep)

		# format of file name "{img_num}_{jumped}_{reward}_{last_jump}_capture.jpg"

		# save all the game data, sort by frame number
		files = list(filter((lambda filename: True if "capture" in filename else False), files))
		files = sorted(files, key=lambda name: int(name.split("_")[0]))

		if len(files) > 0:
			# append an empty array to all the information arrays
			actions.append([])
			rewards.append([])
			images.append([])
			last_jumps.append([])

			for file in files:
				# load image from file
				image = np.load(os.path.join(*path, file))
				_, jumped, reward, frames_since_jump, _ = file.split("_")
				# append the information to the different arrays
				actions[iter_counter].append(1 if int(jumped) == 1 else 0)
				rewards[iter_counter].append(float(reward))
				images[iter_counter].append(image)
				last_jumps[iter_counter].append(frames_since_jump)
				num_saved += 1
				#print("Parsed file: {}".format(file))
			# lambda function to convert to float array

			# converts these 2d arrays into numpy arrays - one row for each game. Stored as object though, as array is not rectangular
			convert = lambda arr: np.array(arr[iter_counter], np.float32)
			rewards[iter_counter], actions[iter_counter], last_jumps[iter_counter], images[iter_counter] = map(convert, 
				[rewards, actions, last_jumps, images])
			iter_counter += 1

	print("{} frames saved.".format(num_saved))
	return actions, last_jumps, images, rewards

# calculate adjusted rewards to account for future rewards
def calculate_adjusted_rewards(actions, rewards):
	adjusted_rewards = []
	gamma = 0.95
	iter_counter = 0
	# loop through each game
	for i, game in enumerate(rewards):
		# add a new array for this game
		adjusted_rewards.append([])

		# loop through rewards in a game
		for j, reward in enumerate(rewards[i]):
			adj_reward = 0
			# calculate the adjusted reward, accounting for future frames.
			for k, future_reward in enumerate(rewards[i][j:]):
				#print("gamma ** j: {}, future reward: {}".format(gamma ** j, future_reward))
				adj_reward += (gamma ** k) * future_reward

			adjusted_rewards[iter_counter].append(adj_reward)
		adjusted_rewards[iter_counter] = np.array(adjusted_rewards[iter_counter], dtype=np.float32)
		iter_counter += 1

	return np.array(adjusted_rewards)

def parse_data():
	# get the training data
	print("Processing training data...")
	actions, last_jumps, images, rewards = map(np.array, get_training_data(TRAIN_SCREEN_DIR))

	# calculate adjusted rewards
	print("Calculating adjusted rewards..")
	adjusted_rewards = calculate_adjusted_rewards(actions, rewards)

	# save information to files
	if not os.path.exists(DATA_DIR):
	   os.makedirs(DATA_DIR)
	print("Saving data...")
	np.save(os.path.join(DATA_DIR, "actions.npy"), actions)
	np.save(os.path.join(DATA_DIR, "last_jumps.npy"), last_jumps)
	np.save(os.path.join(DATA_DIR, "images.npy"), images)
	np.save(os.path.join(DATA_DIR, "rewards.npy"), rewards)
	np.save(os.path.join(DATA_DIR, "adjusted_rewards.npy"), adjusted_rewards)
	print("Completed data parsing.")

if __name__ == "__main__":
	parse_data()