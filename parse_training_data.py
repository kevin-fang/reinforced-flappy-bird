import os
import sys
sys.path.append('./FlappyBird')
import numpy as np
from config import *

def get_training_data(directory):
	# counter for number of saved games
	num_saved = 0

	actions = []
	last_jumps = []
	rewards = []
	iter_counter = 0
	images = []

	for game_num in range(1, NUM_GAMES + 1):
		actions.append([])
		rewards.append([])
		last_jumps.append([])
		images.append([])

		game_action = np.load(os.path.join(TRAIN_SCREEN_DIR, "game{}_action.npy".format(game_num)))
		game_data = np.load(os.path.join(TRAIN_SCREEN_DIR, "game{}_data.npy".format(game_num)))
		images[iter_counter].append(game_data)

		for i, action in enumerate(game_action):
			#print(action)
			actions[iter_counter].append(action[1])
			rewards[iter_counter].append(action[2])
			num_saved += 1

		convert = lambda arr: np.array(arr[iter_counter], np.float32)
		rewards[iter_counter], actions[iter_counter] = map(convert, 
			[rewards, actions])

		iter_counter += 1

	print("{} frames saved.".format(num_saved))
	return actions, last_jumps, images, rewards

# calculate adjusted rewards to account for future rewards
def calculate_adjusted_rewards(actions, rewards):
	adjusted_rewards = []
	discount = 0.99
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
				adj_reward += (discount ** k) * future_reward

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