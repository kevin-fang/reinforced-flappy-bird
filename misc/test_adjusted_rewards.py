import numpy as np


actions = np.load('./data/actions.npy')
rewards = np.load('./data/rewards.npy')

def calculate_adjusted_rewards(actions, rewards):
	adjusted_rewards = []
	gamma = 0.85
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
			print(adj_reward)
			adjusted_rewards[iter_counter].append(adj_reward)
		adjusted_rewards[iter_counter] = np.array(adjusted_rewards[iter_counter], dtype=np.float32)
		iter_counter += 1

	return np.array(adjusted_rewards)

print(calculate_adjusted_rewards([actions[6]], [rewards[6]]))