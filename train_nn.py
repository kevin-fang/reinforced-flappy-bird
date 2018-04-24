import random, os, numpy as np, tensorflow as tf
from tf_graph import FlappyGraph
from config import *
import sys
from datetime import datetime

# add frames since last jump to training data
def add_jumps_to_training(training_images, last_jumps):
    print("Parsing data...")
    iter_counter = 0
    X_data = []
    # (game frames, height, width)
    for i, game in enumerate(training_images):
        X_data.append([])
        for j, image in enumerate(game):
            X_data[iter_counter].append(np.append(image.ravel(), last_jumps[i][j]))
        X_data[iter_counter] = np.array(X_data[iter_counter], dtype=np.float32)
        iter_counter += 1

    return np.array(X_data)

# initialize the neural network
flappy_graph = FlappyGraph(NUM_NEURAL_DIMS)
init = tf.global_variables_initializer()
global sess
sess = tf.Session()
sess.run(init)

# functions to load/save a model
global saver
saver = tf.train.Saver()
def save_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    saver.save(sess, MODEL_PATH)

def restore_model():
    saver.restore(sess, MODEL_PATH)

# training iteration
def train_iteration():
    print("Loading data...")
    training_images = np.load(os.path.join(DATA_DIR, "images.npy"))
    actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
    rewards = np.load(os.path.join(DATA_DIR, "adjusted_rewards.npy"))
    last_jumps = np.load(os.path.join(DATA_DIR, "last_jumps.npy"))
    X_data = add_jumps_to_training(training_images = training_images, last_jumps = last_jumps)

    # combine all games into one array for training
    all_x_data = np.array([])
    for i, _ in enumerate(X_data):
        game = X_data[i].ravel().reshape(-1, X_data[i].shape[0])
        all_x_data = np.append(all_x_data, game)
    all_x_data = all_x_data.reshape([-1, NUM_NEURAL_DIMS])

    all_actions = np.array([])
    for i, _ in enumerate(actions):
        game = actions[i].ravel()
        all_actions = np.append(all_actions, game)

    all_rewards = np.array([])
    for i, _ in enumerate(rewards):
        game = rewards[i].ravel()
        all_rewards = np.append(all_rewards, game)

    for j in range(2):

        # shuffle frames in the game data
        randomize = np.arange(len(all_rewards))
        np.random.shuffle(randomize)
        all_x_data = all_x_data[randomize]
        all_rewards = all_rewards[randomize]
        all_actions = all_actions[randomize]

        #print(all_x_data.shape, all_rewards.shape, all_actions.shape)

        # run the training
        grads, b3, rwds, new_prob, _, train_loss = sess.run([flappy_graph.grads, flappy_graph.b3, flappy_graph.rewards, flappy_graph.new_prob, flappy_graph.train_step, flappy_graph.loss], 
                    feed_dict = {
                        flappy_graph.inputs: all_x_data, 
                        flappy_graph.actions: all_actions, 
                        flappy_graph.rewards: all_rewards, 
                        flappy_graph.lr: 1e-2
                        }
                    )
        
        # print debugging information
        print("loss", train_loss, "new_prob and rewards: ", list(zip(new_prob, rwds)))
        print("grads", grads)
        print(b3)

import run_agent

# used for logging
def get_time():
    return str(datetime.now())

# determine whether to save the model or generate a new one
if len(sys.argv) == 1:
    save_model()
    with open('training_log.txt', 'w') as log:
        log.write("[{}] Generating new model...\n".format(get_time()))
else:
    restore_model()
    with open('training_log.txt', 'w') as log:
        log.write("[{}] Loading pretrained model...\n".format(get_time()))

num_iterations = 1

while True:
    run_agent.run()
    train_iteration()
    save_model()
    with open('training_log.txt', 'a') as log:
        log.write("[{}] Finished iteration: {}\n".format(get_time(), num_iterations))
    num_iterations += 1