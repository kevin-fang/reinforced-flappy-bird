import random, os, numpy as np, tensorflow as tf
import sys
sys.path.append('./FlappyBird')
from tf_graph import FlappyGraph
from config import *
from datetime import datetime

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

    # combine all games into one array for training
    all_x_data = np.array([])
    for i, _ in enumerate(training_images):
        game = training_images[i][0].ravel().reshape(-1, training_images[i].shape[0])
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


    losses = []

    # perform n_epochs over the data
    n_epochs = 4
    for j in range(n_epochs):

        # shuffle frames in the game data
        randomize = np.arange(len(all_rewards))
        np.random.shuffle(randomize)
        all_x_data = all_x_data[randomize]
        temp_rewards = all_rewards
        all_rewards = all_rewards[randomize]
        all_actions = all_actions[randomize]

        # run the training
        rwds, _, train_loss = sess.run([flappy_graph.rewards, flappy_graph.train_step, flappy_graph.loss], 
                    feed_dict = {
                        flappy_graph.inputs: all_x_data, 
                        flappy_graph.actions: all_actions, 
                        flappy_graph.rewards: all_rewards, 
                        flappy_graph.lr: 1e-3
                        }
                    )
        
        if j == 1: 
            print(temp_rewards)
        losses.append(train_loss)
        if train_loss == 0:
            print("Loss = 0. Exiting...")
            sys.exit(0)
    return losses

import run_agent

# used for logging
def get_time():
    return str(datetime.now())

# determine whether to save the model or generate a new one
timestamp = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())

# make directory for logs
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_name = os.path.join(LOG_DIR, 'training_log_{}.txt'.format(timestamp))

# choose whether to save or load a model
if len(sys.argv) == 1:
    print("Usage: \nGenerate new model: python train_nn.py -n\nLoad existing model: python train_nn.py -l")
    sys.exit(1)
elif sys.argv[1] == "-l" or sys.argv[1] == "--load":
    restore_model()
    with open(log_name, 'w') as log:
        log.write("[{}] Loading pretrained model...\n".format(get_time()))
elif sys.argv[1] == "-n" or sys.argv[1] == "--new":
    save_model()
    with open(log_name, 'w') as log:
        log.write("[{}] Generating new model...\n".format(get_time()))

num_iterations = 1

while True:
    run_agent.run()
    loss = train_iteration()
    save_model()
    with open(log_name, 'a') as log:
        log.write("[{}] Finished iteration: {}, loss = {}\n".format(get_time(), num_iterations, loss))
    num_iterations += 1