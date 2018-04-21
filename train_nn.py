import random, os, numpy as np, tensorflow as tf
from tf_graph import FlappyGraph
from config import *

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

flappy_graph = FlappyGraph(int((CANVAS_WIDTH * IMG_SCALE_FACTOR) * round(CANVAS_HEIGHT * IMG_SCALE_FACTOR)) + 1)
init = tf.global_variables_initializer()
global sess
sess = tf.Session()
sess.run(init)

global saver
saver = tf.train.Saver()
def save_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    saver.save(sess, MODEL_PATH)

def train_iteration():
    print("Loading data...")
    training_images = np.load(os.path.join(DATA_DIR, "images.npy"))
    actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
    rewards = np.load(os.path.join(DATA_DIR, "adjusted_rewards.npy"))
    last_jumps = np.load(os.path.join(DATA_DIR, "last_jumps.npy"))
    X_data = add_jumps_to_training(training_images = training_images, last_jumps = last_jumps)

    all_x_data = np.array([])
    for i, _ in enumerate(X_data):
        game = X_data[i].ravel().reshape(-1, X_data[i].shape[0])
        all_x_data = np.append(all_x_data, game)
    all_x_data = all_x_data.reshape([-1, 11361])

    all_actions = np.array([])
    for i, _ in enumerate(actions):
        game = actions[i].ravel()
        all_actions = np.append(all_actions, game)

    all_rewards = np.array([])
    for i, _ in enumerate(rewards):
        game = rewards[i].ravel()
        all_rewards = np.append(all_rewards, game)

    for j in range(5):
        randomize = np.arange(len(all_rewards))
        np.random.shuffle(randomize)
        all_x_data = all_x_data[randomize]
        all_rewards = all_rewards[randomize]
        all_actions = all_actions[randomize]

        #print(all_x_data.shape, all_rewards.shape, all_actions.shape)

        rwds, new_prob, _, train_loss = sess.run([flappy_graph.rewards, flappy_graph.new_prob, flappy_graph.train_step, flappy_graph.loss], 
                    feed_dict={
                        flappy_graph.inputs: all_x_data, 
                        flappy_graph.actions: all_actions, 
                        flappy_graph.rewards: all_rewards, 
                        flappy_graph.lr: 1e-4}
                    )
            #print(new_prob.shape, rwds.shape)
            #print("loss", train_loss, "new_prob and rewards: ", list(zip(new_prob, rwds)))


    '''
    for i in range(NUM_GAMES):
        print("Training on game {}".format(i))
        for j in range(5):
            randomize = np.arange(len(X_data[i]))
            np.random.shuffle(randomize)
            X_i = X_data[i][randomize]
            actions_i = actions[i][randomize]
            rewards_i = rewards[i][randomize]
            #print(sess.run([tf.shape(flappy_graph.y_logits), tf.shape(flappy_graph.sigmoid), tf.shape(flappy_graph.actions)], feed_dict={
            #                flappy_graph.inputs: X_data_i 
            #                flappy_graph.actions: actions_, 
            #                flappy_graph.rewards: rewards[i], 
            #                flappy_graph.lr: 1e-4}))
            rwds, new_prob, _, train_loss = sess.run([flappy_graph.rewards, flappy_graph.new_prob, flappy_graph.train_step, flappy_graph.loss], 
                        feed_dict={
                            flappy_graph.inputs: X_i, 
                            flappy_graph.actions: actions_i, 
                            flappy_graph.rewards: rewards_i, 
                            flappy_graph.lr: 1e-5}
                        )
            #print(new_prob.shape, rwds.shape)
            #print("loss", train_loss, "new_prob and rewards: ", list(zip(new_prob, rwds)))
    '''


import run_agent
save_model()

for i in range(100):
    run_agent.run()
    train_iteration()
    save_model()