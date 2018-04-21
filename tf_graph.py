import tensorflow as tf
from config import *
import numpy as np

class FlappyGraph:
    def __init__(self, img_size):
        L1 = 200
        L2 = 50
        output_dim = 1
        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32, [None, img_size], name='inputs')
        self.actions = tf.placeholder(tf.float32, [None], name='actions')
        self.rewards = tf.placeholder(tf.float32, [None], name='rewards')

        # single layer neural network
        W1 = tf.Variable(tf.truncated_normal([img_size, L1], stddev=0.001, dtype=tf.float32))
        b1 = tf.Variable(tf.ones(L1))

        W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.001, dtype=tf.float32))
        b2 = tf.Variable(tf.ones(L2))
        
        W3 = tf.Variable(tf.truncated_normal([L2, output_dim], stddev=0.001, dtype=tf.float32))
        b3 = tf.Variable(tf.ones(output_dim))

        y1 = tf.nn.relu(tf.matmul(self.inputs, W1) + b1, name='fc1')
        y2 = tf.nn.relu(tf.matmul(y1, W2) + b2, name='fc2')
        
        self.y_logits = tf.matmul(y2, W3) + b3
        self.sigmoid = tf.sigmoid(self.y_logits)
        
        self.new_prob = ((tf.reshape(self.actions, [-1, 1]) - 1) + self.sigmoid) * (2 * tf.reshape(self.actions, [-1, 1]) - 1)
        self.loss = tf.reduce_mean(self.rewards * tf.log(self.new_prob))
    
        self.lr = tf.placeholder(tf.float32)
        self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)