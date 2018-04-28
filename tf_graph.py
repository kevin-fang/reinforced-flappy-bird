import tensorflow as tf
from config import *
import numpy as np

class FlappyGraph:
    def __init__(self, input_dims):
        L1 = 5
        output_dim = 1
        tf.reset_default_graph()
        self.inputs = tf.placeholder(tf.float32, [None, input_dims], name='inputs')

        self.actions = tf.placeholder(tf.float32, [None], name='actions')
        self.rewards = tf.placeholder(tf.float32, [None], name='rewards')

        # single layer neural network
        W1 = tf.Variable(tf.truncated_normal([input_dims, L1], stddev=0.01, dtype=tf.float32))
        b1 = tf.Variable(tf.ones(L1))

        self.b1 = b1
        self.W1 = W1
        
        W3 = tf.Variable(tf.truncated_normal([input_dims, output_dim], stddev=0.01, dtype=tf.float32))
        b3 = tf.Variable(tf.ones(output_dim))
        self.b3 = b3

        y1 = tf.nn.leaky_relu(tf.matmul(self.inputs, W1) + b1, name='fc1')

        self.W3 = W3
        
        self.y_logits = tf.matmul(self.inputs, W3) + b3
        self.sigmoid = tf.sigmoid(self.y_logits)
        
        reshaped_actions = tf.reshape(self.actions, [-1, 1])
        self.new_prob = ((reshaped_actions - 1) + self.sigmoid) * (2 * reshaped_actions - 1)
        self.loss = tf.reduce_mean(self.rewards * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_logits, labels=reshaped_actions))
    
        self.grads = tf.gradients(self.loss, [self.b3, self.W3])
        self.lr = tf.placeholder(tf.float32)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)