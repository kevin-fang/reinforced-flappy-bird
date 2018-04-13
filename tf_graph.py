import tensorflow as tf

class FlappyGraph:
	def __init__(self, img_size):
		L1 = 200
		output_dim = 1

		tf.reset_default_graph()
		self.inputs = tf.placeholder(tf.float32, [None, img_size])
		self.actions = tf.placeholder(tf.float32, [None, output_dim])

		W1 = tf.Variable(tf.truncated_normal([img_size, L1], stddev=0.001, dtype=tf.float32))
		b1 = tf.Variable(tf.zeros(L1))

		W2 = tf.Variable(tf.truncated_normal([L1, output_dim], stddev=0.001, dtype=tf.float32))
		b2 = tf.Variable(tf.ones(output_dim) * -2.8)

		y1 = tf.nn.relu(tf.matmul(self.inputs, W1) + b1)
		self.y_logits = tf.matmul(y1, W2) + b2

		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_logits, labels=self.actions))

		self.lr = tf.placeholder(tf.float32)
		self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
		self.correct_prediction = tf.equal(tf.argmax(self.actions, 1), tf.argmax(self.y_logits, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))