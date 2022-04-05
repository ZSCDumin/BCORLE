# coding: utf-8
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.tf_utils import Smooth_L1_Loss
import math

class FQE:
	def __init__(
            self,
			replay_buffer,
			replay_buffer_train,
			ckpt_path,
			save_dir,
            action_dim,
            n_features,
            n_w,
            learning_rate,
            reward_decay,
			number_users, Lambda_size, Lambda_interval
    ):
		self.ckpt_path = ckpt_path
		self.save_dir = save_dir
		self.replay_buffer = replay_buffer
		self.action_dim = action_dim
		self.n_features = n_features
		self.n_w = n_w
		self.lr = learning_rate
		self.gamma = reward_decay
		self.number_users = number_users
		self.Lambda_size = Lambda_size
		self.Lambda_interval = Lambda_interval

		self._next_action()

		# consist of [target_net, evaluate_net]
		self._build_net()

		t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

		with tf.variable_scope('hard_replacement'):
			self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		self.sess = tf.Session()

		self.sess.run(tf.global_variables_initializer())

	def _next_action(self):
		# 读取已保存的模型
		sess = tf.Session()
		# 先加载图和参数变量
		self.ckpt_path = '.'.join([self.ckpt_path, 'meta'])

		saver = tf.train.import_meta_graph(self.ckpt_path)
		saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))

		# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
		graph = tf.get_default_graph()

		# 模型数值导入
		next_action = graph.get_tensor_by_name("next_action:0")

		state = graph.get_tensor_by_name("state:0")
		Lambda = graph.get_tensor_by_name("lambda:0")


		feed_dict = {state: self.replay_buffer.new_batch_data["state"],
						 Lambda: self.replay_buffer.new_batch_data["Lambda"]}
		Q_next_action = sess.run(next_action, feed_dict)

		self.replay_buffer.new_batch_data["next_action"] = Q_next_action

	def _build_net(self):
		# ------------------ all inputs ------------------------
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
		self.r = tf.placeholder(tf.float32, [None, 1], name='r')  # input Reward
		self.a = tf.placeholder(tf.float32, [None, self.action_dim], name='a')  # input Action
		self.w = tf.placeholder(tf.float32, [None, self.n_w], name='w')  # input Action
		self.a_ = tf.placeholder(tf.float32, [None, self.action_dim], name='a_')  # input Next_Action
		self.done = tf.placeholder(tf.float32, [None, 1], name='done')

		#w_initializer, b_initializer = tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0.01)

		# ------------------ build evaluate_net ------------------
		with tf.variable_scope('eval_net',reuse=tf.AUTO_REUSE):

			r10 = tf.layers.batch_normalization(tf.concat([self.s, self.w], 1), axis=-1, momentum=0.99,
												epsilon=0.001,
												center=True,
												scale=True,
												beta_initializer=tf.zeros_initializer(),
												gamma_initializer=tf.ones_initializer(),
												moving_mean_initializer=tf.zeros_initializer(),
												moving_variance_initializer=tf.ones_initializer(),
												beta_regularizer=None,
												gamma_regularizer=None,
												beta_constraint=None,
												gamma_constraint=None,
												training=False,
												trainable=True,
												name=None,
												reuse=tf.AUTO_REUSE,
												renorm=False,
												renorm_clipping=None,
												renorm_momentum=0.99,
												fused=None)
			r11 = layers.fully_connected(r10, 128, activation_fn=tf.nn.relu)
			r12 = layers.fully_connected(r11, 64, activation_fn=tf.nn.relu)

			r21 = layers.fully_connected(self.a, 64, activation_fn=tf.nn.relu)
			r22 = layers.fully_connected(r21, 64, activation_fn=tf.nn.relu)

			r3 = tf.concat((r12, r22), axis=1)
			self.q_eval = layers.fully_connected(r3, self.action_dim, activation_fn=None)

			# ------------------ build evaluate_net ------------------
		with tf.variable_scope('target_net', reuse=tf.AUTO_REUSE):
			r10 = tf.layers.batch_normalization(tf.concat([self.s_, self.w], 1), axis=-1, momentum=0.99,
												epsilon=0.001,
												center=True,
												scale=True,
												beta_initializer=tf.zeros_initializer(),
												gamma_initializer=tf.ones_initializer(),
												moving_mean_initializer=tf.zeros_initializer(),
												moving_variance_initializer=tf.ones_initializer(),
												beta_regularizer=None,
												gamma_regularizer=None,
												beta_constraint=None,
												gamma_constraint=None,
												training=False,
												trainable=True,
												name=None,
												reuse=tf.AUTO_REUSE,
												renorm=False,
												renorm_clipping=None,
												renorm_momentum=0.99,
												fused=None)
			r11 = layers.fully_connected(r10, 128, activation_fn=tf.nn.relu)
			r12 = layers.fully_connected(r11, 64, activation_fn=tf.nn.relu)

			r21 = layers.fully_connected(self.a_, 64, activation_fn=tf.nn.relu)
			r22 = layers.fully_connected(r21, 64, activation_fn=tf.nn.relu)

			r3 = tf.concat((r12, r22), axis=1)
			self.q_target = layers.fully_connected(r3, self.action_dim, activation_fn=None)

		with tf.variable_scope('q_target'):
			q_target = self.r + self.gamma * self.q_target * self.done    # shape=(None, )
			self.q_target = tf.stop_gradient(q_target)
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval, name='TD_error'))
		with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
			self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


	def learn(self, state, action, reward, Lambda, next_state, next_action, done, index):

		#shuffled_indices = np.random.permutation(len(self.replay_buffer.new_batch_data["state"]))
		if index % 10 == 0:
			self.sess.run(self.target_replace_op)
		_, loss = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: state,
                self.a: action,
                self.r: reward,
                self.w: Lambda,
                self.s_: next_state,
				self.a_: next_action,
				self.done: 1 - done
            })

		return loss

	def evaluate(self, next_action_pi):
		predict_value = np.zeros([self.Lambda_size, self.number_users])
		index = 0
		batch_size = 10000
		train_times = int(math.ceil(len(self.replay_buffer.new_batch_data["state"]) / batch_size))
		value = np.zeros(len(self.replay_buffer.new_batch_data["state"]))
		for j in range(train_times):
			begin = j * batch_size
			if (j + 1) * batch_size < len(self.replay_buffer.new_batch_data["state"]):
				end = (j + 1) * batch_size
			else:
				end = len(self.replay_buffer.new_batch_data["state"])
			value2 = self.sess.run(self.q_eval, feed_dict=
			{self.s: self.replay_buffer.new_batch_data["state"][begin:end],
			 self.a: next_action_pi[begin:end],
			 self.w: self.replay_buffer.new_batch_data["Lambda"][begin:end]})
			value[begin:end] = np.squeeze(value2)
		while index < len(self.replay_buffer.new_batch_data["done"]):
			ini_index = index
			while self.replay_buffer.new_batch_data["done"][index][0] != 1.:
				index += 1
			predict_value[int(round(self.replay_buffer.new_batch_data["Lambda"][index][0] / self.Lambda_interval)),
						  self.replay_buffer.new_batch_data["user_id"][index][0] - 1] = value[ini_index]
			index += 1
		return predict_value


