# coding: utf-8

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from utils.tf_utils import Smooth_L1_Loss
from utils.DataProcess_utils import *
import datetime

class FC_Q(object):
	def __init__(
			self, state_dim, num_actions, Lambda_dim, threshold, lr,
			name, q_loss_weight=2e1, i_regularization_weight=1e-1, i_loss_weight=1.0
	):
		self.name = name
		self.state_dim = state_dim
		self.action_dim = num_actions

		# placeholders for Q amd I networks
		self.state_ = tf.placeholder(tf.float32, [None, state_dim], name="state")
		self.lambda_ = tf.placeholder(tf.float32, [None, Lambda_dim], name="lambda")

		with tf.variable_scope(self.name + 'q_net'):
			# Q network
			self.q0 = tf.layers.batch_normalization(tf.concat([self.state_, self.lambda_], 1), axis=-1, momentum=0.99, epsilon=0.001,
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
													reuse=None,
													renorm=False,
													renorm_clipping=None,
													renorm_momentum=0.99,
													fused=None
													)
			self.q1 = layers.fully_connected(self.q0, 1024, activation_fn=tf.nn.relu)
			self.q2 = layers.fully_connected(self.q1, 512, activation_fn=tf.nn.relu)
			self.q2_ = layers.fully_connected(self.q2, 512, activation_fn=tf.nn.relu)
			self.q3 = layers.fully_connected(self.q2_, num_actions, activation_fn=None)

		with tf.variable_scope(self.name + 'i_net'):
			# I network
			self.q0 = tf.layers.batch_normalization(tf.concat([self.state_, self.lambda_], 1), axis=-1, momentum=0.99, epsilon=0.001,
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
													reuse=None,
													renorm=False,
													renorm_clipping=None,
													renorm_momentum=0.99,
													fused=None
													)
			self.i1 = layers.fully_connected(self.q0, 1024, activation_fn=tf.nn.relu)
			self.i2 = layers.fully_connected(self.i1, 512, activation_fn=tf.nn.relu)
			self.i2_ = layers.fully_connected(self.i2, 512, activation_fn=tf.nn.relu)
			self.i3 = layers.fully_connected(self.i2_, num_actions, activation_fn=tf.nn.relu)

		# next_action_possible is the softmax output of i3, which means the origin probability of chosen actions,
		# just an imitation of the logged data, and the final probability of action is a combination of Q value term and
		# imitation term
		self.next_action_possible = tf.nn.log_softmax(self.i3)
		self.imt = tf.exp(self.next_action_possible)
		self.prob = self.imt / tf.reduce_max(self.imt, axis=1, keep_dims=True)[0]
		self.imt = (self.prob > threshold)
		self.imt = tf.cast(self.imt, dtype=tf.float32)
		self.prob_ = self.imt

		# Use large negative number to mask actions from argmax
		self.output_prob = self.imt * self.q3 + (1 - self.imt) * -1e8

		# next action
		self.next_action_ = tf.expand_dims(tf.argmax(self.output_prob, axis=1), axis=1, name = 'next_action')

		# placeholder for current_action
		self.current_action = tf.placeholder(tf.int32, [None, 1], name='current_action')
		self.current_action_one_hot = tf.one_hot(self.current_action, depth=num_actions)
		self.current_q = tf.squeeze(tf.matmul(self.current_action_one_hot, tf.expand_dims(self.q3, axis=-1)), axis=1, name='current_q')
		self.current_action_reduce_dim = tf.squeeze(self.current_action, squeeze_dims=1)

		self.is_weights = tf.placeholder(tf.float32, [None, 1], name='is_weights')
		self.reward1 = tf.placeholder(tf.float32, [None, 1], name='reward1')

		# i loss
		self.i_loss = i_loss_weight * tf.reduce_mean(
			tf.multiply(
				self.is_weights,
				tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=self.current_action_reduce_dim, logits=self.i3), axis=1)
			)
		)
		self.i3_loss = i_regularization_weight * tf.reduce_mean(self.is_weights * tf.pow(self.i3, 2))

		# placeholder for target_Q
		self.target_q = tf.placeholder(tf.float32, [None, 1], name='target_q')
		# q_loss
		self.q_loss = q_loss_weight * Smooth_L1_Loss(
			self.current_q, self.target_q, self.name, self.is_weights
		) + 1e-2 * tf.reduce_mean(tf.square(self.q3))
		self.Q_loss = self.q_loss + self.i_loss + self.i3_loss

		# Optimize the Q
		self.Q_optim_ = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.Q_loss)
		self.q_optim_ = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.q_loss)
		self.i_optim_ = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.i_loss + self.i3_loss)

		# generative network
		self.preprocessed_obs_ph = tf.placeholder(tf.float32, [None, state_dim + Lambda_dim], name="next_state")

		self.g_embedding = tf.placeholder(tf.float32, [None, num_actions], name="g_embedding")

		# IPS evaluation
		self.ips_ratio, self.behaviour_prob, self.target_prob = self.importance_ratio(
			self.g_embedding, self.current_action
		)

	def importance_ratio(self, estimated_pai, action):
		"""calculate the importance_ratio for policy evaluation.
		Arguments:
			pai: the target policy.
			estimated_pai: estimation of behavior policy.
			action: action taken by behavior policy.
		"""
		target_prob = tf.exp(self.next_action_possible)
		one_hot_actions = tf.squeeze(tf.one_hot(action, self.action_dim), axis=1)
		target_prob = tf.expand_dims(tf.reduce_sum(target_prob * one_hot_actions, axis=1), axis=1)
		behavior_prob = tf.expand_dims(tf.reduce_sum(estimated_pai * one_hot_actions, axis=1), axis=1)
		ratio = tf.clip_by_value(target_prob / behavior_prob, 1.0e-2, 1.0e2)

		return ratio, behavior_prob, target_prob

	def _generative_model(self, input_obs, scope="generative_model"):
		"""
		build a behavioral cloning network to learn from offline batch data.
		Arguments:
			input_obs: the (list, dict)[of] input tensor of observation.
			scope: the name of variable scope.
		"""
		raise NotImplementedError

	def get_network_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name)]

	def get_i_network_variables(self):
		return [t for t in tf.trainable_variables() if t.name.startswith(self.name + 'i_net')]


# class PropensityNet(object):
# 	def __init__(self, state_dim, num_actions, Lambda_dim, lr, name):
# 		self.name = name
# 		self.state_dim = state_dim
# 		self.action_dim = num_actions
# 		self.state_ = tf.placeholder(tf.float32, [None, state_dim], name="propensity_state")
# 		self.lambda_ = tf.placeholder(tf.float32, [None, Lambda_dim], name="propensity_lambda")
# 		with tf.variable_scope(self.name + 'i_net'):
# 			# placeholders for PropensityNet
#
# 			# I network
# 			self.i0 = tf.layers.batch_normalization(tf.concat([self.state_, self.lambda_], 1), axis=-1, momentum=0.99,
# 													epsilon=0.001,
# 													center=True,
# 													scale=True,
# 													beta_initializer=tf.zeros_initializer(),
# 													gamma_initializer=tf.ones_initializer(),
# 													moving_mean_initializer=tf.zeros_initializer(),
# 													moving_variance_initializer=tf.ones_initializer(),
# 													beta_regularizer=None,
# 													gamma_regularizer=None,
# 													beta_constraint=None,
# 													gamma_constraint=None,
# 													training=False,
# 													trainable=True,
# 													name=None,
# 													reuse=None,
# 													renorm=False,
# 													renorm_clipping=None,
# 													renorm_momentum=0.99,
# 													fused=None,
# 													virtual_batch_size=None,
# 													adjustment=None)
# 			self.i1 = layers.fully_connected(self.i0, 1024, activation_fn=tf.nn.relu)
# 			self.i2 = layers.fully_connected(self.i1, 512, activation_fn=tf.nn.relu)
# 			self.i2_ = layers.fully_connected(self.i2, 512, activation_fn=tf.nn.relu)
# 			self.i3 = layers.fully_connected(self.i2_, num_actions, activation_fn=None)
# 		self.i3_ = tf.squeeze(self.i3, name = 'action_probability')
#
# 		# placeholder for current_action
# 		self.current_action = tf.placeholder(tf.float32, [None, num_actions])
#
# 		# i loss
# 		self.i_loss = tf.reduce_mean(
# 			tf.nn.softmax_cross_entropy_with_logits(labels=self.current_action, logits=self.i3)
# 		)
# 		self.i_loss = self.i_loss + 1e-2 * tf.reduce_mean(tf.square(self.i3))
#
# 		# R1 network
# 		# whether come next day
# 		self.current_action_float = tf.placeholder(tf.float32, [None, num_actions], name='propensity_action')
# 		self.r13 = tf.squeeze(self.get_reward_prediction(
# 			self.state_,
# 			self.lambda_,
# 			self.current_action_float,
# 			self.name + 'r_net',
# 			reuse=False), name = "propensity_r13")
#
# 		# placeholder for real R1
# 		self.current_r1 = tf.placeholder(tf.float32, [None, 2], name='propensity_r1_action')
#
# 		# r1 loss
# 		self.r1_loss = tf.reduce_mean(
# 			tf.nn.softmax_cross_entropy_with_logits(labels=self.current_r1, logits=self.r13)
# 		)
# 		self.r1_loss = self.r1_loss + 1e-2 * tf.reduce_mean(tf.square(self.r13))
#
# 		# Optimize the Q
# 		self.i_optim_ = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.i_loss)
# 		self.r1_optim_ = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.r1_loss)
#
# 	def get_i_network_variables(self):
# 		return [t for t in tf.trainable_variables() if t.name.startswith(self.name + 'i_net')]
#
# 	def get_reward_prediction(self, state_, lambda_, current_action_float, scope, reuse=False):
# 		with tf.variable_scope(scope, reuse=reuse):
# 			r10 = tf.layers.batch_normalization(tf.concat([state_, lambda_], 1), axis=-1, momentum=0.99,
# 													epsilon=0.001,
# 													center=True,
# 													scale=True,
# 													beta_initializer=tf.zeros_initializer(),
# 													gamma_initializer=tf.ones_initializer(),
# 													moving_mean_initializer=tf.zeros_initializer(),
# 													moving_variance_initializer=tf.ones_initializer(),
# 													beta_regularizer=None,
# 													gamma_regularizer=None,
# 													beta_constraint=None,
# 													gamma_constraint=None,
# 													training=False,
# 													trainable=True,
# 													name=None,
# 													reuse=None,
# 													renorm=False,
# 													renorm_clipping=None,
# 													renorm_momentum=0.99,
# 													fused=None,
# 													virtual_batch_size=None,
# 													adjustment=None)
# 			r11 = layers.fully_connected(r10, 256, activation_fn=tf.nn.relu)
# 			r12 = layers.fully_connected(r11, 256, activation_fn=tf.nn.relu)
#
# 			r21 = layers.fully_connected(current_action_float, 64, activation_fn=tf.nn.relu)
# 			r22 = layers.fully_connected(r21, 64, activation_fn=tf.nn.relu)
#
# 			r3 = tf.concat((r12, r22), axis=1)
# 			r13 = layers.fully_connected(r3, 2, activation_fn=None)
# 			return r13


class DiscreteBCQ(object):
	def __init__(
			self,
			sess,
			num_actions,
			action_dim,
			Lambda_dim,
			state_dim,
			Estimator,
			max_timesteps,
			Lambda_min,
			Lambda_max,
			Lambda_interval,
			Number_real_evaluation_users,
			fqe_train_steps,
			rem_train_steps,
			BCQ_threshold=0.3,
			discount=0.99,
			optimizer_parameters_lr=3e-4,
			polyak_target_update=False,
			target_update_frequency=8e3,
			tau=0.005,
			q_loss_weight=2e1,
			i_regularization_weight=1e-1,
			i_loss_weight=1.0
	):
		# Estimator, for evaluation
		self.estimator = Estimator

		# lambda, lagrange multiplier
		self.action_dim = action_dim
		self.Lambda_min = Lambda_min
		self.Lambda_max = Lambda_max
		self.Lambda_interval = Lambda_interval
		self.Lambda_dim = Lambda_dim
		self.Lambda_size = int((self.Lambda_max - self.Lambda_min) / self. Lambda_interval + 1)
		self.optimizer_parameters_lr = optimizer_parameters_lr
		self.q_loss_weight =q_loss_weight

		# # pre-train
		# self.propensity_network = PropensityNet(
		# 	state_dim, num_actions,  Lambda_dim, optimizer_parameters_lr, "pre_train_network"
		# )

		# Determine network type
		self.Q = FC_Q(
			state_dim, num_actions, Lambda_dim, BCQ_threshold, optimizer_parameters_lr, "Q_network",
			q_loss_weight, i_regularization_weight, i_loss_weight
		)
		self.Q_target = FC_Q(
			state_dim, num_actions,  Lambda_dim, BCQ_threshold, optimizer_parameters_lr, "Q_target_network",
			q_loss_weight, i_regularization_weight, i_loss_weight
		)
		self.Q_lr = optimizer_parameters_lr

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau
		self.update_network_op = self.maybe_update_target()

		# Evaluation hyper-parameters
		self.state_shape = (-1, state_dim)
		self.state_dim = state_dim
		self.num_actions = num_actions
		self.max_timesteps = max_timesteps
		self.fqe_train_steps = fqe_train_steps
		self.rem_train_steps = rem_train_steps

		# Threshold for "unlikely" actions
		self.threshold = BCQ_threshold

		# Number of training iterations
		self.iterations = 0

		#Generate real evaluation data
		self.Number_real_evaluation_users = Number_real_evaluation_users
		# self.all_user_come, self.all_user_hongbao, self.all_user_liucun, self.all_hongbao_pre30, \
		# self.all_liucun_pre30, self.all_average_liucun, self.all_user_type = \
		# 	self.real_evaluation_data()
		# session
		self.sess = sess

		# Initialize network
		self.sess.run(tf.global_variables_initializer())

		# Initialize networks to start with the same variables:
		self.sess.run(self.copy_target_update())

		# saver
		self.saver = tf.train.Saver(max_to_keep=100000)

	def append_batch_data(self, replay_buffer):
		state = replay_buffer.new_batch_data["state"]
		action = replay_buffer.new_batch_data["action"]
		next_state = replay_buffer.new_batch_data["next_state"]
		reward1 = replay_buffer.new_batch_data["reward1"]
		reward2 = replay_buffer.new_batch_data["reward2"]
		reward = replay_buffer.new_batch_data["reward"]
		done = replay_buffer.new_batch_data["done"]
		Lambda = replay_buffer.new_batch_data["Lambda"]

		if not replay_buffer.is_priority:
			for s, a, s_, r1, r2, r, d, l in zip(
				state, action, next_state, reward1, reward2, reward, done, Lambda
			):
				replay_buffer.add_without_priority(s, a, s_, r1, r2, r, d, l)
			return


		# Compute the current Q value
		current_q = self.sess.run(self.Q.current_q, feed_dict={
			self.Q.state_: state,
			self.Q.current_action: action,
			self.Q.lambda_: Lambda
		})

		next_action = self.sess.run(self.Q.next_action_, feed_dict={
			self.Q.state_: state,
			self.Q.lambda_: Lambda
		})

		# Compute the target Q value
		target_q = self.sess.run(self.Q_target.current_q, feed_dict={
			self.Q_target.state_: next_state,
			self.Q_target.current_action: next_action,
			self.Q_target.lambda_: Lambda
		})

		target_q = reward + (1 - done) * self.discount * target_q
		error = abs(current_q - target_q)
		replay_buffer.add(error, (state, action, reward, next_state, done, reward1, reward2, Lambda))

	def train(self, replay_buffer):
		# Sample replay buffer

		if replay_buffer.is_priority:
			state, action, next_state, reward, done, reward1, reward2, Lambda,\
			idxs, is_weights = replay_buffer.sample_priority()
		else:
			state, action, next_state, reward, done, reward1, reward2, Lambda, \
			idxs, is_weights = replay_buffer.sample_without_priority()

		# r_l = np.zeros(21)
		# for j in range(1000):
		# 	for i in range(21):
		# 		if Lambda[j] == i * 0.05:
		# 			r_l[i] += reward[j]

		action = np.expand_dims(np.squeeze(action), axis=1)
		reward = np.expand_dims(np.squeeze(reward), axis=1)
		reward1 = np.expand_dims(np.squeeze(reward1), axis=1)
		reward2 = np.expand_dims(np.squeeze(reward2), axis=1)
		done = np.expand_dims(np.squeeze(done), axis=1)
		Lambda = np.expand_dims(np.squeeze(Lambda), axis=1)

		# Loss for total reward, reward1, and reward2 respectively
		errors, actor_loss = self._get_loss(
			state, next_state, action, reward, reward1, reward2, Lambda, is_weights, done
		)


		if replay_buffer.is_priority:
			# update priority
			for i in range(replay_buffer.batch_size):
				idx = idxs[i]
				replay_buffer.update(idx, errors[i])

		print("actor_loss: {}: ".format(
			actor_loss)
		)

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.sess.run(self.update_network_op)

	def polyak_target_update(self, use_locking=False):
		Q_parameter_vars = self.Q.get_network_variables()
		Q_target_vars = self.Q_target.get_network_variables()
		update_op = []
		for param, target_param in zip(Q_parameter_vars, Q_target_vars):
			update_op.append(target_param.assign(self.tau * param + (1 - self.tau) * target_param, use_locking))
		return update_op

	def copy_target_update(self, use_locking=False):
		Q_parameter_vars = self.Q.get_network_variables()
		Q_target_vars = self.Q_target.get_network_variables()
		update_op = []
		for param, target_param in zip(Q_parameter_vars, Q_target_vars):
			update_op.append(target_param.assign(param, use_locking))
		return update_op

	def copy_pretrain_update(self, use_locking=False):
		Q_pretrain_vars = self.propensity_network.get_i_network_variables()
		Q_vars = self.Q.get_i_network_variables()
		update_op = []
		for param, target_param in zip(Q_pretrain_vars, Q_vars):
			update_op.append(target_param.assign(param, use_locking))
		return update_op

	def _get_importance_ratio(self, batch_data):
		ratio, logged_prob, model_prob = self.sess.run(
			(self.Q.ips_ratio, self.Q.behaviour_prob, self.Q.target_prob), feed_dict={
				self.Q.state_: np.array(batch_data["obs"]),
				self.Q.current_action: np.array(batch_data["actions"]),
				self.Q.g_embedding: np.array(batch_data["propensities"])
			}
		)
		return list(ratio), list(logged_prob), list(model_prob)

	def _get_loss(self, state, next_state, action, reward, reward1, reward2, Lambda, is_weights, done):
		# Compute the target Q value
		current_q = self.sess.run(self.Q.current_q, feed_dict={
			self.Q.state_: state,
			self.Q.current_action: action,
			self.Q.lambda_: Lambda
		})


		next_action = self.sess.run(self.Q.next_action_, feed_dict={
			self.Q.state_: next_state,
			self.Q.lambda_: Lambda
		})


		target_q = self.sess.run(self.Q_target.current_q, feed_dict={
			self.Q_target.state_: next_state,
			self.Q_target.current_action: next_action,
			self.Q_target.lambda_: Lambda
		})

		target_q = reward + (1 - done) * self.discount * target_q

		# Compute loss
		actor_loss = self.sess.run([self.Q.Q_loss, self.Q.Q_optim_], feed_dict={
			self.Q.state_: state,
			self.Q.lambda_: Lambda,
			self.Q.current_action: action,
			self.Q.target_q: target_q,
			self.Q.is_weights: np.expand_dims(np.squeeze(is_weights), axis=1),
		})

		# i_loss = self.sess.run(self.Q.i_loss, feed_dict={
		# 	self.Q.state_: state,
		# 	self.Q.current_action: action,
		# 	self.Q.lambda_: Lambda,
		# 	self.Q.is_weights: np.expand_dims(is_weights, axis=1),
		# })
		#
		# i3_loss = self.sess.run(self.Q.i3_loss, feed_dict={
		# 	self.Q.state_: state,
		# 	self.Q.lambda_: Lambda,
		# 	self.Q.is_weights: np.expand_dims(is_weights, axis=1)
		# })
		#
		# q_loss = self.sess.run(self.Q.q_loss, feed_dict={
		# 	self.Q.state_: state,
		# 	self.Q.current_action: action,
		# 	self.Q.lambda_: Lambda,
		# 	self.Q.target_q: target_q,
		# 	self.Q.is_weights: np.expand_dims(is_weights, axis=1),
		# })

		# errors
		errors = current_q - target_q


		return errors, actor_loss[0]
		#return errors, actor_loss[0], i_loss, q_loss, i3_loss

