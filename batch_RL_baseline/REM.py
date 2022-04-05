# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi Head DQN agent."""

import datetime

import os

from batch_rl.multi_head import atari_helpers
from dopamine.agents.dqn import dqn_agent_rl
import tensorflow as tf
import numpy as np
import collections
import tensorflow.contrib.layers as layers

class REMAgent(dqn_agent_rl.DQNAgent):
  """DQN agent with multiple heads."""

  def __init__(self, sess,
               num_actions,
               state_dim, Lambda_dim,
               number_users, Lambda_size, Lambda_interval, optimizer_parameters_lr,
               num_heads,
               transform_strategy='IDENTITY',
               num_convex_combinations=1,
               network=None,
               init_checkpoint_dir=None, **kwargs):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: tf.Session, for executing ops.
      num_actions, number of actions the agent can take at any state.
      num_heads, Number of heads per action output of the Q function.
      transform_strategy, Possible options include (1)
      'STOCHASTIC' for multiplication with a left stochastic matrix. (2)
      'IDENTITY', in which case the heads are not transformed.
      num_convex_combinations: If transform_strategy is 'STOCHASTIC',
        then this argument specifies the number of random
        convex combinations to be created. If None, `num_heads` convex
        combinations are created.
      network: tf.Keras.Model. A call to this object will return an
        instantiation of the network provided. The network returned can be run
        with different inputs to create different outputs. See
        atari_helpers.MultiHeadQNetwork as an example.
      init_checkpoint_dir, directory from which initial checkpoint before
        training is loaded if there doesn't exist any checkpoint in the current
        agent directory. If None, no initial checkpoint is loaded.
      **kwargs: Arbitrary keyword arguments.
    """
    tf.logging.info('Creating MultiHeadDQNAgent with following parameters:')
    tf.logging.info('\t num_heads: %d', num_heads)
    tf.logging.info('\t transform_strategy: %s', transform_strategy)
    tf.logging.info('\t num_convex_combinations: %d', num_convex_combinations)
    tf.logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
    self.num_heads = num_heads
    self.state_dim = state_dim
    self.Lambda_dim = Lambda_dim
    self.number_users = number_users
    self.Lambda_size = Lambda_size
    self.Lambda_interval = Lambda_interval
    # # pre-train
    # self.propensity_network = PropensityNet(
    #     state_dim, num_actions, Lambda_dim, 0.001, "pre_train_network"
    # )
    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None
    self._q_heads_transform = None
    self._num_convex_combinations = num_convex_combinations
    self.transform_strategy = transform_strategy
    super(REMAgent, self).__init__(state_dim, num_actions, network=network, **kwargs)
    self._build_networks()
    self.train_op = self._build_train_op()
    self.sess = sess
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver(max_to_keep=100000)

  def _create_network(self, name):
    """Builds a multi-head Q-network that outputs Q-values for multiple heads.

    Args:
      name, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    kwargs = {}  # Used for passing the transformation matrix if any
    if self._q_heads_transform is None:
      if self.transform_strategy == 'STOCHASTIC':
        tf.logging.info('Creating q_heads transformation matrix..')
        self._q_heads_transform = atari_helpers.random_stochastic_matrix(
            self.num_heads, num_cols=self._num_convex_combinations)
    if self._q_heads_transform is not None:
      kwargs.update({'transform_matrix': self._q_heads_transform})
    network = self.network(
        num_actions=self.num_actions,
        num_heads=self.num_heads,
        transform_strategy=self.transform_strategy,
        name=name,
        **kwargs)
    return network

  def _network_template(self, state, Lambda):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    kwargs = {}  # Used for passing the transformation matrix if any
    if self._q_heads_transform is None:
      if self.transform_strategy == 'STOCHASTIC':
        tf.logging.info('Creating q_heads transformation matrix..')
        self._q_heads_transform = atari_helpers.random_stochastic_matrix(
          self.num_heads, num_cols=self._num_convex_combinations)
    if self._q_heads_transform is not None:
      kwargs.update({'transform_matrix': self._q_heads_transform})

    network_template = self._network(
      state=state,
      Lambda= Lambda,
      num_actions=self.num_actions,
      num_heads=self.num_heads,
      transform_strategy=self.transform_strategy,
      **kwargs)
    return network_template

  def _network(self,
               state,
               Lambda,
               num_actions,
               num_heads,
               transform_strategy=None,
               **kwargs):
    self._kernel_initializer = tf.initializers.variance_scaling(
      scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
    )
    activation_fn = tf.nn.relu
    with tf.variable_scope('rem_q_net'):
        input = tf.concat([state, Lambda], axis = 1)
        r10 = tf.layers.batch_normalization(input, axis=-1, momentum=0.99,
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
        r11 = layers.fully_connected(r10, 256, activation_fn=tf.nn.relu)
        net = layers.fully_connected(r11, 256, activation_fn=tf.nn.relu)
        net = layers.fully_connected(net, num_actions * num_heads, activation_fn=None)
        unordered_q_heads = tf.reshape(net, [-1, num_actions, num_heads])
        q_heads, q_values = combine_q_functions(
          unordered_q_heads, transform_strategy, **kwargs)
    return self._get_network_type()(q_heads, unordered_q_heads, q_values)

  def _get_network_type(self):
    return collections.namedtuple(
    'multi_head_dqn_network', ['q_heads', 'unordered_q_heads', 'q_values'])

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension for each head.
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs.q_heads, axis=1)
    #replay_next_qt_max = self._replay_next_target_net_outputs.q_heads
    is_non_terminal = 1. - tf.cast(self.done_ph, tf.float32)
    is_non_terminal = is_non_terminal
    rewards =self.reward_ph
    return rewards + (
        self.cumulative_gamma * replay_next_qt_max * is_non_terminal)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    indices = tf.stack([tf.reshape(tf.range(tf.shape(self.action_ph)[0]), shape = tf.shape(self.action_ph)), self.action_ph], axis=-1)
    self.replay_chosen_q = tf.squeeze(tf.gather_nd(
        self._replay_net_outputs.q_heads, indices=indices))
    self.average_Q = tf.reduce_mean(self.replay_chosen_q, axis=-1, name = 'current_q')
    #self.replay_chosen_q = tf.squeeze(tf.gather(
    #     self._replay_net_outputs.q_heads, indices=self.action_ph), name = 'current_q')  #indices=np.squeeze(self.action_ph)
    self.target = tf.squeeze(tf.stop_gradient(self._build_target_q_op()))
    loss = tf.losses.huber_loss(
        self.target, self.replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    q_head_losses = tf.reduce_mean(loss, axis=0)
    self.final_loss = tf.reduce_mean(q_head_losses)
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', self.final_loss)
    with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
        return self.optimizer.minimize(self.final_loss)

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
      current_q = self.sess.run(self.replay_chosen_q, feed_dict={
          self.state_ph: state,
          self.action_ph: action,
          self.Lambda_ph: Lambda
      })

      # Compute the target Q value
      target_q = self.sess.run(self.target, feed_dict={
          self.next_state_ph: next_state,
          self.Lambda_ph: Lambda,
          self.reward_ph:reward,
          self.done_ph:done
      })
      error = abs(current_q - target_q)
      replay_buffer.add(error, (state, action, reward, next_state, done, reward1, reward2, Lambda))

  def train(self, replay_buffer):

      # start = datetime.datetime.now()
      # Sample replay buffer
      if replay_buffer.is_priority:
          state, action, next_state, reward, done, reward1, reward2, Lambda, \
          idxs, is_weights = replay_buffer.sample_priority()
      else:
          state, action, next_state, reward, done, reward1, reward2, Lambda, \
          idxs, is_weights = replay_buffer.sample_without_priority()
      # end = datetime.datetime.now()
      # print('Sample time cost')
      # print(end - start)

      Lambda = np.expand_dims(Lambda, axis=1)
      done = np.expand_dims(done, axis=1)
      reward = np.expand_dims(reward, axis=1)
      # Loss for total reward, reward1, and reward2 respectively

      # start = datetime.datetime.now()
      # Update target network by polyak or full copy every X iterations.
      _, loss = self.sess.run([self.train_op, self.final_loss], feed_dict =
                {self.state_ph: state,
                self.action_ph: action,
                self.reward_ph: reward,
                self.Lambda_ph: Lambda,
                self.next_state_ph: next_state,
				self.done_ph: done})
      # end = datetime.datetime.now()
      # print('train_op time cost')
      # print(end - start)

      # start = datetime.datetime.now()
      current_q = self.sess.run(self.replay_chosen_q, feed_dict={
          self.state_ph: state,
          self.action_ph: action,
          self.Lambda_ph: Lambda
      })
      # end = datetime.datetime.now()
      # print('current_q time cost')
      # print(end - start)

      # start = datetime.datetime.now()
      # Compute the target Q value
      target_q = self.sess.run(self.target, feed_dict={
          self.next_state_ph: next_state,
          self.Lambda_ph: Lambda,
          self.reward_ph: reward,
          self.done_ph: done
      })
      errors = abs(current_q - target_q)
      # end = datetime.datetime.now()
      # print('target_q time cost')
      # print(end - start)

      # start = datetime.datetime.now()
      if replay_buffer.is_priority:
          # update priority
          for i in range(replay_buffer.batch_size):
              idx = idxs[i]
              replay_buffer.update(idx, errors[i])
      # end = datetime.datetime.now()
      # print('replay_buffer_update time cost')
      # print(end - start)

      print("q_loss: {}".format(loss))

      return loss


def combine_q_functions(q_functions, transform_strategy, **kwargs):
  """Utility function for combining multiple Q functions.

  Args:
    q_functions: Multiple Q-functions concatenated.
    transform_strategy, Possible options include (1) 'IDENTITY' for no
      transformation (2) 'STOCHASTIC' for random convex combination.
    **kwargs: Arbitrary keyword arguments. Used for passing `transform_matrix`,
      the matrix for transforming the Q-values if the passed
      `transform_strategy` is `STOCHASTIC`.

  Returns:
    q_functions: Modified Q-functions.
    q_values: Q-values based on combining the multiple heads.
  """
  # Create q_values before reordering the heads for training
  q_values = tf.reduce_mean(q_functions, axis=-1)

  if transform_strategy == 'STOCHASTIC':
    left_stochastic_matrix = kwargs.get('transform_matrix')
    if left_stochastic_matrix is None:
      raise ValueError('None value provided for stochastic matrix')
    q_functions = tf.tensordot(
        q_functions, left_stochastic_matrix, axes=[[2], [0]])
  elif transform_strategy == 'IDENTITY':
    tf.logging.info('Identity transformation Q-function heads')
  else:
    raise ValueError(
        '{} is not a valid reordering strategy'.format(transform_strategy))
  return q_functions, q_values

#
# class PropensityNet(object):
#     def __init__(self, state_dim, num_actions, Lambda_dim, lr, name):
#         self.name = name
#         self.state_dim = state_dim
#         self.action_dim = num_actions
#         self.state_ = tf.placeholder(tf.float32, [None, state_dim], name="propensity_state")
#         self.lambda_ = tf.placeholder(tf.float32, [None, Lambda_dim], name="propensity_lambda")
#         with tf.variable_scope(self.name + 'i_net'):
#             # placeholders for PropensityNet
#
#             # I network
#             self.i0 = tf.layers.batch_normalization(tf.concat([self.state_, self.lambda_], 1), axis=-1, momentum=0.99,
#                                                     epsilon=0.001,
#                                                     center=True,
#                                                     scale=True,
#                                                     beta_initializer=tf.zeros_initializer(),
#                                                     gamma_initializer=tf.ones_initializer(),
#                                                     moving_mean_initializer=tf.zeros_initializer(),
#                                                     moving_variance_initializer=tf.ones_initializer(),
#                                                     beta_regularizer=None,
#                                                     gamma_regularizer=None,
#                                                     beta_constraint=None,
#                                                     gamma_constraint=None,
#                                                     training=False,
#                                                     trainable=True,
#                                                     name=None,
#                                                     reuse=None,
#                                                     renorm=False,
#                                                     renorm_clipping=None,
#                                                     renorm_momentum=0.99,
#                                                     fused=None,
#                                                     virtual_batch_size=None,
#                                                     adjustment=None)
#             self.i1 = layers.fully_connected(self.i0, 1024, activation_fn=tf.nn.relu)
#             self.i2 = layers.fully_connected(self.i1, 512, activation_fn=tf.nn.relu)
#             self.i2_ = layers.fully_connected(self.i2, 512, activation_fn=tf.nn.relu)
#             self.i3 = layers.fully_connected(self.i2_, num_actions, activation_fn=None)
#         self.i3_ = tf.squeeze(self.i3, name = 'action_probability')
#
#         # placeholder for current_action
#         self.current_action = tf.placeholder(tf.float32, [None, num_actions])
#
#         # i loss
#         self.i_loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=self.current_action, logits=self.i3)
#         )
#         self.i_loss = self.i_loss + 1e-2 * tf.reduce_mean(tf.square(self.i3))
#
#         # R1 network
#         # whether come next day
#         self.current_action_float = tf.placeholder(tf.float32, [None, num_actions], name='propensity_action')
#         self.r13 = tf.squeeze(self.get_reward_prediction(
#         self.state_,
#         self.lambda_,
#         self.current_action_float,
#         self.name + 'r_net',
#         reuse=False), name = "propensity_r13")
#
#         # placeholder for real R1
#         self.current_r1 = tf.placeholder(tf.float32, [None, 2], name='propensity_r1_action')
#
#         # r1 loss
#         self.r1_loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=self.current_r1, logits=self.r13)
#         )
#         self.r1_loss = self.r1_loss + 1e-2 * tf.reduce_mean(tf.square(self.r13))
#
#         # Optimize the Q
#         self.i_optim_ = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.i_loss)
#         self.r1_optim_ = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.r1_loss)
#
#     def get_i_network_variables(self):
#         return [t for t in tf.trainable_variables() if t.name.startswith(self.name + 'i_net')]
#
#     def get_reward_prediction(self, state_, lambda_, current_action_float, scope, reuse=False):
#         with tf.variable_scope(scope, reuse=reuse):
#             r10 = tf.layers.batch_normalization(tf.concat([state_, lambda_], 1), axis=-1, momentum=0.99,
#                                             epsilon=0.001,
#                                             center=True,
#                                             scale=True,
#                                             beta_initializer=tf.zeros_initializer(),
#                                             gamma_initializer=tf.ones_initializer(),
#                                             moving_mean_initializer=tf.zeros_initializer(),
#                                             moving_variance_initializer=tf.ones_initializer(),
#                                             beta_regularizer=None,
#                                             gamma_regularizer=None,
#                                             beta_constraint=None,
#                                             gamma_constraint=None,
#                                             training=False,
#                                             trainable=True,
#                                             name=None,
#                                             reuse=None,
#                                             renorm=False,
#                                             renorm_clipping=None,
#                                             renorm_momentum=0.99,
#                                             fused=None,
#                                             virtual_batch_size=None,
#                                             adjustment=None)
#         r11 = layers.fully_connected(r10, 256, activation_fn=tf.nn.relu)
#         r12 = layers.fully_connected(r11, 256, activation_fn=tf.nn.relu)
#
#         r21 = layers.fully_connected(current_action_float, 64, activation_fn=tf.nn.relu)
#         r22 = layers.fully_connected(r21, 64, activation_fn=tf.nn.relu)
#
#         r3 = tf.concat((r12, r22), axis=1)
#         r13 = layers.fully_connected(r3, 2, activation_fn=None)
#         return r13
