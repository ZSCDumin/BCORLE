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

import os

from batch_rl.multi_head import atari_helpers
from dopamine.agents.dqn import dqn_agent
import tensorflow as tf
import numpy as np
import collections
import tensorflow.contrib.layers as layers
import datetime
import copy


class MultiHeadDQNAgent(dqn_agent.DQNAgent):
  """DQN agent with multiple heads."""

  def __init__(self,
               replay_buffer_train,
               replay_buffer,
               ckpt_path, save_dir,
               num_actions,
               state_dim, Lambda_dim,
               number_users, Lambda_size, Lambda_interval,
               num_heads=10,
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
    self.ckpt_path = ckpt_path
    self.save_dir = save_dir
    self.state_dim = state_dim
    self.Lambda_dim = Lambda_dim
    self.replay_buffer_train = replay_buffer_train
    self.replay_buffer = replay_buffer
    self.number_users = number_users
    self.Lambda_size = Lambda_size
    self.Lambda_interval = Lambda_interval
    #self._next_action()
    #self._build_replay_buffer()

    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None
    self._q_heads_transform = None
    self._num_convex_combinations = num_convex_combinations
    self.transform_strategy = transform_strategy
    start = datetime.datetime.now()
    super(MultiHeadDQNAgent, self).__init__(state_dim, num_actions, replay_buffer, ckpt_path, save_dir, network=network, **kwargs)
    end = datetime.datetime.now()
    print(end - start)
    print('1 finished')

    start = datetime.datetime.now()
    self._build_networks()
    end = datetime.datetime.now()
    print(end - start)
    print('2 finished')
    start = datetime.datetime.now()
    self.train_op = self._build_train_op()

    end = datetime.datetime.now()
    print(end - start)
    print('3 finished')
    start = datetime.datetime.now()
    self.sess = tf.Session()
    init = tf.global_variables_initializer()
    self.sess.run(init)

    end = datetime.datetime.now()
    print(end - start)
    print('4 finished')

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
    indices = tf.stack(
        [tf.reshape(tf.range(tf.shape(self.next_action_ph)[0]), shape=tf.shape(self.next_action_ph)), self.next_action_ph], axis=-1)

    replay_next_qt_max = tf.gather_nd(
        self._replay_next_target_net_outputs.q_heads, indices=indices)
    # replay_next_qt_max = self._replay_next_target_net_outputs.q_heads \
    #                      * tf.transpose(tf.one_hot(self.next_action_ph, depth = self.num_actions), [0,2,1])
    is_non_terminal = 1. - tf.cast(self.done_ph, tf.float32)
    is_non_terminal = tf.expand_dims(is_non_terminal, axis=-1)
    rewards = tf.expand_dims(self.reward_ph, axis=-1)
    return rewards + (
        self.cumulative_gamma * replay_next_qt_max * is_non_terminal)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    # actions = self.replay_buffer.new_batch_data["action"]
    # indices = tf.stack([tf.range(actions.shape[0]), tf.squeeze(actions)], axis=-1)
    indices = tf.stack(
        [tf.reshape(tf.range(tf.shape(self.action_ph)[0]), shape=tf.shape(self.action_ph)), self.action_ph], axis=-1)
    self.replay_chosen_q = tf.squeeze(tf.gather_nd(
        self._replay_net_outputs.q_heads, indices=indices))
    target = tf.squeeze(tf.stop_gradient(self._build_target_q_op()))
    loss = tf.losses.huber_loss(
        target, self.replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    q_head_losses = tf.reduce_mean(loss, axis=0)
    self.final_loss = tf.reduce_mean(q_head_losses)
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', self.final_loss)
    with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
        return self.optimizer.minimize(self.final_loss)

  def learn(self, state,action,reward,Lambda,next_state,next_action,done, train_step):

      _, loss = self.sess.run([self.train_op, self.final_loss], feed_dict =
                {self.state_ph: state,
                self.action_ph: action,
                self.reward_ph: reward,
                self.Lambda_ph: Lambda,
                self.next_state_ph: next_state,
				self.next_action_ph: next_action,
				self.done_ph: done})

      if train_step % 10 == 0:
        self.sess.run(self._sync_qt_ops)

      return loss

  def evaluate(self, next_action_pi):
      predict_value = np.zeros([self.Lambda_size, self.number_users])
      index = 0
      while index < len(self.replay_buffer.new_batch_data["done"]):
          ini_index = index
          while self.replay_buffer.new_batch_data["done"][index][0] != 1.:
              index += 1
          value = self.sess.run(self._replay_net_outputs.q_values, feed_dict=
          {self.state_ph: np.expand_dims(self.replay_buffer.new_batch_data["state"][ini_index], 0),
           self.Lambda_ph: np.expand_dims(self.replay_buffer.new_batch_data["Lambda"][ini_index], 0)})
          value = np.squeeze(value)[next_action_pi[ini_index][0]]
          predict_value[int(round(self.replay_buffer.new_batch_data["Lambda"][index][0] / self.Lambda_interval)),
                         self.replay_buffer.new_batch_data["user_id"][index][0] - 1] = value
          index += 1
      self.sess.close()
      print(predict_value)
      return predict_value



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
