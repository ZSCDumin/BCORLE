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


class MultiHeadDQNAgent(dqn_agent.DQNAgent):
  """DQN agent with multiple heads."""

  def __init__(self,
               sess,
               num_actions,
               num_heads=1,
               transform_strategy='IDENTITY',
               num_convex_combinations=1,
               network=None,
               init_checkpoint_dir=None,
               **kwargs):
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
    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None
    self._q_heads_transform = None
    self._num_convex_combinations = num_convex_combinations
    self.transform_strategy = transform_strategy
    super(MultiHeadDQNAgent, self).__init__(
        sess, num_actions, network=network, **kwargs)

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

  def _network_template(self, state):
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
      num_actions=self.num_actions,
      num_heads=self.num_heads,
      transform_strategy=self.transform_strategy,
      **kwargs)
    return network_template

  def _network(self,
               state,
               num_actions,
               num_heads,
               transform_strategy=None,
               **kwargs):
    self._kernel_initializer = tf.initializers.variance_scaling(
      scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
    )
    activation_fn = tf.nn.relu
    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = tf.layers.conv2d(
      net,
      filters=32,
      kernel_size=[8, 8],
      strides=[4, 4],
      padding='SAME',
      activation=activation_fn,
      kernel_initializer=self._kernel_initializer
    )
    net = tf.layers.conv2d(
      net,
      filters=64,
      kernel_size=[4, 4],
      strides=[2, 2],
      padding='SAME',
      activation=activation_fn,
      kernel_initializer=self._kernel_initializer
    )
    net = tf.layers.conv2d(
      net,
      filters=64,
      kernel_size=[3, 3],
      strides=[1, 2],
      padding='SAME',
      activation=activation_fn,
      kernel_initializer=self._kernel_initializer
    )
    net = tf.layers.flatten(net)
    net = tf.layers.dense(
      net,
      units=512,
      activation=activation_fn,
      kernel_initializer=self._kernel_initializer
    )
    net = tf.layers.dense(
      net,
      units=num_actions * num_heads,
      kernel_initializer=self._kernel_initializer
    )
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
    is_non_terminal = 1. - tf.cast(self._replay.terminals, tf.float32)
    is_non_terminal = tf.expand_dims(is_non_terminal, axis=-1)
    rewards = tf.expand_dims(self._replay.rewards, axis=-1)
    return rewards + (
        self.cumulative_gamma * replay_next_qt_max * is_non_terminal)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    actions = self._replay.actions
    indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
    replay_chosen_q = tf.gather_nd(
        self._replay_net_outputs.q_heads, indices=indices)
    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    q_head_losses = tf.reduce_mean(loss, axis=0)
    final_loss = tf.reduce_mean(q_head_losses)
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', final_loss)
    return self.optimizer.minimize(final_loss)


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
