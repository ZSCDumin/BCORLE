#  coding=utf-8

"""
BCQ agent with fixed replay buffer(s).
"""

from dopamine.agents.dqn import dqn_agent
from replay_memory import fixed_replay_buffer

import os
import collections
import random
import numpy as np
from absl import logging
from batch_rl.multi_head import atari_helpers

import tensorflow as tf


class FixedReplayREMBCQAgent(dqn_agent.DQNAgent):
    """An implementation of the BCQ agent with fixed replay buffer(s)."""

    def __init__(self, sess, num_actions,
                 replay_data_dir,
                 num_heads=1,
                 transform_strategy='IDENTITY',
                 num_convex_combinations=1,
                 replay_suffix=None,
                 init_checkpoint_dir=None,
                 threshold=0.3,
                 q_loss_weight=1.0,
                 i_regularization_weight=1e-2,
                 i_loss_weight=1.0,
                 replay_capacity=1000000,
                 data_set_mode='ALL',
                 **kwargs):
        """Initializes the agent and constructs the components of its graph.

            Args:
                sess: tf.Session, for executing ops.
                num_actions: int, number of actions the agent can take at any state.
                replay_data_dir: str, log Directory from which to load the replay buffer.
                replay_suffix: int, If not None, then only load the replay buffer
                    corresponding to the specific suffix in data directory.
                init_checkpoint_dir: str, directory from which initial checkpoint before
                    training is loaded if there doesn't exist any checkpoint in the current
                    agent directory. If None, no initial checkpoint is loaded.
                threshold: the threshold of BCQ for selecting the actions with prob higher than threshold.
                q_loss_weight: weight for TD-error loss in BCQ
                i_regularization_weight: weight for regularization loss in BCQ
                i_loss_weight: weight for imitation loss in BCQ
                replay_capacity: the capacity of the replay_buffer

                **kwargs: Arbitrary keyword arguments.
            """

        assert replay_data_dir is not None
        # Set replay_log_dir before calling parent's initializer
        logging.info(
            'Creating FixedReplayBCQAgent with replay directory: %s', replay_data_dir)
        logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
        logging.info('\t replay_suffix %s', replay_suffix)
        self._replay_data_dir = replay_data_dir
        self._replay_suffix = replay_suffix
        self._replay_capacity = replay_capacity
        self._threshold = threshold
        self._q_loss_weight = q_loss_weight,
        self._i_regularization_weight = i_regularization_weight,
        self._i_loss_weight = i_loss_weight
        self.num_heads = num_heads
        self._q_heads_transform = None
        self._num_convex_combinations = num_convex_combinations
        self.transform_strategy = transform_strategy
        self._data_set_mode = data_set_mode
        if init_checkpoint_dir is not None:
            self._init_checkpoint_dir = os.path.join(
                init_checkpoint_dir, 'checkpoints')
        else:
            self._init_checkpoint_dir = None
        super(FixedReplayREMBCQAgent, self).__init__(
            sess, num_actions, **kwargs)

    def _build_target_q_op(self):
        """Build an op used as a target for the Q-value.
        Returns:
        target_q_op: An op calculating the Q-value.
        """
        # Get the maximum Q-value across the actions dimension for each head.
        replay_chosen_q_values = self._replay_next_net_outputs.unordered_q_heads
        replay_chosen_imt = self._replay_next_net_outputs.unordered_imt_heads
        replay_chosen_imt = tf.exp(replay_chosen_imt)
        replay_chosen_imt = (
                replay_chosen_imt / tf.reduce_max(replay_chosen_imt, axis=1, keep_dims=True)[0]
                > self._threshold
        )
        replay_chosen_imt = tf.cast(replay_chosen_imt, dtype=tf.float32)

        replay_chosen_imt = replay_chosen_imt * replay_chosen_q_values + (1 - replay_chosen_imt) * -1e8

        next_action = tf.argmax(replay_chosen_imt, axis=1)

        replay_next_action_one_hot = tf.one_hot(
            next_action, self.num_actions, 1., 0., name='next_action_one_hot'
        )
        replay_next_q_values = tf.reduce_sum(
            self._replay_next_target_net_outputs.unordered_q_heads * tf.transpose(replay_next_action_one_hot, [0, 2, 1]),
            axis=1,
            name='replay_next_q_values'
        )

        # multi_head for next_q_values
        kwargs = {}  # Used for passing the transformation matrix if any
        if self._q_heads_transform is None:
            if self.transform_strategy == 'STOCHASTIC':
                tf.logging.info('Creating q_heads transformation matrix..')
                self._q_heads_transform = atari_helpers.random_stochastic_matrix(
                    self.num_heads, num_cols=self._num_convex_combinations)
        if self._q_heads_transform is not None:
            kwargs.update({'transform_matrix': self._q_heads_transform})
        replay_next_q_values, _ = combine_q_functions2(replay_next_q_values, self.transform_strategy, **kwargs)

        is_non_terminal = 1. - tf.cast(self._replay.terminals, tf.float32)
        is_non_terminal = tf.expand_dims(is_non_terminal, axis=-1)
        rewards = tf.expand_dims(self._replay.rewards, axis=-1)
        target_res = rewards + (
                self.cumulative_gamma * replay_next_q_values * is_non_terminal
        )
        return target_res

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """
        actions = self._replay.actions
        indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
        replay_chosen_q = tf.gather_nd(
            self._replay_net_outputs.unordered_q_heads, indices=indices
        )
        replay_chosen_q = tf.squeeze(replay_chosen_q)
        kwargs = {}  # Used for passing the transformation matrix if any
        if self._q_heads_transform is None:
            if self.transform_strategy == 'STOCHASTIC':
                tf.logging.info('Creating q_heads transformation matrix..')
                self._q_heads_transform = atari_helpers.random_stochastic_matrix(
                    self.num_heads, num_cols=self._num_convex_combinations)
        if self._q_heads_transform is not None:
            kwargs.update({'transform_matrix': self._q_heads_transform})
        replay_chosen_q, _ = combine_q_functions2(replay_chosen_q, self.transform_strategy, **kwargs)

        target = tf.stop_gradient(self._build_target_q_op())
        q_loss = tf.losses.huber_loss(
            tf.squeeze(target), tf.squeeze(replay_chosen_q), reduction=tf.losses.Reduction.NONE)

        i_loss = []
        for i in range(self.num_heads):
            i_loss.append(tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._replay.actions, logits=self._replay_net_outputs.unordered_imt_heads[:, :, i]), axis=1))
        i_loss = tf.stack(i_loss, axis=1)
        i_loss, _ = combine_q_functions2(
            tf.squeeze(i_loss, axis=-1), self.transform_strategy, **kwargs)

        i3_loss = []
        for i in range(self.num_heads):
            i3_loss.append(tf.reduce_mean(tf.pow(self._replay_net_outputs.unordered_i_heads[:, :, i], 2), axis=1))
        i3_loss = tf.stack(i3_loss, axis=1)
        i3_loss, _ = combine_q_functions2(
            i3_loss, self.transform_strategy, **kwargs)

        loss = q_loss + self._i_loss_weight * tf.squeeze(i_loss) + self._i_regularization_weight * tf.squeeze(i3_loss)
        final_loss = tf.reduce_mean(loss)
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('HuberLoss', final_loss)

        return self.optimizer.minimize(final_loss)

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
            **kwargs
        )
        return network_template

    def _network(
            self,
            state,
            num_actions,
            num_heads,
            transform_strategy=None,
            **kwargs
    ):
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
        net_out = tf.layers.flatten(net)
        q_values = tf.layers.dense(
            net_out,
            units=512,
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer
        )
        q_values = tf.layers.dense(
            q_values,
            units=num_actions * num_heads,
            kernel_initializer=self._kernel_initializer
        )
        net_i = tf.layers.dense(
            net_out,
            units=512,
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer
        )
        net_i = tf.layers.dense(
            net_i,
            units=num_actions * num_heads,
            kernel_initializer=self._kernel_initializer
        )

        unordered_q_heads = tf.reshape(q_values, [-1, num_actions, num_heads])
        unordered_i_heads = tf.reshape(net_i, [-1, num_actions, num_heads])
        unordered_imt_heads = tf.reshape(net_i, [-1, num_actions, num_heads])
        unordered_imt_heads = tf.nn.log_softmax(unordered_imt_heads, axis=1)

        return self._get_network_type()(
            unordered_q_heads,
            unordered_imt_heads,
            unordered_i_heads)

    def _get_network_type(self):
        """Returns the type of the outputs of a Q value network.

        Returns:
          net_type: _network_type object defining the outputs of the network.
        """
        return collections.namedtuple(
            'rem_bcq_network',
            [
                'unordered_q_heads', 'unordered_imt_heads', 'unordered_i_heads'
            ]
        )

    def step(self, reward, observation):
        self._record_observation(observation)
        self.action = self._select_action()
        return self.action

    def end_episode(self, reward):
        assert self.eval_mode, 'Eval mode is not set to be True.'
        super(FixedReplayREMBCQAgent, self).end_episode(reward)

    def _select_action(self):
        """Select an action from the set of available actions.

        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.

        Returns:
           int, the selected action.
        """
        if self.eval_mode:
            epsilon = self.epsilon_eval
        else:
            epsilon = self.epsilon_fn(
                self.epsilon_decay_period,
                self.training_steps,
                self.min_replay_history,
                self.epsilon_train)
        if random.random() <= epsilon:
            # Choose a random action with probability epsilon.
            return random.randint(0, self.num_actions - 1)
        else:
            # Select action according to policy with probability (1-eps)
            # otherwise, select random action

            return self._sess.run(self.next_action, {self.state_ph: self.state})

    def _build_networks(self):
        """Builds the Q-value network computations needed for acting and training.

        These are:
          self.online_convnet: For computing the current state's Q-values.
          self.target_convnet: For computing the next state's target Q-values.
          self._net_outputs: The actual Q-values.
          self._q_argmax: The action maximizing the current state's Q-values.
          self._replay_net_outputs: The replayed states' Q-values.
          self._replay_next_target_net_outputs: The replayed next states' target
            Q-values (see Mnih et al., 2015 for details).
        """

        # _network_template instantiates the model and returns the network object.
        # The network object can be used to generate different outputs in the graph.
        # At each call to the network, the parameters will be reused.
        self.online_convnet = tf.make_template('Online', self._network_template)
        self.target_convnet = tf.make_template('Target', self._network_template)
        self._net_outputs = self.online_convnet(self.state_ph)

        # TODO(bellemare): Ties should be broken. They are unlikely to happen when
        #  using a deep network, but may affect performance with a linear
        #  approximation scheme.

        self._replay_net_outputs = self.online_convnet(self._replay.states)
        self._replay_next_net_outputs = self.online_convnet(self._replay.next_states)
        self._replay_next_target_net_outputs = self.target_convnet(
            self._replay.next_states)

        chosen_q_values = self._net_outputs.unordered_q_heads
        chosen_imt = self._net_outputs.unordered_imt_heads

        chosen_imt = tf.exp(chosen_imt)
        chosen_imt = (
                chosen_imt / tf.reduce_max(chosen_imt, axis=1, keep_dims=True)[0]
                > self._threshold
        )
        chosen_imt = tf.cast(chosen_imt, dtype=tf.float32)

        chosen_imt = chosen_imt * chosen_q_values + (1 - chosen_imt) * -1e8

        avg_q_values = tf.reduce_mean(chosen_imt, axis=2)

        self.next_action = tf.argmax(avg_q_values, axis=1)

    def _build_replay_buffer(self, use_staging):
        """Creates the replay buffer used by the agent."""
        return fixed_replay_buffer.WrappedFixedReplayBuffer(
            data_dir=self._replay_data_dir,
            replay_suffix=self._replay_suffix,
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype,
            replay_capacity=self._replay_capacity,
            data_set_mode=self._data_set_mode,
        )


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


def combine_q_functions2(q_funtions, transform_strategy, **kwargs):
    """Utility function for combining multiple Q functions (2 dimension).
    Args:
    q_functions: Multiple Q-functions concatenated.
    transform_strategy: str, Possible options include (1) 'IDENTITY' for no
      transformation (2) 'STOCHASTIC' for random convex combination.
    **kwargs: Arbitrary keyword arguments. Used for passing `transform_matrix`,
      the matrix for transforming the Q-values if the passed
      `transform_strategy` is `STOCHASTIC`.

    Returns:
    q_functions: Modified Q-functions.
    q_values: Q-values based on combining the multiple heads.
    """
    # Create q_values before reordering the heads for training
    q_values = tf.reduce_mean(q_funtions, axis=-1)

    if transform_strategy == 'STOCHASTIC':
        left_stochastic_matrix = kwargs.get('transform_matrix')
        if left_stochastic_matrix is None:
            raise ValueError('None value provided for stochastic matrix')
        q_funtions = tf.tensordot(
            q_funtions, left_stochastic_matrix, axes=[[1], [0]])
    elif transform_strategy == 'IDENTITY':
        tf.logging.info('Identity transformation Q-function heads')
    else:
        raise ValueError(
            '{} is not a valid reordering strategy'.format(transform_strategy))
    return q_funtions, q_values

