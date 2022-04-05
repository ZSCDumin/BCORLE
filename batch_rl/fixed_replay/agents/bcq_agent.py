#  coding=utf-8

"""
BCQ agent with fixed replay buffer(s).
"""

from dopamine.agents.dqn import dqn_agent
from replay_memory import fixed_replay_buffer, fixed_replay_buffer_upper_envelop

import os
import collections
import random
import numpy as np
from absl import logging

import tensorflow as tf


class BCQAgent(dqn_agent.DQNAgent):
    """An implementation of the BCQ agent with fixed replay buffer(s)."""
    def __init__(self, sess, num_actions,
                 replay_data_dir,
                 replay_suffix=None,
                 init_checkpoint_dir=None,
                 threshold=0.3,
                 # q_loss_weight=2e1,
                 # i_regularization_weight=1e-1,
                 # i_loss_weight=1.0,
                 q_loss_weight=1.0,
                 i_regularization_weight=1e-2,
                 i_loss_weight=1.0,
                 replay_capacity=1000000,
                 data_set_mode='ALL',
                 name='BCQ',
                 border=None,
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
        self._name = name
        self._border = border
        self._replay_data_dir = replay_data_dir
        self._replay_suffix = replay_suffix
        self._replay_capacity = replay_capacity
        self._data_set_mode = data_set_mode
        self._threshold = threshold
        self._q_loss_weight = q_loss_weight,
        self._i_regularization_weight = i_regularization_weight,
        self._i_loss_weight = i_loss_weight
        if init_checkpoint_dir is not None:
            self._init_checkpoint_dir = os.path.join(
                init_checkpoint_dir, 'checkpoints')
        else:
            self._init_checkpoint_dir = None
        super(BCQAgent, self).__init__(
            sess, num_actions, **kwargs)

    def _build_target_q_op(self):
        """Build an op used as a target for the Q-value.
        Returns:
        target_q_op: An op calculating the Q-value.
        """
        # Get the maximum Q-value across the actions dimension for each head.
        replay_chosen_q_values = self._replay_next_net_outputs.q_values
        replay_chosen_i = self._replay_next_net_outputs.net_i
        replay_action_imt = self._replay_next_net_outputs.action_prob

        replay_action_imt = tf.exp(replay_action_imt)
        replay_action_imt = (
                replay_action_imt / tf.reduce_max(replay_action_imt, axis=1, keep_dims=True)[0]
                > self._threshold
        )
        replay_action_imt = tf.cast(replay_action_imt, dtype=tf.float32)

        replay_action_imt = replay_action_imt * replay_chosen_q_values + (1 - replay_action_imt) * -1e8

        next_action = tf.argmax(replay_action_imt, axis=1)

        replay_next_action_one_hot = tf.one_hot(
            next_action, self.num_actions, 1., 0., name='next_action_one_hot')

        replay_next_q_values = tf.reduce_sum(
            self._replay_next_target_net_outputs.q_values * replay_next_action_one_hot,
            axis=1,
            name='replay_next_q_values')

        return self._replay.rewards + self.cumulative_gamma * replay_next_q_values * (
                1. - tf.cast(self._replay.terminals, tf.float32))

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """
        replay_action_one_hot = tf.one_hot(
            self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
        replay_chosen_q = tf.reduce_sum(
            self._replay_net_outputs.q_values * replay_action_one_hot,
            axis=1,
            name='replay_chosen_q'
        )

        target = tf.stop_gradient(self._build_target_q_op())
        q_loss = tf.losses.huber_loss(
            target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
        tmp = tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self._replay.actions, logits=self._replay_net_outputs.action_prob), axis=1)
        i_loss = self._i_loss_weight * tf.reduce_mean(tmp, axis=1)
        i3_loss = self._i_regularization_weight * tf.reduce_mean(
            tf.pow(self._replay_net_outputs.net_i, 2), axis=1
        )

        loss = q_loss + i_loss + i3_loss
        if self._border:
            ratio = tf.exp(self._replay.returns / self._replay.estimated_returns)
            ratio = tf.nn.softmax(ratio) * 1

            final_loss = tf.reduce_sum(loss * ratio) * 1
        else:
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
        network_template = self._network(
            state=state,
            num_actions=self.num_actions
        )
        return network_template

    def _network(
            self,
            state,
            num_actions
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
        net = tf.layers.dense(
            net_out,
            units=512,
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer
        )
        q_values = tf.layers.dense(
            net,
            units=num_actions,
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
            units=num_actions,
            kernel_initializer=self._kernel_initializer
        )

        action_prob = tf.nn.log_softmax(net_i)

        return self._get_network_type()(q_values, net_i, action_prob)

    def _get_network_type(self):
        """Returns the type of the outputs of a Q value network.

        Returns:
          net_type: _network_type object defining the outputs of the network.
        """
        return collections.namedtuple(
            'bcq_network',
            ['q_values', 'net_i', 'action_prob']
        )

    def step(self, reward, observation):
        self._record_observation(observation)
        self.action = self._select_action()
        return self.action

    def end_episode(self, reward):
        assert self.eval_mode, 'Eval mode is not set to be True.'
        super(BCQAgent, self).end_episode(reward)

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
            q, imt, i = self._sess.run(self._net_outputs, {self.state_ph: self.state})
            imt = np.exp(imt)
            imt = (imt / np.max(imt, axis=1)[0] > self._threshold)
            res = (imt * q + (1. - imt) * -1e8)
            res = np.argmax(res, axis=1)
            return res

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

        self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
        self._replay_net_outputs = self.online_convnet(self._replay.states)
        self._replay_next_net_outputs = self.online_convnet(self._replay.next_states)
        self._replay_next_target_net_outputs = self.target_convnet(
            self._replay.next_states)

    def _build_replay_buffer(self, use_staging):
        """Creates the replay buffer used by the agent."""
        print("self.name: {}".format(self._name))
        if self._name == 'BCQ':
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
                data_set_mode=self._data_set_mode
            )
        elif self._name == 'BAIL_BCQ1':
            return fixed_replay_buffer_upper_envelop.WrappedFixedReplayBuffer(
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
                train_mode='BCQ',
                border=self._border,
            )
        else:
            return fixed_replay_buffer_upper_envelop.WrappedFixedReplayBuffer(
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
                train_mode='BCQ',
            )

