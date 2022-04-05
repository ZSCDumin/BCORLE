#  coding=utf-8

"""
BCQ agent with fixed replay buffer(s).
"""

from dopamine.agents.dqn import dqn_agent_rl_bcqrem

import os
import collections
import random
import numpy as np
from absl import logging
from batch_rl.multi_head import atari_helpers

import tensorflow as tf
import tensorflow.contrib.layers as layers
import copy

class REMBCQAgent(dqn_agent_rl_bcqrem.DQNAgent):
    """An implementation of the BCQ agent with fixed replay buffer(s)."""

    def __init__(self, sess, num_actions,
                 state_dim, Lambda_dim,
                 number_users, Lambda_size, Lambda_interval,
                 num_heads,
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
                 network=None,
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

        logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
        logging.info('\t replay_suffix %s', replay_suffix)
        self.state_dim = state_dim
        self.Lambda_dim = Lambda_dim
        self.number_users = number_users
        self.Lambda_size = Lambda_size
        self.Lambda_interval = Lambda_interval
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
        # self.propensity_network = PropensityNet(
        #     state_dim, num_actions, Lambda_dim, 0.001, "pre_train_network"
        # )
        if init_checkpoint_dir is not None:
            self._init_checkpoint_dir = os.path.join(
                init_checkpoint_dir, 'checkpoints')
        else:
            self._init_checkpoint_dir = None
        super(REMBCQAgent, self).__init__(state_dim, num_actions, network=network, **kwargs)
        self._build_networks()
        self.train_op = self._build_train_op()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100000)


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

        replay_chosen_imt = replay_chosen_imt * tf.reduce_mean(replay_chosen_q_values, axis=-1) + (1 - replay_chosen_imt) * -1e8

        next_action = tf.argmax(replay_chosen_imt, axis=1)

        replay_next_action_one_hot = tf.one_hot(
            next_action, self.num_actions, 1., 0., name='next_action_one_hot'
        )
        replay_next_q_values = tf.reduce_mean(
            tf.reduce_mean(self._replay_next_target_net_outputs.unordered_q_heads, axis=-1) * replay_next_action_one_hot,
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

        is_non_terminal = 1. - tf.cast(tf.squeeze(self.done_ph), tf.float32)
        is_non_terminal = is_non_terminal
        rewards = tf.squeeze(self.reward_ph)
        target_res = rewards + (
                self.cumulative_gamma * replay_next_q_values * is_non_terminal
        )
        return target_res

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """
        indices = tf.stack(
            [tf.reshape(tf.range(tf.shape(self.action_ph)[0]), shape=tf.shape(self.action_ph)), self.action_ph],
            axis=-1)
        replay_chosen_q = tf.gather_nd(
            self._replay_net_outputs.unordered_q_heads, indices=indices
        )
        kwargs = {}  # Used for passing the transformation matrix if any
        if self._q_heads_transform is None:
            if self.transform_strategy == 'STOCHASTIC':
                tf.logging.info('Creating q_heads transformation matrix..')
                self._q_heads_transform = atari_helpers.random_stochastic_matrix(
                    self.num_heads, num_cols=self._num_convex_combinations)
        if self._q_heads_transform is not None:
            kwargs.update({'transform_matrix': self._q_heads_transform})
        self.replay_chosen_q, _ = combine_q_functions2(replay_chosen_q, self.transform_strategy, **kwargs)
        self.average_Q = tf.reduce_mean(self.replay_chosen_q, axis=-1, name='current_q')

        self.target = tf.stop_gradient(self._build_target_q_op())
        q_loss = tf.losses.huber_loss(
            tf.squeeze(self.target), tf.squeeze(tf.reduce_mean(self.replay_chosen_q,axis = -1)), reduction=tf.losses.Reduction.NONE)

        i_loss = []
        i_loss.append(tf.expand_dims(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.squeeze(self.action_ph, squeeze_dims=1), logits=self._replay_net_outputs.unordered_imt_heads)), axis = -1))
        i_loss = tf.stack(i_loss, axis=1)
        i_loss, _ = combine_q_functions2(
            tf.squeeze(i_loss, axis=-1), self.transform_strategy, **kwargs)

        i3_loss = []
        i3_loss.append(tf.expand_dims(tf.pow(tf.reduce_mean(self._replay_net_outputs.unordered_i_heads), 2),axis=-1))
        i3_loss = tf.stack(i3_loss, axis=1)
        i3_loss, _ = combine_q_functions2(
            i3_loss, self.transform_strategy, **kwargs)

        loss = q_loss + self._i_loss_weight * tf.squeeze(i_loss) + self._i_regularization_weight * tf.squeeze(i3_loss)

        self.final_loss = tf.reduce_mean(loss)
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('HuberLoss', self.final_loss)

        return self.optimizer.minimize(self.final_loss)

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
            Lambda=Lambda,
            num_actions=self.num_actions,
            num_heads=self.num_heads,
            transform_strategy=self.transform_strategy,
            **kwargs
        )
        return network_template

    def _network(
            self,
            state,
            Lambda,
            num_actions,
            num_heads,
            transform_strategy=None,
            **kwargs
    ):
        self._kernel_initializer = tf.initializers.variance_scaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
        )
        activation_fn = tf.nn.relu
        with tf.variable_scope('rembcq_q_net'):
            input = tf.concat([state, Lambda], axis=1)
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
            net_out = layers.fully_connected(net, num_actions * num_heads, activation_fn=None)
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
                units=num_actions,
                kernel_initializer=self._kernel_initializer
            )
            unordered_q_heads = tf.reshape(q_values, [-1, num_actions, num_heads])
            unordered_i_heads = net_i
            unordered_imt_heads = net_i
            #unordered_imt_heads = tf.reshape(net_i, [-1, num_actions])
            unordered_imt_heads = tf.nn.log_softmax(unordered_imt_heads)

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
        super(REMBCQAgent, self).end_episode(reward)

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

            return self.sess.run(self.next_action, {self.state_ph: self.state})

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
        self._net_outputs = self.online_convnet(self.state_ph, self.Lambda_ph)

        # TODO(bellemare): Ties should be broken. They are unlikely to happen when
        #  using a deep network, but may affect performance with a linear
        #  approximation scheme.

        self._replay_net_outputs = self.online_convnet(self.state_ph, self.Lambda_ph)
        self._replay_next_net_outputs = self.online_convnet(self.next_state_ph, self.Lambda_ph)
        self._replay_next_target_net_outputs = self.target_convnet(
            self.next_state_ph, self.Lambda_ph)

        chosen_q_values = self._net_outputs.unordered_q_heads
        chosen_imt = self._net_outputs.unordered_imt_heads
        kwargs = {}  # Used for passing the transformation matrix if any
        if self._q_heads_transform is None:
            if self.transform_strategy == 'STOCHASTIC':
                tf.logging.info('Creating q_heads transformation matrix..')
                self._q_heads_transform = atari_helpers.random_stochastic_matrix(
                    self.num_heads, num_cols=self._num_convex_combinations)
        if self._q_heads_transform is not None:
            kwargs.update({'transform_matrix': self._q_heads_transform})
        chosen_q_values,_ = combine_q_functions2(chosen_q_values, self.transform_strategy, **kwargs)
        chosen_imt = tf.exp(chosen_imt)
        chosen_imt = (
                chosen_imt / tf.reduce_max(chosen_imt, axis=1, keep_dims=True)[0]
                > self._threshold
        )
        chosen_imt = tf.cast(chosen_imt, dtype=tf.float32)
        chosen_q_values = tf.reduce_mean(chosen_q_values, axis=-1)
        chosen_imt = chosen_imt * chosen_q_values + (1 - chosen_imt) * -1e8

        #avg_q_values = tf.reduce_mean(chosen_imt, axis=2)

        self.next_action = tf.expand_dims(tf.argmax(chosen_imt, axis=1), axis= 1, name = 'next_action')

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
            self.reward_ph: reward,
            self.done_ph: done
        })
        error = abs(current_q - target_q)
        replay_buffer.add(error, (state, action, reward, next_state, done, reward1, reward2, Lambda))

    def train(self, replay_buffer):

        # Sample replay buffer
        if replay_buffer.is_priority:
            state, action, next_state, reward, done, reward1, reward2, Lambda, \
            idxs, is_weights = replay_buffer.sample_priority()
        else:
            state, action, next_state, reward, done, reward1, reward2, Lambda, \
            idxs, is_weights = replay_buffer.sample_without_priority()

        Lambda = np.expand_dims(Lambda, axis=1)
        done = np.expand_dims(done, axis=1)
        reward = np.expand_dims(reward, axis=1)
        # Loss for total reward, reward1, and reward2 respectively

        # Update target network by polyak or full copy every X iterations.
        _, loss = self.sess.run([self.train_op, self.final_loss], feed_dict=
        {self.state_ph: state,
         self.action_ph: action,
         self.reward_ph: reward,
         self.Lambda_ph: Lambda,
         self.next_state_ph: next_state,
         self.done_ph: done})

        current_q = self.sess.run(self.replay_chosen_q, feed_dict={
            self.state_ph: state,
            self.action_ph: action,
            self.Lambda_ph: Lambda
        })

        # Compute the target Q value
        target_q = self.sess.run(self.target, feed_dict={
            self.next_state_ph: next_state,
            self.Lambda_ph: Lambda,
            self.reward_ph: reward,
            self.done_ph: done
        })
        errors = abs(np.mean(np.squeeze(current_q)) - target_q)

        if replay_buffer.is_priority:
            # update priority
            for i in range(replay_buffer.batch_size):
                idx = idxs[i]
                replay_buffer.update(idx, errors[i])

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
    tf.random_uniform()
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

#
#
# class PropensityNet(object):
#     def __init__(self, state_dim, num_actions, Lambda_dim, lr, name):
#         self.name = name
#         self.state_dim = state_dim
#         self.action_dim = num_actions
#         self.state_ = tf.placeholder(tf.float32, [None, state_dim], name="p_state")
#         self.lambda_ = tf.placeholder(tf.float32, [None, Lambda_dim], name="p_lambda")
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
#         self.i3_ = tf.squeeze(self.i3, name = 'action_p')
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


