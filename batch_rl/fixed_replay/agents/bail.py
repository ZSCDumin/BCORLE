# coding: utf-8

"""
DQN agent with fixed replay buffer(s).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from concurrent import futures

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import gzip
import tensorflow as tf
import collections
import matplotlib.pyplot as plt
import random
import seaborn as sns
import imageio
from replay_memory import fixed_replay_buffer_upper_envelop
from absl import logging

from dopamine.agents.dqn import dqn_agent

gfile = tf.gfile


class FixedReplayBailAgent(dqn_agent.DQNAgent):
    """An implementation of the DQN agent with fixed replay buffer(s)."""

    def __init__(self, sess, num_actions,
                 replay_data_dir,
                 replay_suffix=None,
                 init_checkpoint_dir=None,
                 replay_capacity=1000000,
                 K=1000,
                 data_set_mode='ALL',
                 select_percentage=0.3,
                 **kwargs):
        assert replay_data_dir is not None

        logging.info(
            'Creating FixedReplayDQNAgent with replay directory: %s', replay_data_dir)
        logging.info('\t init_checkpoint_dir %s', init_checkpoint_dir)
        logging.info('\t replay_suffix %s', replay_suffix)
        # Set replay_log_dir before calling parent's initializer
        self._replay_data_dir = replay_data_dir
        self._replay_suffix = replay_suffix
        self._replay_capacity = replay_capacity
        self.select_percentage = select_percentage
        self.K = K
        self._data_set_mode = data_set_mode
        if init_checkpoint_dir is not None:
            self._init_checkpoint_dir = os.path.join(
                init_checkpoint_dir, 'checkpoints')
        else:
            self._init_checkpoint_dir = None

        super(FixedReplayBailAgent, self).__init__(sess, num_actions, **kwargs)

    def _build_replay_buffer(self, use_staging):
        """Creates the replay buffer used by the agent."""
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
            data_set_mode=self._data_set_mode
        )

    def get_mcret(self, dir, suffix, rollout=1000):
        replay_buffer = self._replay.memory._load_buffer(suffix)
        self.cal_mcret(replay_buffer, dir, rollout, suffix)

    def get_mc_estimated_ret(self, dir, suffix, rollout=1000):
        replay_buffer = self._replay.memory._load_buffer(suffix)
        self.cal_mc_estimated_ret(replay_buffer, dir, rollout, suffix)

    def get_bc_data(self, dir, border, suffix):
        replay_buffer = self._replay.memory._load_buffer(suffix, with_return=True)
        self.cal_bcret(replay_buffer, dir, border, suffix)

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

        # value net (for upper envelop)
        self.value_convnet = tf.make_template('Value', self._network_template)
        self.retrain_value_convnet = tf.make_template('Retrain_Value', self._network_template)
        self._value_net_outputs = self.value_convnet(self.state_ph)
        self._batch_value_net_outputs = self.value_convnet(self.batch_size_state_ph)

        self._replay_value_net_outputs = self.value_convnet(self._replay.states)
        self._replay_value_net_outputs_test = self.value_convnet(self._replay.states_test)

        # behavior clone net
        self.bc_convnet = tf.make_template('BehaviorClone', self._network_template_bc)
        # self.retrain_bc_convnet = tf.make_template('Retrain_BehaviorClone', self._network_template_bc)
        self._bc_net_outputs = self.bc_convnet(self.state_ph)
        self._replay_bc_net_outputs = self.bc_convnet(self._replay.state_bc)
        self._replay_bc_net_outputs_test = self.bc_convnet(self._replay.state_bc_test)
        self._action_prob_argmax = tf.argmax(self._bc_net_outputs.actions, axis=1)[0]

    def _network_template(self, state):
        """Builds the convolutional network used to compute the agent's Q-values.

        Args:
          state: `tf.Tensor`, contains the agent's current state.

        Returns:
          net: _network_type object containing the tensors output by the network.
        """
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
        v_values = tf.layers.dense(
            net,
            units=1,
            activation=None,
            kernel_initializer=self._kernel_initializer
        )
        return self._get_network_type()(v_values)

    def _network_template_bc(self, state):
        """Builds the convolutional network used to compute the agent's Q-values.

        Args:
          state: `tf.Tensor`, contains the agent's current state.

        Returns:
          net: _network_type object containing the tensors output by the network.
        """
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
        actions = tf.layers.dense(
            net,
            units=self.num_actions,
            activation=None,
            kernel_initializer=self._kernel_initializer
        )
        actions_prob = tf.nn.log_softmax(actions)
        return self._get_network_type_bc()(actions_prob)

    def _get_network_type(self):
        """Returns the type of the outputs of a Q value network.

        Returns:
          net_type: _network_type object defining the outputs of the network.
        """
        return collections.namedtuple('Bail_network', ['v_values'])

    def _get_network_type_bc(self):
        """Returns the type of the outputs of a Q value network.

        Returns:
          net_type: _network_type object defining the outputs of the network.
        """
        return collections.namedtuple('Bail_BC_network', ['actions'])

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """
        # train the ue network
        self.returns = tf.expand_dims(self._replay.returns, axis=1)
        self.estimated_returns = self._replay_value_net_outputs.v_values
        # loss is the l2 penalty loss, self.K is the big integer, refer https://arxiv.org/abs/1910.12179
        self._ue_loss = self._l2PenaltyLoss(self.estimated_returns, self.returns, k_val=self.K)
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('l2PenaltyLoss', tf.reduce_mean(self._ue_loss))
        self._ue_optim_ = self.optimizer.minimize(tf.reduce_mean(self._ue_loss))

        # test the ue network
        self.returns_test = tf.expand_dims(self._replay.returns_test, axis=1)
        self.estimated_returns_test = self._replay_value_net_outputs_test.v_values
        # loss is the l2 penalty loss, self.K is the big integer, refer https://arxiv.org/abs/1910.12179
        self._ue_loss_test = self._l2PenaltyLoss(self.estimated_returns_test, self.returns_test, k_val=self.K)
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('l2PenaltyLoss_test', tf.reduce_mean(self._ue_loss_test))

        # train the bc network
        self.estimated_actions_prob = self._replay_bc_net_outputs.actions
        # loss is the bc loss
        self._bc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self._replay.action_bc,
            logits=self.estimated_actions_prob
        )
        # test the bc network
        self.estimated_actions_prob_test = self._replay_bc_net_outputs_test.actions

        # loss is the bc loss
        self._bc_loss_test = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self._replay.action_bc_test,
            logits=self.estimated_actions_prob_test
        )
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('bc_loss', tf.reduce_mean(self._bc_loss))
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('bc_loss_test', tf.reduce_mean(self._bc_loss_test))
        self._bc_optim_ = self.optimizer.minimize(tf.reduce_mean(self._bc_loss))

        return self._ue_optim_, self._bc_optim_, self._ue_loss_test, self._bc_loss_test

    # fixme: duoble check whether use this function
    def _build_sync_op(self):
        """Builds ops for assigning weights from online to target network.

        Returns:
          ops: A list of ops assigning weights from online to target network.
        """
        # Get trainable variables from online and target DQNs
        sync_qt_ops = []
        scope = tf.get_default_graph().get_name_scope()
        trainables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, 'Value'))
        trainables_retrain = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, 'Retrain_Value'))

        for (w, w_retrain) in zip(trainables, trainables_retrain):
            # Assign weights from online to target network.
            sync_qt_ops.append(w.assign(w_retrain, use_locking=True))
        return sync_qt_ops

    def _sync_with_given_trainables(self, trainables, network_name='Value'):
        """Builds ops for assigning weights from given variables to given network.

        Returns:
          ops: A list of ops assigning weights from given variables to given network.
        """
        # Get trainable variables from online and target DQNs
        sync_qt_ops = []
        scope = tf.get_default_graph().get_name_scope()
        trainables_target = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, network_name))

        for (w, w_target) in zip(trainables, trainables_target):
            # Assign weights from online to target network.
            sync_qt_ops.append(w.assign(w_target, use_locking=True))
        self._sess.run(sync_qt_ops)

    def _l2PenaltyLoss(self, predicted, target, k_val):
        loss = tf.cast((predicted >= target), tf.float32) * (predicted - target) ** 2 +\
               tf.cast((predicted < target), tf.float32) * k_val * (target - predicted) ** 2
        return loss

    '''Training code for UE is here'''

    def calc_ue_valiloss(self, test_states, test_returns, batch_size, ue_loss_k):
        test_iter = int(np.ceil(test_returns.shape[0] / batch_size))
        validation_loss = 0

        for n in range(test_iter):
            ind = slice(n * batch_size, min((n + 1) * batch_size, test_returns.shape[0]))
            states_t, returns_t = test_states[ind], test_returns[ind]
            Vsi = self._sess.run(self._replay_value_net_outputs.v_values, feed_dict={
                self._replay_state_ph: states_t
            })
            loss = self._l2PenaltyLoss(Vsi, np.expand_dims(returns_t, axis=1), k_val=ue_loss_k)
            loss = self._sess.run(tf.reduce_mean(loss))
            validation_loss += loss

        return validation_loss / test_iter

    def _select_batch_ue(self, states, returns, actions, seed, ue_loss_k, C, select_percentage):

        s_val = self._sess.run(self._replay_value_net_outputs.v_values, feed_dict={
            self._replay_state_ph: states
        })
        returns = np.expand_dims(returns, axis=1)
        ratios = returns / np.min(s_val, C) if C is not None else returns / s_val
        ratios = np.squeeze(ratios, axis=1)
        increasing_ratios = np.sort(ratios)
        increasing_ratio_indices = np.argsort(ratios)
        bor_ind = increasing_ratio_indices[-int(select_percentage * states.shape[0])]  # fixme: 0.25 or 0.3
        border = ratios[bor_ind]

        selected_buffer = []
        '''begin selection'''
        print('Selecting with ue border', border.item())
        for i in range(states.shape[0]):
            rat = ratios[i]
            if rat >= border:
                obs, act = states[i], actions[i]
                selected_buffer.append((obs, act))

        initial_len, selected_len = len(states), len(selected_buffer)
        print('border:', border, 'selecting ratio:', selected_len, '/', initial_len)
        return (selected_buffer, selected_len, border)

    # def bc_train(self, replay_buffer, iterations=500, batch_size=1000):
    #     # fixme: iteration used to be 500
    #     iterations = 5
    #     for it in range(iterations):
    #         ind = np.random.random_integers(0, len(replay_buffer) - 1, batch_size)
    #         states = [replay_buffer[i][0] for i in ind]  # fixme: may out of range
    #         actions = [replay_buffer[i][1] for i in ind]
    #         actions = tf.one_hot(
    #             actions, self.num_actions, 1., 0., name='action_one_hot'
    #         )
    #         actions = self._sess.run(actions)
    #
    #         self._sess.run([self._bc_loss, self._bc_optim_], feed_dict={
    #             self._replay_state_ph: np.array(states),
    #             self.actions: actions
    #         })
    #         # TODO: logger
    #         # logger.store(Loss=actor_loss.cpu().item())

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
            # Choose the action with highest Q-value at the current state.
            return self._sess.run(self._action_prob_argmax, {self.state_ph: self.state})

    def cal_mcret(self, replay_buffer, dir, rollout, suffix):
        gts = []
        gamma = self._replay.memory._replay_buffers[0]._gamma

        g = 0
        prev_s = 0
        # index of the last state in the current episode
        termination_point = 0

        endpoint = []
        dist = []  # L2 distance between the current state and the termination point
        indices = [0, len(replay_buffer._store['observation']) - 3]

        def next_tuple(max_indice, min_indice):
            indice = max_indice
            while indice >= min_indice:
                tuple = replay_buffer.sample_transition_batch(
                    batch_size=1, indices=[indice], mode='mc', single_iter=True)
                yield indice, tuple
                indice -= 1

        for ind, (state, action, reward, next_state, next_action, next_reward,
                  terminal, indice) in next_tuple(indices[-1], indices[0]):
            # fixme: 是否采用并行计算优化取决于实验结果
            # fixme: 是否优化内存: 一次load 一个replay buffer并存储
            if ind % 1000 == 0:
                print('calculating mc..., finished {}%'.format((indices[-1] - ind) / indices[-1] * 100))

            if terminal:
                g = reward
                gts.append(g)
                endpoint.append(ind)
                termination_point = state
                prev_s = state
                dist.append(0.)
                continue

            if np.array_equal(prev_s, next_state):
                g = gamma * g + reward
                prev_s = state
                dist.append(np.linalg.norm(state - termination_point))
            else:
                g = reward
                endpoint.append(ind)
                termination_point = state
                prev_s = state
                dist.append(0.)

            gts.append(g)

        gts, endpoint, dist = gts[::-1], endpoint[::-1], dist[::-1]
        aug_gts = gts[:]

        # Add augmentation terms
        start = 0
        for i in range(len(endpoint)):
            end = endpoint[i]

            # episodes not early terminated
            for j in range(end, start, -1):
                interval = dist[start: start + end - j + 2]
                index = interval.index(min(interval))
                # term = end - j + 1
                # term += rollout - index
                aug_gts[j] += gamma ** (end - j + 1) * gts[start + index]
                if index != end - j + 1:
                    # fixme: whether 1000
                    aug_gts[j] -= gamma ** (27000) * gts[index + j]
                    # term -= end - index - j + 1

                # print("number of terms used to calculate mc ret: ", term)
            start = end + 1
        replay_buffer._store['returns'] = np.squeeze(np.array(aug_gts) * 1)

        replay_buffer.save_return(dir, iteration_number=suffix)

    def cal_mc_estimated_ret(self, replay_buffer, dir, rollout, suffix):
        estimated_returns = []

        indices = [0, len(replay_buffer._store['observation']) - 3]

        def next_tuple(max_indice, min_indice):
            indice = max_indice
            while indice >= min_indice:
                indices = list(range(max(indice - 3200, min_indice), indice))
                indices.reverse()
                tuple = replay_buffer.sample_transition_batch(
                    batch_size=len(range(max(indice - 3200, min_indice), indice)),
                    indices=indices,
                    mode='mc',
                    single_iter=True
                )
                yield indice, tuple
                indice -= 3200

        verbose_steps = indices[-1]
        for ind, (state, action, reward, next_state, next_action, next_reward,
                  terminal, indice) in next_tuple(indices[-1], indices[0]):
            # fixme: 是否采用并行计算优化取决于实验结果
            # fixme: 是否优化内存: 一次load 一个replay buffer并存储
            if ind < verbose_steps:
                print('calculating mc estimated return..., finished {}%'.format((indices[-1] - ind) / indices[-1] * 100))
                verbose_steps -= self.verbose_steps
            estimated_return = self._sess.run(self._batch_value_net_outputs,
                                              feed_dict={
                                                  self.batch_size_state_ph: state,
                                              })
            estimated_returns.extend(np.squeeze(estimated_return))

        estimated_returns = estimated_returns[::-1]

        replay_buffer._store['estimated_returns'] = np.squeeze(np.array(estimated_returns) * 1)

        replay_buffer.save_estimated_return(dir, iteration_number=suffix)

    def cal_bcret(self, replay_buffer, dir, border, suffix):
        state_bc = []
        action_bc = []
        reward_bc = []
        terminal_bc = []
        next_state_bc = []

        indices = [0, len(replay_buffer._store['observation']) - 3]

        def next_tuple(max_indice, min_indice):
            indice = max_indice
            while indice >= min_indice:
                indices = list(range(max(indice - 32, min_indice), indice))
                indices.reverse()
                tuple = replay_buffer.sample_transition_batch(
                    batch_size=len(range(max(indice - 32, min_indice), indice)),
                    indices=indices,
                    mode='bc',
                    single_iter=True
                )
                yield indice, tuple
                indice -= 32

        verbose_steps = indices[-1]
        for ind, (state, action, reward, next_state, next_action, next_reward, terminal, returns, indice) in next_tuple(indices[-1], indices[0]):
            # fixme: 是否采用并行计算优化取决于实验结果
            # fixme: 是否优化内存: 一次load 一个replay buffer并存储
            if ind < verbose_steps:
                tf.logging.info('calculating mc..., finished {}%'.format((indices[-1] - ind) / indices[-1] * 100))
                verbose_steps -= self.verbose_steps

            reward_estimated = self._sess.run(self._batch_value_net_outputs,
                                                              feed_dict={
                                                                  self.batch_size_state_ph: state,
                                                              })
            ratios = returns / np.squeeze(reward_estimated)
            for i, ratio in enumerate(ratios):
                if ratio > border:
                    state_bc.append(state[i])
                    action_bc.append(action[i])
                    reward_bc.append(reward[i])
                    terminal_bc.append(terminal[i])
                    next_state_bc.append(next_state[i])

        state_bc, action_bc, reward_bc, terminal_bc,\
        next_state_bc = state_bc[::-1], action_bc[::-1], reward_bc[::-1], terminal_bc[::-1], next_state_bc[::-1]

        # Add augmentation terms
        replay_buffer._store['state_bc'] = np.squeeze(np.array(state_bc) * 1)
        replay_buffer._store['action_bc'] = np.squeeze(np.array(action_bc) * 1)
        replay_buffer._store['reward_bc'] = np.squeeze(np.array(reward_bc) * 1)
        replay_buffer._store['terminal_bc'] = np.squeeze(np.array(terminal_bc) * 1)
        replay_buffer._store['next_state_bc'] = np.squeeze(np.array(next_state_bc) * 1)

        print("len_state_bc", len(state_bc))

        replay_buffer.save_bc(dir, iteration_number=suffix)

    def _train_step(self):
        """Runs a single training step.

        Runs a training op if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        self._sess.run(self._train_op[0])
        if (self.summary_writer is not None and
                self.training_steps > 0 and
                self.training_steps % self.summary_writing_frequency == 0):
            summary = self._sess.run(self._merged_summaries)
            self.summary_writer.add_summary(summary, self.training_steps)

        self.training_steps += 1

    def _train_step_bc(self):
        """Runs a single training step.

        Runs a training op if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        self._sess.run(self._train_op[1])
        if (self.summary_writer is not None and
                self.training_steps > 0 and
                self.training_steps % self.summary_writing_frequency == 0):
            summary = self._sess.run(self._merged_summaries)
            self.summary_writer.add_summary(summary, self.training_steps)

        self.training_steps += 1


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image

