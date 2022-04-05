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

"""Runner for experiments with a fixed replay buffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import gzip

from dopamine.discrete_domains import logger
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment

import tensorflow as tf
import logging
import numpy as np
import collections
from dopamine.discrete_domains import atari_lib

gfile = tf.gfile


class BailFixedReplayRunner(run_experiment.Runner):
    """Object that handles running Dopamine experiments with fixed replay buffer."""

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        """Reloads the latest checkpoint if it exists.

        This method will first create a `Checkpointer` object and then call
        `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
        checkpoint in self._checkpoint_dir, and what the largest file number is.
        If a valid checkpoint file is found, it will load the bundled data from this
        file and will pass it to the agent for it to reload its data.
        If the agent is able to successfully unbundle, this method will verify that
        the unbundled data contains the keys,'logs' and 'current_iteration'. It will
        then load the `Logger`'s data from the bundle, and will return the iteration
        number keyed by 'current_iteration' as one of the return values (along with
        the `Checkpointer` object).

        Args:
          checkpoint_file_prefix, the checkpoint file prefix.

        Returns:
          start_iteration, the iteration number to start the experiment from.
          experiment_checkpointer: `Checkpointer` object for the experiment.
        """
        self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                       checkpoint_file_prefix)
        self._start_iteration = 0

        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished iteration 0 (so we will start from iteration 1).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
            self._checkpoint_dir)
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version)
            if self._agent.unbundle(
                    self._checkpoint_dir, latest_checkpoint_version, experiment_data):
                if experiment_data is not None:
                    assert 'logs' in experiment_data
                    assert 'current_iteration' in experiment_data
                    self._logger.data = experiment_data['logs']
                    self._start_iteration = experiment_data['current_iteration'] + 1

                logging.info('Reloaded upper envelop calculation checkpoint and will start from iteration %d',
                             self._start_iteration)

        # Code for the loading a checkpoint at initialization
        init_checkpoint_dir = self._agent._init_checkpoint_dir  # pylint: disable=protected-access
        if (self._start_iteration == 0) and (init_checkpoint_dir is not None):
            if checkpointer.get_latest_checkpoint_number(self._checkpoint_dir) < 0:
                # No checkpoint loaded yet, read init_checkpoint_dir
                init_checkpointer = checkpointer.Checkpointer(
                    init_checkpoint_dir, checkpoint_file_prefix)
                latest_init_checkpoint = checkpointer.get_latest_checkpoint_number(
                    init_checkpoint_dir)
                if latest_init_checkpoint >= 0:
                    experiment_data = init_checkpointer.load_checkpoint(
                        latest_init_checkpoint)
                    if self._agent.unbundle(
                            init_checkpoint_dir, latest_init_checkpoint, experiment_data):
                        if experiment_data is not None:
                            assert 'logs' in experiment_data
                            assert 'current_iteration' in experiment_data
                            self._logger.data = experiment_data['logs']
                            self._start_iteration = experiment_data['current_iteration'] + 1
                        tf.logging.info(
                            'Reloaded checkpoint from %s and will start from iteration %d',
                            init_checkpoint_dir, self._start_iteration)

    def _initialize_checkpointer_and_maybe_resume_bc(self, checkpoint_file_prefix):
        """Reloads the latest checkpoint if it exists.

        This method will first create a `Checkpointer` object and then call
        `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
        checkpoint in self._checkpoint_dir, and what the largest file number is.
        If a valid checkpoint file is found, it will load the bundled data from this
        file and will pass it to the agent for it to reload its data.
        If the agent is able to successfully unbundle, this method will verify that
        the unbundled data contains the keys,'logs' and 'current_iteration'. It will
        then load the `Logger`'s data from the bundle, and will return the iteration
        number keyed by 'current_iteration' as one of the return values (along with
        the `Checkpointer` object).

        Args:
          checkpoint_file_prefix, the checkpoint file prefix.

        Returns:
          start_iteration, the iteration number to start the experiment from.
          experiment_checkpointer: `Checkpointer` object for the experiment.
        """
        self._checkpointer = checkpointer.Checkpointer(self._bail_ckpt_dir,
                                                       checkpoint_file_prefix)
        self._start_iteration_bc = 0

        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished iteration 0 (so we will start from iteration 1).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
            self._bail_ckpt_dir)
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(
                latest_checkpoint_version)
            if self._agent.unbundle(
                    self._bail_ckpt_dir, latest_checkpoint_version, experiment_data):
                if experiment_data is not None:
                    assert 'logs' in experiment_data
                    assert 'current_iteration' in experiment_data
                    self._logger.data = experiment_data['logs']
                    self._start_iteration_bc = experiment_data['current_iteration'] + 1

                logging.info('Reloaded upper envelop calculation checkpoint and will start from iteration %d',
                             self._start_iteration_bc)

        # Code for the loading a checkpoint at initialization
        init_checkpoint_dir = self._agent._init_checkpoint_dir  # pylint: disable=protected-access
        if (self._start_iteration_bc == 0) and (init_checkpoint_dir is not None):
            if checkpointer.get_latest_checkpoint_number(self._bail_ckpt_dir) < 0:
                # No checkpoint loaded yet, read init_checkpoint_dir
                init_checkpointer = checkpointer.Checkpointer(
                    init_checkpoint_dir, checkpoint_file_prefix)
                latest_init_checkpoint = checkpointer.get_latest_checkpoint_number(
                    init_checkpoint_dir)
                if latest_init_checkpoint >= 0:
                    experiment_data = init_checkpointer.load_checkpoint(
                        latest_init_checkpoint)
                    if self._agent.unbundle(
                            init_checkpoint_dir, latest_init_checkpoint, experiment_data):
                        if experiment_data is not None:
                            assert 'logs' in experiment_data
                            assert 'current_iteration' in experiment_data
                            self._logger.data = experiment_data['logs']
                            self._start_iteration_bc = experiment_data['current_iteration'] + 1
                        tf.logging.info(
                            'Reloaded checkpoint from %s and will start from iteration %d',
                            init_checkpoint_dir, self._start_iteration_bc)

    def _create_directories(self):
        """Create necessary sub-directories."""
        try:
            base_dir_this_agent = os.path.join(self._base_dir,
                                               self.cfg['ATARI_ENV'] + str(self.cfg['NUM_EXPERIMENT'])
                                               )
            gfile.MakeDirs(base_dir_this_agent)
            base_dir_this_agent = os.path.join(base_dir_this_agent,
                                               self.cfg['AGENT']
                                               )
            gfile.MakeDirs(base_dir_this_agent)
            self._checkpoint_dir = os.path.join(base_dir_this_agent,
                                                'checkpoints')
            self._logger = logger.Logger(os.path.join(base_dir_this_agent,
                                                      'logs'))

            self._bail_mc_gain_dir = os.path.join(self.cfg['replay_dir'],
                                                  self.cfg['ATARI_ENV'] + str(self.cfg['NUM_EXPERIMENT']),
                                                  'replay_logs'
                                                  )
            self._bail_ue_dir = os.path.join(base_dir_this_agent,
                                             'upper_envelop')
            base_dir_this_agent = os.path.join(base_dir_this_agent,
                                               str(self.cfg['select_percentage'])
                                               )
            gfile.MakeDirs(base_dir_this_agent)
            self._bail_bc_dir = os.path.join(base_dir_this_agent,
                                             'bc_data')
            self._bail_log_dir = os.path.join(base_dir_this_agent,
                                              'bc_logs')
            self._bail_ckpt_dir = os.path.join(base_dir_this_agent,
                                              'bc_ckpt')
            gfile.MakeDirs(self._bail_ue_dir)
            gfile.MakeDirs(self._bail_bc_dir)
            gfile.MakeDirs(self._bail_log_dir)
            gfile.MakeDirs(self._bail_ckpt_dir)
            self.base_dir_this_agent = base_dir_this_agent
        except tf.errors.PermissionDeniedError:
            # If it already exists, ignore exception.
            pass

    def run_experiment(self, ret_border=False):
        """Runs a full experiment, spread over multiple iterations."""
        print('Beginning training...')
        # list of upper ue
        is_train_ue = True
        is_plot_ue = True
        self._agent.verbose_steps = self.verbose_steps

        mc_rets = gfile.ListDirectory(self._bail_ue_dir)
        if mc_rets and str(mc_rets[0]) == 'ue_visual.png':
            logging.info('The upper envelop training has already been finished')
            is_plot_ue = False

        # list of exist mc files
        mc_rets = gfile.ListDirectory(self._bail_mc_gain_dir)
        mc_ret_counters = collections.Counter(
            [name.split('.')[-2] for name in mc_rets]
        )
        mc_ret_suffixes = [x for x in mc_ret_counters if mc_ret_counters[x] >= 8]

        # list of exist ckpt files fixme: only given one temporally
        ckpt_suffixes = [x for x in mc_ret_counters if mc_ret_counters[x] >= 6]

        # ckpt_suffixes = [str(self.cfg['checkpointDir_suffix'])]
        print("mc_ret_suffixes: ", mc_ret_suffixes, "  ckpt_suffixes:, ", ckpt_suffixes)

        if is_plot_ue:
            if self._num_iterations < self._start_iteration:
                logging.warning('num_iterations (%d) < start_iteration(%d)',
                                self._num_iterations, self._start_iteration)
                return
            elif self._num_iterations == self._start_iteration:
                is_train_ue = False
                logging.info('The upper envelop training has already been finished,'
                             ' but the upper envelop figure is not plotted')

            if is_train_ue:
                for suffix in ckpt_suffixes:
                    if str(suffix) not in mc_ret_suffixes:
                        print('Starting MC calculation')
                        self._agent.get_mcret(self._bail_mc_gain_dir, suffix=suffix)
                    else:
                        print("MC calculation already finished")

                print("ALL MC calculation finished")

                # After MC-return calculation, we train the upper envelop
                self.train_upper_envelop()
            self.plot_upper_envelop()

        # save estimated returns
        for suffix in ckpt_suffixes:
            if str(suffix) not in mc_ret_suffixes:
                print('Starting MC calculation')
                self._agent.get_mc_estimated_ret(self._bail_mc_gain_dir, suffix=suffix)
            else:
                print("Estimated returns for suffix {} already finished".format(suffix))

        filename = os.path.join(self.base_dir_this_agent)
        border = gfile.ListDirectory(filename)
        if border and ('border_new.gz' in border):
            logging.info('The border has already been calculated')

            with gfile.GFile(os.path.join(filename, 'border_new.gz'), 'rb') as f:
                with gzip.GzipFile(fileobj=f) as infile:
                    border = np.load(infile, allow_pickle=False)
                    print("Successfully loaded border: {}".format(border))

        else:  # the border has not been calculated yet
            filename = os.path.join(filename, 'border_new.gz')
            print("Staring calculate the border")
            # Train behavior cloning
            border = self._get_border(select_percentage=self._agent.select_percentage)
            print("Finished calculate the border: {}".format(border))

            with gfile.GFile(filename, 'wb') as f:
                with gzip.GzipFile(fileobj=f) as outfile:
                    np.save(outfile, border, allow_pickle=False)
                    print("Successfully saved")

        self.border = border
        if ret_border:  # the border has already been calculated
            return border
        #
        #
        # if self.cfg['only_cal_bc_data'] == 'False':
        #     print("Start training behavior cloning")
        #
        #     self._logger._logging_dir = self._bail_log_dir
        #     self._initialize_checkpointer_and_maybe_resume_bc(self.checkpoint_file_prefix)
        #
        #     is_train_bc = True
        #     if self._num_iterations_bc < self._start_iteration_bc:
        #         logging.warning('num_iterations (%d) < start_iteration(%d)',
        #                         self._num_iterations_bc, self._start_iteration_bc)
        #         return
        #     elif self._num_iterations_bc == self._start_iteration_bc:
        #         is_train_bc = False
        #         logging.info('The upper envelop training has already been finished,'
        #                      ' but the upper envelop figure is not plotted')
        #
        #     if is_train_bc:
        #         # Reload the replay buffer
        #         if self._agent._data_set_mode == 'ALL':
        #             num_buffers = 5
        #         else:
        #             num_buffers = 1
        #         tf.logging.info("The number of buffers: {}".format(num_buffers))
        #         self._agent._replay.memory.reload_buffer(num_buffers=num_buffers, with_return=True,
        #                                                  border=self.border,
        #                                                  runner_bail=self._agent)
        #         self.train_behavior_cloning()

    def train_upper_envelop(self):
        self.previous_loss = float('inf')
        self.num_increases = 0
        consecutive_steps = 4
        scope = tf.get_default_graph().get_name_scope()
        best_parameters = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, 'Value'))
        self._agent._sync_with_given_trainables(best_parameters, 'Retrain_Value')

        for iteration in range(self._start_iteration, self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)
            self._checkpoint_experiment(iteration)

            validation_loss = statistics['eval_episode_validation_loss'][0]
            if validation_loss < self.previous_loss:
                self.previous_loss = validation_loss
                scope = tf.get_default_graph().get_name_scope()
                best_parameters = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope=os.path.join(scope, 'Value')
                )
                self._agent._sync_with_given_trainables(best_parameters, 'Retrain_Value')
                self.num_increases = 0
            else:
                self.num_increases += 1
            if self.num_increases == consecutive_steps:
                self._agent._sync_with_given_trainables(best_parameters, 'Value')
                break

        print("Policy training is complete.")

    def train_behavior_cloning(self):
        for iteration in range(self._start_iteration_bc, self._num_iterations_bc):
            statistics = self._run_one_iteration_bc(iteration)
            self._log_experiment(iteration, statistics)
            self._checkpoint_experiment(iteration, dir=self._bail_ckpt_dir)

        print("Policy training is complete.")

    def plot_upper_envelop(self, ue_lr=3e-3, ue_wd=2e-2, ue_loss_k=10, ue_train_epoch=50, consecutive_steps=4,
                           num_iteration=20, num_samples_each_iteration=100):
        dir = self._bail_ue_dir
        current_palette = sns.color_palette('bright', 10)
        sns.set(color_codes=True, palette='dark', style='darkgrid', font_scale=0.4)

        upper_learning_rate, weight_decay, k_val, num_epoches, consecutive_steps = ue_lr, ue_wd, ue_loss_k,\
                                                                                   ue_train_epoch, consecutive_steps
        states = []
        MC_returns = []
        upper_envelope_r = []

        for iteration in range(0, num_iteration):
            if not self._agent._replay_suffix:
                # Reload the replay buffer
                if self._agent._data_set_mode == 'ALL':
                    num_buffers = 5
                else:
                    num_buffers = 1
                tf.logging.info("The number of buffers: {}".format(num_buffers))
                self._agent._replay.memory.reload_buffer(num_buffers=num_buffers, with_return=True)
            for step_count in range(num_samples_each_iteration):
                sampled_states, sample_returns = self._agent._sess.run(
                    [
                        self._agent._replay.states,
                        self._agent._replay.returns
                    ]
                )
                sample_estimated_returns = self._agent._sess.run(self._agent._replay_value_net_outputs, feed_dict={
                    self._agent._replay.states: sampled_states
                })
                sample_estimated_returns = np.squeeze(sample_estimated_returns)
                states.extend(list(sampled_states))
                MC_returns.extend(list(sample_returns))
                upper_envelope_r.extend(list(sample_estimated_returns))

        states = np.array(states)
        upper_envelope_r = np.array(upper_envelope_r)
        MC_returns = np.array(MC_returns)


        increasing_ue_vals = np.sort(upper_envelope_r)
        increasing_ue_indices = np.argsort(upper_envelope_r)
        MC_returns = MC_returns[increasing_ue_indices]

        all_ue_loss = self._agent._l2PenaltyLoss(np.expand_dims(increasing_ue_vals, axis=1),
                                          np.expand_dims(MC_returns, axis=1), k_val=k_val)
        all_ue_loss = self._sess.run(tf.reduce_mean(all_ue_loss))
        print("all_ue_loss is {}".format(all_ue_loss))
        plt.rc('legend', fontsize=14)  # legend fontsize
        fig, axs = plt.subplots()

        axs.set_xlabel('state', fontsize=28)
        axs.set_ylabel('Returns and \n Upper Envelope', fontsize=28, multialignment="center")

        fig, ax = plt.subplots()
        plot_s = list(np.arange(states.shape[0]))
        sns.scatterplot(x=plot_s, y=list(MC_returns), label='MC Returns')
        sns.lineplot(x=plot_s, y=list(increasing_ue_vals), color='palevioletred', label="Upper Envelope")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:], labels=labels[:])
        image = fig2data(fig)


        with gfile.GFile(os.path.join(dir, 'ue_visual.png'), "w") as f:
            imageio.imsave(f, image, 'PNG')

        print('Plotted current UE in', os.path.join(dir, "ue_visual.png"))

        return

    def _get_border(self, num_iteration=30, num_samples_each_iteration=100, select_percentage=0.3):
        ratios = []

        for iteration in range(0, num_iteration):
            if not self._agent._replay_suffix:
                # Reload the replay buffer
                if self._agent._data_set_mode == 'ALL':
                    num_buffers = 5
                else:
                    num_buffers = 1
                tf.logging.info("The number of buffers: {}".format(num_buffers))
                self._agent._replay.memory.reload_buffer(
                    num_buffers=num_buffers,
                    with_return=True,
                    with_estimated_return=True)
            for step_count in range(num_samples_each_iteration):
                sampled_states, sample_returns = self._agent._sess.run(
                    [
                        self._agent._replay.states,
                        self._agent._replay.returns
                    ]
                )
                sample_estimated_returns = self._agent._sess.run(self._agent._replay_value_net_outputs, feed_dict={
                    self._agent._replay.states: sampled_states
                })
                sample_estimated_returns = np.squeeze(sample_estimated_returns)
                ratios.extend(list(sample_returns / (sample_estimated_returns + 1e-6)))

        ratios = np.array(ratios)

        increasing_ratios = np.sort(ratios)
        increasing_ratio_indices = np.argsort(ratios)
        bor_ind = increasing_ratio_indices[-int(select_percentage * len(ratios))]
        border = ratios[bor_ind]

        return border

    def _run_one_phase_mc(self, min_steps, statistics, run_mode_str):
        """Runs the agent/environment loop until a desired number of steps.

        Args:
          min_steps, minimum number of steps to generate in this phase.
          statistics: `IterationStatistics` object which records the experimental
            results.
          run_mode_str, describes the run mode for this agent.

        Returns:
          Tuple containing the number of steps taken in this phase (int), the sum of
            returns (float), and the number of episodes performed (int).
        """
        step_count = 0
        validation_loss = 0.
        verbose_step = self.verbose_steps

        while step_count < min_steps:
            returns_test, estimated_returns_test = self._agent._sess.run(
                [
                    tf.expand_dims(self._agent._replay.returns_test, axis=1),
                    self._agent._replay_value_net_outputs_test.v_values,

                ])
            loss_test = self._agent._sess.run(
                tf.reduce_mean(self._agent._l2PenaltyLoss(estimated_returns_test, returns_test, k_val=self._agent.K))
            )

            validation_loss += loss_test
            step_count += 1
            if step_count > verbose_step:
                tf.logging.info(
                    'Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S  ", time.localtime())) +
                    'Steps percent of this phase in this iteration: {} % '.format(float(step_count) /
                                                                                  float(min_steps) * 100.0) +
                    'Steps executed: {} '.format(step_count) +
                    'Validation_loss: {} '.format(loss_test)
                )
                verbose_step += self.verbose_steps
        statistics.append({
            '{}_episode_validation_loss'.format(run_mode_str): validation_loss / step_count
        })
        return step_count, validation_loss / step_count

    def _run_train_phase(self):
        """Run training phase."""
        self._agent.eval_mode = False
        verbose_step = self.verbose_steps

        start_time = time.time()
        for step_count in range(self._training_steps):
            self._agent._train_step()  # pylint: disable=protected-access
            if step_count > verbose_step:
                tf.logging.info(
                    'Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S  ", time.localtime())) +
                    'Steps percent of this phase in this iteration: {} % '.format(float(step_count) /
                                                                                  float(self._training_steps) * 100.0)
                )
                verbose_step += self.verbose_steps
            if step_count % 10000 == 0:
                returns_b_estimation, returns_b_true = self._agent._sess.run(
                    [self._agent._replay_value_net_outputs.v_values,
                     tf.expand_dims(self._agent._replay.returns, axis=1)]
                )
                loss_test = self._agent._sess.run(
                    tf.reduce_mean(
                        self._agent._l2PenaltyLoss(returns_b_estimation, returns_b_true, k_val=self._agent.K))
                )

                loss_trian, _ = self._sess.run([tf.reduce_mean(self._agent._ue_loss), self._agent._ue_optim_])

                print("---------------------checking estimation........---------------------")
                # print("returns_b", returns_b_true)
                # print("estimated returns_b", returns_b_estimation)
                print("step_count: {}, loss_test: {}".format(step_count, loss_test))
                print("step_count: {}, loss_train: {}".format(step_count, loss_trian))
                logging.info('training upper envelop....{}%'.format(
                    step_count / self._training_steps * 100
                ))
        time_delta = time.time() - start_time
        tf.logging.info('Average training steps per second: %.2f',
                        self._training_steps / time_delta)

    def _run_train_phase_bc(self):
        """Run training phase."""
        self._agent.eval_mode = False
        verbose_step = self.verbose_steps

        loss_train, loss_test = None, None

        start_time = time.time()
        for step_count in range(self._training_steps):
            # self._agent._replay.memory.sample_transition_batch_bc()
            self._agent._train_step_bc()  # pylint: disable=protected-access
            if step_count > verbose_step:
                tf.logging.info(
                    'Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S  ", time.localtime())) +
                    'Steps percent of this phase in this iteration: {} % '.format(float(step_count) /
                                                                                  float(self._training_steps) * 100.0)
                )
                verbose_step += self.verbose_steps
            if step_count % 10000 == 0:
                actions_estimation, actions_true = self._agent._sess.run(
                    [self._agent._replay_bc_net_outputs_test.actions,
                     tf.expand_dims(self._agent._replay.action_bc_test, axis=1)]
                )
                loss_test = self._agent._sess.run(
                    tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf.one_hot(
                                actions_true, self._agent.num_actions, 1., 0., name='action_one_hot'
                            ),
                            logits=actions_estimation
                        )
                    )
                )

                loss_trian, _ = self._sess.run([tf.reduce_mean(self._agent._bc_loss), self._agent._bc_optim_])

                print("---------------------checking estimation........---------------------")
                print("step_count: {}, loss_train: {}".format(step_count, loss_trian))
                print("step_count: {}, loss_test: {}".format(step_count, loss_test))
                logging.info('training behavior cloning....{}%'.format(
                    step_count / self._training_steps * 100
                ))
        time_delta = time.time() - start_time
        tf.logging.info('Average training steps per second: %.2f',
                        self._training_steps / time_delta)

        return loss_train, loss_test

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction."""
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        # pylint: disable=protected-access
        # Reload the replay buffer
        if not self._agent._replay_suffix:
            # Reload the replay buffer
            if self._agent._data_set_mode == 'ALL':
                num_buffers = 5
            else:
                num_buffers = 1
            tf.logging.info("The number of buffers: {}".format(num_buffers))
            self._agent._replay.memory.reload_buffer(num_buffers=num_buffers, with_return=True)
        # pylint: enable=protected-access
        self._run_train_phase()

        validation_loss = self._run_eval_phase_mc(statistics, iteration)

        self._save_tensorboard_summaries(
            iteration, validation_loss)
        return statistics.data_lists

    def _run_one_iteration_bc(self, iteration):
        """Runs one iteration of agent/environment interaction."""
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        # pylint: disable=protected-access
        # Reload the replay buffer
        if not self._agent._replay_suffix:
            # Reload the replay buffer
            num_buffers = 5  # fixme: must be 5 or more
            tf.logging.info("The number of buffers: {}".format(num_buffers))
            self._agent._replay.memory.reload_buffer(num_buffers=num_buffers, with_return=True, border=self.border,
                                                             runner_bail=self._agent)

        # pylint: enable=protected-access
        loss_train, loss_test = self._run_train_phase_bc()

        num_episodes_eval, average_reward_eval = self._run_eval_phase_bc(statistics, iteration)

        self._save_tensorboard_summaries_bc(iteration, loss_train, loss_test,
                                            num_episodes_eval,
                                            average_reward_eval)
        return statistics.data_lists

    def _run_eval_phase_mc(self, statistics, iteration):
        """Run evaluation phase.

        Args:
          statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.

        Returns:
          num_episodes, The number of episodes run in this phase.
          average_reward: float, The average reward generated in this phase.
        """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        _, validation_loss = self._run_one_phase_mc(
            self._evaluation_steps, statistics, 'eval')
        logging.info('Averaged validation loss in evaluation episode: %.2f',
                     validation_loss)

        # logging
        logging.info('Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logging.info('Averaged validation loss is {}'.format(validation_loss))
        print('Time: {}, Iteration: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), iteration))
        print(
            "Evaluation Phase: Averaged validation loss is {}".format(validation_loss)
        )

        return validation_loss

    def _run_eval_phase_bc(self, statistics, iteration):
        """Run evaluation phase.

        Args:
          statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.

        Returns:
          num_episodes, The number of episodes run in this phase.
          average_reward: float, The average reward generated in this phase.
        """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        _, sum_returns, num_episodes = self._run_one_phase(
            self._evaluation_steps, statistics, 'eval')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        logging.info('Average undiscounted return per evaluation episode: %.2f',
                     average_return)
        statistics.append({'eval_average_return': average_return})

        # store log
        logging.info('Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logging.info('the average_return is {}'.format(average_return))
        print('Time: {}, Iteration: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), iteration))
        print(
            "Evaluation Phase: the average_return is {}".format(average_return)
        )

        return num_episodes, average_return

    def _save_tensorboard_summaries(self, iteration, validation_loss):
        """Save statistics as tensorboard summaries.

        Args:
          iteration: int, The current iteration number.
          num_episodes_eval: int, number of evaluation episodes run.
          average_reward_eval: float, The average evaluation reward.
        """
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Eval/Validation_loss',
                             simple_value=validation_loss),
        ])
        self._summary_writer.add_summary(summary, iteration)

    def _save_tensorboard_summaries_bc(self, iteration, loss_train, loss_test,
                                            num_episodes_eval,
                                            average_reward_eval):
        """Save statistics as tensorboard summaries.

        Args:
          iteration: int, The current iteration number.
          num_episodes_eval: int, number of evaluation episodes run.
          average_reward_eval: float, The average evaluation reward.
        """
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/train_loss',
                             simple_value=loss_train),
            tf.Summary.Value(tag='Train/test_loss',
                             simple_value=loss_test),
            tf.Summary.Value(tag='Val/NumEpisodes',
                             simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Val/AverageReturns',
                             simple_value=average_reward_eval),
        ])

        self._summary_writer.add_summary(summary, iteration)

    def get_ckpt_suffixes_bc_data(self, suffixes):
        if int(self.cfg['checkpointDir_suffix']) + 1 > len(suffixes):
            return []
        else:
            return [str(suffixes[int(self.cfg['checkpointDir_suffix'])])]


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