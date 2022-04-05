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

import time

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
from plot_utils.figtodata import fig2data

import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import imageio
import json
import gzip
gfile = tf.gfile


class FixedReplayRunner(run_experiment.Runner):
  """Object that handles running Dopamine experiments with fixed replay buffer."""

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    super(FixedReplayRunner, self)._initialize_checkpointer_and_maybe_resume(
        checkpoint_file_prefix)

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
          print(
              'Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S  ", time.localtime())) +
              'Steps percent of this phase in this iteration: {} % '.format(float(step_count) /
                                                                            float(self._training_steps) * 100.0)
          )
          verbose_step += self.verbose_steps
    time_delta = time.time() - start_time
    tf.logging.info('Average training steps per second: %.2f',
                    self._training_steps / (time_delta + 1e-6))

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction."""
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    print("Starting iteration %d", iteration)

    # pylint: disable=protected-access
    if not self._agent._replay_suffix:
      # Reload the replay buffer
      if self._agent._data_set_mode == 'ALL':
          num_buffers = 5
      else:
          num_buffers = 1
      tf.logging.info("The number of buffers: {}".format(num_buffers))
      self._agent._replay.memory.reload_buffer(num_buffers=num_buffers)
    # pylint: enable=protected-access
    self._run_train_phase()

    num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics, iteration)

    self._save_tensorboard_summaries(
        iteration, num_episodes_eval, average_reward_eval)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_eval,
                                  average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='Eval/AverageReturns',
                         simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

  def plot_entropy_distribution(self, num_iteration=10, num_samples_each_iteration=200):
    # create dir for saving entropy distribution
    dir = os.path.join(self.base_dir_this_env, 'entropy')
    gfile.MakeDirs(dir)

    current_palette = sns.color_palette('bright', 10)
    sns.set(color_codes=True, palette='muted', style='darkgrid', font_scale=0.4)

    probabilities = []

    suffixes = self._agent._replay.memory.my_chosen_ckpt_suffixes

    for iteration in range(0, num_iteration):
        if not self._agent._replay_suffix:
            # Reload the replay buffer
            if self._agent._data_set_mode == 'ALL':
                num_buffers = 5
            else:
                num_buffers = 1
            tf.logging.info("The number of buffers: {}".format(num_buffers))
            self._agent._replay.memory.reload_buffer(num_buffers=num_buffers)
        for step_count in range(num_samples_each_iteration):
            if step_count % 100 == 0:
                print('In iteration {}..., alrady {} %'.format(
                    iteration,
                    step_count / num_samples_each_iteration * 100.0)
                )
            sampled_actions_probability = self._agent._sess.run(
                tf.exp(self._agent._replay_net_outputs.action_prob)
            )
            probabilities.extend(list(sampled_actions_probability))

    probabilities = np.array(probabilities)

    entropy = -np.sum(np.log(probabilities) * probabilities, axis=1)
    our_defined_entropy = -np.sum(1/np.sqrt(probabilities), axis=1) * 1

    with gfile.Open(os.path.join(dir, 'entropy.json'), 'w') as f:
        json.dump({"entropy": float(np.mean(entropy)), "out_defined_entropy": float(np.mean(our_defined_entropy))}, f)

    self.plot(entropy, dir, 'Entropy')
    self.plot(our_defined_entropy, dir, 'Our_defined_entropy')

  def plot(self, entropy, dir, name):
    increasing_entropy = np.sort(entropy)
    index_increasing_entropy = np.argsort(entropy)
    entropy = entropy[index_increasing_entropy]

    current_palette = sns.color_palette('bright', 10)
    sns.set(color_codes=True, palette='muted', style='darkgrid', font_scale=3)

    fig, ax = plt.subplots(figsize=(20, 12))
    plot_s = list(np.arange(entropy.shape[0]))
    sns.lineplot(x=plot_s, y=list(entropy), color='palevioletred', label="Entropy")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:], loc='best')
    image = fig2data(fig)

    with gfile.GFile(os.path.join(dir, '{}.png'.format(name)), "w") as f:
        imageio.imsave(f, image, 'PNG')

    print('Plotted current UE in', os.path.join(dir, '{}.png'.format(name)))

