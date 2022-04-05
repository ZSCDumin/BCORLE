# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from absl import logging

from batch_rl.baselines.agents.dqn_agent import LoggedDQNAgent
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import logger

import numpy as np
import tensorflow as tf

gfile = tf.gfile


def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


def create_runner(base_dir, schedule='continuous_train_and_eval'):
  """Creates an experiment Runner.

  Args:
    base_dir, base directory for hosting all subdirectories.
    scheduleing, which type of Runner to use.

  Returns:
    runner: A `Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if schedule == 'continuous_train_and_eval':
    return Runner(base_dir, create_agent)
  # Continuously runs training until max num_iterations is hit.
  elif schedule == 'continuous_train':
    return TrainRunner(base_dir, create_agent)
  else:
    raise ValueError('Unknown schedule: {}'.format(schedule))


class Runner(object):
  """Object that handles running Dopamine experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.

  A simple scenario to train a DQN agent is as follows:

  ```python
  import dopamine.discrete_domains.atari_lib
  base_dir = '/tmp/simple_example'
  def create_agent(sess, environment):
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n)
  runner = Runner(base_dir, create_agent, atari_lib.create_atari_environment)
  runner.run()
  ```
  """

  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=1000,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=108000,
               cfg=None,
               mode=None):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
      checkpoint_file_prefix, the prefix to use for checkpoint files.
      logging_file_prefix, prefix to use for the log files.
      log_every_n, the frequency for writing logs.
      num_iterations, the iteration number threshold (must be greater than
        start_iteration).
      training_steps, the number of training steps to perform.
      evaluation_steps, the number of evaluation steps to perform.
      max_steps_per_episode, maximum number of steps after which an episode
        terminates.

    This constructor will take the following actions:
    - Initialize an environment.
    - Initialize a `tf.Session`.
    - Initialize a logger.
    - Initialize an agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None
    self.cfg = cfg
    self.mode = mode
    # For BAIL
    if self.cfg['AGENT'] == 'BAIL':
        num_iterations = 30
        training_steps = 250000
        evaluation_steps = 125000
        max_steps_per_episode = 108000
        self._num_iterations_bc = 1000

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir

    self.checkpoint_file_prefix = checkpoint_file_prefix

    self.verbose_steps = 10000
    self._create_directories()
    if self.cfg['AGENT'] in ['BAIL_BCQ_weighted']:
        self._summary_writer = tf.summary.FileWriter(os.path.join(self._base_dir,
                                                 self.cfg['ATARI_ENV'] + str(self.cfg['NUM_EXPERIMENT']),
                                                 self.cfg['AGENT'],
                                                 str(self.cfg['select_percentage']),
                                                 'tensorboard'))
    else:
        if self.mode == 'generator':
            self._summary_writer = tf.summary.FileWriter(os.path.join(self._base_dir,
                                                                      self.cfg['ATARI_ENV'] + str(
                                                                          self.cfg['NUM_EXPERIMENT']),
                                                                      'tensorboard'))
        else:
            self._summary_writer = tf.summary.FileWriter(os.path.join(self._base_dir,
                                                                      self.cfg['ATARI_ENV'] + str(
                                                                          self.cfg['NUM_EXPERIMENT']),
                                                                      self.cfg['AGENT'],
                                                                      'tensorboard'))

    print('Loading environment {}.'.format(cfg['ATARI_ENV']))
    self._environment = create_environment_fn(game_name=cfg['ATARI_ENV'])
    config = tf.ConfigProto(allow_soft_placement=True)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple agents/workers on the same GPU.
    config.gpu_options.allow_growth = True
    # Set up a session and initialize variables.
    self._sess = tf.Session('', config=config)
    print('Creating agent {}.'.format(cfg['AGENT']))
    self._agent = create_agent_fn(self._sess, self._environment,
                                  summary_writer=self._summary_writer)

    # todo: define training steps and evaluation steps in the main function
    if not type(self._agent) == LoggedDQNAgent and (not self._agent._data_set_mode == 'ALL'):
        self._training_steps = int(training_steps / 5)

    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

  def _create_directories(self):
    """Create necessary sub-directories."""
    try:
        if self.mode == 'generator':
            print("generator mode")
            base_dir_this_agent = os.path.join(self._base_dir,
                                               self.cfg['ATARI_ENV'] + str(self.cfg['NUM_EXPERIMENT'])
                                               )
            gfile.MakeDirs(base_dir_this_agent)
        else:
            print("off line mode")
            base_dir_this_agent = os.path.join(self._base_dir,
                                               self.cfg['ATARI_ENV'] + str(self.cfg['NUM_EXPERIMENT'])
                                               )
            gfile.MakeDirs(base_dir_this_agent)
            self.base_dir_this_env = base_dir_this_agent

            base_dir_this_agent = os.path.join(base_dir_this_agent,
                                               self.cfg['AGENT']
                                               )
            gfile.MakeDirs(base_dir_this_agent)
            if self.cfg['AGENT'] in ['BAIL_BCQ_weighted']:
                base_dir_this_agent = os.path.join(base_dir_this_agent,
                                                   str(self.cfg['select_percentage'])
                                                   )
                gfile.MakeDirs(base_dir_this_agent)

        self._checkpoint_dir = os.path.join(base_dir_this_agent,
                                            'checkpoints')
        self._logger = logger.Logger(os.path.join(base_dir_this_agent,
                                                  'logs'))
        if self.cfg['AGENT'] == 'BAIL':
            self._bail_mc_gain_dir = os.path.join(base_dir_this_agent, 'bail_mc_gain')
    except tf.errors.PermissionDeniedError:
        # If it already exists, ignore exception.
        pass

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
        logging.info('Reloaded checkpoint and will start from iteration %d',
                     self._start_iteration)

  def _initialize_episode(self):
    """Initialization for a new episode.

    Returns:
      action, the initial action chosen by the agent.
    """
    initial_observation = self._environment.reset()
    return self._agent.begin_episode(initial_observation)

  def _run_one_step(self, action):
    """Executes a single step in the environment.

    Args:
      action, the action to perform in the environment.

    Returns:
      The observation, reward, and is_terminal values returned from the
        environment.
    """
    observation, reward, is_terminal, _ = self._environment.step(action)
    return observation, reward, is_terminal

  def _end_episode(self, reward):
    """Finalizes an episode run.

    Args:
      reward: float, the last reward from the environment.
    """
    self._agent.end_episode(reward)

  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal = self._run_one_step(action)

      total_reward += reward
      step_number += 1

      # Perform reward clipping.
      reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or
          step_number == self._max_steps_per_episode):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._agent.end_episode(reward)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)

    self._end_episode(reward)

    return step_number, total_reward

  def _run_one_phase(self, min_steps, statistics, run_mode_str):
    """Runs the agent/environment loop until a desired number of steps.

    We follow the Machado et al., 2017 convention of running full episodes,
    and terminating once we've run a minimum number of steps.

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
    num_episodes = 0
    sum_returns = 0.
    verbose_step = self.verbose_steps

    while step_count < min_steps:
      episode_length, episode_return = self._run_one_episode()
      statistics.append({
          '{}_episode_lengths'.format(run_mode_str): episode_length,
          '{}_episode_returns'.format(run_mode_str): episode_return
      })
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1

      if step_count > verbose_step:
          tf.logging.info(
              'Time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S  ", time.localtime())) +
              'Steps percent of this phase in this iteration: {} % '.format(float(step_count) /
                                                                            float(min_steps) * 100.0) +
              'Steps executed: {} '.format(step_count) +
              'Episode length: {} '.format(episode_length) +
              'Return: {}\r'.format(sum_returns / num_episodes)
          )
          verbose_step += self.verbose_steps

      # # We use sys.stdout.write instead of logging so as to flush frequently
      # # without generating a line break.
      # sys.stdout.write('Steps executed: {} '.format(step_count) +
      #                  'Episode length: {} '.format(episode_length) +
      #                  'Return: {}\r'.format(episode_return))
      # sys.stdout.flush()
    return step_count, sum_returns, num_episodes

  def _run_train_phase(self, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.

    Returns:
      num_episodes, The number of episodes run in this phase.
      average_reward: float, The average reward generated in this phase.
      average_steps_per_second: float, The average number of steps per second.
    """
    # Perform the training phase, during which the agent learns.
    self._agent.eval_mode = False
    start_time = time.time()
    number_steps, sum_returns, num_episodes = self._run_one_phase(
        self._training_steps, statistics, 'train')
    average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    statistics.append({'train_average_return': average_return})
    time_delta = time.time() - start_time
    average_steps_per_second = number_steps / time_delta
    statistics.append(
        {'train_average_steps_per_second': average_steps_per_second})
    logging.info('Average undiscounted return per training episode: %.2f',
                 average_return)
    logging.info('Average training steps per second: %.2f',
                 average_steps_per_second)
    return num_episodes, average_return, average_steps_per_second

  def _run_eval_phase(self, statistics, iteration):
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

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. The interleaving of train/eval phases implemented here
    are to match the implementation of (Mnih et al., 2015).

    Args:
      iteration, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))
    num_episodes_eval, average_reward_eval = self._run_eval_phase(
        statistics, iteration)

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train, num_episodes_eval,
                                     average_reward_eval,
                                     average_steps_per_second)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_train,
                                  average_reward_train,
                                  num_episodes_eval,
                                  average_reward_eval,
                                  average_steps_per_second):
    """Save statistics as tensorboard summaries.

    Args:
      iteration, The current iteration number.
      num_episodes_train, number of training episodes run.
      average_reward_train: float, The average training reward.
      num_episodes_eval, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
      average_steps_per_second: float, The average number of steps per second.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(
            tag='Train/NumEpisodes', simple_value=num_episodes_train),
        tf.Summary.Value(
            tag='Train/AverageReturns', simple_value=average_reward_train),
        tf.Summary.Value(
            tag='Train/AverageStepsPerSecond',
            simple_value=average_steps_per_second),
        tf.Summary.Value(
            tag='Eval/NumEpisodes', simple_value=num_episodes_eval),
        tf.Summary.Value(
            tag='Eval/AverageReturns', simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.

    Args:
      iteration, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)

  def _checkpoint_experiment(self, iteration, dir=None):
    """Checkpoint experiment data.

    Args:
      iteration, iteration number for checkpointing.
    """
    checkpointer_dir = dir if dir else self._checkpoint_dir
    experiment_data = self._agent.bundle_and_checkpoint(checkpointer_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)


class TrainRunner(Runner):
  """Object that handles running experiments.

  The `TrainRunner` differs from the base `Runner` class in that it does not
  the evaluation phase. Checkpointing and logging for the train phase are
  preserved as before.
  """

  def __init__(self, base_dir, create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment):
    """Initialize the TrainRunner object in charge of running a full experiment.

    Args:
      base_dir, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      create_environment_fn: A function which receives a problem name and
        creates a Gym environment for that problem (e.g. an Atari 2600 game).
    """
    logging.info('Creating TrainRunner ...')
    super(TrainRunner, self).__init__(base_dir, create_agent_fn,
                                      create_environment_fn)
    self._agent.eval_mode = False

  def _run_one_iteration(self, iteration):
    """Runs one iteration of agent/environment interaction.

    An iteration involves running several episodes until a certain number of
    steps are obtained. This method differs from the `_run_one_iteration` method
    in the base `Runner` class in that it only runs the train phase.

    Args:
      iteration, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    num_episodes_train, average_reward_train, average_steps_per_second = (
        self._run_train_phase(statistics))

    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train,
                                     average_steps_per_second)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, num_episodes,
                                  average_reward, average_steps_per_second):
    """Save statistics as tensorboard summaries."""
    summary = tf.Summary(value=[
        tf.Summary.Value(
            tag='Train/NumEpisodes', simple_value=num_episodes),
        tf.Summary.Value(
            tag='Train/AverageReturns', simple_value=average_reward),
        tf.Summary.Value(
            tag='Train/AverageStepsPerSecond',
            simple_value=average_steps_per_second),
    ])
    self._summary_writer.add_summary(summary, iteration)
