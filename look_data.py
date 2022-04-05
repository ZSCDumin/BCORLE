# coding: utf-8

import tensorflow as tf
import utils.replay_buffer_utils as utils
import discrete_BCQ
import os
from utils.tf_utils import mkdir, fig2data
import matplotlib
matplotlib.use('Agg')
import numpy as np
# from utils.log_utils import *
import datetime
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tensorflow.python.platform import gfile
import gzip

from batch_RL_baseline import REM
from batch_RL_baseline import rem_bcq

from pre_train2 import PropensityNet
from Eval_trajectory_generator import eval_data_generator
from Real_evaluation import real_evaluation
from evaluation import evaluation


flags = tf.app.flags

# distributed reader setting
flags.DEFINE_string("job_name", '0', "job name: worker or ps")
flags.DEFINE_integer("task_index", 0, "Worker or server index")
flags.DEFINE_string("worker_hosts", '', "worker_hosts")

# environment
flags.DEFINE_string("env", "Bonus Allocation", "name of the env")
flags.DEFINE_integer("seed", 3, "numpy and tf sees")
flags.DEFINE_integer("max_trainningsteps", 500, "max time steps for training")
flags.DEFINE_integer("max_timesteps", 30, "max time steps for one user")
flags.DEFINE_integer("Action_dim", 10, "action dimension of env")
flags.DEFINE_integer("State_dim", 549, "state dimension of env")
flags.DEFINE_integer("Action_dim_test", 21, "action dimension of env")
flags.DEFINE_integer("State_dim_test", 44, "state dimension of env")
flags.DEFINE_integer("Number_real_evaluation_users", 10000, "Number of users in evaluation")

# models
flags.DEFINE_string("model_name", "Default", "buffer file")
flags.DEFINE_float("BCQ_threshold", 0.3, "threshold in BCQ")
flags.DEFINE_float("Min_Lambda", 0.0, "The Miximum of lambda")
flags.DEFINE_float("Max_Lambda", 1.0, "The Maximum of lambda")
flags.DEFINE_float("Interval_Lambda", 0.05, "The Interval of lambda")
flags.DEFINE_bool("Is_prioritized_replay_buffer", True, "whether use prioritized replay buffer")
# weights for loss
flags.DEFINE_float("q_loss_weight", 1.0, "q_loss_weight")
flags.DEFINE_float("i_regularization_weight", 0.03, "i_regularization_weight")
flags.DEFINE_float("i_loss_weight", 0.03, "i_loss_weight")
# batch_size
flags.DEFINE_integer("batch_size", 1000, "batch_size")
# pre_train
flags.DEFINE_bool("pretrain", True, "whether implement pre_train")
flags.DEFINE_integer("pre_train_epochs", 200, "pre_train_epochs")
flags.DEFINE_integer("pre_train_size", 10000, "pre_train_batch_size")
flags.DEFINE_integer("pre_train_minibatch_size", 1000, "pre_train_minibatch_size")
# learning
flags.DEFINE_float("discount", 1.00, "discount")
flags.DEFINE_integer("buffer_size", 1000, "maximum replay buffer size")
flags.DEFINE_float("optimizer_parameters_lr", 3e-4, "optimizer_parameters_lr")
flags.DEFINE_integer("train_freq", 50, "train_freq")
flags.DEFINE_bool("polyak_target_update", True, "polyak_target_update")
flags.DEFINE_integer("target_update_freq", 1, "target_update_freq")
flags.DEFINE_float("tau", 0.005, "tau")
flags.DEFINE_string('checkpointDir', '',  "Directory from which to save the replay data")

flags.DEFINE_string('buckets', '',  'bucketDir, store results')


#batch_RL
#REM,discrete_BCQ,REMBCQ
flags.DEFINE_string("Batch_RL_learner", "REMBCQ", "Batch_RL_learner")
#IPSEstimator,DirectMethodEstimator,DoublyRobustEstimator,FQEstimator,REMEstimator
# Off-policy evaluation
flags.DEFINE_string("Estimator", "IPSEstimator,DirectMethodEstimator,DoublyRobustEstimator,FQEstimator,REMEstimator", "Evaluation Estimator")
flags.DEFINE_integer("eval_freq", 100, "eval_freq")
flags.DEFINE_integer("fqe_train_steps", 500, "fqe_train_steps")
flags.DEFINE_integer("rem_train_steps", 500, "rem_train_steps")

# input ODPS tables and checkpoint directory
#tf.python_io.TableReader
#第一个表中存放训练数据，第二个表存放真实评估data，第三个表存放评估轨迹
flags.DEFINE_string(
    "tables",
    ",,",
    "tables info"
)
flags.DEFINE_string(
    "selected_cols",
    "user_id,ds,step_in_episode,state,action,reward1,reward2,next_state,terminal",
    "selected_cols"
)
flags.DEFINE_string(
    "selected_cols_simulation",
    "user_id,state,action,reward_liucun,next_state,terminal,liucun_rate",
    "selected_cols"
)

#flags.DEFINE_string("checkpointDir", 'oss://...', "checkpointDir")


# mode setting
flags.DEFINE_bool("test_mode", False, "whether test locally. (We use this mode for debug)")
flags.DEFINE_bool("simulation_mode", True, "whether simulation. (We use this mode for simulation)")


FLAGS = tf.app.flags.FLAGS


# train function
def train(learner,eval_data_dir,replay_buffer, replay_buffer_pretrain, replay_buffer_evaluation,
        estimator, state_dim, num_actions, action_dim, Max_timesteps, Lambda_dim, Lambda_min, Lambda_max,
          Lambda_interval, Number_real_evaluation_users, save_dir, all_user_come, all_user_hongbao, all_user_liucun, all_hongbao_pre30, all_liucun_pre30, all_average_liucun, \
    all_user_type
):
    Lambda_size = int((Lambda_max - Lambda_min) / Lambda_interval + 1)
    with tf.Session() as sess:

        propensity = PropensityNet(sess, state_dim, num_actions, Lambda_dim, FLAGS.optimizer_parameters_lr,
                                   "pre_train_network")

        # pre-train the replay_buffer_evaluation model, for predicting the logged propensities
        if FLAGS.pretrain:
            propensity.pre_train(replay_buffer_pretrain,
                replay_buffer_evaluation,
                FLAGS.pre_train_epochs,
                FLAGS.pre_train_minibatch_size,
                FLAGS.pre_train_size,
                FLAGS.test_mode,
                FLAGS.tables.split(",")[0])

        # reset replay_buffer_evaluation table reader
        if not FLAGS.test_mode:
            replay_buffer_evaluation.reset_table_reader(FLAGS.tables.split(",")[2])

        if learner == 'REM':
            policy = REM.REMAgent(sess, num_actions, state_dim, Lambda_dim,
            Number_real_evaluation_users, Lambda_size, Lambda_interval)
        if learner == 'discrete_BCQ':
            # Initialize and load policy
            policy = discrete_BCQ.DiscreteBCQ(sess, num_actions, action_dim, Lambda_dim, state_dim, estimator, Max_timesteps,
                                              Lambda_min, Lambda_max, Lambda_interval, Number_real_evaluation_users, FLAGS.fqe_train_steps,
                                              FLAGS.rem_train_steps, FLAGS.BCQ_threshold,
                                              FLAGS.discount,
                                              FLAGS.optimizer_parameters_lr, FLAGS.polyak_target_update,
                                              FLAGS.target_update_freq, FLAGS.tau,
                                              q_loss_weight=FLAGS.q_loss_weight,
                                              i_regularization_weight=FLAGS.i_regularization_weight,
                                              i_loss_weight=FLAGS.i_loss_weight
                                              )
        if learner == 'REMBCQ':
            policy = rem_bcq.REMBCQAgent(sess, num_actions, state_dim, Lambda_dim,
            Number_real_evaluation_users, Lambda_size, Lambda_interval)

        if not FLAGS.test_mode:
            setting = "{}_seed_{}_w1_{}_w2_{}_w3_{}".format(
                FLAGS.env, FLAGS.seed,
                FLAGS.q_loss_weight, FLAGS.i_regularization_weight,
                FLAGS.i_loss_weight
            )
            model_name = "{}_{}".format(FLAGS.model_name, setting)
            save_dir = FLAGS.checkpointDir + model_name + '/'
            save_dir = os.path.join(save_dir, learner)
            mkdir(save_dir)
        else:
            save_dir = os.path.join('./ckpt', learner)

        eval_data_come = np.zeros([FLAGS.max_trainningsteps, Lambda_size])
        eval_data_hongbao = np.zeros([FLAGS.max_trainningsteps, Lambda_size])

        # Start train
        training_iters = 0
        while training_iters < FLAGS.max_trainningsteps:

            for _ in range(int(FLAGS.eval_freq)):
                if FLAGS.test_mode:
                    replay_buffer.get_batch_data_from_local_npz()
                else:
                    replay_buffer.get_batch_data_from_odps()
                policy.append_batch_data(replay_buffer)
                policy.train(replay_buffer)

            # Save Model
            if not FLAGS.test_mode:

                # save ckpt path
                ckpt_path = os.path.join(
                    save_dir,
                    FLAGS.env + '_date_' + str(datetime.date.today()) + '_step_' + str(training_iters) + '.ckpt'
                )
                save_path = policy.saver.save(sess, ckpt_path)
                print("Model saved in file: %s" % save_path)
            else:
                ckpt_path = os.path.join(
                    save_dir,
                    FLAGS.env + '_date_' + str(datetime.date.today())  + '_step_' + str(training_iters) + '.ckpt'
                )
                print("Model saved in file: %s" % ckpt_path)
                policy.saver.save(sess, ckpt_path)


            if training_iters == FLAGS.max_trainningsteps - 1:
                Plot = True
            else:
                Plot = False

            #real_evaluation
            true_value, total_come, total_hongbao = real_evaluation(FLAGS.Number_real_evaluation_users,  Lambda_min, Lambda_max, Lambda_interval, Lambda_size,
                            all_user_come, all_user_hongbao, all_user_liucun, all_hongbao_pre30, all_liucun_pre30, all_average_liucun,
            all_user_type, training_iters, Plot, FLAGS.discount, ckpt_path, save_dir)

            eval_data_come[training_iters,:] = total_come
            eval_data_hongbao[training_iters, :] = total_hongbao

            # Evaluation
            predict_value_IPS, predict_value_DM, predict_value_DR, predict_value_FQE, predict_value_REM = \
                evaluation(eval_data_dir, replay_buffer_evaluation, training_iters, FLAGS.test_mode, ckpt_path, save_dir, estimator, num_actions,
                           FLAGS.Number_real_evaluation_users, Lambda_interval, Lambda_size,FLAGS.discount, Max_timesteps,
                           action_dim, Lambda_dim, state_dim, FLAGS.fqe_train_steps, FLAGS.rem_train_steps)
            # predict_value_IPS, predict_value_DM, predict_value_DR, predict_value_FQE, predict_value_REM =
            #     policy.evaluation(replay_buffer_evaluation, training_iters, FLAGS.test_mode, ckpt_path, save_dir)

            error_IPS = np.abs(predict_value_IPS - true_value)
            error_DM = np.abs(predict_value_DM - true_value)
            error_DR = np.abs(predict_value_DR - true_value)
            error_FQE = np.abs(predict_value_FQE - true_value)
            error_REM = np.abs(predict_value_REM - true_value)

            print("IPS OPE errors: {}".format(np.sum(error_IPS)))
            print("DM OPE errors: {}".format(np.sum(error_DM)))
            print("DR OPE errors: {}".format(np.sum(error_DR)))
            print("FQE OPE errors: {}".format(np.sum(error_FQE)))
            print("REM OPE errors: {}".format(np.sum(error_REM)))
            print("Training iterations: {}".format(training_iters))

            training_iters += FLAGS.eval_freq

    return eval_data_come, eval_data_hongbao



def main(unused_argv):
    print("tables:" + FLAGS.tables)
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    print("---------------------------------------")
    print("Setting: Training BCQ, Env: {}, Seed: {}".format(FLAGS.env, FLAGS.seed))
    print("---------------------------------------")
    print("Start Training")
    print("---------------------------------------")

    # Define state and action dimension


    action_dim = 1
    # Set worker
    worker_count = None
    if not FLAGS.test_mode:
        worker_spec = FLAGS.worker_hosts.split(",")
        worker_count = len(worker_spec)
        print("worker_count:{}".format(worker_count))



    table_reader = tf.python_io.TableReader(
        table=FLAGS.tables.split(",")[1],
        selected_cols=FLAGS.selected_cols_simulation,
        slice_id=0,
        slice_count=1
    )

    row_count = table_reader.get_row_count()
    tuple_data = table_reader.read(row_count)
    all_user_come_str, all_user_hongbao_str, all_user_liucun_str, all_hongbao_pre30_str, \
    all_liucun_pre30_str, all_average_liucun_str, all_user_type_str = zip(*tuple_data)

    print(all_user_come_str)
    print(all_user_hongbao_str)
    print(all_user_liucun_str)
    print(all_hongbao_pre30_str)
    print(all_liucun_pre30_str)
    print(all_average_liucun_str)
    print(all_user_type_str)

if __name__ == '__main__':
    tf.app.run()


