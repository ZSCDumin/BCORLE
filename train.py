# coding: utf-8
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import utils.replay_buffer_utils as utils
import discrete_BCQ
import os
from utils.tf_utils import mkdir, fig2data
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import imageio
# from utils.log_utils import *
import datetime
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tensorflow.python.platform import gfile
import gzip
import copy
import math

from batch_RL_baseline import REM
from batch_RL_baseline import rem_bcq
from batch_RL_baseline import mopo
from offlinerl.algo import algo_select
#from offlinerl.data.d4rl import load_d4rl_buffer
#from offlinerl.evaluation import OnlineCallBackFunction

from pre_train2 import PropensityNet
from Eval_trajectory_generator import eval_data_generator
from Real_evaluation2 import real_evaluation
from evaluation import evaluation
from time import time


flags = tf.app.flags

# distributed reader setting
flags.DEFINE_string("job_name", '0', "job name: worker or ps")
flags.DEFINE_integer("task_index", 0, "Worker or server index")
flags.DEFINE_string("worker_hosts", '', "worker_hosts")

# environment
flags.DEFINE_string("env", "Bonus Allocation", "name of the env")
flags.DEFINE_integer("seed", 5, "numpy and tf seeds")
flags.DEFINE_integer("max_trainningsteps", 10000, "max time steps for training")
flags.DEFINE_integer("max_timesteps", 30, "max time steps for one user")
flags.DEFINE_integer("Action_dim", 10, "action dimension of env")
flags.DEFINE_integer("State_dim", 549, "state dimension of env")
flags.DEFINE_integer("Action_dim_test", 21, "action dimension of env")
flags.DEFINE_integer("State_dim_test", 44, "state dimension of env")
flags.DEFINE_integer("Number_real_evaluation_users", 100, "Number of users in evaluation")

# models
flags.DEFINE_string("model_name", "Default", "buffer file")
flags.DEFINE_float("BCQ_threshold", 0.3, "threshold in BCQ")
flags.DEFINE_integer("num_heads", 10, "Number of heads in REM")
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
flags.DEFINE_bool("pretrain", False, "whether implement pre_train")
flags.DEFINE_integer("pre_train_epochs", 2000, "pre_train_epochs")
flags.DEFINE_integer("pre_train_size", 10000, "pre_train_batch_size")
flags.DEFINE_integer("pre_train_minibatch_size", 1000, "pre_train_minibatch_size")
flags.DEFINE_bool("real_evaluation", True, "whether implement real_evaluation")
flags.DEFINE_bool("evaluation", False, "whether implement evaluation")
# learning
flags.DEFINE_float("discount", 0.99, "discount")
flags.DEFINE_integer("buffer_size", 1000, "maximum replay buffer size")
flags.DEFINE_float("optimizer_parameters_lr", 3e-4, "optimizer_parameters_lr")
flags.DEFINE_integer("train_freq", 50, "train_freq")
flags.DEFINE_integer("choose_evaluation_lambda", 0, "eval_lambda")
flags.DEFINE_bool("polyak_target_update", True, "polyak_target_update")
flags.DEFINE_integer("target_update_freq", 1, "target_update_freq")
flags.DEFINE_float("tau", 0.005, "tau")
flags.DEFINE_string('checkpointDir', '',  "Directory from which to save the replay data")

flags.DEFINE_string('buckets', '',  'bucketDir, store results')

#batch_RL
#REM,discrete_BCQ,REMBCQ,MOPO
flags.DEFINE_string("Batch_RL_learner", "REMBCQ", "Batch_RL_learner")
#IPSEstimator,DirectMethodEstimator,DoublyRobustEstimator,FQEstimator,REMEstimator
# Off-policy evaluation
flags.DEFINE_string("Estimator", "IPSEstimator,DirectMethodEstimator,DoublyRobustEstimator,FQEstimator,REMEstimator", "Evaluation Estimator")
flags.DEFINE_integer("eval_freq", 100, "eval_freq")
flags.DEFINE_integer("fqe_train_steps", 4000, "fqe_train_steps")
flags.DEFINE_integer("rem_train_steps", 4000, "rem_train_steps")

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
flags.DEFINE_bool("test_mode", True, "whether test locally. (We use this mode for debug)")
flags.DEFINE_bool("simulation_mode", True, "whether simulation. (We use this mode for simulation)")


FLAGS = tf.app.flags.FLAGS


# train function
def train(learner,eval_data_dir,replay_buffer, replay_buffer_pretrain, replay_buffer_evaluation,
        estimator, state_dim, num_actions, action_dim, Max_timesteps, Lambda_dim, Lambda_min, Lambda_max,
          Lambda_interval, Number_real_evaluation_users, save_dir, all_user_come, all_user_hongbao, all_user_liucun,
          all_hongbao_pre30, all_liucun_pre30, all_average_liucun,all_user_type, result_dir
):
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['AR PL KaitiM GB']})
    Lambda_size = int((Lambda_max - Lambda_min) / Lambda_interval + 1)

    with tf.Session() as sess:

        start = time()
        print("Start: " + str(start))
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
            Number_real_evaluation_users, Lambda_size, Lambda_interval,FLAGS.optimizer_parameters_lr, num_heads= FLAGS.num_heads)
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
            Number_real_evaluation_users, Lambda_size, Lambda_interval,  num_heads = FLAGS.num_heads)


        if learner == 'MOPO':
            dict = {"algo_name" : mopo}
            algo_init_fn, algo_trainer_obj, algo_config = algo_select(dict)
            #train_buffer = replay_buffer
            algo_init = algo_init_fn(algo_config)
            policy = algo_trainer_obj(algo_init, algo_config)
            callback = None
            #callback.initialize(train_buffer=train_buffer, val_buffer=None, task=algo_config["task"])


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

        eval_data_come = []
        eval_data_hongbao = []
        eval_data_convertion = []

        # Start train
        training_iters = 0

        # if not FLAGS.test_mode:
        #     replay_buffer.get_batch_data_from_odps(is_evaluation=False)
        #
        #     shuffled_indices = np.random.permutation(len(replay_buffer.new_batch_data2["state"]))
        #     eval_train_state = np.array(replay_buffer.new_batch_data2["state"])[shuffled_indices]
        #     eval_train_action = np.array(replay_buffer.new_batch_data2["action"])[shuffled_indices]
        #     eval_train_reward1 = np.array(replay_buffer.new_batch_data2["reward1"])[shuffled_indices]
        #     eval_train_reward2 = np.array(replay_buffer.new_batch_data2["reward2"])[shuffled_indices]
        #     eval_train_Lambda = np.array(replay_buffer.new_batch_data2["Lambda"])[shuffled_indices]
        #     eval_train_next_state = np.array(replay_buffer.new_batch_data2["next_state"])[shuffled_indices]
        #     eval_train_done = np.array(replay_buffer.new_batch_data2["done"])[shuffled_indices]
        #
        # else:
        #     shuffled_indices = np.random.permutation(len(replay_buffer.state))
        #     eval_train_state = np.array(replay_buffer.state)[shuffled_indices]
        #     eval_train_action = np.array(replay_buffer.action)[shuffled_indices]
        #     eval_train_reward1 = np.array(replay_buffer.reward1)[shuffled_indices]
        #     eval_train_reward2 = np.array(replay_buffer.reward2)[shuffled_indices]
        #     eval_train_Lambda = np.array(replay_buffer.Lambda)[shuffled_indices]
        #     eval_train_next_state = np.array(replay_buffer.next_state)[shuffled_indices]
        #     eval_train_done = np.array(replay_buffer.done)[shuffled_indices]

        if FLAGS.evaluation:
            is_evaluation = True
            # get a batch
            if FLAGS.test_mode:
                replay_buffer_evaluation.get_batch_data_from_local_eval_npz()
            else:
                # replay_buffer.get_batch_data_from_d2_eval_npz(eval_data_dir)
                replay_buffer_evaluation.get_batch_data_from_eval_odps(is_evaluation)



        total_true_value = []
        total_true_cost_value = []
        total_REM_value = []
        total_REM_cost_value = []

        total_error_IPS = []
        total_error_DM = []
        total_error_DR = []
        total_error_FQE = []
        total_error_REM = []
        if FLAGS.test_mode:
            replay_buffer.get_batch_data_from_local_npz(learner)
        if learner == 'MOPO':
            transition2 = policy.pre_train(replay_buffer)
        while training_iters < FLAGS.max_trainningsteps:

            for _ in range(int(FLAGS.eval_freq)):
                # if FLAGS.test_mode:
                #     replay_buffer.get_batch_local_data()
                # else:
                #     replay_buffer.get_train_data(learner)
                #policy.append_batch_data(replay_buffer)
                if learner == 'MOPO':
                    replay_buffer.get_batch_local_data()
                    policy.train(replay_buffer,transition2)
                else:
                    policy.train(replay_buffer)

            if learner == 'MOPO':

                all_user_come2 = copy.deepcopy(all_user_come)
                all_user_hongbao2 = copy.deepcopy(all_user_hongbao)
                all_user_liucun2 = copy.deepcopy(all_user_liucun)
                all_hongbao_pre302 = copy.deepcopy(all_hongbao_pre30)
                all_liucun_pre302 = copy.deepcopy(all_liucun_pre30)
                all_average_liucun2 = copy.deepcopy(all_average_liucun)
                all_user_type2 = copy.deepcopy(all_user_type)

                true_value, true_value_cost, total_come, total_hongbao, total_convertion = policy.real_evaluation(sess, FLAGS.Number_real_evaluation_users,  Lambda_min, Lambda_max, Lambda_interval,
                                    Lambda_size, all_user_come2, all_user_hongbao2, all_user_liucun2, all_hongbao_pre302,
                                    all_liucun_pre302, all_average_liucun2,
                all_user_type, training_iters + FLAGS.eval_freq, False, FLAGS.discount, result_dir)

                eval_data_come.append(total_come)
                eval_data_hongbao.append(total_hongbao)
                eval_data_convertion.append(total_convertion)
                total_true_value.append(np.sum(true_value, axis=1))
                total_true_cost_value.append(np.sum(true_value_cost, axis=1))

                training_iters += FLAGS.eval_freq

                continue

            # Save Model
            if not FLAGS.test_mode:

                # save ckpt path
                ckpt_path = os.path.join(
                    save_dir,
                    FLAGS.env + '_date_' + str(datetime.date.today()) + '_step_' + str(training_iters + FLAGS.eval_freq) + '.ckpt'
                )
                if training_iters == 0:
                    save_path = policy.saver.save(sess, ckpt_path)
                    ckpt_path0 = ckpt_path
                else:
                    save_path = policy.saver.save(sess, ckpt_path, write_meta_graph=False)
                print("Model saved in file: %s" % save_path)
            else:
                ckpt_path = os.path.join(
                    save_dir,
                    FLAGS.env + '_date_' + str(datetime.date.today())  + '_step_' + str(training_iters + FLAGS.eval_freq) + '.ckpt'
                )
                print("Model saved in file: %s" % ckpt_path)
                if training_iters == 0:
                    if not FLAGS.test_mode:
                        policy.saver.save(sess, ckpt_path, max_to_keep=None)
                        ckpt_path0 = ckpt_path
                    else:
                        policy.saver.save(sess, ckpt_path)
                        ckpt_path0 = ckpt_path
                else:
                    if not FLAGS.test_mode:
                        policy.saver.save(sess, ckpt_path, write_meta_graph=False, max_to_keep=None)
                    else:
                        policy.saver.save(sess, ckpt_path, write_meta_graph=False)


            if training_iters == FLAGS.max_trainningsteps - FLAGS.eval_freq:
                Plot = True
            else:
                Plot = False

            if FLAGS.real_evaluation:

                all_user_come2 = copy.deepcopy(all_user_come)
                all_user_hongbao2 = copy.deepcopy(all_user_hongbao)
                all_user_liucun2 = copy.deepcopy(all_user_liucun)
                all_hongbao_pre302 = copy.deepcopy(all_hongbao_pre30)
                all_liucun_pre302 = copy.deepcopy(all_liucun_pre30)
                all_average_liucun2 = copy.deepcopy(all_average_liucun)
                all_user_type2 = copy.deepcopy(all_user_type)

                #real_evaluation
                true_value, true_value_cost, total_come, total_hongbao, total_convertion = \
                    real_evaluation(sess, FLAGS.Number_real_evaluation_users,  Lambda_min, Lambda_max, Lambda_interval,
                                    Lambda_size, all_user_come2, all_user_hongbao2, all_user_liucun2, all_hongbao_pre302,
                                    all_liucun_pre302, all_average_liucun2,
                all_user_type, training_iters + FLAGS.eval_freq, False, FLAGS.discount, ckpt_path0, save_dir, result_dir)

                stop = time()
                print("Stop: " + str(stop))
                print(str(stop - start) + "秒")

                eval_data_come.append(total_come)
                eval_data_hongbao.append(total_hongbao)
                eval_data_convertion.append(total_convertion)
                total_true_value.append(np.sum(true_value, axis=1))
                total_true_cost_value.append(np.sum(true_value_cost, axis=1))
                print('Real_evaluation finished')

            if FLAGS.evaluation:
                # Evaluation
                predict_value_IPS, predict_value_DM, predict_value_DR, predict_value_FQE, predict_value_REM,\
                predict_cost_value_REM = \
                    evaluation(eval_data_dir, replay_buffer, replay_buffer_evaluation, training_iters,
                               FLAGS.test_mode,ckpt_path0, save_dir, estimator, num_actions,
                               FLAGS.Number_real_evaluation_users, FLAGS.discount, Max_timesteps,
                               action_dim, Lambda_dim, state_dim, FLAGS.fqe_train_steps, FLAGS.rem_train_steps,
                               Lambda_min, Lambda_max, Lambda_interval, Lambda_size, result_dir,
                               eval_train_state, eval_train_action,eval_train_reward1,eval_train_reward2,eval_train_Lambda,
                                eval_train_next_state,eval_train_done
                )
                # predict_value_IPS, predict_value_DM, predict_value_DR, predict_value_FQE, predict_value_REM =
                #     policy.evaluation(replay_buffer_evaluation, training_iters, FLAGS.test_mode, ckpt_path, save_dir)

                total_REM_value.append(np.sum(predict_value_REM,axis = 1))
                total_REM_cost_value.append(np.sum(predict_cost_value_REM, axis=1))

                if FLAGS.real_evaluation:
                    error_IPS = np.abs(predict_value_IPS - true_value_cost)
                    error_DM = np.abs(predict_value_DM - true_value_cost)
                    error_DR = np.abs(predict_value_DR - true_value_cost)
                    error_FQE = np.abs(predict_value_FQE - true_value_cost)
                    error_REM = np.abs(predict_cost_value_REM - true_value_cost)

                    total_error_IPS.append(error_IPS)
                    total_error_DM.append(error_DM)
                    total_error_DR.append(error_DR)
                    total_error_FQE.append(error_FQE)
                    total_error_REM.append(error_REM)

                    print("IPS OPE errors: {}".format(np.sum(error_IPS)))
                    print("DM OPE errors: {}".format(np.sum(error_DM)))
                    print("DR OPE errors: {}".format(np.sum(error_DR)))
                    print("FQE OPE errors: {}".format(np.sum(error_FQE)))
                    print("REM OPE errors: {}".format(np.sum(error_REM)))
                    print("Training iterations: {}".format(training_iters))

            training_iters += FLAGS.eval_freq

    return eval_data_come, eval_data_hongbao, eval_data_convertion, total_true_value, total_true_cost_value,\
           total_REM_value, total_REM_cost_value,total_error_IPS,total_error_DM,total_error_DR,total_error_FQE,total_error_REM



def main(unused_argv):
    print("tables:" + FLAGS.tables)
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    print("---------------------------------------")
    print("Setting: Training: {}, Env: {}, Seed: {}".format(FLAGS.Batch_RL_learner, FLAGS.env, FLAGS.seed))
    print("---------------------------------------")
    print("Start Training")
    print("---------------------------------------")

    # Define state and action dimension

    if FLAGS.simulation_mode:
        state_dim = int(FLAGS.State_dim_test)
        num_actions = int(FLAGS.Action_dim_test)
    else:
        state_dim = int(FLAGS.State_dim)
        num_actions = int(FLAGS.Action_dim)

    action_dim = 1
    # Set worker
    worker_count = None
    if not FLAGS.test_mode:
        worker_spec = FLAGS.worker_hosts.split(",")
        worker_count = len(worker_spec)
        print("worker_count:{}".format(worker_count))

    # Get the Lambda
    Lambda_dim = 1
    Lambda_min = FLAGS.Min_Lambda
    Lambda_max = FLAGS.Max_Lambda
    Lambda_interval = FLAGS.Interval_Lambda
    Number_real_evaluation_users = FLAGS.Number_real_evaluation_users
    Max_timesteps = FLAGS.max_timesteps

    # Set seed
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Whether use prioritized replay buffer
    is_prioritized_replay_buffer = True if FLAGS.Is_prioritized_replay_buffer == 'True' else False

    # Buffer for training BCQ
    replay_buffer = utils.StandardBuffer(
        state_dim, num_actions, FLAGS.batch_size,
        FLAGS.buffer_size, FLAGS.tables.split(",")[0], FLAGS.selected_cols_simulation, FLAGS.discount,
        Lambda_min=Lambda_min, Lambda_max=Lambda_max, Lambda_interval=Lambda_interval, test_mode=FLAGS.test_mode,
        is_prioritized_replay_buffer=is_prioritized_replay_buffer
    )
    # Buffer for pre-train the reward function in logged data
    replay_buffer_pretrain = utils.StandardBuffer(
        state_dim, num_actions, FLAGS.batch_size,
        FLAGS.buffer_size, FLAGS.tables.split(",")[0], FLAGS.selected_cols_simulation,FLAGS.discount,
        Lambda_min=Lambda_min, Lambda_max=Lambda_max, Lambda_interval=Lambda_interval, test_mode=FLAGS.test_mode,
        is_prioritized_replay_buffer=is_prioritized_replay_buffer
    )
    # Buffer for off-policy evaluation(OPE)
    replay_buffer_evaluation = utils.StandardBuffer(
        state_dim, num_actions, FLAGS.batch_size,
        FLAGS.buffer_size, FLAGS.tables.split(",")[2], FLAGS.selected_cols_simulation,FLAGS.discount,
        Lambda_min=Lambda_min, Lambda_max=Lambda_max, Lambda_interval=Lambda_interval, test_mode=FLAGS.test_mode,
        is_prioritized_replay_buffer=is_prioritized_replay_buffer
    )

    estimators = FLAGS.Estimator

    # Creating saving files. AssertionError will occur if the export_dir already exists.
    save_dir = ""
    if not FLAGS.test_mode:
        setting = "{}_seed_{}_w1_{}_w2_{}_w3_{}".format(
            FLAGS.env, FLAGS.seed,
            FLAGS.q_loss_weight, FLAGS.i_regularization_weight,
            FLAGS.i_loss_weight
        )
        model_name = "{}_{}".format(FLAGS.model_name, setting)
        save_dir = FLAGS.checkpointDir + model_name + '/'
        mkdir(save_dir)
    else:
        save_dir = './ckpt'

    eval_data_dir = os.path.join(FLAGS.checkpointDir, '{}'.format('eval_trajectory'))

    if not FLAGS.test_mode:
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

        # all_user_come_str = list(all_user_come_str)
        # all_user_hongbao_str = list(all_user_hongbao_str)
        # all_user_liucun_str = list(all_user_liucun_str)
        # all_hongbao_pre30_str =list(all_hongbao_pre30_str)
        # all_average_liucun_str = list(all_average_liucun_str)
        # all_liucun_pre30_str = list(all_liucun_pre30_str)
        # all_user_type_str = list(all_user_type_str)

        all_user_come = []
        all_user_hongbao = []
        all_user_liucun = []
        all_hongbao_pre30 = []
        all_average_liucun = []
        all_liucun_pre30 = []
        all_user_type = []

        for i in range(Number_real_evaluation_users):
            all_user_come.append(list(eval(all_user_come_str[i].decode().replace(" ", ","))))
            all_user_hongbao.append(list(eval(all_user_hongbao_str[i].decode().replace(" ", ","))))
            all_user_liucun.append(list(eval(all_user_liucun_str[i].decode().replace(" ", ","))))
            all_hongbao_pre30.append(list(eval(all_hongbao_pre30_str[i].decode().replace(" ", ","))))
            all_average_liucun.append(float(all_average_liucun_str[i]))
            all_liucun_pre30.append(list(eval(all_liucun_pre30_str[i].decode().replace(" ", ","))))
            all_user_type.append(float(all_user_type_str[i]))

    else:
        #Generate eval data
        all_user_come, all_user_hongbao, all_user_liucun, all_hongbao_pre30, all_liucun_pre30, all_average_liucun, \
        all_user_type = eval_data_generator(eval_data_dir,FLAGS.Number_real_evaluation_users, FLAGS.test_mode)


    for learner in FLAGS.Batch_RL_learner.split(','):
        # train
        result_dir = os.path.join('G:\code\simulation_CMDP2/results', learner)

        eval_data_come, eval_data_hongbao, eval_data_convertion, total_true_value,  total_true_cost_value, \
        total_REM_value, total_REM_cost_value, total_error_IPS,total_error_DM,total_error_DR,total_error_FQE,\
        total_error_REM = train(learner,eval_data_dir,
            replay_buffer, replay_buffer_pretrain, replay_buffer_evaluation, estimators, state_dim, num_actions, action_dim, Max_timesteps,
            Lambda_dim, Lambda_min, Lambda_max, Lambda_interval, Number_real_evaluation_users, save_dir, all_user_come, all_user_hongbao,
                                                  all_user_liucun, all_hongbao_pre30, all_liucun_pre30, all_average_liucun, \
            all_user_type, result_dir)

        come_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + '_' + learner +'_come.npy')
        np.save(come_file_name, eval_data_come)
        hongbao_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + '_' + learner +'_hongbao.npy')
        np.save(hongbao_file_name, eval_data_hongbao)
        convertion_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + '_' + learner + '_convertion.npy')
        np.save(convertion_file_name, eval_data_convertion)
        true_value_file_name = os.path.join(result_dir, 'seed_'  + str(FLAGS.seed) + '_' + learner + '_true_value.npy')
        np.save(true_value_file_name, total_true_value)
        true_value_cost_file_name = os.path.join(result_dir, 'seed_'  + str(FLAGS.seed) + '_' + learner + '_true_cost_value.npy')
        np.save(true_value_cost_file_name, total_true_cost_value)

        # with gfile.GFile(come_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save('G:\code\simulation_CMDP2\data/trajectory_user.npz', eval_data_come, allow_pickle=False)
        # hongbao_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed)+'_hongbao')
        # with gfile.GFile(hongbao_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, eval_data_hongbao, allow_pickle=False)
        # convertion_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + '_convertion')
        # with gfile.GFile(convertion_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, eval_data_convertion, allow_pickle=False)
        # true_value_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + '_true_value')
        # with gfile.GFile(true_value_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, total_true_value, allow_pickle=False)
        # true_value_cost_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + '_true_cost_value')
        # with gfile.GFile(true_value_cost_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, total_true_cost_value, allow_pickle=False)
        # REM_value_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + '_REM_value')
        # with gfile.GFile(REM_value_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, total_REM_value, allow_pickle=False)
        # REM_cost_value_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + '_REM_cost_value')
        # with gfile.GFile(REM_cost_value_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, total_REM_cost_value, allow_pickle=False)
        # IPS_error_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + 'error_IPS')
        # with gfile.GFile(IPS_error_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, total_error_IPS, allow_pickle=False)
        # DM_error_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + 'error_DM')
        # with gfile.GFile(DM_error_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, total_error_DM, allow_pickle=False)
        # DR_error_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + 'error_DR')
        # with gfile.GFile(DR_error_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, total_error_DR, allow_pickle=False)
        # FQE_error_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + 'error_FQE')
        # with gfile.GFile(FQE_error_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, total_error_FQE, allow_pickle=False)
        # REM_error_file_name = os.path.join(result_dir, 'seed_' + str(FLAGS.seed) + 'error_REM')
        # with gfile.GFile(REM_error_file_name, 'wb') as f:
        #     with gzip.GzipFile(fileobj=f) as outfile:
        #         np.save(outfile, total_error_REM, allow_pickle=False)


if __name__ == '__main__':
    tf.app.run()


