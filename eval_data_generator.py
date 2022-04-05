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
flags.DEFINE_string(
    "outputs",
    ",",
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


def main(unused_argv):

    action_dim = 1
    # Set worker
    worker_count = None
    if not FLAGS.test_mode:
        worker_spec = FLAGS.worker_hosts.split(",")
        worker_count = len(worker_spec)
        print("worker_count:{}".format(worker_count))


    # Set seed
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Generate eval data
    all_user_come, all_user_hongbao, all_user_liucun, all_hongbao_pre30, all_liucun_pre30, all_average_liucun, \
    all_user_type = eval_data_generator(FLAGS.outputs.split(",")[1], FLAGS.Number_real_evaluation_users, FLAGS.test_mode)

    saved_all_user_come = []
    saved_all_user_hongbao = []
    saved_all_user_liucun = []
    saved_all_hongbao_pre30 = []
    saved_all_liucun_pre30 = []
    saved_all_average_liucun = []
    saved_all_user_type = []

    saved_data = []
    for j in range(all_user_come.shape[0]):
        # saved_all_user_come.append([' '.join(str(i) for i in all_user_come[j, :])])
        # saved_all_user_hongbao.append([' '.join(str(i) for i in all_user_hongbao[j, :])])
        # saved_all_user_liucun.append([' '.join(str(i) for i in all_user_liucun[j, :])])
        # saved_all_hongbao_pre30.append([' '.join(str(i) for i in all_hongbao_pre30[j, :])])
        # saved_all_liucun_pre30.append([' '.join(str(i) for i in all_liucun_pre30[j, :])])
        # saved_all_average_liucun.append([str(all_average_liucun[j])])
        # saved_all_user_type.append([str(all_user_type[j])])
        #
        saved_data.append([' '.join(str(i) for i in all_user_come[j, :]),
                           ' '.join(str(i) for i in all_user_hongbao[j, :]),
                           ' '.join(str(i) for i in all_user_liucun[j, :]),
                           ' '.join(str(i) for i in all_hongbao_pre30[j, :]),
                           ' '.join(str(i) for i in all_liucun_pre30[j, :]),
                           str(all_average_liucun[j]),
                           str(all_user_type[j])
                           ])

    # Open a table，打开一个表，返回writer对象
    writer = tf.python_io.TableWriter(FLAGS.outputs.split(",")[0])

    # Write records to the 0-3 columns of the table. 将数据写至表中的第0-3列。
    writer.write(saved_data, indices=[0,1,2,3,4,5,6])


    # Close the table 关闭表和writer
    writer.close()



    # result_dir = os.path.join(FLAGS.buckets, 'eval_data')
    #
    # all_user_come_file_name = os.path.join(result_dir, 'all_user_come')
    # with gfile.GFile(all_user_come_file_name, 'wb') as f:
    #     with gzip.GzipFile(fileobj=f) as outfile:
    #         np.save(outfile, all_user_come, allow_pickle=False)
    # all_user_hongbao_file_name = os.path.join(result_dir, 'all_user_hongbao')
    # with gfile.GFile(all_user_hongbao_file_name, 'wb') as f:
    #     with gzip.GzipFile(fileobj=f) as outfile:
    #         np.save(outfile, all_user_hongbao, allow_pickle=False)
    # all_user_liucun_file_name = os.path.join(result_dir, 'all_user_come')
    # with gfile.GFile(all_user_liucun_file_name, 'wb') as f:
    #     with gzip.GzipFile(fileobj=f) as outfile:
    #         np.save(outfile, all_user_liucun, allow_pickle=False)
    # all_hongbao_pre30_file_name = os.path.join(result_dir, 'all_hongbao_pre30')
    # with gfile.GFile(all_hongbao_pre30_file_name, 'wb') as f:
    #     with gzip.GzipFile(fileobj=f) as outfile:
    #         np.save(outfile, all_hongbao_pre30, allow_pickle=False)
    # all_user_come_file_name = os.path.join(result_dir, 'all_user_come')
    # with gfile.GFile(all_user_come_file_name, 'wb') as f:
    #     with gzip.GzipFile(fileobj=f) as outfile:
    #         np.save(outfile, all_user_come, allow_pickle=False)
    # all_user_come_file_name = os.path.join(result_dir, 'all_user_come')
    # with gfile.GFile(all_user_come_file_name, 'wb') as f:
    #     with gzip.GzipFile(fileobj=f) as outfile:
    #         np.save(outfile, all_user_come, allow_pickle=False)
    # all_user_come_file_name = os.path.join(result_dir, 'all_user_come')
    # with gfile.GFile(all_user_come_file_name, 'wb') as f:
    #     with gzip.GzipFile(fileobj=f) as outfile:
    #         np.save(outfile, all_user_come, allow_pickle=False)




if __name__ == '__main__':
    tf.app.run()


