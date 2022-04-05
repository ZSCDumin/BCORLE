# coding: utf-8

import matplotlib
matplotlib.use('Agg')
from OPE.IPS_origin import importance_sampling
from OPE.DoublyRobust import DoublyRobust
#from OPE.DirectMethod import DirectMethod
from OPE.DirectMethod2 import DirectMethod
from OPE.FQE import FQE
from OPE.REM import MultiHeadDQNAgent
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import rc
from tensorflow.python.platform import gfile
import imageio
import numpy as np
import os
import math
import copy
import datetime
import PIL.Image as Image

def evaluation(eval_data_dir, replay_buffer_train, replay_buffer, learn_count, test_mode, ckpt_path, save_dir,
               Estimator, num_actions,
               Number_real_evaluation_users, discount, max_timesteps,
               action_dim, Lambda_dim, state_dim, fqe_train_steps, rem_train_steps, Lambda_min, Lambda_max,
               Lambda_interval, Lambda_size, result_dir,eval_train_state,eval_train_action,eval_train_reward1,eval_train_reward2,
               eval_train_Lambda, eval_train_next_state,eval_train_done):
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['AR PL KaitiM GB']})
    """
    		:param
    		eval_num: the num of batch for evaluation;
    		replay_buffer: buffer for evaluation
    		learn_count: the number of training epoch
    		"""

    # is_evaluation = True
    # # get a batch
    # if test_mode:
    #     replay_buffer.get_batch_data_from_local_eval_npz()
    # else:
    #     #replay_buffer.get_batch_data_from_d2_eval_npz(eval_data_dir)
    #     replay_buffer.get_batch_data_from_eval_odps(is_evaluation)
    #     #replay_buffer.get_batch_data_from_odps()

    total_importance_product = None
    predict_value_IPS = None
    predict_value_DM = None
    predict_value_DM_day = None
    predict_value_DR = None
    predict_value_FQE = None
    predict_value_REM = None
    predict_cost_value_REM = None
    for estimator in Estimator.split(','):

        if estimator == 'IPSEstimator':
            pi_b = 1.0 / num_actions
            predict_value_IPS, total_importance_product = \
                importance_sampling(replay_buffer, pi_b, ckpt_path, save_dir, num_actions,
                                    Number_real_evaluation_users, Lambda_size,
                                    Lambda_interval, discount, max_timesteps)

            print('IPS OPE finished')

        # Model-based DM method
        if estimator == 'DirectMethodEstimator':
            # Model-based DM method
            predict_value_DM_day, predict_value_DM = DirectMethod(replay_buffer, ckpt_path, save_dir, num_actions,
                                                                  Number_real_evaluation_users, Lambda_size,
                                                                  Lambda_interval, discount, learn_count,
                                                                  max_timesteps)

            print('DM OPE finished')

        if estimator == 'DoublyRobustEstimator':
            pi_b = 1.0 / num_actions
            if total_importance_product is None:
                _, total_importance_product = importance_sampling(replay_buffer, pi_b, ckpt_path,
                                                                  save_dir, num_actions,
                                                                  Number_real_evaluation_users,
                                                                  Lambda_size,
                                                                  Lambda_interval, discount,
                                                                  max_timesteps)
            if predict_value_DM_day is None:
                predict_value_DM_day, predict_value_DM = DirectMethod(replay_buffer, ckpt_path, save_dir,
                                                                      num_actions,
                                                                      Number_real_evaluation_users,
                                                                      Lambda_size,
                                                                      Lambda_interval, discount, learn_count,
                                                                      max_timesteps)

            predict_value_DR = DoublyRobust(replay_buffer, total_importance_product, predict_value_DM_day,
                                            Number_real_evaluation_users,
                                            Lambda_size, discount, max_timesteps)

            print('DR OPE finished')

        if estimator == 'FQEstimator':
            batch_size = 10000
            learn_size = 10000
            fqe = FQE(replay_buffer, replay_buffer_train, ckpt_path, save_dir, action_dim, state_dim, Lambda_dim, 0.001,
                      discount, Number_real_evaluation_users, Lambda_size, Lambda_interval)

            train_times = int(math.ceil(len(replay_buffer_train.new_batch_data2["state"]) / learn_size))

            state = eval_train_state
            action = eval_train_action
            reward = eval_train_reward1
            reward_cost = eval_train_reward2
            Lambda = eval_train_Lambda
            next_state = eval_train_next_state
            done = eval_train_done

            sess = tf.Session()
            # 先加载图和参数变量
            ckpt_path2 = copy.deepcopy(ckpt_path)
            ckpt_path2 = '.'.join([ckpt_path2, 'meta'])
            saver = tf.train.import_meta_graph(ckpt_path2)
            saver.restore(sess, tf.train.latest_checkpoint(save_dir))
            # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
            graph = tf.get_default_graph()
            # 模型数值导入
            next_action_ = graph.get_tensor_by_name("next_action:0")
            state_ = graph.get_tensor_by_name("state:0")
            Lambda_ = graph.get_tensor_by_name("lambda:0")

            next_action = np.zeros(len(replay_buffer_train.new_batch_data2["state"]))

            for j in range(train_times):
                begin = j * learn_size
                if (j + 1) * learn_size < len(replay_buffer_train.new_batch_data2["state"]):
                    end = (j + 1) * learn_size
                else:
                    end = len(replay_buffer_train.new_batch_data2["state"])
                feed_state = next_state[begin:end]
                feed_lambda = Lambda[begin:end]
                feed_dict = {state_: feed_state,
                             Lambda_: feed_lambda}
                Q_2 = sess.run(next_action_, feed_dict)
                next_action[begin:end] = np.squeeze(Q_2)

            for index in range(fqe_train_steps):
                rows = np.random.choice(state.shape[0], batch_size)
                state2 = state[rows, :]
                # self.new_batch_data["time_id"] = step_in_episode[:self.max_size]
                action2 = action[rows, :]
                reward2 = reward_cost[rows, :]
                Lambda2 = Lambda[rows, :]
                next_state2 = next_state[rows, :]
                next_action2 = np.expand_dims(np.squeeze(next_action[rows]),-1)
                done2 = done[rows, :]

                loss = fqe.learn(state2, action2, reward2, Lambda2, next_state2, next_action2, done2, index)
                print("FQE_iteration:{}, Loss:{}".format(index, loss))
            feed_state = replay_buffer.new_batch_data["state"]
            feed_lambda = replay_buffer.new_batch_data["Lambda"]
            feed_dict = {state_: feed_state,
                         Lambda_: feed_lambda}
            next_action_pi = sess.run(next_action_, feed_dict)
            predict_value_FQE = fqe.evaluate(next_action_pi)
            print('FQE OPE finished')

        if estimator == 'REMEstimator':
            predict_value_REM2 = 0
            batch_size = 1000
            learn_size = 10000
            start = datetime.datetime.now()
            rem = MultiHeadDQNAgent(replay_buffer_train, replay_buffer, ckpt_path, save_dir, num_actions, state_dim,
                                    Lambda_dim,
                                    Number_real_evaluation_users, Lambda_size, Lambda_interval)
            rem2 = MultiHeadDQNAgent(replay_buffer_train, replay_buffer, ckpt_path, save_dir, num_actions, state_dim,
                                    Lambda_dim,
                                    Number_real_evaluation_users, Lambda_size, Lambda_interval)
            end = datetime.datetime.now()
            print(end - start)
            print('REM class finished')
            start = datetime.datetime.now()
            state = eval_train_state
            action = eval_train_action
            reward = eval_train_reward1
            reward_cost = eval_train_reward2
            Lambda = eval_train_Lambda
            next_state = eval_train_next_state
            done = eval_train_done
            end = datetime.datetime.now()
            print(end - start)
            print('data export finished')

            sess = tf.Session()
            # 先加载图和参数变量
            ckpt_path2 = copy.deepcopy(ckpt_path)
            ckpt_path2 = '.'.join([ckpt_path2, 'meta'])
            saver = tf.train.import_meta_graph(ckpt_path2)
            saver.restore(sess, tf.train.latest_checkpoint(save_dir))
            # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
            graph = tf.get_default_graph()
            # 模型数值导入
            next_action_ = graph.get_tensor_by_name("next_action:0")
            state_ = graph.get_tensor_by_name("state:0")
            Lambda_ = graph.get_tensor_by_name("lambda:0")

            next_action = np.zeros(len(replay_buffer_train.new_batch_data2["state"]))

            train_times = int(math.ceil(len(replay_buffer_train.new_batch_data2["state"]) / learn_size))
            for j in range(train_times):
                begin = j * learn_size
                if (j + 1) * learn_size < len(replay_buffer_train.new_batch_data2["state"]):
                    end = (j + 1) * learn_size
                else:
                    end = len(replay_buffer_train.new_batch_data2["state"])
                feed_state = next_state[begin:end]
                feed_lambda = Lambda[begin:end]
                feed_dict = {state_: feed_state,
                             Lambda_: feed_lambda}
                Q_2 = sess.run(next_action_, feed_dict)
                next_action[begin:end] = np.squeeze(Q_2)
            print(train_times)


            for index in range(rem_train_steps):

                rows = np.random.choice(state.shape[0], batch_size)
                state2 = state[rows, :]
                # self.new_batch_data["time_id"] = step_in_episode[:self.max_size]
                action2 = action[rows, :]
                reward2 = reward[rows, :]
                reward_cost2 = reward_cost[rows,:]
                Lambda2 = Lambda[rows, :]
                next_state2 = next_state[rows, :]
                next_action2 = np.expand_dims(np.squeeze(next_action[rows]), -1)
                done2 = done[rows, :]

                loss = rem.learn(state2, action2, reward2, Lambda2, next_state2, next_action2, done2, index)
                loss2 = rem2.learn(state2, action2, reward_cost2, Lambda2, next_state2, next_action2, done2, index)
                print("train_iters: {}, REM OPE_liucun loss: {}, REM OPE_cost loss: {}".format(index, loss, loss2))
            feed_state = replay_buffer.new_batch_data["state"]
            feed_lambda = replay_buffer.new_batch_data["Lambda"]
            feed_dict = {state_: feed_state,
                         Lambda_: feed_lambda}
            next_action_pi = sess.run(next_action_, feed_dict)
            predict_value_REM = rem.evaluate(next_action_pi)
            predict_cost_value_REM = rem2.evaluate(next_action_pi)
            #predict_value_REM2 += predict_value_REM
            print('REM OPE finished')
            #predict_value_REM2 /= 5

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    # 画图
    fig = plt.figure(figsize=(12, 8))

    x_label = []
    y_label_REM = []
    # y_label_FQE = []
    y_label_DR = []
    y_label_DM = []
    y_label_IPS = []
    for i in range(np.sum(predict_value_REM, axis = 1).shape[0]):
        y_label_REM.append(np.sum(predict_value_REM, axis = 1)[i])
        # y_label_FQE.append(np.sum(predict_value_FQE, axis=1)[i])
        # y_label_DR.append(np.sum(predict_value_DR, axis=1)[i])
        # y_label_DM.append(np.sum(predict_value_DM, axis=1)[i])
        # y_label_IPS.append(np.sum(predict_value_IPS, axis=1)[i])
    for l in range(Lambda_size):
        x_label.append(round(Lambda_min + l * Lambda_interval, 2))
    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(time=x_label, data=y_label_REM, color="r", condition="OPE_REM")
    # sns.tsplot(time=x_label, data=y_label_FQE, color="g", condition="OPE_FQE")
    # sns.tsplot(time=x_label, data=y_label_DR, color="y", condition="OPE_DR")
    # sns.tsplot(time=x_label, data=y_label_DM, color="b", condition="OPE_DM")
    # sns.tsplot(time=x_label, data=y_label_IPS, color="c", condition="OPE_IPS")

    plt.ylabel("Q_value")
    plt.xlabel("Lambda")
    plt.title("Q_value(liucun)")

    # draw the renderer
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


    result_dir = result_dir
    with gfile.GFile(os.path.join(result_dir, 'output_OPE_REM.png'), "w") as f:
        imageio.imsave(f, image, 'PNG')

    plt.close(fig)



    fig2 = plt.figure(figsize=(12, 8))

    x_label = []
    y_label_REM = []
    # y_label_FQE = []
    y_label_DR = []
    y_label_DM = []
    y_label_IPS = []
    for i in range(np.sum(predict_cost_value_REM, axis=1).shape[0]):
        y_label_REM.append(np.sum(predict_cost_value_REM, axis=1)[i])
        # y_label_FQE.append(np.sum(predict_value_FQE, axis=1)[i])
        # y_label_DR.append(np.sum(predict_value_DR, axis=1)[i])
        # y_label_DM.append(np.sum(predict_value_DM, axis=1)[i])
        # y_label_IPS.append(np.sum(predict_value_IPS, axis=1)[i])
    for l in range(Lambda_size):
        x_label.append(round(Lambda_min + l * Lambda_interval, 2))
    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(time=x_label, data=y_label_REM, color="r", condition="OPE_REM")
    # sns.tsplot(time=x_label, data=y_label_FQE, color="g", condition="OPE_FQE")
    # sns.tsplot(time=x_label, data=y_label_DR, color="y", condition="OPE_DR")
    # sns.tsplot(time=x_label, data=y_label_DM, color="b", condition="OPE_DM")
    # sns.tsplot(time=x_label, data=y_label_IPS, color="c", condition="OPE_IPS")

    plt.ylabel("Q_value")
    plt.xlabel("Lambda")
    plt.title("Q_value(cost)")

    # draw the renderer
    import PIL.Image as Image
    # draw the renderer
    fig2.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig2.canvas.get_width_height()
    buf = np.fromstring(fig2.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)

    result_dir = result_dir
    with gfile.GFile(os.path.join(result_dir, 'output_cost_OPE_REM.png'), "w") as f:
        imageio.imsave(f, image, 'PNG')

    return predict_value_IPS, predict_value_DM, predict_value_DR, predict_value_FQE, predict_value_REM, predict_cost_value_REM

if __name__ == '__main__':
    tf.app.run()