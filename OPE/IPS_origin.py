# coding: utf-8
import numpy as np
import tensorflow as tf
import os
import math


def importance_sampling(replay_buffer, pi_b, ckpt_path, save_dir, number_actions,
                        number_users, Lambda_size, Lambda_interval, gamma, max_time_steps):

    batch_size = 10000

    #读取已保存的模型
    sess = tf.Session()
    # 先加载图和参数变量
    ckpt_path = '.'.join([ckpt_path, 'meta'])

    saver = tf.train.import_meta_graph(ckpt_path)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
    graph = tf.get_default_graph()

    # 模型数值导入
    Q_action = graph.get_tensor_by_name("current_q:0")

    state = graph.get_tensor_by_name("state:0")
    Lambda = graph.get_tensor_by_name("lambda:0")
    action = graph.get_tensor_by_name("current_action:0")

    pi_b_Q_action = graph.get_tensor_by_name("action_probability:0")
    pi_b_state = graph.get_tensor_by_name("propensity_state:0")
    pi_b_Lambda = graph.get_tensor_by_name("propensity_lambda:0")

    index = 0
    importance_reward = np.zeros([Lambda_size, number_users])
    total_importance_product = np.zeros([Lambda_size, number_users, max_time_steps])

    Q_ = np.zeros([number_actions, len(replay_buffer.new_batch_data["state"])])
    train_times = int(math.ceil(len(replay_buffer.new_batch_data["state"])/batch_size))
    for i in range(number_actions):
        Q_3 = np.zeros(len(replay_buffer.new_batch_data["state"]))
        for j in range(train_times):
            begin = j*batch_size
            if (j+1)*batch_size < len(replay_buffer.new_batch_data["state"]):
                end = (j+1)*batch_size
            else:
                end = len(replay_buffer.new_batch_data["state"])
            feed_state = replay_buffer.new_batch_data["state"][begin:end]
            feed_lambda = replay_buffer.new_batch_data["Lambda"][begin:end]
            feed_action = np.expand_dims(np.tile(i, len(replay_buffer.new_batch_data["state"]))[begin:end],1)
            feed_dict = {state: feed_state,
                     Lambda: feed_lambda,
                     action: feed_action}
            Q_2 = sess.run(Q_action, feed_dict)
            Q_3[begin:end] = np.squeeze(Q_2)
        Q_[i,:] = Q_3.T
        print('calculate Q_value_actions_{} finished'.format(i))
    Q_pi_b_ = np.zeros([len(replay_buffer.new_batch_data["state"]),number_actions])

    for j in range(train_times):
        begin = j * batch_size
        if (j+1)*batch_size < len(replay_buffer.new_batch_data["state"]):
            end = (j+1)*batch_size
        else:
            end = len(replay_buffer.new_batch_data["state"])
        feed_dict = {pi_b_state: replay_buffer.new_batch_data["state"][begin:end],
                 pi_b_Lambda: replay_buffer.new_batch_data["Lambda"][begin:end]}
        Q_pi_b_2 = sess.run(pi_b_Q_action, feed_dict)
        Q_pi_b_[begin:end,:] = Q_pi_b_2
        print('calculate Q_pi_b_{} finished'.format(j))
    while index < len(replay_buffer.new_batch_data["done"]):
        importance_product = 1
        sum_reward = 0
        discount = 1
        timestep = 0

        while replay_buffer.new_batch_data["done"][index][0] != 1.:
            # 行为选取
            Q2 = Q_[:,index]
            Q = np.exp(Q2) / np.sum(np.exp(Q2), axis=0)
            #pi_b预测
            Q_pi_b = np.exp(Q_pi_b_[index]) / np.sum(np.exp(Q_pi_b_[index]), axis=0)
            ##importance计算
            importance = Q[replay_buffer.new_batch_data["action"][index][0]] /\
                         Q_pi_b[replay_buffer.new_batch_data["action"][index][0]]
            importance_product = importance_product * importance
            total_importance_product[int(round(replay_buffer.new_batch_data["Lambda"][index][0] / Lambda_interval)),
                          replay_buffer.new_batch_data["user_id"][index][0] - 1,timestep] = importance_product
            sum_reward += discount * replay_buffer.new_batch_data["reward2"][index][0]
            discount = discount * gamma
            index += 1
            timestep += 1

        Q2 = Q_[:, index]
        Q = np.exp(Q2) / np.sum(np.exp(Q2), axis=0)
        # pi_b预测
        Q_pi_b = np.exp(Q_pi_b_[index]) / np.sum(np.exp(Q_pi_b_[index]), axis=0)
        ##importance计算
        importance = Q[replay_buffer.new_batch_data["action"][index][0]] / Q_pi_b[
            replay_buffer.new_batch_data["action"][index][0]]
        importance_product = importance_product * importance
        total_importance_product[int(replay_buffer.new_batch_data["Lambda"][index][0] / Lambda_interval),
                                 replay_buffer.new_batch_data["user_id"][index][0] - 1, timestep] = importance_product
        sum_reward += discount * replay_buffer.new_batch_data["reward2"][index][0]
        importance_reward[int(replay_buffer.new_batch_data["Lambda"][index][0] / Lambda_interval),
                          replay_buffer.new_batch_data["user_id"][index][0] - 1] = importance_product * sum_reward

        index += 1

        print(index)
    return importance_reward, total_importance_product