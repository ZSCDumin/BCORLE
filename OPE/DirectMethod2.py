# coding: utf-8
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from collections import Counter
import math
#from utils.evaluation_utils import create_from_batch


def DirectMethod(replay_buffer, ckpt_path, save_dir, num_actions,
                 Number_real_evaluation_users, Lambda_size, Lambda_interval, discount, learn_count, max_timesteps):
    batch_size = 10000
    number_users = Number_real_evaluation_users

    # 读取已保存的模型
    sess = tf.Session()
    # 先加载图和参数变量
    ckpt_path = '.'.join([ckpt_path, 'meta'])

    saver = tf.train.import_meta_graph(ckpt_path)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
    graph = tf.get_default_graph()

    # 模型数值导入
    Q_action = graph.get_tensor_by_name("propensity_r13:0")

    propensity_state = graph.get_tensor_by_name("propensity_state:0")
    propensity_Lambda = graph.get_tensor_by_name("propensity_lambda:0")
    propensity_action = graph.get_tensor_by_name("propensity_action:0")

    state = graph.get_tensor_by_name("state:0")
    Lambda = graph.get_tensor_by_name("lambda:0")

    # Estimate
    index = 0
    importance_reward = np.zeros([Lambda_size, number_users, max_timesteps])
    predict_reward = np.zeros([Lambda_size,number_users])


    next_action_ = graph.get_tensor_by_name("next_action:0")
    train_times = int(math.ceil(len(replay_buffer.new_batch_data["state"]) / batch_size))
    next_action2 = np.zeros([len(replay_buffer.new_batch_data["state"]), num_actions])
    for j in range(train_times):
        begin = j * batch_size
        if (j + 1) * batch_size < len(replay_buffer.new_batch_data["state"]):
            end = (j + 1) * batch_size
        else:
            end = len(replay_buffer.new_batch_data["state"])
        feed_dict = {state: replay_buffer.new_batch_data["state"][begin:end],
                     Lambda: replay_buffer.new_batch_data["Lambda"][begin:end]}
        next_action = sess.run(
            tf.one_hot(next_action_, depth=num_actions, axis=-1)
            , feed_dict
        )
        next_action2[begin:end] = np.squeeze(next_action)

    print('calculate next_action finished')
    reward1_action = np.zeros(len(replay_buffer.new_batch_data["state"]))
    for j in range(train_times):
        begin = j * batch_size
        if (j + 1) * batch_size < len(replay_buffer.new_batch_data["state"]):
            end = (j + 1) * batch_size
        else:
            end = len(replay_buffer.new_batch_data["state"])
        feed_dict = {propensity_state: replay_buffer.new_batch_data["state"][begin:end],
                 propensity_Lambda: replay_buffer.new_batch_data["Lambda"][begin:end],
                 propensity_action: np.squeeze(next_action2[begin:end])}
        reward1_action2 = sess.run(Q_action, feed_dict)
        reward1_action[begin:end] = reward1_action2
    print('calculate Q_value finished')

    while index < len(replay_buffer.new_batch_data["done"]):

        timestep = 0

        predict_reward[int(replay_buffer.new_batch_data["Lambda"][index][0] / Lambda_interval),
                       replay_buffer.new_batch_data["user_id"][index][0] - 1] = reward1_action[index]
        importance_reward[int(replay_buffer.new_batch_data["Lambda"][index][0] / Lambda_interval),
                          replay_buffer.new_batch_data["user_id"][index][0] - 1, timestep] = reward1_action[index]

        if replay_buffer.new_batch_data["done"][index][0] != 1.:
            timestep += 1
            index += 1
            while replay_buffer.new_batch_data["done"][index][0] != 1.:

                importance_reward[int(replay_buffer.new_batch_data["Lambda"][index][0] / Lambda_interval),
                replay_buffer.new_batch_data["user_id"][index][0] - 1, timestep] = reward1_action[index]
                index += 1
                timestep += 1
            importance_reward[int(round(replay_buffer.new_batch_data["Lambda"][index][0] / Lambda_interval)),
                              replay_buffer.new_batch_data["user_id"][index][0] - 1, timestep] = reward1_action[index]
            index += 1
        else:
            index += 1
        print(index)

    return importance_reward, predict_reward