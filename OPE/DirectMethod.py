# coding: utf-8
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from collections import Counter
#from utils.evaluation_utils import create_from_batch


def DirectMethod(replay_buffer, ckpt_path, save_dir, num_actions,
                 Number_real_evaluation_users, Lambda_size, Lambda_interval, discount, learn_count, max_timesteps):

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

    model_reward1 = []
    model_reward2 = []
    model_reward = []
    for a in range(num_actions): #所有replay全都执行当前action会产生多大的reward
        action_one_hot = to_categorical(a, num_classes=num_actions)
        feed_dict = {propensity_state: replay_buffer.new_batch_data["state"],
                     propensity_Lambda: replay_buffer.new_batch_data["Lambda"],
                     propensity_action: np.tile(action_one_hot, (len(replay_buffer.new_batch_data["state"]), 1))}
        reward1_action = sess.run(Q_action, feed_dict)
        reward1_action = -np.argmax(reward1_action, 1)
        reward2_action = -replay_buffer.ActionEmbedding.mapping_back[a]
        #在这个地方我认为应该也对reward2进行整个轨迹上的扩展
        model_reward1.append(reward1_action)
        model_reward2.append(reward2_action)
    model_reward1 = np.array(model_reward1).T

    state = graph.get_tensor_by_name("state:0")
    Lambda = graph.get_tensor_by_name("lambda:0")
    feed_dict = {state: replay_buffer.new_batch_data["state"],
                 Lambda: replay_buffer.new_batch_data["Lambda"]}

    next_action_ = graph.get_tensor_by_name("next_action:0")
    replay_buffer.new_batch_data["model_propensities"] = sess.run(
        tf.one_hot(next_action_, depth=num_actions)
        , feed_dict
    )
    # Count the number of actions in a batch
    actions = np.argmax(np.squeeze(replay_buffer.new_batch_data["model_propensities"]), axis=1)
    actions_count = Counter(sorted(actions))
    print("========================================================================")
    print("count_actions", actions_count)
    batch_for_estimator = {
        "reward1": model_reward1, "reward2": model_reward2,
        "model_propensities": replay_buffer.new_batch_data["model_propensities"],
        "Lambda": replay_buffer.new_batch_data["Lambda"],
        "user": replay_buffer.new_batch_data["user_id"],
        "done": replay_buffer.new_batch_data["done"]
    }

    # Estimate
    batch = create_from_batch_DM(batch_for_estimator)
    predict_value_day, predict_value = estimate(batch, learn_count, Number_real_evaluation_users,
                             Lambda_size, discount, Lambda_interval, max_timesteps)
    return predict_value

def create_from_batch_DM(batch):
    res_batch = dict()
    res_batch["reward1"] = batch["reward1"]
    res_batch["reward2"] = batch["reward2"]
    res_batch["model_propensities"] = batch["model_propensities"]
    res_batch["Lambda"] = batch["Lambda"]
    res_batch["user"] = batch["user"]
    res_batch["done"] = batch["done"]
    return res_batch

def estimate(batch, learn_count, number_users, Lambda_size, gamma, Lambda_interval, max_timesteps):
    dr_value_reward1 = np.zeros([batch["model_propensities"].shape[0], 1])
    dr_value_reward2 = np.zeros([batch["model_propensities"].shape[0], 1])
    dr_value_reward = np.zeros([batch["model_propensities"].shape[0], 1])

    for i in range(batch["model_propensities"].shape[0]):
        learned_action = np.argmax(batch["model_propensities"][i,:])
        dr_value_reward1[i] = batch["reward1"][i][learned_action]
        dr_value_reward2[i] = batch["reward2"][learned_action]
        dr_value_reward[i] = dr_value_reward1[i] + batch["Lambda"][i] * dr_value_reward2[i]

    index = 0
    importance_reward = np.zeros([Lambda_size, number_users, max_timesteps])
    predict_reward = np.zeros([Lambda_size,number_users])
    while index < len(batch["done"]):
        sum_reward = 0
        discount = np.ones(max_timesteps)
        sum_reward = np.zeros(max_timesteps)
        timestep = 0

        while batch["done"][index][0] != 1.:
            for i in range(timestep+1):
                sum_reward[i] += dr_value_reward1[index] * discount[i]
                discount[i] *= gamma
            index += 1
            timestep += 1

        for i in range(timestep + 1):
            sum_reward[i] += dr_value_reward1[index] * discount[i]

        importance_reward[int(batch["Lambda"][index][0] / Lambda_interval), batch["user"][index][0] - 1, :] = sum_reward
        predict_reward[int(batch["Lambda"][index][0] / Lambda_interval), batch["user"][index][0] - 1] = sum_reward[0]
        index += 1

    return importance_reward, predict_reward