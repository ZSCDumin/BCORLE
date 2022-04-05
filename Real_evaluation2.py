# coding: utf-8

import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import rc
from tensorflow.python.platform import gfile
import imageio

def real_evaluation(sess, Number_real_evaluation_users,  Lambda_min, Lambda_max, Lambda_interval, Lambda_size,
                    all_user_come_str, all_user_hongbao_str, all_user_liucun_str, all_hongbao_pre30_str,
                    all_liucun_pre30_str,all_average_liucun_str, all_user_type_str, training_iters, Plot,
                    discount, ckpt_path, save_dir, result_dir):
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['AR PL KaitiM GB']})
    Number_days = 30
    high_threshold_liucun = 0.8
    low_threshold_liucun = 0.2
    max_size_hongbao = 2.1
    max_size_hongbao = max_size_hongbao  # 最大面额红包2.1元
    size_hongbao = int(max_size_hongbao * 10)  # 21个红包
    Number_users = Number_real_evaluation_users
    all_state = np.zeros([Number_users, Number_days, size_hongbao * 2 + 2])
    all_action = np.zeros([Number_users, Number_days, 1])
    all_reward_liucun = np.zeros([Number_users, Number_days, 1])
    all_next_state = np.zeros([Number_users, Number_days, size_hongbao * 2 + 2])
    all_terminal = np.zeros([Number_users, Number_days, 1])
    true_value = np.zeros([Lambda_size, Number_users])
    true_value_cost = np.zeros([Lambda_size, Number_users])

    total_hongbao = []
    total_come = []
    total_convertion = []

    # sess = tf.Session()
    # 先加载图和参数变量
    ckpt_path = '.'.join([ckpt_path, 'meta'])

    print(ckpt_path)

    saver = tf.train.import_meta_graph(ckpt_path)
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
    graph = tf.get_default_graph()

    # 模型数值导入
    next_action_ = graph.get_tensor_by_name("next_action:0")

    state_ = graph.get_tensor_by_name("state:0")
    Lambda_ = graph.get_tensor_by_name("lambda:0")

    all_user_type2 = np.zeros(Number_users)
    all_user_liucun2 = np.zeros([Number_users, size_hongbao])
    all_user_come2 = np.zeros([Number_users, Number_days])
    all_user_hongbao2 = np.zeros([Number_users, Number_days])
    all_hongbao_pre302 = np.zeros([Number_users, size_hongbao])
    all_liucun_pre302 = np.zeros([Number_users, size_hongbao])
    all_average_liucun2 = np.zeros(Number_users)

    for i in range(Number_users):
        all_user_type2[i] = all_user_type_str[i]
        all_user_liucun2[i,:] = all_user_liucun_str[i]
        all_user_come2[i,:] = all_user_come_str[i]
        all_user_hongbao2[i,:] = all_user_hongbao_str[i]
        all_hongbao_pre302[i,:] = all_hongbao_pre30_str[i]
        all_liucun_pre302[i,:] = all_liucun_pre30_str[i]
        all_average_liucun2[i] = all_average_liucun_str[i]

    for l in range(Lambda_size):

        all_user_type = copy.deepcopy(all_user_type2)
        all_user_liucun = copy.deepcopy(all_user_liucun2)
        all_user_come = copy.deepcopy(all_user_come2)
        all_user_hongbao = copy.deepcopy(all_user_hongbao2)
        all_hongbao_pre30 = copy.deepcopy(all_hongbao_pre302)
        all_liucun_pre30 = copy.deepcopy(all_liucun_pre302)
        all_average_liucun = copy.deepcopy(all_average_liucun2)

        Lambda = Lambda_min + l * Lambda_interval
        Lambda = np.ones([1, 1]) * Lambda
        come_lambda = []
        hongbao_lambda = []
        for user in range(Number_users):
            all_state[user, 0, :] = np.concatenate((np.zeros(1), np.ones(1) * all_user_type[user],
                                                    all_hongbao_pre30[user, :], all_liucun_pre30[user, :]))
        liucun_rate_lambda = np.ones([Number_users,Number_days])
        hongbao_cost_lambda = np.zeros([Number_users, Number_days])
        for d in range(Number_days):
            feed_state_ = np.zeros([Number_users,size_hongbao * 2 + 2])
            new_come = np.zeros(Number_users)
            new_hongbao = np.zeros(Number_users)
            for user in range(Number_users):
                if all_user_come[user, -1] == 0:  # 前一天没来
                    if np.random.rand() < all_average_liucun[user]:  # 前一天没来今天来了
                        new_come[user] = 1
                        feed_state = np.expand_dims(all_state[user, d, :], axis=0)
                        feed_state_[user,:] = feed_state

                else:  # 前一天来了
                    if np.random.rand() < all_user_liucun[user, int(all_user_hongbao[user, -1] * 10) - 1]:  # 前一天来了今天也来了
                        all_liucun_pre30[user, int(all_user_hongbao[user, -1] * 10) - 1] += 1
                        new_come[user] = 1
                        feed_state = np.expand_dims(all_state[user, d, :], axis=0)
                        feed_state_[user, :] = feed_state
                        liucun_rate_lambda[user,d] = 0
                    else:
                        liucun_rate_lambda[user, d] = -1
            feed_Lambda_ = np.ones([Number_users, 1]) * Lambda
            feed_dict = {state_: feed_state_, Lambda_: feed_Lambda_}
            next_action = sess.run(next_action_, feed_dict)
            for user in range(Number_users):
                if new_come[user] == 1:
                    new_hongbao[user] = (np.squeeze(next_action[user]) + 1) / 10.0
                    hongbao_cost_lambda[user,d] = new_hongbao[user]
                    all_hongbao_pre30[user, int(new_hongbao[user] * 10) - 1] += 1

                    come_lambda.append(new_come[user])
                    hongbao_lambda.append(new_hongbao[user])
                if d > 0:
                    all_reward_liucun[user, d - 1] = new_come[user] - 1
                    all_next_state[user, d - 1, :] = all_state[user, d, :]
                if all_user_come[user, 0] == 1:
                    all_hongbao_pre30[user, int(all_user_hongbao[user, 0] * 10) - 1] -= 1
                    if all_user_come[user, 1] == 1:
                        all_liucun_pre30[user, int(all_user_hongbao[user, 0] * 10) - 1] -= 1
                user_come_temp = all_user_come[user, :]
                user_come_temp = np.delete(user_come_temp, [0])
                user_come_temp = np.append(user_come_temp, [new_come[user]])
                all_user_come[user, :] = user_come_temp
                user_hongbao_temp = all_user_hongbao[user, :]
                user_hongbao_temp = np.append(user_hongbao_temp, [new_hongbao[user]])
                user_hongbao_temp = np.delete(user_hongbao_temp, [0])
                all_user_hongbao[user, :] = user_hongbao_temp
                if d != Number_days - 1:
                    all_state[user, d + 1, :] = np.concatenate(
                        (np.ones(1) * (d + 1), np.ones(1) * all_user_type[user], all_hongbao_pre30[user, :],
                         all_liucun_pre30[user, :]))
        for user in range(Number_users):
            liucun_rate_lambda_ = liucun_rate_lambda[user,:]
            hongbao_cost_lambda_ = hongbao_cost_lambda[user,:]
            delete_1 = np.array([1])
            liucun_rate_lambda_ = np.setdiff1d(liucun_rate_lambda_, delete_1)
            true_value[l, user] = sum(np.logspace(0, len(liucun_rate_lambda_) - 1, len(liucun_rate_lambda_),
                                                  base=discount) * liucun_rate_lambda_)
            delete_1 = np.array([0])
            hongbao_cost_lambda_ = np.setdiff1d(hongbao_cost_lambda_, delete_1)
            true_value_cost[l,user] = sum(np.logspace(0, len(hongbao_cost_lambda_) - 1, len(hongbao_cost_lambda_),
                                                  base=discount) * hongbao_cost_lambda_)

        print("[Real Evaluation @learn_count={}], Lambda={}, Monthly_come={}, Monthly_hongbao_cost={}, convertion={}".format(
            training_iters,
            round(Lambda_min + l *Lambda_interval, 2),
            sum(come_lambda),
            sum(hongbao_lambda),
            sum(come_lambda)/sum(hongbao_lambda)))
        total_come.append(sum(come_lambda))
        total_hongbao.append(sum(hongbao_lambda))
        total_convertion.append(sum(come_lambda)/sum(hongbao_lambda))

    if Plot:
        x_label = []
        for l in range(Lambda_size):
            x_label.append(round(Lambda_min + l * Lambda_interval, 2))
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 16,
                 }
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 30,
                 }
        # 画图
        fig2 = plt.figure(figsize=(26, 8))
        ax1_1 = fig2.add_subplot(1, 3, 1)  # t
        plt.tick_params(labelsize=18)
        plt.grid(linestyle='-.')
        ax1_2 = fig2.add_subplot(1, 3, 2)  # hongbao cost
        plt.tick_params(labelsize=18)
        plt.grid(linestyle='-.')
        ax1_3 = fig2.add_subplot(1, 3, 3)  # hongbao cost
        plt.tick_params(labelsize=18)
        plt.grid(linestyle='-.')


        ax1_1.plot(x_label, total_come, marker='*', markersize='10', linewidth='2')
        ax1_1.set_xlabel(u'$λ_1$', font2)
        ax1_1.set_ylabel(u'Visit', font2)
        ax1_1.set_title(u'Visit', font2)

        ax1_2.plot(x_label, total_hongbao, marker='*', markersize='10', linewidth='2')
        ax1_2.set_xlabel(u'$λ_1$', font2)
        ax1_2.set_ylabel(u'Cost', font2)
        ax1_2.set_title(u'Cost', font2)

        ax1_3.plot(x_label, total_convertion, marker='*', markersize='10', linewidth='2')
        ax1_3.set_xlabel(u'$λ_1$', font2)
        ax1_3.set_ylabel(u'Convertion', font2)
        ax1_3.set_title(u'Convertion', font2)

        fig2.tight_layout()
        fig2.subplots_adjust(wspace=0.22)
        image = fig2data(fig2)
        with gfile.GFile(os.path.join(result_dir, 'output_lambda.png'), "w") as f:
          imageio.imsave(f, image, 'PNG')
    return true_value, true_value_cost, total_come, total_hongbao, total_convertion


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
