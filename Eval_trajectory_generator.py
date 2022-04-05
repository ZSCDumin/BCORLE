# coding: utf-8
import numpy as np
import os
import gzip
import tensorflow as tf

def eval_data_generator(table_name, Number_real_evaluation_users, test_mode):
    gfile = tf.gfile
    Number_days = 30
    high_threshold_liucun = 0.8
    low_threshold_liucun = 0.2
    max_size_hongbao = 2.1
    max_size_hongbao = max_size_hongbao  # 最大面额红包2.1元
    size_hongbao = int(max_size_hongbao * 10)  # 21个红包
    Number_users = Number_real_evaluation_users
    all_user_type = np.zeros(Number_users)
    all_user_liucun = np.zeros([Number_users, size_hongbao])
    all_user_come = np.zeros([Number_users,Number_days])
    all_user_hongbao = np.zeros([Number_users, Number_days])
    all_hongbao_pre30 = np.zeros([Number_users, size_hongbao])
    all_liucun_pre30 = np.zeros([Number_users, size_hongbao])
    all_average_liucun = np.zeros(Number_users)

    saved_user_ID = []
    saved_state = []
    saved_action = []
    saved_reward_liucun = []
    saved_next_state = []
    saved_terminal = []
    saved_liucun_rate = []
    saved_data = []

    for user in range(Number_users):
        user_ID = user + 1
        random_value = np.random.rand()  # 判断该用户属于哪一类留存率
        if random_value < 1.0 / 3.0:  # 低留存
            user_liucun = np.random.rand(size_hongbao) * low_threshold_liucun
            user_type = 0
        elif random_value < 2.0 / 3.0:
            user_liucun = np.random.rand(size_hongbao)
            user_type = 1
        else:
            user_liucun = high_threshold_liucun + np.random.rand(size_hongbao) * low_threshold_liucun
            user_type = 2
        user_liucun.sort()
        all_user_type[user] = user_type
        all_user_liucun[user,:] = user_liucun
        user_come = np.zeros(Number_days)  # 前30天用户到来数据
        user_hongbao = np.zeros(Number_days, dtype=float)  # 前30天用户收到红包数据
        # 前30天用户是否到来以及到来后发放的红包大小
        hongbao_pre30 = np.zeros(size_hongbao)  # 前30天用户领到的红包面额数目
        liucun_pre30 = np.zeros(size_hongbao)  # 前30天用户第二天到来统计
        average_liucun = np.mean(user_liucun)
        all_average_liucun[user] = average_liucun
        for i in range(Number_days):
            if i == 0 or user_come[i - 1] == 0:
                if np.random.rand() < average_liucun:  # 用户到来
                    user_come[i] = 1
                    user_hongbao[i] = (np.random.randint(size_hongbao) + 1) / 10.0
                    hongbao_pre30[int(user_hongbao[i] * 10) - 1] += 1
            else:
                if np.random.rand() < user_liucun[int(user_hongbao[i - 1] * 10) - 1]:  # 用户到来
                    liucun_pre30[int(user_hongbao[i - 1] * 10) - 1] += 1
                    user_come[i] = 1
                    user_hongbao[i] = (np.random.randint(size_hongbao) + 1) / 10.0
                    hongbao_pre30[int(user_hongbao[i] * 10) - 1] += 1
        all_user_come[user,:] = user_come
        all_user_hongbao[user,:] = user_hongbao
        all_hongbao_pre30[user,:] = hongbao_pre30
        all_liucun_pre30[user,:] = liucun_pre30

        # 开始统计进行中的30天
        state = np.zeros([Number_days, size_hongbao * 2 + 2])
        state[0, :] = np.concatenate((np.zeros(1), np.ones(1) * user_type, hongbao_pre30, liucun_pre30))  # 第一个状态是天数
        action = np.zeros([Number_days, 1])
        reward_liucun = np.zeros([Number_days, 1])
        next_state = np.zeros([Number_days, size_hongbao * 2 + 2])
        terminal = np.zeros([Number_days, 1])
        for d in range(Number_days):
            new_come = 0
            new_hongbao = 0
            if user_come[-1] == 0:  # 前一天没来
                if np.random.rand() < average_liucun:  # 前一天没来今天来了
                    new_come = 1
                    new_hongbao = (np.random.randint(size_hongbao) + 1) / 10.0
                    hongbao_pre30[int(new_hongbao * 10) - 1] += 1
            else:  # 前一天来了
                if np.random.rand() < user_liucun[int(user_hongbao[-1] * 10) - 1]:  # 前一天来了今天也来了
                    liucun_pre30[int(user_hongbao[-1] * 10) - 1] += 1
                    new_come = 1
                    new_hongbao = (np.random.randint(size_hongbao) + 1) / 10.0
                    hongbao_pre30[int(new_hongbao * 10) - 1] += 1
            action[d] = new_hongbao
            if d > 0:
                reward_liucun[d - 1] = new_come - 1
                next_state[d - 1, :] = state[d, :]
            if user_come[0] == 1:
                hongbao_pre30[int(user_hongbao[0] * 10) - 1] -= 1
                if user_come[1] == 1:
                    liucun_pre30[int(user_hongbao[0] * 10) - 1] -= 1
            user_come = np.append(user_come, [new_come])
            user_come = np.delete(user_come, [0])
            user_hongbao = np.append(user_hongbao, [new_hongbao])
            user_hongbao = np.delete(user_hongbao, [0])
            if d != Number_days - 1:
                state[d + 1, :] = np.concatenate(
                    (np.ones(1) * (d + 1), np.ones(1) * user_type, hongbao_pre30, liucun_pre30))
            else:
                next_state[d, :] = state[d, :]
                terminal[d] = 1
        for d in range(Number_days):  # 用来设定每个用户的最后一个replay的terminal均为1
            if user_come[-1 - d] == 1:
                terminal[-1 - d] = 1
                break

        for d in range(Number_days):
            if user_come[d] == 1:
                # saved_user_ID.append([str(user_ID)])
                # saved_state.append([' '.join(str(i) for i in state[d, :])])
                # saved_action.append([' '.join(str(i) for i in action[d, :])])
                # saved_reward_liucun.append([' '.join(str(i) for i in reward_liucun[d, :])])
                # saved_next_state.append([' '.join(str(i) for i in next_state[d, :])])
                # saved_terminal.append([' '.join(str(i) for i in terminal[d, :])])
                # saved_liucun_rate.append([' '.join(str(i) for i in user_liucun)])

                saved_data.append([str(user_ID),
                                   ' '.join(str(i) for i in state[d, :]),
                                   ' '.join(str(i) for i in action[d, :]),
                                   ' '.join(str(i) for i in reward_liucun[d, :]),
                                   ' '.join(str(i) for i in next_state[d, :]),
                                   ' '.join(str(i) for i in terminal[d, :]),
                                   ' '.join(str(i) for i in user_liucun)])
    #保存数据至D2做离线策略评估
    # if test_mode:
    #     filename = os.path.join('D:/code/simulation_CMDP/data', '{}_{}'.format('eval_trajectory', 'user'))
    #     np.savez(filename, user_ID=saved_user_ID, state=saved_state, action=saved_action,
    #              reward_liucun=saved_reward_liucun, next_state=saved_next_state, terminal=saved_terminal,
    #              liucun_rate=saved_liucun_rate)
    # else:
    #     # Open a table，打开一个表，返回writer对象
    #     writer = tf.python_io.TableWriter(table_name)
    #
    #     # Write records to the 0-3 columns of the table. 将数据写至表中的第0-3列。
    #     writer.write(saved_data, indices=[0,1,2,3,4,5,6])
    #
    #     # Close the table 关闭表和writer
    #     writer.close()



    #将前30天的数据返回做真实评估
    return all_user_come, all_user_hongbao, all_user_liucun, all_hongbao_pre30, all_liucun_pre30, all_average_liucun, all_user_type
