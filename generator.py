import argparse
import numpy as np
import os
import pickle
import gzip
import tensorflow as tf
from collections import Counter

gfile = tf.gfile

parser = argparse.ArgumentParser(description='Simulation_CMDP')
parser.add_argument('--Generatrator_Policy', default='Random',type=str,help='Name of policy that generating replay')
parser.add_argument('--Number_users', default=10000, type=int, help='The number of consumers in the simulation')
parser.add_argument('--Number_days', default=30, type=int, help='The trajectory of each agent')
parser.add_argument('--High_threshold_liucun', default=0.8, type=float, help='The high threshold of liucun rate')
parser.add_argument('--Low_threshold_liucun', default=0.2, type=float, help='The low threshold of liucun rate')
parser.add_argument('--max_size_hongbao',default=2.1,type=float, help='The max value of hongbao')


def main(args):
    Number_users = args.Number_users
    Number_days = args.Number_days
    high_threshold_liucun = args.High_threshold_liucun
    low_threshold_liucun = args.Low_threshold_liucun
    max_size_hongbao = args.max_size_hongbao
    max_size_hongbao = max_size_hongbao  # 最大面额红包2.1元
    size_hongbao = int(max_size_hongbao * 10)  # 21个红包
    saved_user_ID = []
    saved_state = []
    saved_action = []
    saved_reward_liucun = []
    saved_next_state = []
    saved_terminal = []
    saved_liucun_rate = []
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
        user_come = np.zeros(Number_days)  # 前30天用户到来数据
        user_hongbao = np.zeros(Number_days, dtype=float)  # 前30天用户收到红包数据
        # 前30天用户是否到来以及到来后发放的红包大小
        hongbao_pre30 = np.zeros(size_hongbao)  # 前30天用户领到的红包面额数目
        liucun_pre30 = np.zeros(size_hongbao)  # 前30天用户第二天到来统计
        average_liucun = np.mean(user_liucun)
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
        # 开始统计进行中的30天
        state = np.zeros([Number_days, size_hongbao * 2 + 2])
        state[0, :] = np.concatenate((np.zeros(1), np.ones(1)*user_type, hongbao_pre30, liucun_pre30))#第一个状态是天数
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
                state[d + 1, :] = np.concatenate((np.ones(1) * (d+1), np.ones(1)*user_type, hongbao_pre30, liucun_pre30))
            else:
                next_state[d, :] = state[d, :]
                terminal[d] = 1
        for d in range(Number_days):  # 用来设定每个用户的最后一个replay的terminal均为1
            if user_come[-1 - d] == 1:
                terminal[-1 - d] = 1
                break
        for d in range(Number_days):
            if user_come[d] == 1:
                saved_user_ID.append(user_ID)
                saved_state.append(' '.join(str(i) for i in state[d,:]))
                saved_action.append(' '.join(str(i) for i in action[d,:]))
                saved_reward_liucun.append(' '.join(str(i) for i in reward_liucun[d,:]))
                saved_next_state.append(' '.join(str(i) for i in next_state[d,:]))
                saved_terminal.append(' '.join(str(i) for i in terminal[d,:]))
                saved_liucun_rate.append(' '.join(str(i) for i in user_liucun))
    #filename = os.path.join('G:/阿里实习/披露数据/code/simulation_CDP2/data/trajectory_user.npz'.format('trajectory', 'user'))
    np.savez('G:\code\simulation_CMDP2\data/trajectory_user.npz', user_ID = saved_user_ID,
             state = saved_state, action = saved_action, reward_liucun = saved_reward_liucun, next_state = saved_next_state,
             terminal = saved_terminal, liucun_rate = saved_liucun_rate)
    #filename = os.path.join('D:/code/simulation_CMDP/data', '{}_{}.npz'.format('small_eval_trajectory', 'user'))

    # with gfile.GFile(filename, 'wb') as f:
    #     with gzip.GzipFile(fileobj=f) as outfile:
    #         np.save(outfile, saved_liucun_rate, allow_pickle=False)
    #
    # with gfile.GFile(filename, 'rb') as f:
    #     with gzip.GzipFile(fileobj=f) as infile:
    #         liucun_rate = np.load(infile, allow_pickle=False)
    #         print(liucun_rate)


if __name__=='__main__':
    args = parser.parse_args()
    main(args)