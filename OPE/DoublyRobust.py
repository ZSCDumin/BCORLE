# coding: utf-8
import numpy as np
import tensorflow as tf


def DoublyRobust(replay_buffer, total_importance_product, predict_value_DM, number_users, Lambda_size, gamma, timesteps):
    predict_value = np.zeros([Lambda_size, number_users])
    index = 0
    for l in range(Lambda_size):
        for u in range(number_users):
            predict_reward = 0
            discount = 1
            timestep = 0
            for t in range(timesteps):
                if t == timestep:
                    r = replay_buffer.new_batch_data["reward2"][index][0]
                    if replay_buffer.new_batch_data["done"][index][0] != 1.:
                        timestep += 1
                    index += 1
                else:
                    r = 0
                predict_reward += discount * total_importance_product[l,u,t] * (r - predict_value_DM[l,u,t] + gamma * predict_value_DM[l,u,min(t+1, timesteps-1)])
                discount *= gamma
                if index == len(replay_buffer.new_batch_data["done"]):
                    break
            predict_value[l,u] = predict_value_DM[l,u,0] + predict_reward
            if index == len(replay_buffer.new_batch_data["done"]):
                break
        if index == len(replay_buffer.new_batch_data["done"]):
            break
    return predict_value