# coding: utf-8

import numpy as np
import pandas as pd
from utils.SumTree import SumTree
from utils.DataProcess_utils import *
import tensorflow as tf
import copy
import os
import gzip

# Generic replay buffer for standard gym tasks
class StandardBuffer(object):
	e = 0.01
	a = 0.6
	beta = 0.4
	beta_increment_per_sampling = 0.001

	def __init__(
			self, state_dim, action_dim, batch_size,
			buffer_size, tables, selected_cols, discount, Lambda_min, Lambda_max, Lambda_interval,
			n_step=1, test_mode=False, is_prioritized_replay_buffer=True,
			slice_id=0, slice_count=1
	):
		# create an offline_env to do fake interaction with agent
		self.num_epoch = 0
		self.num_record = 0

		# how many records to read from table at one time
		self.batch_size = batch_size
		self.max_size = int(buffer_size + 1)

		self.ptr = 0
		self.crt_size = 0
		self.is_priority = is_prioritized_replay_buffer

		# action dimension
		self.action_dim = action_dim

		# Lambda
		self.Lambda_min = Lambda_min
		self.Lambda_max = Lambda_max
		self.Lambda_interval = Lambda_interval
		self.discount = discount

		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, 1))
		self.next_state = np.array(self.state)
		self.reward = np.zeros((self.max_size, 1))
		self.reward1 = np.zeros((self.max_size, 1))
		self.reward2 = np.zeros((self.max_size, 1))
		self.done = np.zeros((self.max_size, 1))
		self.Lambda = np.zeros((self.max_size, 1))
		self.ActionEmbedding = ActionEmbedding(self.action_dim)

		self.tree = SumTree(self.max_size)

		self.test_mode = test_mode
		self.slice_id = slice_id
		self.slice_count = slice_count
		if not self.test_mode:  # run in PAI
			# number of step to reserved for n_step dqn
			self.n_step = n_step
			self.tables = tables
			self.selected_cols = selected_cols
			self.table_reader = tf.python_io.TableReader(
				table=self.tables,
				selected_cols=self.selected_cols,
				slice_id=self.slice_id,
				slice_count=self.slice_count
			)
			self.row_count = self.table_reader.get_row_count()
		else:
			self.num_iter = 0  # for sampling sequentially

		self.new_batch_data = dict(
			state=None,
			action=None,
			reward=None,
			done=None,
			next_state=None)

		self.replay = dict(
			state=None,
			action=None,
			reward=None,
			done=None,
			next_state=None)

		self.new_batch_data2 = dict(
			state=None,
			action=None,
			reward=None,
			done=None,
			next_state=None)

		self.state_max = None
		self.state_min = None
		self.state_max_minus_min = None
		self.load_regularizer()

	def load_regularizer(self):
		self.state_max = np.load("./utils/state_max.npy")
		self.state_min = np.load("./utils/state_min.npy")
		self.state_max_minus_min = np.load("./utils/state_max_minus_min.npy")
		self.state_max_minus_min = np.array([max(length, 1.0) for length in self.state_max_minus_min])

	def _get_priority(self, error):
		return (np.abs(error) + self.e) ** self.a

	def reset(self):
		self.new_batch_data = dict(
			state=None,
			actions=None,
			rewards=None,
			not_dones=None,
			next_state=None)

	def reset_table_reader(self, table):
		self.table_reader = tf.python_io.TableReader(
			table=table,
			selected_cols=self.selected_cols,
			slice_id=self.slice_id,
			slice_count=self.slice_count
		)
		self.row_count = self.table_reader.get_row_count()

	def update(self, idx, error):
		# error: TD error
		# add the tuples in priority
		p = self._get_priority(error)
		self.tree.update(idx, p)

	def add(self, error, tuples):
		# error: TD error
		# tuples: (state, action, reward, next_state, done, reward1, reward2)
		# add the tuples in priority
		p = self._get_priority(error)
		for i in range(len(error)):
			self.tree.add(
				p[i], (
					tuples[0][i], tuples[1][i], tuples[2][i], tuples[3][i], tuples[4][i], tuples[5][i], tuples[6][i], tuples[7][i]
				)
			)

	def add_without_priority(self, s, a, s_, r1, r2, r, d, l):
		self.state[self.ptr] = s
		self.action[self.ptr] = a
		self.next_state[self.ptr] = s_
		self.reward1[self.ptr] = r1
		self.reward2[self.ptr] = r2
		self.reward[self.ptr] = r
		self.done[self.ptr] = d
		self.Lambda[self.ptr] = l

		self.ptr = (self.ptr + 1) % self.max_size
		self.crt_size = min(self.crt_size + 1, self.max_size)

	def get_batch_data_from_eval_odps(self, is_evaluation, pre_train_batch_size=None):
		table_reader = self.table_reader
		row_count = table_reader.get_row_count()
		self.tuple_data = table_reader.read(row_count)
		self.parse_tuple_data(self.tuple_data, is_evaluation, with_appendix=True, pre_train_batch_size=pre_train_batch_size)

	def get_batch_data_from_odps(self, is_evaluation):
		table_reader = self.table_reader
		row_count = table_reader.get_row_count()
		self.tuple_data = table_reader.read(row_count)
		# self.parse_tuple_data(self.tuple_data, is_evaluation=is_evaluation, with_appendix=True,
		# 					  pre_train_batch_size=pre_train_batch_size)

		user_id_str, state_str, action_str, reward_liucun_str, next_state_str, terminal_str, \
		liucun_rate_str = zip(*self.tuple_data)

		lambda_size = (self.Lambda_max - self.Lambda_min) / self.Lambda_interval + 1

		reward_hongbao_str = copy.deepcopy(action_str)

		# transform str to list
		user_id = []
		state = []
		action = []
		reward_liucun = []
		reward_hongbao = []
		next_state = []
		terminal = []
		liucun_rate = []
		Lambda = []  # save the lambda

		for i in range(len(user_id_str)):
			user_id.append(int(user_id_str[i]))
			state.append(list(eval(state_str[i].decode().replace(" ", ","))))
			action.append(float(action_str[i]))
			reward_liucun.append(float(reward_liucun_str[i]))
			reward_hongbao.append(float(reward_hongbao_str[i]))
			next_state.append(list(eval(next_state_str[i].decode().replace(" ", ","))))
			terminal.append(float(terminal_str[i]))
			liucun_rate.append(list(eval(liucun_rate_str[i].decode().replace(" ", ","))))
			Lambda.append(0)

		# print('read data finished')

		# Extend data based on lambda
		user_id = user_id * int(lambda_size)
		state = state * int(lambda_size)
		action = action * int(lambda_size)
		reward_liucun = reward_liucun * int(lambda_size)
		reward_hongbao = reward_hongbao * int(lambda_size)
		next_state = next_state * int(lambda_size)
		terminal = terminal * int(lambda_size)
		Lambda = Lambda * int(lambda_size)

		reward = copy.deepcopy(reward_liucun)
		Q_value = copy.deepcopy(reward_liucun)

		# print('extend Lambda finished')
		for i in range(len(user_id)):
			Lambda[i] = self.Lambda_min + (i // len(user_id_str)) * self.Lambda_interval
			reward[i] = reward_liucun[i] - Lambda[i] * reward_hongbao[i]
			index = i
			sum_Q_value = 0
			discount_ = 1
			while terminal[index] != 1.:
				sum_Q_value += reward_hongbao[index] * discount_
				discount_ *= self.discount
				index += 1
				if index == len(user_id):
					break
			if index == len(user_id):
				break
			sum_Q_value += reward_hongbao[index] * discount_
			Q_value[i] = sum_Q_value

		# print('extend Q_value finished')

		if is_evaluation:

			# shuffled_indices = np.random.permutation(len(user_id))
			user_id = np.array(user_id)
			# ds = ds[shuffled_indices]
			# step_in_episode = step_in_episode[shuffled_indices]
			state = np.array(state)
			action = np.array(action)
			reward_liucun = np.array(reward_liucun)
			reward_hongbao = np.array(reward_hongbao)
			next_state = np.array(next_state)
			terminal = np.array(terminal)
			reward = np.array(reward)
			Lambda = np.array(Lambda)
			Q_value = np.array(Q_value)

		else:
			shuffled_indices = np.random.permutation(len(user_id))
			user_id = np.array(user_id)[shuffled_indices]
			# ds = ds[shuffled_indices]
			# step_in_episode = step_in_episode[shuffled_indices]
			state = np.array(state)[shuffled_indices]
			action = np.array(action)[shuffled_indices]
			reward_liucun = np.array(reward_liucun)[shuffled_indices]
			reward_hongbao = np.array(reward_hongbao)[shuffled_indices]
			next_state = np.array(next_state)[shuffled_indices]
			terminal = np.array(terminal)[shuffled_indices]
			reward = np.array(reward)[shuffled_indices]
			Lambda = np.array(Lambda)[shuffled_indices]
			Q_value = np.array(Q_value)[shuffled_indices]

		# self.num_iter += 1

		self.user_id = np.expand_dims(user_id, axis=1)
		# step_in_episode = np.expand_dims(step_in_episode, axis=1)

		self.action_one_hot = self.ActionEmbedding.action_embedding(action.astype(np.float32)).astype(np.int32)

		state_list = list(state)  # transfer to list
		# from sting to numpy array for each data
		state = [np.expand_dims(state_list[i], axis=1) for i in range(len(state_list))]

		next_state_list = list(next_state)  # transfer to list

		next_state = [np.expand_dims(next_state_list[i], axis=1) for i in range(len(next_state_list))]

		self.state = np.concatenate(state, axis=1).T  # transfer to numpy array
		self.next_state = np.concatenate(next_state, axis=1).T

		# whether come next day
		self.reward_liucun = np.expand_dims(reward_liucun, axis=1).astype(np.float32)  # reshape
		self.reward_hongbao = np.expand_dims(reward_hongbao, axis=1).astype(np.float32)
		self.reward = np.expand_dims(reward, axis=1).astype(np.float32)
		self.Lambda = np.expand_dims(Lambda, axis=1).astype(np.float32)
		self.Q_value = np.expand_dims(Q_value, axis=1).astype(np.float32)
		# bonus amount

		# reward_hongbao = self.Lambda * np.expand_dims(-reward2, axis=1).astype(np.float32)  # reshape
		# reward = reward_liucun + reward_hongbao
		# terminal = [0 if t in ['False', 'false', 'FALSE', 0, '0'] else 1 for t in terminal]
		self.done = np.expand_dims(terminal, axis=1).astype(np.float32)  # reward
		self.new_batch_data2["user_id"] = self.user_id
		# self.new_batch_data["time_id"] = step_in_episode[:self.max_size]
		self.new_batch_data2["state"] = self.state
		self.new_batch_data2["action"] = self.action_one_hot
		self.new_batch_data2["reward1"] = self.reward_liucun
		self.new_batch_data2["reward2"] = self.reward_hongbao
		self.new_batch_data2["reward"] = self.reward
		self.new_batch_data2["done"] = self.done
		self.new_batch_data2["next_state"] = self.next_state
		self.new_batch_data2["Lambda"] = self.Lambda
		self.new_batch_data2["Q_value"] = self.Q_value

	# read_batch_size = (self.batch_size + self.n_step - 1) if not pre_train_batch_size \
		# 	else (pre_train_batch_size + self.n_step - 1)
		# try:
		# 	self.tuple_data = table_reader.read(read_batch_size)
		#
		# 	if len(self.tuple_data) < read_batch_size:
		# 		# reach the end of data
		# 		self.num_epoch += 1
		# 		table_reader.close()
		# 		table_reader = tf.python_io.TableReader(
		# 			table=self.tables,
		# 			selected_cols=self.selected_cols,
		# 			slice_id=0,
		# 			slice_count=1
		# 		)
		# 		self.tuple_data.extend(table_reader.read(read_batch_size - len(self.tuple_data)))
		#
		# except tf.errors.OutOfRangeError:
		# 	# reach the end of data
		# 	self.num_epoch += 1
		# 	table_reader.close()
		# 	table_reader = tf.python_io.TableReader(
		# 		table=self.tables,
		# 		selected_cols=self.selected_cols,
		# 		slice_id=0,
		# 		slice_count=1
		# 	)
		# 	self.tuple_data = table_reader.read(read_batch_size)
		# self.num_record += self.batch_size

	def get_train_data(self,learner):

		rows = np.random.choice(self.user_id.shape[0],self.max_size)
		self.new_batch_data["user_id"] = self.user_id[rows,:]
		# self.new_batch_data["time_id"] = step_in_episode[:self.max_size]
		self.new_batch_data["state"] = self.state[rows,:]
		self.new_batch_data["action"] = self.action_one_hot[rows,:]
		self.new_batch_data["reward1"] = self.reward_liucun[rows,:]
		self.new_batch_data["reward2"] = self.reward_hongbao[rows,:]
		self.new_batch_data["reward"] = self.reward[rows,:]
		self.new_batch_data["done"] = self.done[rows,:]
		self.new_batch_data["next_state"] = self.next_state[rows,:]
		self.new_batch_data["Lambda"] = self.Lambda[rows,:]
		self.new_batch_data["Q_value"] = self.Q_value[rows,:]




	def get_batch_data_from_odps_predict(self):
		table_reader = self.table_reader
		try:
			tuple_data = table_reader.read(self.batch_size)
		except tf.errors.OutOfRangeError:
			# reach the end of data
			return True
		self.num_record += len(tuple_data)
		# assert len(tuple_data) == self.batch_size + self.n_step - 1

		user_id, state = zip(*tuple_data)
		self.new_batch_data["state_origin"] = state

		# Adjust crt_size if we're using a custom size
		user_id = np.expand_dims(user_id, axis=1)

		state_list = list(state)  # transfer to list
		# from sting to numpy array for each data
		state = [np.expand_dims(state_list[i].split(','), axis=1) for i in range(len(state_list))]

		state = np.concatenate(state, axis=1).T  # transfer to numpy array
		state_one_hot = []
		for i in range(len(state[0])):
			# one_hot_data = embedding_state(state_dict[i], state[:, i])
			one_hot_data = np.expand_dims(state[:, i], axis=1)
			if one_hot_data is not None:
				state_one_hot.append(one_hot_data)
		state_one_hot = np.concatenate(state_one_hot, axis=1).astype(np.float32)
		self.whether_double = state_one_hot[:, 0]
		state_one_hot = (state_one_hot - self.state_min) / self.state_max_minus_min

		self.new_batch_data["user_id"] = user_id
		self.new_batch_data["state"] = state_one_hot

		return False

	def get_batch_data_from_local_eval_npz(self, table=None):
		# Load replay buffer from local npz
		data = np.load('D:/code/simulation_CMDP/data/eval_trajectory_user.npz')
		user_id_str = data['user_ID']
		state_str = data['state']
		action_str = data['action']
		reward_liucun_str = data['reward_liucun']
		reward_hongbao_str = copy.deepcopy(action_str)
		next_state_str = data['next_state']
		terminal_str = data['terminal']
		liucun_rate_str = data['liucun_rate']

		# if (self.num_iter + 1) * self.batch_size > len(user_id_str):
		# 	self.num_iter = 0
		# start = self.num_iter * self.batch_size
		# end = (self.num_iter + 1) * self.batch_size

		lambda_size = (self.Lambda_max - self.Lambda_min) / self.Lambda_interval + 1

		#transform str to list
		user_id = []
		state = []
		action = []
		reward_liucun = []
		reward_hongbao = []
		next_state = []
		terminal = []
		liucun_rate = []
		Lambda = [] #save the lambda

		for i in range(len(user_id_str)):
			user_id.append(user_id_str[i])
			state.append(list(eval(state_str[i].replace(" ", ","))))
			action.append(float(action_str[i]))
			reward_liucun.append(float(reward_liucun_str[i]))
			reward_hongbao.append(float(reward_hongbao_str[i]))
			next_state.append(list(eval(next_state_str[i].replace(" ", ","))))
			terminal.append(float(terminal_str[i]))
			liucun_rate.append(list(eval(liucun_rate_str[i].replace(" ", ","))))
			Lambda.append(0)

		#Extend data based on lambda
		user_id = user_id * int(lambda_size)
		state = state * int(lambda_size)
		action = action * int(lambda_size)
		reward_liucun = reward_liucun * int(lambda_size)
		reward_hongbao = reward_hongbao * int(lambda_size)
		next_state = next_state * int(lambda_size)
		terminal = terminal * int(lambda_size)
		Lambda = Lambda * int(lambda_size)

		reward = copy.deepcopy(reward_liucun)
		for i in range(len(user_id)):
			Lambda[i] = self.Lambda_min + (i // len(user_id_str)) * self.Lambda_interval
			reward[i] = reward_liucun[i] - Lambda[i] * reward_hongbao[i]

		user_id = np.expand_dims(user_id, axis=1)

		action_one_hot = self.ActionEmbedding.action_embedding(np.array(action).astype(np.float32)).astype(np.int32)

		state_list = list(state)  # transfer to list
		state = [np.expand_dims(state_list[i], axis=1) for i in range(len(state_list))]

		next_state_list = list(next_state)  # transfer to list
		next_state = [np.expand_dims(next_state_list[i], axis=1) for i in range(len(next_state_list))]

		state = np.concatenate(state, axis=1).T  # transfer to numpy array
		next_state = np.concatenate(next_state, axis=1).T

		# whether come next day
		reward_liucun = np.expand_dims(reward_liucun, axis=1).astype(np.float32)  # reshape
		reward_hongbao = np.expand_dims(reward_hongbao, axis=1).astype(np.float32)
		reward = np.expand_dims(reward, axis=1).astype(np.float32)
		Lambda = np.expand_dims(Lambda, axis=1).astype(np.float32)
		# bonus amount
		done = np.expand_dims(terminal, axis=1).astype(np.float32)  # reward

		self.new_batch_data["user_id"] = user_id
		self.new_batch_data["state"] = state
		self.new_batch_data["action"] = action_one_hot
		self.new_batch_data["reward1"] = reward_liucun
		self.new_batch_data["reward2"] = reward_hongbao
		self.new_batch_data["reward"] = reward
		self.new_batch_data["done"] = done
		self.new_batch_data["next_state"] = next_state
		self.new_batch_data["Lambda"] = Lambda

	def get_batch_data_from_d2_eval_npz(self, eval_data_dir, table=None):
		gfile = tf.gfile
		user_ID_file_name = os.path.join(eval_data_dir, 'user_ID')
		with gfile.GFile(user_ID_file_name, 'rb') as f:
			with gzip.GzipFile(fileobj=f) as outfile:
				user_id_str = np.load(outfile,  allow_pickle=False)
		state_file_name = os.path.join(eval_data_dir, 'state')
		with gfile.GFile(state_file_name, 'rb') as f:
			with gzip.GzipFile(fileobj=f) as outfile:
				state_str = np.load(outfile,  allow_pickle=False)
		action_file_name = os.path.join(eval_data_dir, 'action')
		with gfile.GFile(action_file_name, 'rb') as f:
			with gzip.GzipFile(fileobj=f) as outfile:
				action_str = np.load(outfile,  allow_pickle=False)
		reward_liucun_file_name = os.path.join(eval_data_dir, 'reward_liucun')
		with gfile.GFile(reward_liucun_file_name, 'rb') as f:
			with gzip.GzipFile(fileobj=f) as outfile:
				reward_liucun_str = np.load(outfile,  allow_pickle=False)
		next_state_file_name = os.path.join(eval_data_dir, 'next_state')
		with gfile.GFile(next_state_file_name, 'rb') as f:
			with gzip.GzipFile(fileobj=f) as outfile:
				next_state_str = np.load(outfile,  allow_pickle=False)
		terminal_file_name = os.path.join(eval_data_dir, 'terminal')
		with gfile.GFile(terminal_file_name, 'rb') as f:
			with gzip.GzipFile(fileobj=f) as outfile:
				terminal_str = np.load(outfile, allow_pickle=False)
		liucun_rate_file_name = os.path.join(eval_data_dir, 'liucun_rate')
		with gfile.GFile(liucun_rate_file_name, 'rb') as f:
			with gzip.GzipFile(fileobj=f) as outfile:
				liucun_rate_str = np.load(outfile, allow_pickle=False)

		reward_hongbao_str = copy.deepcopy(action_str)

		lambda_size = (self.Lambda_max - self.Lambda_min) / self.Lambda_interval + 1

		#transform str to list
		user_id = []
		state = []
		action = []
		reward_liucun = []
		reward_hongbao = []
		next_state = []
		terminal = []
		liucun_rate = []
		Lambda = [] #save the lambda

		for i in range(len(user_id_str)):
			user_id.append(user_id_str[i])
			state.append(list(eval(state_str[i].replace(" ", ","))))
			action.append(float(action_str[i]))
			reward_liucun.append(float(reward_liucun_str[i]))
			reward_hongbao.append(float(reward_hongbao_str[i]))
			next_state.append(list(eval(next_state_str[i].replace(" ", ","))))
			terminal.append(float(terminal_str[i]))
			liucun_rate.append(list(eval(liucun_rate_str[i].replace(" ", ","))))
			Lambda.append(0)

		#Extend data based on lambda
		user_id = user_id * int(lambda_size)
		state = state * int(lambda_size)
		action = action * int(lambda_size)
		reward_liucun = reward_liucun * int(lambda_size)
		reward_hongbao = reward_hongbao * int(lambda_size)
		next_state = next_state * int(lambda_size)
		terminal = terminal * int(lambda_size)
		Lambda = Lambda * int(lambda_size)

		reward = copy.deepcopy(reward_liucun)
		for i in range(len(user_id)):
			Lambda[i] = self.Lambda_min + (i // len(user_id_str)) * self.Lambda_interval
			reward[i] = reward_liucun[i] - Lambda[i] * reward_hongbao[i]

		user_id = np.expand_dims(user_id, axis=1)

		action_one_hot = self.ActionEmbedding.action_embedding(np.array(action).astype(np.float32)).astype(np.int32)

		state_list = list(state)  # transfer to list
		state = [np.expand_dims(state_list[i], axis=1) for i in range(len(state_list))]

		next_state_list = list(next_state)  # transfer to list
		next_state = [np.expand_dims(next_state_list[i], axis=1) for i in range(len(next_state_list))]

		state = np.concatenate(state, axis=1).T  # transfer to numpy array
		next_state = np.concatenate(next_state, axis=1).T

		# whether come next day
		reward_liucun = np.expand_dims(reward_liucun, axis=1).astype(np.float32)  # reshape
		reward_hongbao = np.expand_dims(reward_hongbao, axis=1).astype(np.float32)
		reward = np.expand_dims(reward, axis=1).astype(np.float32)
		Lambda = np.expand_dims(Lambda, axis=1).astype(np.float32)
		# bonus amount
		done = np.expand_dims(terminal, axis=1).astype(np.float32)  # reward

		self.new_batch_data["user_id"] = user_id
		self.new_batch_data["state"] = state
		self.new_batch_data["action"] = action_one_hot
		self.new_batch_data["reward1"] = reward_liucun
		self.new_batch_data["reward2"] = reward_hongbao
		self.new_batch_data["reward"] = reward
		self.new_batch_data["done"] = done
		self.new_batch_data["next_state"] = next_state
		self.new_batch_data["Lambda"] = Lambda

	def get_batch_data_from_local_npz(self, learner, table=None):
		# Load replay buffer from local npz
		data = np.load('G:\code\simulation_CMDP2\data/trajectory_user.npz')
		user_id_str = data['user_ID']
		state_str = data['state']
		action_str = data['action']
		reward_liucun_str = data['reward_liucun']
		reward_hongbao_str = copy.deepcopy(action_str)
		next_state_str = data['next_state']
		terminal_str = data['terminal']
		liucun_rate_str = data['liucun_rate']

		lambda_size = (self.Lambda_max - self.Lambda_min) / self.Lambda_interval + 1

		#transform str to list
		user_id = []
		state = []
		action = []
		reward_liucun = []
		reward_hongbao = []
		next_state = []
		terminal = []
		liucun_rate = []
		Lambda = [] #save the lambda

		for i in range(len(user_id_str)):
			user_id.append(user_id_str[i])
			state.append(list(eval(state_str[i].replace(" ", ","))))
			action.append(eval(action_str[i]))
			reward_liucun.append(eval(reward_liucun_str[i]))
			reward_hongbao.append(eval(reward_hongbao_str[i]))
			next_state.append(list(eval(next_state_str[i].replace(" ", ","))))
			terminal.append(eval(terminal_str[i]))
			liucun_rate.append(list(eval(liucun_rate_str[i].replace(" ", ","))))
			Lambda.append(0)

		#Extend data based on lambda
		user_id = user_id * int(lambda_size)
		state = state * int(lambda_size)
		action = action * int(lambda_size)
		reward_liucun = reward_liucun * int(lambda_size)
		reward_hongbao = reward_hongbao * int(lambda_size)
		next_state = next_state * int(lambda_size)
		terminal = terminal * int(lambda_size)
		Lambda = Lambda * int(lambda_size)

		reward = copy.deepcopy(reward_liucun)
		Q_value = copy.deepcopy(reward_liucun)
		for i in range(len(user_id)):
			Lambda[i] = self.Lambda_min + (i // len(user_id_str)) * self.Lambda_interval
			reward[i] = reward_liucun[i] - Lambda[i] * reward_hongbao[i]
			index = i
			sum_Q_value = 0
			discount_ = 1
			while terminal[index] != 1:
				sum_Q_value += reward_hongbao[index] * discount_
				discount_ *= self.discount
				index += 1
			sum_Q_value += reward_hongbao[index] * discount_
			Q_value[i] = sum_Q_value


		shuffled_indices = np.random.permutation(len(user_id))
		user_id = np.array(user_id)[shuffled_indices]
		#ds = ds[shuffled_indices]
		#step_in_episode = step_in_episode[shuffled_indices]
		state = np.array(state)[shuffled_indices]
		action = np.array(action)[shuffled_indices]
		reward_liucun = np.array(reward_liucun)[shuffled_indices]
		reward_hongbao = np.array(reward_hongbao)[shuffled_indices]
		next_state = np.array(next_state)[shuffled_indices]
		terminal = np.array(terminal)[shuffled_indices]
		reward = np.array(reward)[shuffled_indices]
		Lambda = np.array(Lambda)[shuffled_indices]
		Q_value = np.array(Q_value)[shuffled_indices]

		state_max = np.max(state.astype(np.float32), axis=0)
		state_min = np.min(state.astype(np.float32), axis=0)
		state_max_minus_min = state_max - state_min

		self.num_iter += 1

		self.user_id = np.expand_dims(user_id, axis=1)
		#step_in_episode = np.expand_dims(step_in_episode, axis=1)

		self.action_one_hot = self.ActionEmbedding.action_embedding(action.astype(np.float32)).astype(np.int32)

		state_list = list(state)  # transfer to list
		# from sting to numpy array for each data
		state = [np.expand_dims(state_list[i], axis=1) for i in range(len(state_list))]

		next_state_list = list(next_state)  # transfer to list
		# for i in range(len(next_state_list)):
		# 	if not isinstance(next_state_list[i], str) or next_state_list[i] is None or len(next_state_list[i]) == 1\
		# 			or next_state_list[i] == '\\N':
		# 		next_state_list[i] = state_list[i]
		# from sting to numpy array for each data
		next_state = [np.expand_dims(next_state_list[i], axis=1) for i in range(len(next_state_list))]

		self.state = np.concatenate(state, axis=1).T  # transfer to numpy array
		self.next_state = np.concatenate(next_state, axis=1).T

		# whether come next day
		self.reward_liucun = np.expand_dims(reward_liucun, axis=1).astype(np.float32)  # reshape
		self.reward_hongbao = np.expand_dims(reward_hongbao, axis=1).astype(np.float32)
		self.reward = np.expand_dims(reward, axis=1).astype(np.float32)
		self.Lambda = np.expand_dims(Lambda, axis=1).astype(np.float32)
		self.Q_value = np.expand_dims(Q_value, axis=1).astype(np.float32)
		# bonus amount

		#reward_hongbao = self.Lambda * np.expand_dims(-reward2, axis=1).astype(np.float32)  # reshape
		#reward = reward_liucun + reward_hongbao
		#terminal = [0 if t in ['False', 'false', 'FALSE', 0, '0'] else 1 for t in terminal]
		self.done = np.expand_dims(terminal, axis=1).astype(np.float32)  # reward

		if learner == 'MOPO':
			self.replay["user_id"] = self.user_id
			self.replay["state"] = self.state
			self.replay["action"] = self.action_one_hot
			self.replay["reward1"] = self.reward_liucun
			self.replay["reward2"] = self.reward_hongbao
			self.replay["reward"] = self.reward
			self.replay["done"] = self.done
			self.replay["next_state"] = self.next_state
			self.replay["Lambda"] = self.Lambda
			self.replay["Q_value"] = self.Q_value


	def get_batch_local_data(self):

		rows = np.random.choice(self.user_id.shape[0], self.max_size)
		self.new_batch_data["user_id"] = self.user_id[rows, :]
		# self.new_batch_data["time_id"] = step_in_episode[:self.max_size]
		self.new_batch_data["state"] = self.state[rows, :]
		self.new_batch_data["action"] = self.action_one_hot[rows, :]
		self.new_batch_data["reward1"] = self.reward_liucun[rows, :]
		self.new_batch_data["reward2"] = self.reward_hongbao[rows, :]
		self.new_batch_data["reward"] = self.reward[rows, :]
		self.new_batch_data["done"] = self.done[rows, :]
		self.new_batch_data["next_state"] = self.next_state[rows, :]
		self.new_batch_data["Lambda"] = self.Lambda[rows, :]
		self.new_batch_data["Q_value"] = self.Q_value[rows, :]

	def get_batch_data_from_excel(self, table=None):
		# Load replay buffer from local excel

		user_id, ds, step_in_episode, state, action, reward1, reward2,\
		next_state, terminal = read_data_excel(
			'buffers/hongbao_data.xlsx' if not table else table
		)

		# replace reward1 with the liucun_rate derived by action:

		if (self.num_iter + 1) * self.batch_size > len(user_id):
			self.num_iter = 0
		start = self.num_iter * self.batch_size
		end = (self.num_iter + 1) * self.batch_size

		shuffled_indices = np.random.permutation(len(user_id))
		user_id = user_id[shuffled_indices]
		ds = ds[shuffled_indices]
		step_in_episode = step_in_episode[shuffled_indices]
		state = state[shuffled_indices]
		action = action[shuffled_indices]
		reward1 = reward1[shuffled_indices]
		reward2 = reward2[shuffled_indices]
		next_state = next_state[shuffled_indices]
		terminal = terminal[shuffled_indices]

		user_id = user_id[start:end]
		ds = ds[start:end]
		step_in_episode = step_in_episode[start:end]
		state = state[start:end]
		action = action[start:end]
		reward1 = reward1[start:end]
		reward2 = reward2[start:end]
		next_state = next_state[start:end]
		terminal = terminal[start:end]

		self.num_iter += 1

		user_id = np.expand_dims(user_id, axis=1)
		step_in_episode = np.expand_dims(step_in_episode, axis=1)

		action_one_hot = self.ActionEmbedding.action_embedding(action.astype(np.float32)).astype(np.int32)

		state_list = list(state)  # transfer to list
		# from sting to numpy array for each data
		state = [np.expand_dims(state_list[i].split(','), axis=1) for i in range(len(state_list))]

		next_state_list = list(next_state)  # transfer to list
		for i in range(len(next_state_list)):
			if not isinstance(next_state_list[i], str) or next_state_list[i] is None or len(next_state_list[i]) == 1\
					or next_state_list[i] == '\\N':
				next_state_list[i] = state_list[i]
		# from sting to numpy array for each data
		next_state = [np.expand_dims(next_state_list[i].split(','), axis=1) for i in range(len(next_state_list))]

		state = np.concatenate(state, axis=1).T  # transfer to numpy array

		# state_max = np.max(state.astype(np.float32), axis=0)
		# state_min = np.min(state.astype(np.float32), axis=0)
		# state_max_minus_min = state_max - state_min
		# np.save("./utils/state_max.npy", state_max)
		# np.save("./utils/state_min.npy", state_min)
		# np.save("./utils/state_max_minus_min.npy", state_max_minus_min)

		state_one_hot = []
		for i in range(len(state[0])):
			# one_hot_data = embedding_state(state_dict[i], state[:, i])
			one_hot_data = np.expand_dims(state[:, i], axis=1)
			if one_hot_data is not None:
				state_one_hot.append(one_hot_data)
		state_one_hot = np.concatenate(state_one_hot, axis=1).astype(np.float32)
		state_one_hot = (state_one_hot - self.state_min) / self.state_max_minus_min

		next_state = np.concatenate(next_state, axis=1).T  # transfer to numpy array

		next_state_one_hot = []
		for i in range(len(next_state[0])):
			# next_one_hot_data = embedding_state(state_dict[i], next_state[:, i])
			next_one_hot_data = np.expand_dims(next_state[:, i], axis=1)
			if next_one_hot_data is not None:
				next_state_one_hot.append(next_one_hot_data)
		next_state_one_hot = np.concatenate(next_state_one_hot, axis=1).astype(np.float32)
		next_state_one_hot = (next_state_one_hot - self.state_min) / self.state_max_minus_min

		# whether come next day
		reward1 = np.expand_dims(reward1, axis=1).astype(np.float32)  # reshape
		# bonus amount
		reward2 = self.Lambda * np.expand_dims(-reward2, axis=1).astype(np.float32)  # reshape
		reward = reward1 + reward2
		terminal = [0 if t in ['False', 'false', 'FALSE', 0, '0'] else 1 for t in terminal]
		done = np.expand_dims(terminal, axis=1).astype(np.float32)  # reward

		self.new_batch_data["user_id"] = user_id[:self.max_size]
		self.new_batch_data["time_id"] = step_in_episode[:self.max_size]
		self.new_batch_data["state"] = state_one_hot[:self.max_size]
		self.new_batch_data["action"] = action_one_hot[:self.max_size]
		self.new_batch_data["reward1"] = reward1[:self.max_size]
		self.new_batch_data["reward2"] = reward2[:self.max_size]
		self.new_batch_data["reward"] = reward[:self.max_size]
		self.new_batch_data["done"] = done[:self.max_size]
		self.new_batch_data["next_state"] = next_state_one_hot[:self.max_size]


	def get_batch_data_from_excel_predict(self):
		# Load replay buffer from local excel

		batch_values = pd.read_excel('buffers/hongbao_data_predict.xlsx', header=0).to_numpy()
		user_id = batch_values[:, 0]
		state = batch_values[:, 1]

		self.new_batch_data["state_origin"] = state

		# Adjust crt_size if we're using a custom size
		user_id = np.expand_dims(user_id, axis=1)

		state_list = list(state)  # transfer to list
		# from sting to numpy array for each data
		state = [np.expand_dims(state_list[i].split(','), axis=1) for i in range(len(state_list))]
		state = np.concatenate(state, axis=1).T  # transfer to numpy array
		state_one_hot = []
		for i in range(len(state[0])):
			# one_hot_data = embedding_state(state_dict[i], state[:, i])
			one_hot_data = np.expand_dims(state[:, i], axis=1)
			if one_hot_data is not None:
				state_one_hot.append(one_hot_data)
		state_one_hot = np.concatenate(state_one_hot, axis=1).astype(np.float32)
		self.whether_double = state_one_hot[:, 0]
		state_one_hot = (state_one_hot - self.state_min) / self.state_max_minus_min

		self.new_batch_data["user_id"] = user_id
		self.new_batch_data["state"] = state_one_hot

		return True

	def parse_tuple_data(self, tuple_data, is_evaluation = False, with_appendix=False, pre_train_batch_size=None):
		# assert len(tuple_data) == self.batch_size + self.n_step - 1

		user_id_str, state_str, action_str, reward_liucun_str, next_state_str, terminal_str,\
		liucun_rate_str = zip(*tuple_data)
		# else:
		# 	user_id = ds = None
		# 	step_in_episode, state, action, reward1, reward2, \
		# 	next_state, terminal = zip(*tuple_data)

		# if (self.num_iter + 1) * self.batch_size > len(user_id_str):
		# 	self.num_iter = 0
		# start = self.num_iter * self.batch_size
		# end = (self.num_iter + 1) * self.batch_size

		lambda_size = (self.Lambda_max - self.Lambda_min) / self.Lambda_interval + 1

		reward_hongbao_str = copy.deepcopy(action_str)

		#transform str to list
		user_id = []
		state = []
		action = []
		reward_liucun = []
		reward_hongbao = []
		next_state = []
		terminal = []
		liucun_rate = []
		Lambda = [] #save the lambda

		for i in range(len(user_id_str)):
			user_id.append(int(user_id_str[i]))
			state.append(list(eval(state_str[i].decode().replace(" ", ","))))
			action.append(float(action_str[i]))
			reward_liucun.append(float(reward_liucun_str[i]))
			reward_hongbao.append(float(reward_hongbao_str[i]))
			next_state.append(list(eval(next_state_str[i].decode().replace(" ", ","))))
			terminal.append(float(terminal_str[i]))
			liucun_rate.append(list(eval(liucun_rate_str[i].decode().replace(" ", ","))))
			Lambda.append(0)



		# print('read data finished')

		#Extend data based on lambda
		user_id = user_id * int(lambda_size)
		state = state * int(lambda_size)
		action = action * int(lambda_size)
		reward_liucun = reward_liucun * int(lambda_size)
		reward_hongbao = reward_hongbao * int(lambda_size)
		next_state = next_state * int(lambda_size)
		terminal = terminal * int(lambda_size)
		Lambda = Lambda * int(lambda_size)

		reward = copy.deepcopy(reward_liucun)
		Q_value = copy.deepcopy(reward_liucun)

		# print('extend Lambda finished')
		for i in range(len(user_id)):
			Lambda[i] = self.Lambda_min + (i // len(user_id_str)) * self.Lambda_interval
			reward[i] = reward_liucun[i] - Lambda[i] * reward_hongbao[i]
			index = i
			sum_Q_value = 0
			discount_ = 1
			while terminal[index] != 1.:
				sum_Q_value += reward_hongbao[index] * discount_
				discount_ *= self.discount
				index += 1
				if index == len(user_id):
					break
			if index == len(user_id):
				break
			sum_Q_value += reward_hongbao[index] * discount_
			Q_value[i] = sum_Q_value

		#print('extend Q_value finished')

		if is_evaluation:

			#shuffled_indices = np.random.permutation(len(user_id))
			user_id = np.array(user_id)
			#ds = ds[shuffled_indices]
			#step_in_episode = step_in_episode[shuffled_indices]
			state = np.array(state)
			action = np.array(action)
			reward_liucun = np.array(reward_liucun)
			reward_hongbao = np.array(reward_hongbao)
			next_state = np.array(next_state)
			terminal = np.array(terminal)
			reward = np.array(reward)
			Lambda = np.array(Lambda)
			Q_value = np.array(Q_value)

		else:
			shuffled_indices = np.random.permutation(len(user_id))
			user_id = np.array(user_id)[shuffled_indices]
			# ds = ds[shuffled_indices]
			# step_in_episode = step_in_episode[shuffled_indices]
			state = np.array(state)[shuffled_indices]
			action = np.array(action)[shuffled_indices]
			reward_liucun = np.array(reward_liucun)[shuffled_indices]
			reward_hongbao = np.array(reward_hongbao)[shuffled_indices]
			next_state = np.array(next_state)[shuffled_indices]
			terminal = np.array(terminal)[shuffled_indices]
			reward = np.array(reward)[shuffled_indices]
			Lambda = np.array(Lambda)[shuffled_indices]
			Q_value = np.array(Q_value)[shuffled_indices]



		# state_max = np.max(state.astype(np.float32), axis=0)
		# state_min = np.min(state.astype(np.float32), axis=0)
		# state_max_minus_min = state_max - state_min

		# user_id = user_id[start:end]
		# #ds = ds[start:end]
		# #step_in_episode = step_in_episode[start:end]
		# state = state[start:end]
		# action = action[start:end]
		# reward_liucun = reward_liucun[start:end]
		# reward_hongbao = reward_hongbao[start:end]
		# next_state = next_state[start:end]
		# terminal = terminal[start:end]
		# reward = reward[start:end]
		# Lambda = Lambda[start:end]

		# self.num_iter += 1

		user_id = np.expand_dims(user_id, axis=1)
		#step_in_episode = np.expand_dims(step_in_episode, axis=1)

		action_one_hot = self.ActionEmbedding.action_embedding(action.astype(np.float32)).astype(np.int32)

		state_list = list(state)  # transfer to list
		# from sting to numpy array for each data
		state = [np.expand_dims(state_list[i], axis=1) for i in range(len(state_list))]

		next_state_list = list(next_state)  # transfer to list
		# for i in range(len(next_state_list)):
		# 	if not isinstance(next_state_list[i], str) or next_state_list[i] is None or len(next_state_list[i]) == 1\
		# 			or next_state_list[i] == '\\N':
		# 		next_state_list[i] = state_list[i]
		# from sting to numpy array for each data
		next_state = [np.expand_dims(next_state_list[i], axis=1) for i in range(len(next_state_list))]

		state = np.concatenate(state, axis=1).T  # transfer to numpy array
		next_state = np.concatenate(next_state, axis=1).T

		# whether come next day
		reward_liucun = np.expand_dims(reward_liucun, axis=1).astype(np.float32)  # reshape
		reward_hongbao = np.expand_dims(reward_hongbao, axis=1).astype(np.float32)
		reward = np.expand_dims(reward, axis=1).astype(np.float32)
		Lambda = np.expand_dims(Lambda, axis=1).astype(np.float32)
		Q_value = np.expand_dims(Q_value, axis=1).astype(np.float32)
		# bonus amount

		#reward_hongbao = self.Lambda * np.expand_dims(-reward2, axis=1).astype(np.float32)  # reshape
		#reward = reward_liucun + reward_hongbao
		#terminal = [0 if t in ['False', 'false', 'FALSE', 0, '0'] else 1 for t in terminal]
		done = np.expand_dims(terminal, axis=1).astype(np.float32)  # reward

		if is_evaluation:
			self.new_batch_data["user_id"] = user_id
			# self.new_batch_data["time_id"] = step_in_episode[:self.max_size]
			self.new_batch_data["state"] = state
			self.new_batch_data["action"] = action_one_hot
			self.new_batch_data["reward1"] = reward_liucun
			self.new_batch_data["reward2"] = reward_hongbao
			self.new_batch_data["reward"] = reward
			self.new_batch_data["done"] = done
			self.new_batch_data["next_state"] = next_state
			self.new_batch_data["Lambda"] = Lambda
			self.new_batch_data["Q_value"] = Q_value
		else:
			self.new_batch_data["user_id"] = user_id[:self.max_size]
			#self.new_batch_data["time_id"] = step_in_episode[:self.max_size]
			self.new_batch_data["state"] = state[:self.max_size]
			self.new_batch_data["action"] = action_one_hot[:self.max_size]
			self.new_batch_data["reward1"] = reward_liucun[:self.max_size]
			self.new_batch_data["reward2"] = reward_hongbao[:self.max_size]
			self.new_batch_data["reward"] = reward[:self.max_size]
			self.new_batch_data["done"] = done[:self.max_size]
			self.new_batch_data["next_state"] = next_state[:self.max_size]
			self.new_batch_data["Lambda"] = Lambda[:self.max_size]
			self.new_batch_data["Q_value"] = Q_value[:self.max_size]

		# print('data exported finished')








		# # Adjust crt_size if we're using a custom size
		# user_id = np.expand_dims(user_id, axis=1)
		# step_in_episode = np.expand_dims(step_in_episode, axis=1)
		#
		# action_one_hot = self.ActionEmbedding.action_embedding(np.array(action, dtype=np.float32)).astype(np.int32)
		#
		# state_list = list(state)  # transfer to list
		# # from sting to numpy array for each data
		# state = [np.expand_dims(state_list[i].split(','), axis=1) for i in range(len(state_list))]
		#
		# next_state_list = list(next_state)  # transfer to list
		# # from sting to numpy array for each data
		# for i in range(len(next_state_list)):
		# 	if next_state_list[i] == '\\N' or not isinstance(next_state_list[i], str) or not next_state_list[i]:
		# 		next_state_list[i] = state_list[i]
		# next_state = [np.expand_dims(next_state_list[i].split(','), axis=1) for i in range(len(next_state_list))]
		#
		# state = np.concatenate(state, axis=1).T  # transfer to numpy array
		#
		# state_one_hot = []
		# for i in range(len(state[0])):
		# 	# one_hot_data = embedding_state(state_dict[i], state[:, i])
		# 	one_hot_data = np.expand_dims(state[:, i], axis=1)
		# 	if one_hot_data is not None:
		# 		state_one_hot.append(one_hot_data)
		# state_one_hot = np.concatenate(state_one_hot, axis=1).astype(np.float32)
		# state_one_hot = (state_one_hot - self.state_min) / self.state_max_minus_min
		#
		# next_state = np.concatenate(next_state, axis=1).T  # transfer to numpy array
		#
		# next_state_one_hot = []
		# for i in range(len(next_state[0])):
		# 	# next_one_hot_data = embedding_state(state_dict[i], next_state[:, i])
		# 	next_one_hot_data = np.expand_dims(next_state[:, i], axis=1)
		# 	if next_one_hot_data is not None:
		# 		next_state_one_hot.append(next_one_hot_data)
		# next_state_one_hot = np.concatenate(next_state_one_hot, axis=1).astype(np.float32)
		# next_state_one_hot = (next_state_one_hot - self.state_min) / self.state_max_minus_min
		#
		# # whether come next day
		# reward1 = np.expand_dims(reward1, axis=1).astype(np.float32)  # reshape
		# # bonus amount
		# reward2 = -self.Lambda * np.expand_dims(reward2, axis=1).astype(np.float32)  # reshape
		# reward = reward1 + reward2
		# terminal = [0 if t in ['False', 'false', 'FALSE', 0, '0'] else 1 for t in terminal]
		# done = np.expand_dims(terminal, axis=1).astype(np.float32)  # reward
		#
		# if pre_train_batch_size:
		# 	batch_size = pre_train_batch_size
		# else:
		# 	batch_size = self.batch_size
		#
		# self.new_batch_data["user_id"] = user_id[:batch_size]
		# self.new_batch_data["time_id"] = step_in_episode[:batch_size]
		# self.new_batch_data["state"] = state_one_hot[:batch_size]
		# self.new_batch_data["action"] = action_one_hot[:batch_size]
		# self.new_batch_data["reward1"] = reward1[:batch_size]
		# self.new_batch_data["reward2"] = reward2[:batch_size]
		# self.new_batch_data["reward"] = reward[:batch_size]
		# self.new_batch_data["done"] = done[:batch_size]
		# self.new_batch_data["next_state"] = next_state_one_hot[:batch_size]

	# def parse_tuple_data(self, tuple_data, with_appendix=False, pre_train_batch_size=None):
	# 	# assert len(tuple_data) == self.batch_size + self.n_step - 1
	#
	# 	if with_appendix:
	# 		user_id, ds, step_in_episode, state, action, reward1, reward2, \
	# 		next_state, terminal = zip(*tuple_data)
	# 		time_id = [int(e) for e in step_in_episode]
	# 	else:
	# 		user_id = ds = None
	# 		step_in_episode, state, action, reward1, reward2, \
	# 		next_state, terminal = zip(*tuple_data)
	#
	# 	# Adjust crt_size if we're using a custom size
	# 	user_id = np.expand_dims(user_id, axis=1)
	# 	step_in_episode = np.expand_dims(step_in_episode, axis=1)
	#
	# 	action_one_hot = self.ActionEmbedding.action_embedding(np.array(action, dtype=np.float32)).astype(np.int32)
	#
	# 	state_list = list(state)  # transfer to list
	# 	# from sting to numpy array for each data
	# 	state = [np.expand_dims(state_list[i].split(','), axis=1) for i in range(len(state_list))]
	#
	# 	next_state_list = list(next_state)  # transfer to list
	# 	# from sting to numpy array for each data
	# 	for i in range(len(next_state_list)):
	# 		if next_state_list[i] == '\\N' or not isinstance(next_state_list[i], str) or not next_state_list[i]:
	# 			next_state_list[i] = state_list[i]
	# 	next_state = [np.expand_dims(next_state_list[i].split(','), axis=1) for i in range(len(next_state_list))]
	#
	# 	state = np.concatenate(state, axis=1).T  # transfer to numpy array
	#
	# 	state_one_hot = []
	# 	for i in range(len(state[0])):
	# 		# one_hot_data = embedding_state(state_dict[i], state[:, i])
	# 		one_hot_data = np.expand_dims(state[:, i], axis=1)
	# 		if one_hot_data is not None:
	# 			state_one_hot.append(one_hot_data)
	# 	state_one_hot = np.concatenate(state_one_hot, axis=1).astype(np.float32)
	# 	state_one_hot = (state_one_hot - self.state_min) / self.state_max_minus_min
	#
	# 	next_state = np.concatenate(next_state, axis=1).T  # transfer to numpy array
	#
	# 	next_state_one_hot = []
	# 	for i in range(len(next_state[0])):
	# 		# next_one_hot_data = embedding_state(state_dict[i], next_state[:, i])
	# 		next_one_hot_data = np.expand_dims(next_state[:, i], axis=1)
	# 		if next_one_hot_data is not None:
	# 			next_state_one_hot.append(next_one_hot_data)
	# 	next_state_one_hot = np.concatenate(next_state_one_hot, axis=1).astype(np.float32)
	# 	next_state_one_hot = (next_state_one_hot - self.state_min) / self.state_max_minus_min
	#
	# 	# whether come next day
	# 	reward1 = np.expand_dims(reward1, axis=1).astype(np.float32)  # reshape
	# 	# bonus amount
	# 	reward2 = -self.Lambda * np.expand_dims(reward2, axis=1).astype(np.float32)  # reshape
	# 	reward = reward1 + reward2
	# 	terminal = [0 if t in ['False', 'false', 'FALSE', 0, '0'] else 1 for t in terminal]
	# 	done = np.expand_dims(terminal, axis=1).astype(np.float32)  # reward
	#
	# 	if pre_train_batch_size:
	# 		batch_size = pre_train_batch_size
	# 	else:
	# 		batch_size = self.batch_size
	#
	# 	self.new_batch_data["user_id"] = user_id[:batch_size]
	# 	self.new_batch_data["time_id"] = step_in_episode[:batch_size]
	# 	self.new_batch_data["state"] = state_one_hot[:batch_size]
	# 	self.new_batch_data["action"] = action_one_hot[:batch_size]
	# 	self.new_batch_data["reward1"] = reward1[:batch_size]
	# 	self.new_batch_data["reward2"] = reward2[:batch_size]
	# 	self.new_batch_data["reward"] = reward[:batch_size]
	# 	self.new_batch_data["done"] = done[:batch_size]
	# 	self.new_batch_data["next_state"] = next_state_one_hot[:batch_size]

	def sample_random(self):
		ind = np.random.randint(0, self.crt_size, size=self.batch_size)
		state, next_state, action, reward, not_done = [], [], [], [], []
		for i in ind:
			s, s2, a, r, d = self.state[i], self.next_state[i], self.action[i], \
							self.reward[i], self.not_done[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			reward.append(np.array(r, copy=False))
			not_done.append(np.array(d, copy=False))
		return (np.array(state),
				np.array(action),
				np.array(next_state),
				np.array(reward).reshape(-1, ),  # np.array(reward).reshape(-1, 1)
				np.array(not_done).reshape(-1, ))  # np.array(done).reshape(-1, 1))

	def load(self, save_folder, size=-1):
		reward_buffer = np.load("{}_reward.npy".format(save_folder))

		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.crt_size = min(reward_buffer.shape[0], size)

		self.new_batch_data["state"] = np.load("{}_state.npy".format(save_folder))[:self.crt_size]
		self.new_batch_data["action"] = np.load("{}_action.npy".format(save_folder))[:self.crt_size]
		self.new_batch_data["next_state"] = np.load("{}_next_state.npy".format(save_folder))[:self.crt_size]
		self.new_batch_data["reward"] = reward_buffer[:self.crt_size]
		self.new_batch_data["reward1"] = np.load("{}_reward.npy".format(save_folder))[:self.crt_size]
		self.new_batch_data["reward2"] = np.load("{}_reward.npy".format(save_folder))[:self.crt_size]
		self.new_batch_data["done"] = np.load("{}_not_done.npy".format(save_folder))[:self.crt_size]
		self.new_batch_data["done"] = 1 - self.new_batch_data["done"]

		print("Replay Buffer loaded with {} elements.".format(self.crt_size))

	def sample_priority(self):
		idxs = []
		segment = self.tree.total() / self.batch_size
		priorities = []
		state, action, reward, next_state, done, reward1, reward2, Lambda = [], [], [], [], [], [], [], []
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

		for i in range(self.batch_size):
			a = segment * i
			b = segment * (i + 1)

			while True:
				s = np.random.uniform(a, b)
				(idx, p, data) = self.tree.get(s)
				if not isinstance(data, int):
					break
			priorities.append(p)
			state.append(data[0])
			action.append(data[1])
			reward.append(data[2])
			next_state.append(data[3])
			done.append(data[4])
			reward1.append(data[5])
			reward2.append(data[6])
			Lambda.append(data[7])
			idxs.append(idx)

		sampling_probabilities = priorities / self.tree.total()
		is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
		is_weight /= is_weight.max()


		return (np.array(state),
				np.array(action),
				np.array(next_state),
				np.array(reward).reshape(-1, ),  # np.array(reward).reshape(-1, 1)
				np.array(done).reshape(-1, ),
				np.array(reward1).reshape(-1, ),
				np.array(reward2).reshape(-1, ),
				np.array(Lambda).reshape(-1, ),
				idxs, is_weight)  # np.array(done).reshape(-1, 1))

	def sample_without_priority(self):
		#ind = np.random.randint(0, 512, size=self.batch_size)
		ind = np.random.choice(self.user_id.shape[0], self.batch_size)
		state, action, next_state, reward1, reward2, reward, done, Lambda = [], [], [], [], [], [], [], []
		fake_idxs, fake_is_weight = [], []
		for i in ind:
			s, a, s_, r1, r2, r, d, l = self.state[i], self.action_one_hot[i], self.next_state[i],\
									self.reward_liucun[i], self.reward_hongbao[i], self.reward[i], self.done[i], self.Lambda[i]
			state.append(np.array(s, copy=False))
			action.append(np.array(a, copy=False))
			next_state.append(np.array(s_, copy=False))
			reward1.append(np.array(r1, copy=False))
			reward2.append(np.array(r2, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))
			Lambda.append(np.array(l, copy=False))
			fake_idxs.append((np.array(1.)))
			fake_is_weight.append((np.array(1.)))
		return (np.array(state),
				np.array(action),
				np.array(next_state),
				np.array(reward).reshape(-1, ),  # np.array(reward).reshape(-1, 1)
				np.array(done).reshape(-1, ),
				np.array(reward1).reshape(-1, ),
				np.array(reward2).reshape(-1, ),
				np.array(Lambda).reshape(-1, ),
				np.array(fake_idxs).reshape(-1, ),
				np.array(fake_is_weight).reshape(-1, )
				)


def read_data_excel(filename):
	batch_values = pd.read_excel(filename, header=0).to_numpy()
	return batch_values[:, 0], batch_values[:, 1], batch_values[:, 2], batch_values[:, 3], batch_values[:, 4], \
		batch_values[:, 5], batch_values[:, 6], batch_values[:, 7], batch_values[:, 8]

