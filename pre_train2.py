# coding: utf-8
import tensorflow as tf
import tensorflow.contrib.layers as layers
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import to_categorical
import numpy as np

class PropensityNet(object):

    def __init__(self, sess, state_dim, num_actions, Lambda_dim, lr, name):
        self.sess = sess
        self.name = name
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.lr = lr
        self.state_ = tf.placeholder(tf.float32, [None, state_dim], name="propensity_state")
        self.lambda_ = tf.placeholder(tf.float32, [None, Lambda_dim], name="propensity_lambda")
        self._create_network()
        # Initialize network
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _create_network(self):
        with tf.variable_scope(self.name + 'i_net', reuse=tf.AUTO_REUSE):
            #placeholders for PropensityNet

            # I network
            self.i0 = tf.layers.batch_normalization(tf.concat([self.state_, self.lambda_], 1), axis=-1, momentum=0.99,
                                                    epsilon=0.001,
                                                    center=True,
                                                    scale=True,
                                                    beta_initializer=tf.zeros_initializer(),
                                                    gamma_initializer=tf.ones_initializer(),
                                                    moving_mean_initializer=tf.zeros_initializer(),
                                                    moving_variance_initializer=tf.ones_initializer(),
                                                    beta_regularizer=None,
                                                    gamma_regularizer=None,
                                                    beta_constraint=None,
                                                    gamma_constraint=None,
                                                    training=False,
                                                    trainable=True,
                                                    name=None,
                                                    reuse=tf.AUTO_REUSE,
                                                    renorm=False,
                                                    renorm_clipping=None,
                                                    renorm_momentum=0.99,
                                                    fused=None)
            self.i1 = layers.fully_connected(self.i0, 1024, activation_fn=tf.nn.relu)
            self.i2 = layers.fully_connected(self.i1, 512, activation_fn=tf.nn.relu)
            self.i2_ = layers.fully_connected(self.i2, 512, activation_fn=tf.nn.relu)
            self.i3 = layers.fully_connected(self.i2_, self.num_actions, activation_fn=None)
        #self.i3_ = tf.squeeze(self.i3, name = 'action_probability')
        self.i3_ = tf.expand_dims(self.i3, axis=-1)

        self.i3_ = tf.squeeze(self.i3_, name='action_probability')

        # placeholder for current_action
        self.current_action = tf.placeholder(tf.float32, [None, self.num_actions])

        # i loss
        self.i_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=self.current_action, logits=self.i3)
        )
        self.i_loss = self.i_loss + 1e-2 * tf.reduce_mean(tf.square(self.i3))

        # R1 network
        # whether come next day
        self.current_action_float = tf.placeholder(tf.float32, [None, self.num_actions], name='propensity_action')
        self.r13 = tf.squeeze(self.get_reward_prediction(self.state_,self.lambda_,self.current_action_float,
                                                         self.name + 'r_net',reuse=False), name = "propensity_r13")

        # placeholder for real R1
        self.current_r1 = tf.placeholder(tf.float32, [None], name='propensity_r1_action')

        # r1 lossu
        self.r1_loss = tf.reduce_mean(tf.square(self.current_r1 - self.r13))
        # self.r1_loss = self.r1_loss + 1e-2 * tf.reduce_mean(tf.square(self.r13))

        # Optimize the Q
        with tf.variable_scope('i_train', reuse=tf.AUTO_REUSE):
            self.i_optim_ = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.i_loss)
        with tf.variable_scope('r_train', reuse=tf.AUTO_REUSE):
            self.r1_optim_ = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.r1_loss)

    def get_i_network_variables(self):
        return [t for t in tf.trainable_variables() if t.name.startswith(self.name + 'i_net')]

    def get_reward_prediction(self, state_, lambda_, current_action_float, scope, reuse=False):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            r10 = tf.layers.batch_normalization(tf.concat([state_, lambda_], 1), axis=-1, momentum=0.99,
                                                epsilon=0.001,
                                                center=True,
                                                scale=True,
                                                beta_initializer=tf.zeros_initializer(),
                                                gamma_initializer=tf.ones_initializer(),
                                                moving_mean_initializer=tf.zeros_initializer(),
                                                moving_variance_initializer=tf.ones_initializer(),
                                                beta_regularizer=None,
                                                gamma_regularizer=None,
                                                beta_constraint=None,
                                                gamma_constraint=None,
                                                training=False,
                                                trainable=True,
                                                name=None,
                                                reuse=tf.AUTO_REUSE,
                                                renorm=False,
                                                renorm_clipping=None,
                                                renorm_momentum=0.99,
                                                fused=None)
            r11 = layers.fully_connected(r10, 256, activation_fn=tf.nn.relu)
            r12 = layers.fully_connected(r11, 256, activation_fn=tf.nn.relu)

            r21 = layers.fully_connected(current_action_float, 64, activation_fn=tf.nn.relu)
            r22 = layers.fully_connected(r21, 64, activation_fn=tf.nn.relu)

            r3 = tf.concat((r12, r22), axis=1)
            r13 = layers.fully_connected(r3, 1, activation_fn=None)
            return r13

    def pre_train(
            self, replay_buffer_pretrain, replay_buffer_evaluation,
            pre_train_epochs,
            pre_train_minibatch_size,
            pre_train_batch_size, test_mode, table):
        if not test_mode:
            replay_buffer_pretrain.reset_table_reader(table)

        if test_mode:
            replay_buffer_pretrain.get_batch_data_from_local_npz()
            replay_buffer_evaluation.get_batch_data_from_local_npz()
        else:
            replay_buffer_pretrain.get_batch_data_from_odps(
                pre_train_minibatch_size)

        for step in range(pre_train_epochs):
            # split train and test set
            # Randomize data point

            # get data
            if not test_mode:
                replay_buffer_pretrain.get_train_data()
                # replay_buffer_evaluation.get_batch_data_from_odps(
                #     pre_train_minibatch_size)

            # shuffled_indices = np.random.permutation(
            #     min(
            #         len(replay_buffer_pretrain.new_batch_data["state"]),
            #         len(replay_buffer_evaluation.new_batch_data["state"])
            #     ))
            shuffled_indices = np.random.permutation(
                    len(replay_buffer_pretrain.new_batch_data["state"])
                )
            state_shuffled = replay_buffer_pretrain.new_batch_data["state"][shuffled_indices]
            action_shuffled = replay_buffer_pretrain.new_batch_data["action"][shuffled_indices]
            reward_shuffled = replay_buffer_pretrain.new_batch_data["reward1"][shuffled_indices]
            Lambda_shuffled = replay_buffer_pretrain.new_batch_data["Lambda"][shuffled_indices]
            Q_value_shuffled = replay_buffer_pretrain.new_batch_data["Q_value"][shuffled_indices]
            # state_shuffled_evaluation = replay_buffer_evaluation.new_batch_data["state"][shuffled_indices]
            # action_shuffled_evaluation = replay_buffer_evaluation.new_batch_data["action"][shuffled_indices]
            # reward1_shuffled_evaluation = abs(replay_buffer_evaluation.new_batch_data["reward1"][shuffled_indices])
            # Lambda_shuffled_evaluation = replay_buffer_evaluation.new_batch_data["Lambda"][shuffled_indices]
            # Q_value_shuffled_evaluation = replay_buffer_evaluation.new_batch_data["Q_value"][shuffled_indices]

            state_train, state_test, Lambda_train, Lambda_test, action_reward1_train, action_reward1_test = train_test_split(
                state_shuffled,
                Lambda_shuffled,
                np.concatenate(
                    (action_shuffled,
                     Q_value_shuffled), axis=1
                ),
                test_size=0.2,
                random_state=0,
                shuffle=True
            )
            X_train, l_train, y_train = shuffle(state_train, Lambda_train, action_reward1_train)

            action_train = y_train[:, 0]
            action_train_onehot = to_categorical(action_train, num_classes=self.num_actions)
            reward1_train = y_train[:, 1]
            reward1_train_onehot = reward1_train

            action_test = action_reward1_test[:, 0]
            action_test_onehot = to_categorical(action_test, num_classes=self.num_actions)
            reward1_test = action_reward1_test[:, 1]
            reward1_test_onehot = reward1_test

            # action_test_validation = np.squeeze(action_shuffled_evaluation)
            # action_test_validation_onehot = to_categorical(action_test_validation, num_classes=self.num_actions)
            # reward1_test_validation = Q_value_shuffled_evaluation
            # reward1_test_validation_onehot = reward1_test_validation

            state_train_mini = X_train
            Lambda_train_mini = l_train
            action_train_mini = action_train_onehot
            reward1_train_mini = reward1_train_onehot
            # train the classifier
            i_loss, _ = self.sess.run(
                [self.i_loss, self.i_optim_], feed_dict={
                    self.state_: state_train_mini,
                    self.current_action: action_train_mini,
                    self.lambda_: Lambda_train_mini
                }
            )
            r1_loss, _ = self.sess.run(
                [self.r1_loss, self.r1_optim_], feed_dict={
                    self.state_: state_train_mini,
                    self.current_action_float: action_train_mini,
                    self.current_r1: reward1_train_mini,
                    self.lambda_: Lambda_train_mini
                }
            )

            if step % 10 == 0:
                print("step = {}\tpre_train_loss = {}".format(step, i_loss))
                print("step = {}\tpre_train_loss_r1 = {}".format(step, r1_loss))

                # next_action_possible = self.sess.run(tf.nn.softmax(self.i3), feed_dict={
                #     self.state_: state_test,
                #     self.lambda_: Lambda_test
                # })
                # f1_actions = f1_score(
                #     np.argmax(action_test_onehot, axis=1),
                #     np.argmax(next_action_possible, axis=1),
                #     average='macro'
                # )
                # next_reward_possible = self.sess.run(tf.nn.softmax(self.r13), feed_dict={
                #     self.state_: state_test,
                #     self.lambda_: Lambda_test,
                #     self.current_action_float: action_test_onehot
                # })
                # next_reward_model = next_reward_possible
                # next_reward_possible_validation = self.sess.run(tf.nn.softmax(self.r13), feed_dict={
                #     self.state_: state_shuffled_evaluation,
                #     self.lambda_: Lambda_shuffled_evaluation,
                #     self.current_action_float: action_test_validation_onehot
                # })
                # next_reward_model_validation = next_reward_possible_validation
                #
                # print(
                #     "reward_mean, reward_mean_real, reward_mean_validation, reward_mean_real_validation",
                #     np.mean(next_reward_model), np.mean(reward1_test),
                #     np.mean(reward1_test_validation), np.mean(next_reward_model_validation),
                # )
                # reward1_test_onehot_ = []
                # next_reward_possible_ = []
                # reward1_test_validation_onehot_ = []
                # next_reward_possible_validation_ = []
                # for i in reward1_test_onehot:
                #     reward1_test_onehot_.append(float(i))
                # for i in next_reward_possible:
                #     next_reward_possible_.append(float(i))
                # for i in reward1_test_validation_onehot:
                #     reward1_test_validation_onehot_.append(float(i[0]))
                # for i in next_reward_possible_validation:
                #     next_reward_possible_validation_.append(float(i))
                # f1_reward = f1_score(
                #     reward1_test_onehot_,
                #     next_reward_possible_,
                #     average='macro'
                # )
                # f1_reward_validation = f1_score(
                #     reward1_test_validation_onehot_,
                #     next_reward_possible_validation_,
                #     average='macro'
                # )
                # print(
                #     "f1_actions_score: {}, f1_reward1_score: {},"
                #     " f1_reward1_validation_score: {}".format(f1_actions, f1_reward, f1_reward_validation)
                # )

        # self.sess.run(self.copy_pretrain_update())
        # self.sess.run(self.copy_target_update())
        replay_buffer_pretrain.reset()

    def get_logged_propensities(self, ob):
        res = self.sess.run(tf.nn.softmax(self.i3), feed_dict={
            self.state_: np.expand_dims(ob, 1).T
        })
        return np.squeeze(res)

