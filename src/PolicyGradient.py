"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Input, Dense, Activation, InputLayer
# reproducible
np.random.seed(1)
# tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_action,
            n_feature,
            hidden_layers=None,
            batch_size=512,
            learning_rate=0.01,
            reward_decay=1.0,
            output_graph=False,
            reuse=False
    ):
        self.n_action = n_action
        self.n_feature = n_feature
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size

        self.ep_obs, self.ep_algo_action, self.ep_real_action, self.ep_rs = [], [], [], []
        # with tf.device('/device:GPU:0'):
        self._build_net()
        self.losses = []
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):

        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(
                tf.float32, [None, self.n_feature], name="observations")
            self.tf_acts = tf.placeholder(
                tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(
                tf.float32, [None, ], name="actions_value")
            self.tf_lbs = tf.placeholder(
                tf.float32, [None, self.n_action], name="labels_value")

        with tf.variable_scope("policy_net"):
            # self.model = self._build_shared_network(
            #     self.tf_obs)
            # all_act = self.model.get_layer("dense_3").output
            # fc1
            layer_1 = tf.layers.dense(
                inputs=self.tf_obs,
                units=512,
                activation=tf.nn.relu,  # tanh activation
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name='fc1'
            )
            # bn = tf.contrib.layers.batch_norm(layer_1)
            # layer_2 = tf.layers.dense(
            #     inputs=layer_1,
            #     units=64,
            #     activation=tf.nn.relu,  # tanh activation
            #     kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
            #     name='fc2'
            # )
            # bn = tf.contrib.layers.batch_norm(layer_1)
            layer_2 = tf.layers.dense(
                inputs=layer_1,
                units=64,
                activation=tf.nn.relu,  # tanh activation
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                # kernel_initializer=tf.random_uniform_initializer(),
                bias_initializer=tf.constant_initializer(0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                name='fc2'
            )
            # fc2
            # dropout = tf.nn.dropout(layer_2, keep_prob=0.5)
            all_act = tf.layers.dense(
                inputs=layer_1,
                units=self.n_action,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0, 1),
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc3'
            )
            # use softmax to convert to probability
            self.all_act_prob_soft = tf.nn.softmax(all_act, name='act_prob')
            self.all_act_prob = tf.clip_by_value(
                self.all_act_prob_soft, 1e-4, 1)

            self.entropy = - \
                tf.reduce_sum(self.all_act_prob *
                              tf.log(self.all_act_prob), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(
                self.entropy, name="entropy_mean")

            # self.variance =
            self.prob_avg = tf.reduce_mean(self.all_act_prob_soft, axis=0)
            self.loss_avg = tf.abs(self.prob_avg[0] - 0.5)

        with tf.name_scope('loss'):

               # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
            self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=all_act, labels=self.tf_acts)  # 所选 action 的概率 -log 值
            # 下面的方式是一样的:
            # self.neg_log_prob = tf.reduce_sum(
            #     -tf.log(self.all_act_prob) * tf.one_hot(self.tf_acts, self.n_action), axis=1)
            # (vt = 本reward + 衰减的未来reward) 引导参数的梯度下降
            self.loss = tf.reduce_mean(
                self.neg_log_prob * self.tf_vt) + self.entropy_mean
                 # + tf.losses.get_regularization_loss()
            self.train_op = tf.train.AdamOptimizer(
                self.lr).minimize(self.loss)

    def choose_action(self, observation, mode='random', greedy=None):
        # print('observation', observation.shape)
        prob_weights = self.sess.run(self.all_act_prob_soft, feed_dict={
                                     self.tf_obs: observation})
        # print('prob_weights', prob_weights)
        actions = []
        probs = []
        for i, p in enumerate(prob_weights):
            # print('self.n_action', self.n_action, p)
            p /= np.sum(p)
            # print('p', p)
            # try:
            if mode == 'random':
                # prob_rand = np.ones(self.n_action).astype(np.floating) * 1. / self.n_action
                # if greedy is not None and np.random.rand() < greedy:
                #     action = np.random.choice(self.n_action)
                # else:
                # action = np.random.choice(self.n_action, p=p.ravel())
                action = np.random.choice(self.n_action, p=p)
                # p = prob_rand
            elif mode == 'max':
                action = np.argmax(p)
            else:
                print('choose_action mode is Wrong')
                exit()
            actions.append(action)  # select action w.r.t the actions prob
            probs.append(p)
            # except:
            #     print(p)
        return actions, probs

    def store_transition(self, state, algo_action, real_action, reward):
        self.ep_obs.append(state)
        self.ep_algo_action.append(algo_action)
        self.ep_rs.append(reward)
        self.ep_real_action.append(real_action)

    def learn(self, tran=None):
        if tran is not None:
            self.ep_obs, self.ep_rs, self.ep_algo_action, self.ep_real_action = tran.all_data()
            # self.ep_obs, self.ep_rs, self.ep_algo_action, self.ep_real_action = tran.sample(batch_size=self.batch_size)
            # print('ep_rs', self.ep_rs[0])
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # print('discounted_ep_rs_norm', discounted_ep_rs_norm[:10])

        labels = np.ones((len(discounted_ep_rs_norm), self.n_action))
        _, loss, prob_avg, loss_avg = self.sess.run([self.train_op, self.loss, self.prob_avg, self.loss_avg], feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_algo_action),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
            self.tf_lbs: labels
        })

        # print('prob_avg', prob_avg)
        # print('loss_avg', loss_avg)
        self.ep_obs, self.ep_algo_action, self.ep_real_action, self.ep_rs = [
        ], [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self, terminal=None):
        discounted_ep_rs = np.ones_like(self.ep_rs)
        for t in reversed(range(0, len(self.ep_rs))):
            discounted_ep_rs[t] = self.ep_rs[t]
        return discounted_ep_rs
