import os
import sys
import numpy as np
# from A3C.A3C import A3C
from PolicyGradient import PolicyGradient
# from RL_brain import PolicyGradient
from KevinDataReader import DataReader
from SimulationData import SimulationData
# import logging
# import logging.handlers
# from metrics import same_diff, general_respone, qini_q, qini_Q, qini_discrete
from metrics import IPS, SN_IPS, SN_IPS_v2, same_diff
# import A3C.config as drlflow_conf
import tensorflow as tf
import time
# LOG_FILE = 'output.log'
#
# handler = logging.handlers.RotatingFileHandler(
#     LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
# fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
#
# formatter = logging.Formatter(fmt)   # 实例化formatter
# handler.setFormatter(formatter)      # 为handler添加formatter
#
# logger = logging.getLogger('tst')    # 获取名为tst的logger
# logger.addHandler(handler)           # 为logger添加handler
# logger.setLevel(logging.DEBUG)
# tf.set_random_seed(int(time.time()))


class Transition:
    def __init__(self):
        self.ep_obs = []
        self.ep_algo_action = []
        self.ep_real_action = []
        self.ep_rs = []

    def append(self, state, algo_action, real_action, reward):
        self.ep_obs.append(state)
        self.ep_rs.append(reward)
        self.ep_algo_action.append(algo_action)
        self.ep_real_action.append(real_action)

    def clear(self):
        self.ep_obs = []
        self.ep_algo_action = []
        self.ep_real_action = []
        self.ep_rs = []

    def all_data(self):
        return self.ep_obs, self.ep_rs, self.ep_algo_action, self.ep_real_action

    def sample(self, batch_size):
        indexs = np.arange(self.length())
        np.random.shuffle(indexs)
        sp_obs = [self.ep_obs[i] for i in indexs[:batch_size]]
        sp_rs = [self.ep_rs[i] for i in indexs[:batch_size]]
        sp_algo_action = [self.ep_algo_action[i]
                          for i in indexs[:batch_size]]
        sp_real_action = [self.ep_real_action[i]
                          for i in indexs[:batch_size]]
        return sp_obs, sp_rs, sp_algo_action, sp_real_action

    def length(self):
        return len(self.ep_obs)

    def avg_reward(self, mean, std):
        # print('mean', mean, 'std', std)
        try:
            self.ep_rs = (np.array(self.ep_rs) - mean) / std
        except:
            print('ep_rs', self.ep_rs)
            print('mean', mean)
            print('std', std)
            exit()


class RLift:
    def __init__(self,
                 trains,
                 validates,
                 tests,
                 n_feature,
                 n_action,
                 name_data,
                 metric,
                 reward_design,
                 Zero_is_Action,
                 hidden_layers,
                 train_eval_func,
                 test_eval_func,
                 model_base=None,
                 validate_max_steps=1000,
                 size_bag=5000,
                 parallel=False,
                 learner=None,
                 sess=None,
                 coor=None,
                 n_bags=1,
                 isLoad=False):
        self.name_data = name_data
        self.label = 'visit'
        self.n_action = n_action
        self.n_feature = n_feature
        self.Zero_is_Action = Zero_is_Action

        self.isLoad = isLoad
        # self.model_save
        self.saved_model_dir = None

        self.trains = trains
        self.tests = tests
        self.validates = validates
        self.treatment_weight = [1.0]
        self.treatment_keys = ['reward']
        # if self.isLoad is True:
        #     self.sess = tf.Session()
        #
        #     ans = self.load_model(
        #         self.sess, self.trains[0][0].reshape((-1, 8)))
        #     print('ans', ans)
        #     print('..', self.trains[0][0])
        #     exit()
        #
        # else:
        if self.Zero_is_Action is True:
            self.learner = PolicyGradient(
                n_action=self.n_action,
                n_feature=self.n_feature,
                hidden_layers=hidden_layers)
        else:
            self.learner = PolicyGradient(
                n_action=self.n_action - 1,
                n_feature=self.n_feature,
                hidden_layers=hidden_layers)
        self.sess = self.learner.sess

        self.n_train = len(self.trains)
        self.n_test = len(self.tests)
        self.n_validate = len(self.validates)
        self.max_epoch = 1000000
        self.validate_max_steps = validate_max_steps
        self.n_bags = n_bags
        self.output_steps = 10
        self.parallel = parallel
        self.batch_size = 512
        self.size_bag = size_bag
        self.train_eval_func = train_eval_func
        self.test_eval_func = test_eval_func
        self.n_step_repeat = n_bags
        self.metric = metric
        self.reward_design = reward_design
        self.start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

        # self.saved_model_dir
        # os.mknod("test.txt")
        # self.log_file =
        self.log = open(
            'logs/log_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'a')

        print('train eval_func', getattr(self.train_eval_func, '__name__'))
        print('test eval_func', getattr(self.test_eval_func, '__name__'))
        # self.start_state = [np.zeros_like(self.trains[0][0]), 0, 0]
        # self.end_state = [np.ones_like(self.trains[0][0]), 0, 0]

        # print('train', self.trains)
        # print('test', self.tests)
        # print('train', self.trains)

    def split_bags(self):
        bags = [[]] * self.n_bags
        self.n_train = len(self.trains)
        indexs = np.arange(self.n_train)
        np.random.shuffle(indexs)
        for i in range(self.n_bags):
            bags[i] = [self.trains[j]
                       for j in range(self.n_train) if (j % self.n_bags) == i]
        return bags

    def next_batch(self):
        '''
        Return:
        batch = [state, real_action, response]
        '''
        indexs = np.arange(self.n_train)
        np.random.shuffle(indexs)
        batch = [self.trains[i] for i in indexs[:self.size_bag]]
        return batch

    def stat_actions(self, actions):
        p = [np.sum(actions == i) for i in
             range(self.n_action)]
        p = np.array(p)
        p = p / np.sum(p)
        return p

    def train(self):
        uplift_validate_max = -1
        max_result_record = None
        uplift_train = -1
        uplift_train_actions = [-1] * self.n_action
        validate_max_steps = self.validate_max_steps
        validate_cur_steps = 0
        for epoch in range(self.max_epoch):
            self.log.write('Epoch' + str(epoch) + '\n')
            bag_rewards = np.zeros((self.n_step_repeat,))
            trans = []
            for eid in range(self.n_step_repeat):
                bag = self.next_batch()
                datas = np.array([data[0] for data in bag])
                # lifts = data[4]
                actions, probs = self.learner.choose_action(
                    datas, mode='random', greedy=0.1)

                if self.Zero_is_Action is False:
                    actions = [x + 1 for x in actions]
                    tmp_probs = []
                    for prob in probs:
                        tmp = [0.0]
                        tmp.extend(prob.tolist())
                        tmp_probs.append(tmp)
                    probs = np.array(tmp_probs)
                records = []
                real_probs = np.ones(self.n_action) * 0.2
                for a, data, p in zip(actions, bag, probs):
                    # Record: [Algo Action, Real Action, {Reaction}, Prob_sample, Prob_Algo]
                    records.append(
                        [int(a), data[1], {'reward': data[2]}, real_probs, p])
                    # exit()
                # record = [[a, bag[i][1], bag[i][2], probs[i][1]]
                #           for i, a in enumerate(actions)]
                # if self.metric == 'same_diff':
                # bag_rewards[eid], lifts_actions, pro_actions, lift_treatment, algo_treatment, algo_control, algo_treatment = self.eval_func(
                #     record=records, n_action=self.n_action)
                # elif self.metric == 'qini':
                # bag_rewards_qini[eid] = qini_Q(
                #     record=record, n_action=self.n_action)
                eval_res = self.train_eval_func(records=records, treatment_weight=self.treatment_weight,
                                          treatment_keys=self.treatment_keys, n_action=self.n_action)

                bag_rewards[eid] = eval_res['reward']
                algo_treatment = eval_res['response']
                algo_control = eval_res['control']
                # variance = eval_res['variance']
                algo_probs = eval_res['algo_action_prob']
                # print('variance', variance)
                algo_action_base = eval_res['algo_action_base']
                # print('eval_res', eval_res['reward'], eval_res['response'],
                #       eval_res['control'], eval_res['algo_action_prob'])
                tran = Transition()
                # print('actions', actions)
                for i, (data, algo_prob) in enumerate(zip(bag, probs)):
                    feature = data[0]
                    algo_action = actions[i]
                    real_action = data[1]
                    response = data[2]
                    lifts = data[4]

                    val_next = bag_rewards[eid]

                    if self.reward_design == 'same_diff':
                        # print('reward_design is same diff')
                        # print('algo_action', algo_action, 'real_action', real_action)
                        if algo_action == real_action or real_action == 0:
                            if algo_action == real_action:
                                rwd = (response - algo_control) + val_next
                            elif real_action == 0:
                                rwd = -(response - algo_control) + val_next

                            if self.Zero_is_Action is False:
                                algo_action -= 1

                            tran.append(
                                state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)

                    elif self.reward_design == 'action_depend_baseline':
                        # if algo_action == real_action or real_action == 0:
                            # print('algo_action_base[0]', algo_action_base[0])
                        if algo_action == real_action or real_action == 0:
                            if algo_action == real_action:
                                rwd = (
                                    response - algo_action_base[0][algo_action]) + val_next
                            elif real_action == 0:
                                rwd = - \
                                    (response -
                                     algo_action_base[0][algo_action]) + val_next

                            if self.Zero_is_Action is False:
                                algo_action -= 1
                            tran.append(
                                state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)

                    elif self.reward_design == 'response_yebai':
                        rwd = (response - value_next)
                        if self.Zero_is_Action is False:
                            algo_action -= 1
                        tran.append(
                            state=data[0], real_action=real_action, algo_action=real_action, reward=rwd)

                    elif self.reward_design == 'response_only':
                        if real_action == algo_action:
                            rwd = response - algo_action_base[0][algo_action]
                            if self.Zero_is_Action is False:
                                algo_action -= 1
                            tran.append(
                                state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)

                    elif self.reward_design == 'delayed_reward_only':
                        if real_action == algo_action:
                            rwd = val_next
                            if self.Zero_is_Action is False:
                                algo_action -= 1
                            tran.append(
                                state=data[0], real_action=real_action, algo_action=real_action, reward=rwd)
                    elif self.reward_design == 'variance_add':
                        if real_action == algo_action:
                            if algo_action == real_action or real_action == 0:
                                if real_action > 0:
                                    rwd = (response - algo_control) + val_next - 0.1 * variance
                                else:
                                    rwd = -(response - algo_control) + val_next - 0.1 * variance

                                if self.Zero_is_Action is False:
                                    algo_action -= 1
                                tran.append(
                                    state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)
                    elif self.reward_design == 'response_baseline':
                        # if algo_action > 0:
                        if algo_action == real_action or real_action == 0:
                            if real_action > 0:
                                rwd = (response - algo_control)
                            else:
                                rwd = -(response - algo_control)

                            if self.Zero_is_Action is False:
                                algo_action -= 1

                            # print('reward', rwd)
                            tran.append(
                                state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)
                    elif self.reward_design == 'response_baseline':
                        if algo_action == real_action or real_action == 0:
                            base = self.model_base.predict(feature)
                            if real_action > 0:
                                rwd = (response - base) + val_next
                            else:
                                rwd = -(response - base) + val_next

                            if self.Zero_is_Action is False:
                                algo_action -= 1
                            tran.append(
                                state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)

                    elif self.reward_design == 'action_prob_control':
                        # if algo_action == real_action or real_action == 0:
                            if algo_action == real_action:
                                rwd = (response - algo_control) + val_next - (max(algo_probs) - 1.0 / self.n_action)
                            else:
                                rwd = -(response - algo_treatment) + val_next - (max(algo_probs) - 1.0 / self.n_action)

                            if self.Zero_is_Action is False:
                                algo_action -= 1
                            tran.append(
                                state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)
                    elif self.reward_design == 'same_diff_all':
                        if algo_action == real_action or real_action == 0:
                            if algo_action == real_action:
                                rwd = (response - algo_control) + val_next
                            else:
                                rwd = -(response - algo_control) + val_next
                            if self.Zero_is_Action is False:
                                algo_action -= 1
                            tran.append(
                                state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)
                    elif self.reward_design == 'response_substract_baseline':
                        if algo_action == real_action or real_action == 0:
                            if algo_action == real_action:
                                rwd = (response - val_next) + val_next
                            elif real_action == 0:
                                rwd = -(response - val_next) + val_next
                            if self.Zero_is_Action is False:
                                algo_action -= 1
                            tran.append(
                                state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)
                    elif self.reward_design == 'pure_lift':
                        # if algo_action == real_action:
                        rwd = (response - algo_control) + val_next
                        # else:
                        #     rwd = -(response - algo_control) + val_next
                        if self.Zero_is_Action is False:
                            algo_action -= 1

                        tran.append(
                            state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)

                    elif self.reward_design == 'self_normalize_action_baseline_no_prob':
                        if real_action > 0:
                            rwd = val_next + (response)
                        else:
                            rwd = val_next - (response)

                        if self.Zero_is_Action is False:
                            algo_action -= 1

                        tran.append(
                            state=data[0], real_action=real_action, algo_action=real_action, reward=rwd)

                    elif self.reward_design == 'self_normalize_action_baseline':
                        if real_action > 0:
                            rwd = val_next + (response * algo_prob[real_action] / 0.2 - algo_action_base[0][real_action])
                        else:
                            rwd = val_next - (response * algo_prob[real_action] / 0.2- algo_action_base[0][real_action])

                        if self.Zero_is_Action is False:
                            algo_action -= 1

                        tran.append(
                            state=data[0], real_action=real_action, algo_action=real_action, reward=rwd)

                    elif self.reward_design == 'self_normalize_total_baseline':
                        if real_action > 0:
                            rwd = val_next + (response * algo_prob[real_action] / 0.2 - algo_control)
                        else:
                            rwd = val_next - (response * algo_prob[real_action] / 0.2- algo_control)

                        if self.Zero_is_Action is False:
                            algo_action -= 1

                        tran.append(
                            state=data[0], real_action=real_action, algo_action=real_action, reward=rwd)
                    elif self.reward_design == 'important_sampling_identity_action_baseline':
                        if algo_action == real_action or real_action > 0:
                            if algo_action == real_action:
                                rwd = val_next + (response - algo_action_base[0][algo_action]) / 0.2
                            else:
                                rwd = val_next - (response - algo_action_base[0][algo_action]) / 0.2

                            if self.Zero_is_Action is False:
                                algo_action -= 1

                            tran.append(
                                state=data[0], real_action=real_action, algo_action=algo_action, reward=rwd)
                    else:
                        print('Error! Reward Design is not found!')
                        exit()

                trans.append(tran)
            # print('bag_rewards', bag_rewards)
            reward_mean = np.mean(bag_rewards).astype(np.floating)
            reward_std = 1
            print('mean', reward_mean, 'std', max(1e-4, reward_std))
            self.log.write('mean:' + str(reward_mean) +
                           ' std:' + str(max(1e-4, reward_std)) + '\n')
            for tran in trans:
                tran.avg_reward(reward_mean, reward_std)
                self.learner.learn(tran)
            # if we only use RLift, then it need to record the best result by itself.
            # if self.parallel is False:
            eval_res = self.test(eval_func=self.test_eval_func,
                                 datas=self.validates, epoch=epoch, result_output=False)

            print('validate eval_res', eval_res['reward'], eval_res['response'],
                  eval_res['control'], eval_res['algo_action_prob'], eval_res['algo_action_nums'])
            uplift_validate = eval_res['reward']

            print('uplift_validate', uplift_validate)
            if uplift_validate > uplift_validate_max:
                uplift_validate_max = uplift_validate
                if self.saved_model_dir is not None:
                    self.save_model(sess=self.learner.sess)
                    # exit()

                print('uplift_validate_max', uplift_validate_max)
                self.log.write('uplift_validate_max:' +
                               str(uplift_validate_max) + '\n')
                validate_cur_steps = 0
                max_result_record = self.results_calc(eval_func=self.test_eval_func,
                                                      epoch=epoch, result_output=True,
                                                      outputs_list=['test'],
                                                      isMax=True)
                print('max result test')
                uplift_test_max = max_result_record['test']
                # print('uplift_test_max', uplift_test_max)
                print('tests', uplift_test_max['reward'], uplift_test_max['reward'],
                      uplift_test_max['response'], uplift_test_max['control'], uplift_test_max['algo_action_prob'])
                print('max result train')
                # uplift_test_max = max_result_record['train']
                # # print('uplift_test_max', uplift_test_max)
                # print('trains', uplift_test_max['reward'], uplift_test_max['reward'],
                #       uplift_test_max['response'], uplift_test_max['control'], uplift_test_max['algo_action_prob'])
                self.log.write('max_result_record:' +
                               str(max_result_record) + '\n')

            else:
                validate_cur_steps += 1

            if validate_cur_steps >= validate_max_steps:
                print('Training Finished', epoch)
                print('uplift_test_max', uplift_test_max,
                      'uplift_validate_max', uplift_validate_max)
                print('max result', max_result_record)
                self.log.write('Training Finished:' + str(epoch) + '\n')
                self.log.write('uplift_validate_max:' +
                               str(uplift_validate_max) + '\n')
                self.sess.close()
                sys.exit()

            if epoch % self.output_steps == 0 and epoch > 0:
                print('Epoch', epoch)
                print('uplift_test_max', uplift_test_max)
                print('uplift_validate_max', uplift_validate_max)
                print('max result', max_result_record)
                # self.results_calc(eval_func=self.eval_func, outputs_list=[
                #                   'test', 'validate'], epoch=epoch, result_output=False)

    # def results_store(self):
    #     self.results_trains =
    def save_model(self, sess):
        builder = tf.saved_model.builder.SavedModelBuilder(
            self.saved_model_dir)
        # x 为输入tensor, keep_prob为dropout的prob tensor
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(
            self.learner.tf_obs)}

        # y 为最终需要的输出结果tensor
        outputs = {'output': tf.saved_model.utils.build_tensor_info(
            self.learner.all_act_prob)}

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs, outputs, 'test_sig_name')
        # signature = None

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING], signature_def_map=signature)
        builder.save()

    def load_model(self, sess, _x):
        signature_key = 'test_signature'
        input_key = 'input_x:0'
        output_key = 'output:0'

        meta_graph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], self.saved_model_dir)
        # 从meta_graph_def中取出SignatureDef对象
        signature = meta_graph_def.signature_def

        # 从signature中找出具体输入输出的tensor name
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name

        # 获取tensor 并inference
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)

        res = sess.run(y, feed_dict={x: _x})
        print('res shape', res.shape)

        return res

    def results_calc(self, epoch, eval_func, outputs_list=['train', 'test', 'validate'], result_output=False, isMax=False):
        res = {}
        if isMax is True:
            suffix = 'max'
        else:
            suffix = str(epoch).zfill(5)
        if 'train' in outputs_list:
            uplift_train = self.test(eval_func=eval_func,
                                     datas=self.trains, epoch=epoch, output_filename='train_' + suffix, result_output=result_output)
            res['train'] = uplift_train

        if 'validate' in outputs_list:
            uplift_validate = self.test(eval_func=eval_func,
                                        datas=self.validates, epoch=epoch, output_filename='validate_' + suffix, result_output=result_output)
            res['validate'] = uplift_validate

        if 'test' in outputs_list:
            uplift_test = self.test(eval_func=eval_func,
                                    datas=self.tests, epoch=epoch, output_filename='test_' + suffix, result_output=result_output)
            res['test'] = uplift_test

        print('Epoch', epoch)
        self.log.write('Epoch:' + str(epoch) + '\n')
        for name in res.keys():
            print(name, res[name])
        return res

    def test(self, datas, epoch, eval_func, output_filename=None, result_output=False):
        '''
        Test on the datas = [feature, action_real, reaction]
        '''
        real_probs = np.ones(self.n_action) / self.n_action
        records = []
        features = [data[0] for data in datas]
        actions_algo, probs = self.learner.choose_action(
            features, mode='random', greedy=None)

        # print('probs', probs)
        # for i, a in enumerate(actions_algo):
        #     record.append([int(a), datas[i][1], datas[i][2], None])


            # for i, (a, data, prob) in enumerate(zip(actions_algo, datas, probs)):
        for i, (data, action, prob) in enumerate(zip(datas, actions_algo, probs)):
            # action = int(actions_algo[0])
            # prob = probs[0]
            if self.Zero_is_Action is False:
                action = action + 1
                tmp = [0.0]
                tmp.extend(prob.tolist())
                prob = np.array(tmp)

            records.append(
                [action, data[1], {'reward': data[2]}, real_probs, prob])

        # reactions = [datas[i][2] for i, a in enumerate(actions_algo)]
        # reactions = np.array(reactions)
        # reactions = np.reshape(reactions, (len(datas), 1))

        # if output_filename is not None and result_output is True:
        #     name_func = getattr(self.eval_func, '__name__')
        #     np.save('../output/' + output_filename + '_' + name_func + '_' + self.start_time,
        #             np.hstack((probs, reactions, actions_real)))
        #     print(output_filename + '_' + name_func + ' saved')
        ans = eval_func(records=records, treatment_weight=self.treatment_weight,
                        treatment_keys=self.treatment_keys, n_action=self.n_action)
        return ans
        # probs, reactions, actions_real


def model_baseline(datas):
    X = []
    y = []
    for data in datas:
        if data[1] == 0:
            X.append(data[0])
            y.append(data[2])
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(max_depth=5, random_state=0)
    regr.fit(np.array(X), np.array(y))
    return regr

if __name__ == '__main__':
    # reader = DataReader(action=['womens'], label='visit')
    # trains, validates, tests = reader.get_datas()
    # reader = SimulationData(path_load='../dataset/SimulationData_random.npy')
    reader = SimulationData(path_load=None)
    n_feature = reader.n_feature
    n_action = reader.n_action
    print('n_feature', n_feature)
    print('n_action', n_action)
    trains, validates, tests = reader.get_datas()
    baseline = model_baseline(trains)
    rlift = RLift(name_data=reader.name,
                  trains=trains, validates=validates, tests=tests, hidden_layers=[
                      64],
                  n_feature=reader.n_feature, size_bag=10000, validate_max_steps=1000, reward_design='important_sampling_identity_action_baseline',
                  n_action=5, n_bags=10, train_eval_func=SN_IPS_v2, test_eval_func=IPS,
                  model_base=baseline,
                  metric='SN_IPS_v2', Zero_is_Action=True, isLoad=False)
    rlift.train()
