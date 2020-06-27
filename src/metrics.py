import numpy as np
# import matplotlib.pyplot as plt


def general_respone_origin(record, treatment_weight, n_action):
    n_treat = len(treatment_weight)
    same_treatment = {}
    lift_treatment = {}
    sample_treatment = {}
    sum_actions = {}
    for a in range(n_action):
        same_treatment[a] = [[] for i in range(n_treat)]
        sample_treatment[a] = [[] for i in range(n_treat)]
        lift_treatment[a] = [0] * n_treat

    sum_actions = [0] * n_action
    sample_actions = [0] * n_action

    for rec in record:
        sample_actions[int(rec[1])] += 1
        sum_actions[int(rec[0])] += 1
        algo_action = int(rec[0])
        real_action = int(rec[0])
        for a in range(n_action):
            for i, key in enumerate(rec[2].keys()):
                same_treatment[a][i] += (algo_action == a) * \
                    (real_action == a) * rec[2][key]

            for i, key in enumerate(rec[2].keys()):
                sample_treatment[a][i] += (real_action == a) * rec[2][key]

    pro_algo_actions = np.array(sum_actions).astype(
        np.float) * 1.0 / np.sum(sum_actions).astype(np.float)
    pro_real_actions = np.array(sample_actions).astype(
        np.float) * 1.0 / np.sum(sample_actions).astype(np.float)

    for a in range(n_action):
        for t in range(n_treat):
            algo_lifts[t] += same_treatment[a][t] / pro_real_actions[a]
            real_lifts[t] += sample_treatment[a][t] * pro_real_actions[a]


def general_respone(record, treatment_weight, n_action=5):
    '''
    Defined in the paper "Uplift Modeling with Multiple Treatments and General Response Types"
    Inputs:
    Record: [Algo Action, Real Action, Reaction]
    Return:
    The value of same and diffs
    '''
    n_treat = len(treatment_weight)
    same_treatment = {}
    lift_treatment = {}
    sample_treatment = {}
    sum_actions = {}
    for a in range(n_action):
        same_treatment[a] = [[] for i in range(n_treat)]
        sample_treatment[a] = [[] for i in range(n_treat)]
        lift_treatment[a] = [0] * n_treat

    sum_actions = [0] * n_action
    sample_actions = [0] * n_action
    for rec in record:
        sample_actions[int(rec[1])] += 1
        sum_actions[int(rec[0])] += 1
        if int(rec[0]) == int(rec[1]):
            for i, key in enumerate(rec[2].keys()):
                same_treatment[int(rec[0])][i].append(rec[2][key])
        for i, key in enumerate(rec[2].keys()):
            sample_treatment[int(rec[1])][i].append(rec[2][key])

    pro_algo_actions = np.array(sum_actions).astype(
        np.float) * 1.0 / np.sum(sum_actions).astype(np.float)
    pro_real_actions = np.array(sample_actions).astype(
        np.float) * 1.0 / np.sum(sample_actions).astype(np.float)
    algo_lifts = [0] * n_treat
    real_lifts = [0] * n_treat
    for a in range(n_action):
        for i in range(n_treat):
            lift_treatment[a][i] = sum(
                same_treatment[a][i]) * 1. / (len(same_treatment[a][i]) + 1)
            sample_treatment[a][i] = sum(
                sample_treatment[a][i]) * 1. / (len(sample_treatment[a][i]) + 1)
            algo_lifts[i] += lift_treatment[a][i] * pro_algo_actions[a]
            real_lifts[i] += sample_treatment[a][i] * pro_real_actions[a]
    reward_algo = 0.0
    reward_real = 0.0
    for i in range(n_treat):
        reward_algo += algo_lifts[i] * treatment_weight[i]
        reward_real += real_lifts[i] * treatment_weight[i]

    res = {}
    res['reward'] = reward_algo
    res['algo_lifts'] = algo_lifts
    res['sample_lifts'] = real_lifts

    res['sum_actions'] = sum_actions
    res['sample_actions'] = sample_actions
    res['lift_treatment'] = lift_treatment
    res['sample_treatment'] = sample_treatment
    res['pro_algo_actions'] = pro_algo_actions
    res['pro_real_actions'] = pro_real_actions
    return res


def same_diff(records, treatment_weight=None,
              treatment_keys=None, n_action=5):
    '''
    The HBI index
    Inputs:
    Record: [Algo Action, Real Action, Reaction]
    Return:
    The value of same and difs
    '''
    same_treatment = {}
    same_control = {}
    sample_response = {}
    record = []
    for rec in records:
        # print('rec', rec, rec[2])
        rec[2] = rec[2]['reward']
        record.append(rec)
    records = record
    for a in range(n_action):
        same_treatment[a] = []
        same_control[a] = []
        sample_response[a] = []
    lift_treatment = [0] * n_action
    lift_control = [0] * n_action
    sample_treatment = [0] * n_action
    sample_control = [0] * n_action
    sum_actions = [0] * n_action
    sample_actions = [0] * n_action
    random = []
    for rec in record:
        sample_actions[int(rec[1])] += 1
        sample_response[int(rec[1])].append(rec[2])
        if int(rec[0]) > 0:
            if int(rec[0]) == int(rec[1]):
                same_treatment[int(rec[0])].append(rec[2])

            elif rec[1] == 0:
                same_control[int(rec[0])].append(rec[2])

        sum_actions[int(rec[0])] += 1

        random.append(rec[2])

    # print('Total samples: ', np.sum(sum_actions), np.sum(sample_actions))
    # print('algo_actions: ', sum_actions)
    pro_algo_actions = np.array(sum_actions).astype(
        np.float) / np.sum(sum_actions)
    # print('algo_actions_pro: ', pro_algo_actions)
    # print('sample_actions:', sample_actions)
    pro_sample_actions = np.array(sample_actions).astype(
        np.float) / np.sum(sample_actions)
    # print('sample_actions_pro: ', pro_sample_actions)

    # print('For algo actions: ')
    algo_lifts = 0
    algo_resps = 0
    algo_bases = 0
    for a in range(n_action)[1:]:
        lift_treatment[a] = sum(same_treatment[a]) * 1. / \
            (max(len(same_treatment[a]), 1))
        lift_control[a] = sum(same_control[a]) * 1. / \
            (max(len(same_control[a]), 1))
        # print('Action ', a, ', same_in_sample: ', len(same_treatment[a]), ', estimate lift: ', (lift_treatment[a] - lift_control[a]), 'pro_action', pro_algo_actions[a])
        algo_lifts += (lift_treatment[a] -
                       lift_control[a]) * pro_algo_actions[a]
        algo_resps += lift_treatment[a] * pro_algo_actions[a]
        algo_bases += lift_control[a] * pro_algo_actions[a]
    # print('Algo estimate lift: ', algo_lifts)

    # print('For sample actions: ')
    sample_lifts = 0
    for a in range(n_action)[1:]:
        sample_treatment[a] = sum(sample_response[a]) * 1. / \
            (max(len(sample_response[a]), 1))
        sample_control[a] = sum(sample_response[0]) * 1. / \
            (max(len(sample_response[0]), 1))
        # print('Action ', a, ', samples: ', len(sample_response[a]), ', real lift: ', (sample_treatment[a] - sample_control[a]), 'pro_action', pro_sample_actions[a])
        sample_lifts += (sample_treatment[a] -
                         sample_control[a]) * pro_sample_actions[a]
    # print('Sample real lift: ', sample_lifts)

    # print('For simulated controlled experiment: ')
    exp_lifts = 0
    for a in range(n_action)[1:]:
        # print('Action ', a, ': simulate lift: ', (sample_treatment[a] - sample_control[a]), 'pro_action', pro_algo_actions[a])
        exp_lifts += (sample_treatment[a] -
                      sample_control[a]) * pro_algo_actions[a]
    # print('Experiment simulate lift: ', exp_lifts)
    res = {}
    res['reward'] = algo_lifts
    res['response'] = algo_resps
    res['control'] = algo_bases
    res['variance'] = None

    res['algo_total_lift'] = algo_lifts
    res['algo_total_resp'] = algo_resps
    res['algo_total_base'] = algo_bases
    res['algo_treat_lift'] = None
    res['algo_treat_resp'] = None
    res['algo_treat_base'] = None
    res['algo_action_prob'] = None
    res['algo_action_nums'] = None
    res['algo_action_resp'] = lift_treatment
    res['algo_action_base'] = lift_control

    res['rand_total_lift'] = None
    res['rand_total_resp'] = None
    res['rand_total_base'] = None
    res['rand_treat_lift'] = None
    res['rand_treat_resp'] = None
    res['rand_treat_base'] = None
    res['rand_action_prob'] = None
    res['rand_action_nums'] = None
    res['rand_action_resp'] = None
    res['rand_action_base'] = None

    res['real_total_resp'] = None
    res['real_treat_resp'] = None
    res['real_action_resp'] = None
    res['real_action_nums'] = None
    res['real_action_prob'] = None
    return res


def same_diff_v2(records, treatment_weight, treatment_keys, n_action=5):
    '''
    The HBI index
    Inputs:
    Record: [Algo Action, Real Action, Reaction]
    Return:
    The value of same and difs
    '''
    same_treat_treatment = {}
    same_treat_control = {}
    sample_response = {}
    for a in range(n_action):
        same_treatment[a] = []
        same_control[a] = []
        sample_response[a] = []
    lift_treatment = [0] * n_action
    lift_control = [0] * n_action
    sample_treatment = [0] * n_action
    sample_control = [0] * n_action
    sum_actions = [0] * n_action
    sample_actions = [0] * n_action
    random = []
    for rec in record:
        sample_actions[int(rec[1])] += 1
        sample_response[int(rec[1])].append(rec[2])
        if int(rec[0]) > 0:
            if int(rec[0]) == int(rec[1]):
                same_treatment[int(rec[0])].append(rec[2])

            elif rec[1] == 0:
                same_control[int(rec[0])].append(rec[2])

        sum_actions[int(rec[0])] += 1

        random.append(rec[2])

    # print('Total samples: ', np.sum(sum_actions), np.sum(sample_actions))
    # print('algo_actions: ', sum_actions)
    pro_algo_actions = np.array(sum_actions).astype(
        np.float) / np.sum(sum_actions)
    # print('algo_actions_pro: ', pro_algo_actions)
    # print('sample_actions:', sample_actions)
    pro_sample_actions = np.array(sample_actions).astype(
        np.float) / np.sum(sample_actions)
    # print('sample_actions_pro: ', pro_sample_actions)

    # print('For algo actions: ')
    algo_lifts = 0
    algo_resps = 0
    algo_bases = 0
    for a in range(n_action)[1:]:
        lift_treatment[a] = sum(same_treatment[a]) * 1. / \
            (max(len(same_treatment[a]), 1))
        lift_control[a] = sum(same_control[a]) * 1. / \
            (max(len(same_control[a]), 1))
        # print('Action ', a, ', same_in_sample: ', len(same_treatment[a]), ', estimate lift: ', (lift_treatment[a] - lift_control[a]), 'pro_action', pro_algo_actions[a])
        algo_lifts += (lift_treatment[a] -
                       lift_control[a]) * pro_algo_actions[a]
        algo_resps += lift_treatment[a] * pro_algo_actions[a]
        algo_bases += lift_control[a] * pro_algo_actions[a]
    # print('Algo estimate lift: ', algo_lifts)

    # print('For sample actions: ')
    sample_lifts = 0
    for a in range(n_action)[1:]:
        sample_treatment[a] = sum(sample_response[a]) * 1. / \
            (max(len(sample_response[a]), 1))
        sample_control[a] = sum(sample_response[0]) * 1. / \
            (max(len(sample_response[0]), 1))
        # print('Action ', a, ', samples: ', len(sample_response[a]), ', real lift: ', (sample_treatment[a] - sample_control[a]), 'pro_action', pro_sample_actions[a])
        sample_lifts += (sample_treatment[a] -
                         sample_control[a]) * pro_sample_actions[a]
    # print('Sample real lift: ', sample_lifts)

    # print('For simulated controlled experiment: ')
    exp_lifts = 0
    for a in range(n_action)[1:]:
        # print('Action ', a, ': simulate lift: ', (sample_treatment[a] - sample_control[a]), 'pro_action', pro_algo_actions[a])
        exp_lifts += (sample_treatment[a] -
                      sample_control[a]) * pro_algo_actions[a]
    # print('Experiment simulate lift: ', exp_lifts)
    res = {}
    res['reward'] = algo_lifts
    res['response'] = algo_resps
    res['control'] = algo_bases
    res['variance'] = None

    res['algo_total_lift'] = algo_lifts
    res['algo_total_resp'] = algo_resps
    res['algo_total_base'] = algo_bases
    res['algo_treat_lift'] = algo_treat_lift
    res['algo_treat_resp'] = algo_treat_resp
    res['algo_treat_base'] = algo_treat_base
    res['algo_action_prob'] = algo_action_prob
    res['algo_action_nums'] = algo_action_nums
    res['algo_action_resp'] = algo_action_resp
    res['algo_action_base'] = algo_action_base

    res['rand_total_lift'] = rand_total_lift
    res['rand_total_resp'] = rand_total_resp
    res['rand_total_base'] = rand_total_base
    res['rand_treat_lift'] = rand_treat_lift
    res['rand_treat_resp'] = rand_treat_resp
    res['rand_treat_base'] = rand_treat_base
    res['rand_action_prob'] = rand_action_prob
    res['rand_action_nums'] = rand_action_nums
    res['rand_action_resp'] = rand_action_resp
    res['rand_action_base'] = rand_action_base

    res['real_total_resp'] = real_total_resp
    res['real_treat_resp'] = real_treat_resp
    res['real_action_resp'] = real_action_resp
    res['real_action_nums'] = real_action_nums
    res['real_action_prob'] = real_action_prob
    return algo_lifts, sample_lifts, sum_actions, sample_actions, lift_treatment, lift_control, sample_treatment, sample_control


def SN_IPS(records, treatment_weight, treatment_keys, n_action):
    '''
    Self Normalized IPS metric for unbiased dataset
    Inputs:
    Record: [Algo Action, Real Action, {Reaction}, Prob_sample, Prob_Algo]
    n_action: number of actions
    treatment_weight: weight for each treatment {key1:weight1, ..., keyk:weightk} / [weight1, weight2, ..., weightk]
    Return:
    The value of same and difs
    '''
    n_treat = len(treatment_weight)
    algo_action_resp = {}
    algo_total_norm = 0.0
    algo_total_base_norm = 0.0
    algo_treat_resp = [0.0] * n_treat
    algo_total_resp = 0.0
    algo_action_base = {}
    algo_treat_base = [0.0] * n_treat
    algo_total_base = 0.0
    algo_treat_lift = [0.0] * n_treat
    algo_total_lift = 0.0
    algo_action_nums = np.array([0.0] * n_action).astype(np.floating)
    algo_action_norm = np.array([0.0] * n_action).astype(np.floating)
    algo_action_base_norm = np.array([0.0] * n_action).astype(np.floating)

    rand_action_resp = {}
    rand_total_norm = 0.0
    rand_total_base_norm = 0.0
    rand_treat_resp = [0.0] * n_treat
    rand_total_resp = 0.0
    rand_action_base = {}
    rand_treat_base = [0.0] * n_treat
    rand_total_base = 0.0
    rand_treat_lift = [0.0] * n_treat
    rand_total_lift = 0.0
    rand_action_nums = np.array([0.0] * n_action).astype(np.floating)
    rand_action_norm = np.array([0.0] * n_action).astype(np.floating)
    rand_action_base_norm = np.array([0.0] * n_action).astype(np.floating)

    real_action_resp = {}
    real_treat_resp = [0.0] * n_treat
    real_total_resp = 0.0
    real_action_nums = np.array([0.0] * n_action).astype(np.floating)

    for t in range(n_treat):
        algo_action_resp[t] = np.array([0.0] * n_action).astype(np.floating)
        algo_action_base[t] = np.array([0.0] * n_action).astype(np.floating)

        rand_action_resp[t] = np.array([0.0] * n_action).astype(np.floating)
        rand_action_base[t] = np.array([0.0] * n_action).astype(np.floating)

        real_action_resp[t] = np.array([0.0] * n_action).astype(np.floating)

    print('len of records', len(records))
    for rec in records:
        algo_action = int(rec[0])
        real_action = int(rec[1])
        rand_action = np.random.choice(n_action)
        rand_action = np.random.choice(n_action)
        resp = rec[2]
        real_prob = rec[3]
        algo_prob = rec[4]
        rand_prob = np.ones(n_action).astype(np.float32) / n_action

        # if rec[3] is None:
        #     real_prob = np.ones(n_action).astype(np.floating) / n_action
        # else:

        algo_action_nums[algo_action] += 1
        real_action_nums[real_action] += 1
        rand_action_nums[rand_action] += 1

        for i, key in enumerate(treatment_keys):
            algo_action_resp[i][algo_action] += resp[key] * \
                (real_action == algo_action) / real_prob[real_action]
            algo_treat_resp[i] += resp[key] * \
                (real_action == algo_action) / real_prob[real_action]
            algo_action_base[i][algo_action] += resp[key] * \
                (real_action == 0) / real_prob[0]
            algo_treat_base[i] += resp[key] * \
                (real_action == 0) / real_prob[0]

            rand_action_resp[i][rand_action] += resp[key] * \
                (rand_action == real_action) / real_prob[real_action]
            rand_treat_resp[i] += resp[key] * \
                (rand_action == real_action) / real_prob[real_action]
            rand_action_base[i][rand_action] += resp[key] * \
                (real_action == 0) / real_prob[0]
            rand_treat_base[i] += resp[key] * \
                (real_action == 0) / real_prob[0]

            real_action_resp[i][real_action] += resp[key]
            real_treat_resp[i] += resp[key]

        algo_action_norm[algo_action] += (real_action == algo_action) * \
            1.0 / real_prob[real_action]
        algo_total_norm += (real_action == algo_action) / \
            real_prob[real_action]
        algo_action_base_norm[real_action] += (
            real_action == 0) * 1.0 / real_prob[real_action]
        algo_total_base_norm += (real_action == 0) * \
            1.0 / real_prob[real_action]

        rand_action_norm[rand_action] += rand_prob[real_action] / \
            real_prob[real_action]
        rand_total_norm += rand_prob[real_action] * \
            1.0 / real_prob[real_action]
        rand_action_base_norm[rand_action] += (
            real_action == 0) * 1.0 / real_prob[real_action]
        rand_total_base_norm += (real_action == 0) * \
            1.0 / real_prob[real_action]
    numSample = np.sum(algo_action_nums)
    print('algo_action_nums', algo_action_nums)
    # print('algo_action_resp', algo_action_resp)
    for i, key in enumerate(treatment_keys):
        for a in range(n_action):
            algo_action_resp[i][a] /= (algo_action_norm[a]
                                       if algo_action_norm[a] > 0.0 else 1.0)
            algo_action_base[i][a] /= (algo_action_norm[a]
                                       if algo_action_norm[a] != 0.0 else 1.0)
            rand_action_resp[i][a] /= (rand_action_norm[a]
                                       if rand_action_norm[a] != 0.0 else 1.0)
            rand_action_base[i][a] /= (rand_action_norm[a]
                                       if rand_action_norm[a] != 0.0 else 1.0)
            real_action_resp[i][a] /= (real_action_nums[a]
                                       if real_action_nums[a] != 0.0 else 1.0)
        algo_treat_base[i] /= (algo_total_base_norm if algo_total_base_norm >
                               0.0 else 1.0)
        algo_treat_resp[i] /= (algo_total_norm if algo_total_norm >
                               0.0 else 1.0)
        rand_treat_base[i] /= (rand_total_base_norm if rand_total_base_norm >
                               0.0 else 1.0)
        rand_treat_resp[i] /= (rand_total_norm if rand_total_norm >
                               0.0 else 1.0)
        real_treat_resp[i] /= numSample

    algo_action_prob = algo_action_nums / \
        np.sum(algo_action_nums).astype(np.floating)
    real_action_prob = real_action_nums / \
        np.sum(real_action_nums).astype(np.floating)
    rand_action_prob = rand_action_nums / \
        np.sum(rand_action_nums).astype(np.floating)

    for i in range(n_treat):
        for a in range(n_action):
            algo_treat_lift[i] += (algo_action_resp[i][a] -
                                   algo_action_base[i][a]) * algo_action_prob[a]

            rand_treat_lift[i] += (rand_action_resp[i][a] -
                                   rand_action_base[i][a]) * rand_action_prob[a]

    for i in range(n_treat):
        algo_total_lift += algo_treat_lift[i] * treatment_weight[i]
        algo_total_resp += algo_treat_resp[i] * treatment_weight[i]
        algo_total_base += algo_treat_base[i] * treatment_weight[i]

        rand_total_lift += rand_treat_lift[i] * treatment_weight[i]
        rand_total_resp += rand_treat_resp[i] * treatment_weight[i]
        rand_total_base += rand_treat_base[i] * treatment_weight[i]

        real_total_resp += real_treat_resp[i] * treatment_weight[i]

    res = {}

    res['reward'] = algo_total_lift
    res['response'] = algo_total_resp
    res['control'] = algo_total_base
    res['variance'] = None

    res['algo_total_lift'] = algo_total_lift
    res['algo_total_resp'] = algo_total_resp
    res['algo_total_base'] = algo_total_base
    res['algo_treat_lift'] = algo_treat_lift
    res['algo_treat_resp'] = algo_treat_resp
    res['algo_treat_base'] = algo_treat_base
    res['algo_action_prob'] = algo_action_prob
    res['algo_action_nums'] = algo_action_nums
    res['algo_action_resp'] = algo_action_resp
    res['algo_action_base'] = algo_action_base

    res['rand_total_lift'] = rand_total_lift
    res['rand_total_resp'] = rand_total_resp
    res['rand_total_base'] = rand_total_base
    res['rand_treat_lift'] = rand_treat_lift
    res['rand_treat_resp'] = rand_treat_resp
    res['rand_treat_base'] = rand_treat_base
    res['rand_action_prob'] = rand_action_prob
    res['rand_action_nums'] = rand_action_nums
    res['rand_action_resp'] = rand_action_resp
    res['rand_action_base'] = rand_action_base

    res['real_total_resp'] = real_total_resp
    res['real_treat_resp'] = real_treat_resp
    res['real_action_resp'] = real_action_resp
    res['real_action_nums'] = real_action_nums
    res['real_action_prob'] = real_action_prob

    return res


def SN_IPS_v2(records, treatment_weight, treatment_keys, n_action):
    '''
    Self Normalized IPS metric for unbiased dataset
    Inputs:
    Record: [Algo Action, Real Action, {Reaction}, Prob_sample, Prob_Algo]
    n_action: number of actions
    treatment_weight: weight for each treatment {key1:weight1, ..., keyk:weightk} / [weight1, weight2, ..., weightk]
    Return:
    The value of same and difs
    '''
    n_treat = len(treatment_weight)
    algo_action_resp = {}
    algo_total_norm = 0.0
    algo_total_base_norm = 0.0
    algo_treat_resp = [0.0] * n_treat
    algo_total_resp = 0.0
    algo_action_base = {}
    algo_treat_base = [0.0] * n_treat
    algo_total_base = 0.0
    algo_treat_lift = [0.0] * n_treat
    algo_total_lift = 0.0
    algo_action_nums = np.array([0.0] * n_action).astype(np.floating)
    algo_action_norm = np.array([0.0] * n_action).astype(np.floating)
    algo_action_base_norm = np.array([0.0] * n_action).astype(np.floating)

    rand_action_resp = {}
    rand_total_norm = 0.0
    rand_total_base_norm = 0.0
    rand_treat_resp = [0.0] * n_treat
    rand_total_resp = 0.0
    rand_action_base = {}
    rand_treat_base = [0.0] * n_treat
    rand_total_base = 0.0
    rand_treat_lift = [0.0] * n_treat
    rand_total_lift = 0.0
    rand_action_nums = np.array([0.0] * n_action).astype(np.floating)
    rand_action_norm = np.array([0.0] * n_action).astype(np.floating)
    rand_action_base_norm = np.array([0.0] * n_action).astype(np.floating)

    real_action_resp = {}
    real_treat_resp = [0.0] * n_treat
    real_total_resp = 0.0
    real_action_nums = np.array([0.0] * n_action).astype(np.floating)

    for t in range(n_treat):
        algo_action_resp[t] = np.array([0.0] * n_action).astype(np.floating)
        algo_action_base[t] = np.array([0.0] * n_action).astype(np.floating)

        rand_action_resp[t] = np.array([0.0] * n_action).astype(np.floating)
        rand_action_base[t] = np.array([0.0] * n_action).astype(np.floating)

        real_action_resp[t] = np.array([0.0] * n_action).astype(np.floating)

    respes = []
    for rec in records:
        # print('rec', rec)
        algo_action = int(rec[0])
        real_action = int(rec[1])
        rand_action = np.random.choice(n_action)
        rand_action = np.random.choice(n_action)
        resp = rec[2]
        real_prob = rec[3]
        algo_prob = rec[4]
        rand_prob = np.ones(n_action).astype(np.float32) / n_action
        if algo_action == real_action:
            respes.append(resp['reward'])
        # if rec[3] is None:
        #     real_prob = np.ones(n_action).astype(np.floating) / n_action
        # else:

        algo_action_nums[algo_action] += 1
        real_action_nums[real_action] += 1
        rand_action_nums[rand_action] += 1

        for i, key in enumerate(treatment_keys):
            algo_action_resp[i][real_action] += resp[key] * \
                algo_prob[real_action] / real_prob[real_action]
            algo_treat_resp[i] += resp[key] * \
                algo_prob[real_action] / real_prob[real_action]

            rand_action_resp[i][real_action] += resp[key] * \
                rand_prob[real_action] / real_prob[real_action]
            rand_treat_resp[i] += resp[key] * \
                rand_prob[real_action] / real_prob[real_action]
            if real_action == 0:
                algo_action_base[i][algo_action] += resp[key] / real_prob[0]
                algo_treat_base[i] += resp[key] / real_prob[0]
                rand_action_base[i][rand_action] += resp[key] / real_prob[0]
                rand_treat_base[i] += resp[key] / real_prob[0]

            real_action_resp[i][real_action] += resp[key]
            real_treat_resp[i] += resp[key]

        algo_action_norm[real_action] += algo_prob[real_action] / \
            real_prob[real_action]
        algo_total_norm += algo_prob[real_action] / real_prob[real_action]
        if real_action == 0:
            algo_action_base_norm[algo_action] += 1. / real_prob[0]
            algo_total_base_norm += 1.0 / real_prob[0]

        rand_action_norm[real_action] += rand_prob[real_action] / \
            real_prob[real_action]
        rand_total_norm += rand_prob[real_action] / real_prob[real_action]
        rand_action_base_norm[rand_action] += (
            real_action == 0) * 1.0 / real_prob[0]
        rand_total_base_norm += (real_action == 0) * 1.0 / real_prob[0]
    numSample = np.sum(algo_action_nums)
    # print('algo_action_resp', algo_action_resp)
    for i, key in enumerate(treatment_keys):
        for a in range(n_action):
            algo_action_resp[i][a] /= (algo_action_norm[a]
                                       if algo_action_norm[a] > 0.0 else 1.0)
            algo_action_base[i][a] /= (algo_action_base_norm[a]
                                       if algo_action_base_norm[a] > 0.0 else 1.0)
            rand_action_resp[i][a] /= (rand_action_norm[a]
                                       if rand_action_norm[a] > 0.0 else 1.0)
            rand_action_base[i][a] /= (rand_action_base_norm[a]
                                       if rand_action_base_norm[a] > 0.0 else 1.0)
            real_action_resp[i][a] /= (real_action_nums[a]
                                       if real_action_nums[a] > 0.0 else 1.0)
        algo_treat_base[i] /= (algo_total_base_norm if algo_total_base_norm >
                               0.0 else 1.0)
        algo_treat_resp[i] /= (algo_total_norm if algo_total_norm >
                               0.0 else 1.0)
        rand_treat_base[i] /= (rand_total_base_norm if rand_total_base_norm >
                               0.0 else 1.0)
        rand_treat_resp[i] /= (rand_total_norm if rand_total_norm >
                               0.0 else 1.0)
        real_treat_resp[i] /= numSample
    # print('algo_action_resp', algo_action_resp)
    # for a in range(n_action):
    #     algo_action_norm[a] /= max(algo_action_nums[a], 1)
    #     rand_action_norm[a] /= max(rand_action_nums[a], 1)
    # algo_total_norm /= numSample
    # rand_total_norm /= numSample

    algo_action_prob = algo_action_nums / np.sum(algo_action_nums)
    real_action_prob = real_action_nums / np.sum(real_action_nums)
    rand_action_prob = rand_action_nums / np.sum(rand_action_nums)

    for i in range(n_treat):
        for a in range(n_action):
            algo_treat_lift[i] += (algo_action_resp[i][a] -
                                   algo_action_base[i][a]) * algo_action_prob[a]

            rand_treat_lift[i] += (rand_action_resp[i][a] -
                                   rand_action_base[i][a]) * rand_action_prob[a]

    # print('algo lift', algo_lifts)
    for i in range(n_treat):
        algo_total_lift += algo_treat_lift[i] * treatment_weight[i]
        algo_total_resp += algo_treat_resp[i] * treatment_weight[i]
        algo_total_base += algo_treat_base[i] * treatment_weight[i]

        rand_total_lift += rand_treat_lift[i] * treatment_weight[i]
        rand_total_resp += rand_treat_resp[i] * treatment_weight[i]
        rand_total_base += rand_treat_base[i] * treatment_weight[i]

        real_total_resp += real_treat_resp[i] * treatment_weight[i]
        # real_total_base += real_treat_base[i] * treatment_weight[i]

    res = {}
    res['reward'] = algo_total_resp - algo_total_base
    res['response'] = algo_total_resp
    res['control'] = algo_total_base
    res['variance'] = np.std(respes)

    res['algo_total_lift'] = algo_total_lift
    res['algo_total_resp'] = algo_total_resp
    res['algo_total_base'] = algo_total_base
    res['algo_treat_lift'] = algo_treat_lift
    res['algo_treat_resp'] = algo_treat_resp
    res['algo_treat_base'] = algo_treat_base
    res['algo_action_prob'] = algo_action_prob
    res['algo_action_nums'] = algo_action_nums
    res['algo_action_resp'] = algo_action_resp
    res['algo_action_base'] = algo_action_base

    res['rand_total_lift'] = rand_total_lift
    res['rand_total_resp'] = rand_total_resp
    res['rand_total_base'] = rand_total_base
    res['rand_treat_lift'] = rand_treat_lift
    res['rand_treat_resp'] = rand_treat_resp
    res['rand_treat_base'] = rand_treat_base
    res['rand_action_prob'] = rand_action_prob
    res['rand_action_nums'] = rand_action_nums
    res['rand_action_resp'] = rand_action_resp
    res['rand_action_base'] = rand_action_base

    res['real_total_resp'] = real_total_resp
    res['real_treat_resp'] = real_treat_resp
    res['real_action_resp'] = real_action_resp
    res['real_action_nums'] = real_action_nums
    res['real_action_prob'] = real_action_prob

    return res


def IPS(records, treatment_weight, treatment_keys, n_action):
    '''
    IPS metric for unbiased dataset
    Inputs:
    Record: [Algo Action, Real Action, {Reaction}, Prob_sample]
    n_action: number of actions
    treatment_weight: weight for each treatment {key1:weight1, ..., keyk:weightk} / [weight1, weight2, ..., weightk]
    Return:
    The value of same and difs
    '''
    n_treat = len(treatment_weight)
    algo_action_resp = {}
    algo_treat_resp = [0.0] * n_treat
    algo_total_resp = 0.0
    algo_action_base = {}
    algo_treat_base = [0.0] * n_treat
    algo_total_base = 0.0
    algo_treat_lift = [0.0] * n_treat
    algo_total_lift = 0.0
    algo_action_nums = np.array([0.0] * n_action).astype(np.floating)
    algo_action_base_nums = np.array([0.0] * n_action).astype(np.floating)

    rand_action_resp = {}
    rand_treat_resp = [0.0] * n_treat
    rand_total_resp = 0.0
    rand_action_base = {}
    rand_treat_base = [0.0] * n_treat
    rand_total_base = 0.0
    rand_treat_lift = [0.0] * n_treat
    rand_total_lift = 0.0
    rand_action_nums = np.array([0.0] * n_action).astype(np.floating)
    rand_action_base_nums = np.array([0.0] * n_action).astype(np.floating)

    real_action_resp = {}
    real_treat_resp = [0.0] * n_treat
    real_total_resp = 0.0
    real_action_nums = np.array([0.0] * n_action).astype(np.floating)
    real_action_base_nums = np.array([0.0] * n_action).astype(np.floating)

    for t in range(n_treat):
        algo_action_resp[t] = np.array([0.0] * n_action).astype(np.floating)
        algo_action_base[t] = np.array([0.0] * n_action).astype(np.floating)

        rand_action_resp[t] = np.array([0.0] * n_action).astype(np.floating)
        rand_action_base[t] = np.array([0.0] * n_action).astype(np.floating)

        real_action_resp[t] = np.array([0.0] * n_action).astype(np.floating)

    for rec in records:
        algo_action = int(rec[0])
        real_action = int(rec[1])
        rand_action = np.random.choice(n_action)
        resp = rec[2]
        real_prob = rec[3]

        algo_action_nums[algo_action] += 1
        rand_action_nums[rand_action] += 1
        real_action_nums[real_action] += 1
        if real_action == 0:
            algo_action_base_nums[algo_action] += 1
            rand_action_base_nums[rand_action] += 1

        for i, key in enumerate(treatment_keys):
            algo_action_resp[i][algo_action] += resp[key] * \
                (algo_action == real_action) * 1.0 / real_prob[algo_action]
            algo_treat_resp[i] += resp[key] * \
                (algo_action == real_action) * 1.0 / real_prob[algo_action]
            algo_action_base[i][algo_action] += resp[key] * \
                (real_action == 0) * 1.0 / real_prob[0]
            algo_treat_base += resp[key] * \
                (real_action == 0) * 1.0 / real_prob[0]

            rand_action_resp[i][rand_action] += resp[key] * \
                (rand_action == real_action) / real_prob[rand_action]
            rand_treat_resp[i] += resp[key] * \
                (rand_action == real_action) / real_prob[rand_action]
            rand_action_base[i][rand_action] += resp[key] * \
                (real_action == 0) / real_prob[0]
            rand_treat_base += resp[key] * \
                (real_action == 0) / real_prob[0]

            real_action_resp[i][real_action] += resp[key]
            real_treat_resp[i] += resp[key]

    numSample = np.sum(algo_action_nums)
    for i, key in enumerate(treatment_keys):
        for a in range(n_action):
            algo_action_resp[i][a] /= max(algo_action_nums[a], 1)
            algo_action_base[i][a] /= max(algo_action_nums[a], 1)
            rand_action_resp[i][a] /= max(rand_action_nums[a], 1)
            rand_action_base[i][a] /= max(rand_action_nums[a], 1)
            real_action_resp[i][a] /= max(real_action_nums[a], 1)

        algo_treat_base[i] /= numSample
        algo_treat_resp[i] /= numSample
        rand_treat_base[i] /= numSample
        rand_treat_resp[i] /= numSample
        real_treat_resp[i] /= numSample

    algo_action_prob = algo_action_nums / np.sum(algo_action_nums)
    real_action_prob = real_action_nums / np.sum(real_action_nums)
    rand_action_prob = rand_action_nums / np.sum(rand_action_nums)

    for i in range(n_treat):
        for a in range(n_action):
            algo_treat_lift[i] += (algo_action_resp[i][a] -
                                   algo_action_base[i][a]) * algo_action_prob[a]

            rand_treat_lift[i] += (rand_action_resp[i][a] -
                                   rand_action_base[i][a]) * rand_action_prob[a]

    # print('algo lift', algo_lifts)
    for i in range(n_treat):
        algo_total_lift += algo_treat_lift[i] * treatment_weight[i]
        algo_total_resp += algo_treat_resp[i] * treatment_weight[i]
        algo_total_base += algo_treat_base[i] * treatment_weight[i]

        rand_total_lift += rand_treat_lift[i] * treatment_weight[i]
        rand_total_resp += rand_treat_resp[i] * treatment_weight[i]
        rand_total_base += rand_treat_base[i] * treatment_weight[i]

        real_total_resp += real_treat_resp[i] * treatment_weight[i]
        # real_total_base += real_treat_base[i] * treatment_weight[i]

    res = {}

    res['reward'] = algo_total_lift
    res['response'] = algo_total_resp
    res['control'] = algo_total_base

    res['algo_total_lift'] = algo_total_lift
    res['algo_total_resp'] = algo_total_resp
    res['algo_total_base'] = algo_total_base
    res['algo_treat_lift'] = algo_treat_lift
    res['algo_treat_resp'] = algo_treat_resp
    res['algo_treat_base'] = algo_treat_base
    res['algo_action_prob'] = algo_action_prob
    res['algo_action_nums'] = algo_action_nums
    res['algo_action_resp'] = algo_action_resp
    res['algo_action_base'] = algo_action_base

    res['rand_total_lift'] = rand_total_lift
    res['rand_total_resp'] = rand_total_resp
    res['rand_total_base'] = rand_total_base
    res['rand_treat_lift'] = rand_treat_lift
    res['rand_treat_resp'] = rand_treat_resp
    res['rand_treat_base'] = rand_treat_base
    res['rand_action_prob'] = rand_action_prob
    res['rand_action_nums'] = rand_action_nums
    res['rand_action_resp'] = rand_action_resp
    res['rand_action_base'] = rand_action_base

    res['real_total_resp'] = real_total_resp
    res['real_treat_resp'] = real_treat_resp
    res['real_action_resp'] = real_action_resp
    res['real_action_nums'] = real_action_nums
    res['real_action_prob'] = real_action_prob

    return res


def multi_treatment_v2(records, treatment_weight, treatment_keys, n_action):
    '''
    The HBI index for multi_treatment evaluation, temp is_clk(+), is_sign(+), cost(-)
    For convenience, cost is compared to average cost of random samples
    Inputs:
    Record: [Algo Action, Real Action, {Reaction}]
    n_action: number of actions
    treatment_weight: weight for each treatment {key1:weight1, ..., keyk:weightk} / [weight1, weight2,...,weightk]
    Return:
    The value of same and difs
    '''
    # print('metric multi treatment', n_action)
    n_treat = len(treatment_weight)
    algo_action_resp = {}
    algo_treat_resp = [0.0] * n_treat
    algo_total_resp = 0.0
    algo_action_base = {}
    algo_treat_base = [0.0] * n_treat
    algo_total_base = 0.0
    algo_treat_lift = [0.0] * n_treat
    algo_total_lift = 0.0
    algo_action_nums = np.array([0.0] * n_action).astype(np.floating)

    rand_action_resp = {}
    rand_treat_resp = [0.0] * n_treat
    rand_total_resp = 0.0
    rand_action_base = {}
    rand_treat_base = [0.0] * n_treat
    rand_total_base = 0.0
    rand_treat_lift = [0.0] * n_treat
    rand_total_lift = 0.0
    rand_action_nums = np.array([0.0] * n_action).astype(np.floating)

    real_action_resp = {}
    real_treat_resp = [0.0] * n_treat
    real_total_resp = 0.0
    real_action_nums = np.array([0.0] * n_action).astype(np.floating)

    for t in range(n_treat):
        algo_action_resp[t] = {}
        algo_action_base[t] = {}

        rand_action_resp[t] = {}
        rand_action_base[t] = {}

        real_action_resp[t] = {}
        for a in range(n_action):
            algo_action_resp[t][a] = []
            algo_action_base[t][a] = []
            rand_action_resp[t][a] = []
            rand_action_base[t][a] = []
            real_action_resp[t][a] = []

    algo_treat_resp = [[0] * n_action for i in range(n_treat)]
    algo_treat_base = [[0] * n_action for i in range(n_treat)]

    for rec in records:
        algo_action = int(rec[0])
        real_action = int(rec[1])
        rand_action = int(np.random.choice(n_action))
        resp = rec[2]
        # real_prob = rec[3]
        # algo_prob = rec[4]
        rand_prob = np.ones(n_action).astype(np.float32) / n_action

        algo_action_nums[algo_action] += 1
        real_action_nums[real_action] += 1
        rand_action_nums[rand_action] += 1
        for i, key in enumerate(treatment_keys):
            # print('len of real_action_resp', len(real_action_resp))
            # print('len of real_action_resp', i, len(real_action_resp[i]))
            # print('len of real_action_resp', i, real_action,
            #       len(real_action_resp[i][real_action]))
            real_action_resp[i][real_action].append(resp[key])
        # print('real_action', real_action)
        # print('algo_action', algo_action)
        # print('rand_action', rand_action)
        # print('algo_action == real_action', algo_action == real_action)
        # print('rand_action == real_action', rand_action == real_action)
        # print('real_action == 0', real_action == 0)
        if algo_action > 0:
            if algo_action == real_action:
                for i, key in enumerate(treatment_keys):
                    algo_action_resp[i][algo_action].append(resp[key])
            elif real_action == 0:
                for i, key in enumerate(treatment_keys):
                    algo_action_base[i][algo_action].append(resp[key])
            # else:
            #     print('Error! Func metric.py. line 768')
            #     exit()

        if rand_action > 0:
            if rand_action == real_action:
                for i, key in enumerate(treatment_keys):
                    rand_action_resp[i][rand_action].append(resp[key])
            elif real_action == 0:
                for i, key in enumerate(treatment_keys):
                    rand_action_base[i][rand_action].append(resp[key])
            # else:
            #     print('Error! Func metric.py. line 775')
            #     exit()

    algo_action_prob = algo_action_nums / np.sum(algo_action_nums)
    real_action_prob = real_action_nums / np.sum(real_action_nums)
    rand_action_prob = rand_action_nums / np.sum(rand_action_nums)

    numSample = np.sum(algo_action_nums)

    # algo_treat_lift = [0] * n_treat
    # algo_treat_resp = [0] * n_treat
    # algo_treat_base = [0] * n_treat
    #
    # real_treat_lift = [0] * n_treat
    # real_treat_resp = [0] * n_treat
    # real_treat_base = [0] * n_treat
    # sample_lifts = [0] * n_treat
    # exp_lifts = [0] * n_treat
    for i, key in enumerate(treatment_keys):
        for a in range(n_action):
            algo_action_resp[i][a] = np.sum(algo_action_resp[i][a]) / (
                len(algo_action_resp[i][a]) if len(algo_action_resp[i][a]) > 0.0 else 1.0)
            algo_action_base[i][a] = np.sum(algo_action_resp[i][a]) / (
                len(algo_action_base[i][a]) if len(algo_action_base[i][a]) > 0.0 else 1.0)
            rand_action_resp[i][a] = np.sum(algo_action_resp[i][a]) / (
                len(rand_action_resp[i][a]) if len(rand_action_resp[i][a]) > 0.0 else 1.0)
            rand_action_base[i][a] = np.sum(algo_action_resp[i][a]) / (
                len(rand_action_base[i][a]) if len(rand_action_base[i][a]) > 0.0 else 1.0)
            real_action_resp[i][a] = np.sum(algo_action_resp[i][a]) / (
                len(real_action_resp[i][a]) if len(real_action_resp[i][a]) > 0.0 else 1.0)

    rand_algo_treat_resp = {}
    rand_algo_treat_base = {}
    rand_algo_treat_lift = {}

    rand_control_resp = {}
    rand_control_resp = {}
    for a in range(n_action)[1:]:
        for i in range(n_treat):
            algo_treat_lift[i] += (algo_action_resp[i][a] -
                                   algo_action_base[i][a]) * algo_action_prob[a]
            algo_treat_resp[i] += algo_action_resp[i][a] * algo_action_prob[a]
            algo_treat_base[i] += algo_action_base[i][a] * algo_action_prob[a]

            rand_treat_lift[i] += (rand_action_resp[i][a] -
                                   rand_action_base[i][a]) * rand_action_prob[a]
            rand_treat_resp[i] += rand_action_resp[i][a] * rand_action_prob[a]
            rand_treat_base[i] += rand_action_base[i][a] * rand_action_prob[a]

            real_treat_resp[i] += real_action_resp[i][a] * real_action_prob[a]

            rand_algo_treat_resp[i] = rand_action_resp[i][a] * \
                algo_action_prob[a]
            rand_algo_treat_base[i] = rand_action_base[i][a] * \
                algo_action_prob[a]
            rand_algo_treat_lift[i] = (rand_action_resp[i][a] -
                                       rand_action_base[i][a]) * algo_action_prob[a]
    reward_lift = 0.0
    reward_resp = 0.0
    reward_base = 0.0
    real_total_resp = 0.0

    rand_algo_total_resp = 0.0
    rand_algo_total_base = 0.0
    rand_algo_total_lift = 0.0
    # real_total_base = 0.0
    # real_total_lift = 0.0
    for i in range(n_treat):
        reward_lift += algo_treat_lift[i] * treatment_weight[i]
        reward_resp += algo_treat_resp[i] * treatment_weight[i]
        reward_base += algo_treat_base[i] * treatment_weight[i]

        algo_total_lift += algo_treat_lift[i] * treatment_weight[i]
        algo_total_resp += algo_treat_resp[i] * treatment_weight[i]
        algo_total_base += algo_treat_base[i] * treatment_weight[i]

        rand_total_lift += rand_treat_lift[i] * treatment_weight[i]
        rand_total_resp += rand_treat_resp[i] * treatment_weight[i]
        rand_total_base += rand_treat_base[i] * treatment_weight[i]

        real_total_resp += real_treat_resp[i] * treatment_weight[i]

        rand_algo_total_resp += rand_algo_treat_resp[i] * treatment_weight[i]
        rand_algo_total_base += rand_algo_treat_base[i] * treatment_weight[i]
        rand_algo_total_lift += rand_algo_treat_lift[i] * treatment_weight[i]

    res = {}

    res['reward'] = algo_total_lift
    res['response'] = algo_total_resp
    res['control'] = algo_total_base

    res['algo_total_lift'] = algo_total_lift
    res['algo_total_resp'] = algo_total_resp
    res['algo_total_base'] = algo_total_base
    res['algo_treat_lift'] = algo_treat_lift
    res['algo_treat_resp'] = algo_treat_resp
    res['algo_treat_base'] = algo_treat_base
    res['algo_action_prob'] = algo_action_prob
    res['algo_action_nums'] = algo_action_nums
    res['algo_action_resp'] = algo_action_resp
    res['algo_action_base'] = algo_action_base

    res['rand_total_lift'] = rand_total_lift
    res['rand_total_resp'] = rand_total_resp
    res['rand_total_base'] = rand_total_base
    res['rand_treat_lift'] = rand_treat_lift
    res['rand_treat_resp'] = rand_treat_resp
    res['rand_treat_base'] = rand_treat_base
    res['rand_action_prob'] = rand_action_prob
    res['rand_action_nums'] = rand_action_nums
    res['rand_action_resp'] = rand_action_resp
    res['rand_action_base'] = rand_action_base
    res['rand_algo_total_resp'] = rand_algo_total_resp
    res['rand_algo_total_base'] = rand_algo_total_base
    res['rand_algo_total_lift'] = rand_algo_total_lift

    res['real_total_resp'] = real_total_resp
    res['real_treat_resp'] = real_treat_resp
    res['real_action_resp'] = real_action_resp
    res['real_action_nums'] = real_action_nums
    res['real_action_prob'] = real_action_prob
    return res


def multi_treatment(records, treatment_weight, treatment_keys, n_action):
    '''
    The HBI index for multi_treatment evaluation, temp is_clk(+), is_sign(+), cost(-)
    For convenience, cost is compared to average cost of random samples
    Inputs:
    Record: [Algo Action, Real Action, {Reaction}]
    n_action: number of actions
    treatment_weight: weight for each treatment {key1:weight1, ..., keyk:weightk} / [weight1, weight2,...,weightk]
    Return:
    The value of same and difs
    '''
    # print('metric multi treatment', n_action)
    n_treat = len(treatment_weight)
    # print('n_treat', n_treat, 'n_action', n_action)
    same_treatment = {}
    same_control = {}
    sample_response = {}
    for a in range(n_action):
        same_treatment[a] = [[] for i in range(n_treat)]
        same_control[a] = [[] for i in range(n_treat)]
        sample_response[a] = [[] for i in range(n_treat)]
    lift_treatment = [[0] * n_action for i in range(n_treat)]
    lift_control = [[0] * n_action for i in range(n_treat)]
    sample_treatment = [[0] * n_action for i in range(n_treat)]
    sample_control = [[0] * n_action for i in range(n_treat)]
    sum_actions = [0] * n_action
    sample_actions = [0] * n_action

    for rec in records:
        sample_actions[int(rec[1])] += 1
        for i, key in enumerate(treatment_keys):
            sample_response[int(rec[1])][i].append(rec[2][key])
        if int(rec[0]) > 0:
            if int(rec[0]) == int(rec[1]):
                for i, key in enumerate(treatment_keys):
                    same_treatment[int(rec[0])][i].append(rec[2][key])

            elif rec[1] == 0:
                for i, key in enumerate(treatment_keys):
                    same_control[int(rec[0])][i].append(rec[2][key])

        # print('rec', rec, 'len of sum_actions', len(sum_actions))
        sum_actions[int(rec[0])] += 1

        # random.append(rec[2])

    # print('Total samples: ', np.sum(sum_actions), np.sum(sample_actions))
    # print('algo_actions: ', sum_actions)
    pro_algo_actions = np.array(sum_actions).astype(
        np.float) / np.sum(sum_actions)
    # print('algo_actions_pro: ', pro_algo_actions)
    # print('sample_actions:', sample_actions)
    pro_sample_actions = np.array(sample_actions).astype(
        np.float) / np.sum(sample_actions)
    # print('sample_actions_pro: ', pro_sample_actions)

    # print('For algo actions: ')
    algo_treat_lift = [0] * n_treat
    algo_treat_resp = [0] * n_treat
    algo_treat_base = [0] * n_treat

    real_treat_lift = [0] * n_treat
    real_treat_resp = [0] * n_treat
    real_treat_base = [0] * n_treat
    sample_lifts = [0] * n_treat
    exp_lifts = [0] * n_treat
    for a in range(n_action)[1:]:
        for i in range(n_treat):
            lift_treatment[i][a] = sum(
                same_treatment[a][i]) * 1. / (max(len(same_treatment[a][i]), 1))
            lift_control[i][a] = sum(
                same_control[a][i]) * 1. / (max(len(same_control[a][i]), 1))
            sample_treatment[i][a] = sum(
                sample_response[a][i]) * 1. / (max(len(sample_response[a][i]), 1))
            sample_control[i][a] = sum(
                sample_response[0][i]) * 1. / (max(len(sample_response[0][i]), 1))
            algo_treat_resp[i] += lift_treatment[i][a] * pro_algo_actions[a]
            algo_treat_base[i] += lift_treatment[i][a] * pro_algo_actions[a]
            algo_treat_lift[i] += (lift_treatment[i][a] -
                                   lift_control[i][a]) * pro_algo_actions[a]

            real_treat_resp[i] += sample_treatment[i][a] * pro_algo_actions[a]
            real_treat_base[i] += sample_control[i][a] * pro_algo_actions[a]
            real_treat_lift[i] += (sample_treatment[i][a] -
                                   sample_control[i][a]) * pro_algo_actions[a]
            sample_lifts[i] += (sample_treatment[i][a] -
                                sample_control[i][a]) * pro_sample_actions[a]
            exp_lifts[i] += (sample_treatment[i][a] -
                             sample_control[i][a]) * pro_algo_actions[a]

    reward_lift = 0.0
    reward_resp = 0.0
    reward_base = 0.0
    real_total_resp = 0.0
    real_total_base = 0.0
    real_total_lift = 0.0
    for i in range(n_treat):
        reward_lift += algo_treat_lift[i] * treatment_weight[i]
        reward_resp += algo_treat_resp[i] * treatment_weight[i]
        reward_base += algo_treat_base[i] * treatment_weight[i]

        real_total_resp = real_treat_resp[i] * treatment_weight[i]
        real_total_base = real_treat_base[i] * treatment_weight[i]
        real_total_lift = real_treat_lift[i] * treatment_weight[i]

    res = {}
    res['algo_total_lift'] = reward_lift
    res['algo_total_resp'] = reward_resp
    res['algo_total_base'] = reward_base
    res['algo_treat_lift'] = algo_treat_lift
    res['algo_treat_resp'] = algo_treat_resp
    res['algo_treat_base'] = algo_treat_base
    res['algo_action_resp'] = lift_treatment
    res['algo_action_base'] = lift_control
    res['algo_action_prob'] = sum_actions

    res['real_action_prob'] = sample_actions
    res['real_action_resp'] = sample_treatment
    res['real_action_base'] = sample_control
    res['real_treat_lift'] = real_treat_lift
    res['real_treat_resp'] = real_treat_resp
    res['real_treat_base'] = real_treat_base

    res['rand_total_lift'] = real_total_lift
    res['rand_total_resp'] = real_total_resp
    res['rand_total_base'] = real_total_base
    res['rand_action_prob'] = sample_actions
    res['rand_action_resp'] = sample_treatment
    res['rand_action_base'] = sample_control
    res['rand_treat_lift'] = real_treat_lift
    res['rand_treat_resp'] = real_treat_resp
    res['rand_treat_base'] = real_treat_base

    return res


def uplift_curve(datas):
    '''
    Inputs:
    records_prob: [pro0, pro1, ..., prob_N, reaction, action_real] for each treatment
    reactions: the reaction of customer
    Return:
    The area between random curve and model curve
    '''
    # records_prob[:, 0] = np.clip(records_prob[:, 0], a_min=0, a_max=1)
    # records_prob[:, 1] = np.clip(records_prob[:, 1], a_min=0, a_max=1)
    # val_min = np.min(records_prob)
    # val_max = np.max(records_prob)
    # records_prob[:, 0] = (records_prob[:, 0] - np.min(records_prob[:, 0])) / (np.max(records_prob[:, 0]) - np.min(records_prob[:, 0]))
    treatments = [data for data in datas if int(data[-1]) > 0]
    controls = [data for data in datas if int(data[-1]) == 0]

    # print('treatment', treatments)
    # print('contol', controls)

    def uplift_curve_plot(records_prob, title):
        # records_prob[:, 1] = (records_prob[:, 1] - np.min(records_prob[:, 1])) / \
        #     (np.max(records_prob[:, 1]) - np.min(records_prob[:, 1]))
        recs_uplift = []
        recs_random = []
        for rec in records_prob:
            rec_prob = rec[:2]
            reaction = rec[2]
            act_real = int(rec[3])
            act = np.argmax(rec_prob)
            pro = rec_prob[1]
            recs_uplift.append((act, pro, rec_prob, reaction))
        recs_uplift = sorted(recs_uplift, key=lambda x: -x[1])
        indexs = np.arange(len(recs_uplift))
        np.random.shuffle(indexs)
        for idx in indexs:
            # rec_prob = records_prob[idx]
            # reaction = reactions[idx]
            # act = np.argmax(rec_prob)
            # pro = rec_prob[act]
            recs_random.append(recs_uplift[idx])

        print('recs_random', len(recs_random))
        print('recs_uplift', len(recs_uplift))
        pos = 0
        step = 0.1
        x = np.arange(0, 1, step)
        tmp_uplift = 0
        tmp_random = 0
        sum_random = [0] * len(x)
        sum_uplift = [0] * len(x)
        for i, pth in enumerate(np.arange(0, 1, step)[::-1]):
            # print('pth', pth, len(recs_uplift[pos]), len(recs_uplift))
            # print('pos', pos)
            # print('recs_uplift', recs_uplift[pos], recs_uplift[pos][1])
            while pos + 1 < (len(recs_uplift) * (1 - pth)):
                tmp_uplift += recs_uplift[pos][3]
                tmp_random += recs_random[pos][3]
                pos += 1
            sum_uplift[i] = tmp_uplift
            sum_random[i] = tmp_random
        print('x', len(x), 'sum_uplift', len(sum_uplift))
        plt.title(title)
        up, = plt.plot(x, sum_uplift, 'r-o', label='uplift')
        rd, = plt.plot(x, sum_random, 'g--', label='random')
        plt.legend(handles=[up, rd])
        plt.show()
        return

    uplift_curve_plot(treatments, title='Treatment')
    uplift_curve_plot(controls, 'Control')


def data2rec(datas, n_action=2):
    recs = []
    for data in datas:
        algo_action = np.argmax(data[:n_action])
        real_action = int(data[-1])
        reaction = data[n_action]
        recs.append([algo_action, real_action, reaction])
    return recs


def metric_real(records, treatment_weight, treatment_keys, n_action):
    algo_lift = []
    algo_resp = []
    algo_base = []

    algo_action_lift = {}
    algo_action_resp = {}
    algo_action_base = {}
    for a in range(n_action):
        algo_action_lift[a] = []
        algo_action_resp[a] = []
        algo_action_base[a] = []

    for rec in records:
        algo_action, real_action, Y, B, lift = rec
        tmp_lift = 0.0
        if algo_action > 0:
            tmp_lift = lift[algo_action - 1]
        algo_lift.append(tmp_lift)
        algo_base.append(B)
        algo_resp.append(B + tmp_lift)
        algo_action_lift[algo_action].append(tmp_lift)
        algo_action_base[algo_action].append(B)
        algo_action_resp[algo_action].append(B + tmp_lift)
    print('algo_base', len(algo_base), np.sum(algo_base))
    print('algo_resp', len(algo_resp), np.sum(algo_resp))
    algo_lift = np.sum(algo_lift) * 1. / len(algo_lift)
    algo_base = np.sum(algo_base) * 1. / len(algo_base)
    algo_resp = np.sum(algo_resp) * 1. / len(algo_resp)
    for a in range(n_action):
        algo_action_lift[a] = np.sum(
            algo_action_lift[a]) * 1. / (len(algo_action_lift[a]) + 1)
        algo_action_base[a] = np.sum(
            algo_action_base[a]) * 1. / (len(algo_action_base[a]) + 1)
        algo_action_resp[a] = np.sum(
            algo_action_resp[a]) * 1. / (len(algo_action_resp[a]) + 1)
    res = {}
    res['algo_lift'] = algo_lift
    res['algo_base'] = algo_base
    res['algo_resp'] = algo_resp

    res['algo_action_lift'] = algo_action_lift
    res['algo_action_base'] = algo_action_base
    res['algo_action_resp'] = algo_action_resp

    return res


def data2rec(datas, n_action=2):
    recs = []
    for data in datas:
        algo_action = np.argmax(data[:n_action])
        real_action = int(data[-1])
        reaction = data[n_action]
        recs.append([algo_action, real_action, reaction])
    return recs


def rec2qini(recs):
    '''
    recs = [[algo, real, response, prob]]
    '''
    y_true, d_pred, group = [], [], []
    for rec in recs:
        # print('rec', rec)
        y_true.append(rec[2]['reward'])
        d_pred.append(rec[3])
        group.append(rec[1])

    return y_true, d_pred, group


def qini2recs(y_true, d_pred, group):
    recs = []
    for resp, prob, real in zip(y_true, d_pred, group):
        if prob >= 0.5:
            algo = 1
        else:
            algo = 0
        recs.append([algo, real, resp])
    return recs


def _uplift(responses_control, responses_target, n_control, n_target):
    if n_control == 0:
        return responses_target
    else:
        return responses_target - responses_control * n_target * 1. / n_control


def uplift_curve(y_true, d_pred, group, n_nodes=None):
    if n_nodes is None:
        n_nodes = min(len(y_true) + 1, 200)

    sorted_ds = sorted(zip(d_pred, group, y_true), reverse=True)
    responses_control, responses_target, n_control, n_target = 0, 0, 0, 0
    cumulative_responses = [
        (responses_control, responses_target, n_control, n_target)]

    # print('sorted_ds', sorted_ds)
    for _, is_target, response in sorted_ds:
        if is_target:
            n_target += 1
            responses_target += response
        else:
            n_control += 1
            responses_control += response
        cumulative_responses.append(
            (responses_control, responses_target, n_control, n_target))

    # print('n_nodes', n_nodes)
    # print('cumulative_responses')
    # for ele in cumulative_responses:
    #     print(ele)
    xs = [int(i) for i in np.linspace(0, len(y_true), n_nodes)]
    ys = [_uplift(*cumulative_responses[x]) for x in xs]

    return xs, ys


def number_responses(y_true, group):

    responses_target, responses_control, n_target, n_control = 0, 0, 0, 0
    for is_target, y in zip(group, y_true):
        if is_target:
            n_target += 1
            responses_target += y
        else:
            n_control += 1
            responses_control += y

    rescaled_responses_control = 0 if n_control == 0 else responses_control * \
        n_target / n_control

    return responses_target, rescaled_responses_control


def optimal_uplift_curve(y_true, group, negative_effects=True):

    responses_target, rescaled_responses_control = number_responses(
        y_true, group)

    if negative_effects:
        xs = [0, responses_target, len(
            y_true) - rescaled_responses_control, len(y_true)]
        ys = [0, responses_target, responses_target,
              responses_target - rescaled_responses_control]
    else:
        xs = [0, responses_target - rescaled_responses_control, len(y_true)]
        ys = [0, responses_target - rescaled_responses_control,
              responses_target - rescaled_responses_control]

    return xs, ys


def null_uplift_curve(y_true, group):

    responses_target, rescaled_responses_control = number_responses(
        y_true, group)
    return [0, len(y_true)], [0, responses_target - rescaled_responses_control]


def area_under_curve(xs, ys):
    area = 0
    for i in range(1, len(xs)):
        delta = xs[i] - xs[i - 1]
        y = (ys[i] + ys[i - 1]) / 2
        area += y * delta
    return area


def qini(y_true, d_pred, group, negative_effects):

    # print('y_true', y_true[:10])
    # print('d_pred', d_pred[:10])
    # print('group', group[:10])

    # Xs, ys = uplift_curve(y_true, d_pred, group)

    Xs_null, ys_null = null_uplift_curve(y_true, group)
    Xs_model, ys_model = uplift_curve(y_true, d_pred, group)
    Xs_optimal, ys_optimal = optimal_uplift_curve(
        y_true, group, negative_effects)

    # print('Xs_model', Xs_model)
    # print('ys_model', ys_model)
    # print('Xs', Xs[:10])
    # print('ys', ys[:10])
    #
    # print('area_optimal')
    area_optimal = area_under_curve(
        *optimal_uplift_curve(y_true, group, negative_effects))
    # print('area_model')
    area_model = area_under_curve(*uplift_curve(y_true, d_pred, group))
    # print('area_null')
    area_null = area_under_curve(*null_uplift_curve(y_true, group))
    # print('area_null')

    control_null = area_null / Xs_null[-1]
    control_model = area_model * 1.0 / Xs_model[-1]
    control_optimal = area_optimal * 1.0 / Xs_optimal[-1]

    # print('control_null', control_null)
    # print('control_model', control_model)
    # print('control_optimal', control_optimal)

    # if area_model < 0:
    # print('area_optimal', area_optimal)
    # print('area_model', area_model)
    # print('area_null', area_null)
    #     exit()
    # _ = '_'
    return (area_model - area_null) / (area_optimal - area_null)
    # , None, None, control_optimal, control_model, control_null


def qini_Q(y_true=None, d_pred=None, group=None, record=None, n_action=2):
    # print('qini_Q')
    if record is not None:
        y_true, d_pred, group = rec2qini(record)
        # print('recs is not None')
        # print('record', record)
# print('y_true', y_true)
# print('d_pred', d_pred)
# print('group', group)

    return qini(y_true, d_pred, group, negative_effects=True)


def qini_q(y_true=None, d_pred=None, group=None, record=None, n_action=2):
    if record is not None:
        # print('recs is not None')
        # print('record', record)
        # print('record', record)
        # exit()
        y_true, d_pred, group = rec2qini(record)

    return qini(y_true, d_pred, group, negative_effects=False)


def KL(p, q):
    ans = 0
    for pi, qi in zip(p, q):
        ans += pi * np.log(pi / qi)
    return ans


def qini_discrete(y_true=None, d_pred=None, group=None, record=None, n_action=2):
    '''
    record = [algo, real, response, prob]
    '''
    Y_c = np.array([0, 0])
    Y_t = np.array([0, 0])
    N_ar = np.zeros((2, 2))
    N_a = np.zeros(2)
    for rec in record:
        algo, real, response, prob = rec[0], rec[1], rec[2], rec[3]
        if algo == 1:
            Y_t[real] += response
            N_a[1] += 1
        else:
            Y_c[real] += response
            N_a[0] += 1
        N_ar[algo, real] += 1

    print('Y_t', Y_t, 'Y_c', Y_c)

    val_up = (Y_c[1] - Y_c[0]) / N_a[0]
    val_dn = (Y_t[1] - Y_t[0]) / N_a[1]
    # ans = (val_dn - val_up) - KL(N_ar[0, :], N_ar[1, :]) / 100000
    ans = val_dn * 2 + val_up
    print('ans', ans)
    return ans


def policy(x, lift):
    tmp = [0.0]
    for x in lift:
        tmp.append(x)
    lift = tmp
    n_action = len(lift)
    prob = np.array(lift) / np.sum(lift)
    action = np.random.choice(n_action, p=prob)
    return action, prob





if __name__ == '__main__':
    import sys
    sys.path.append('../src/')
    from SimulationData import SimulationData
    reader = SimulationData(path_load='../dataset/SimulationData.npy')
    trains, validates, test = reader.get_datas()
    records_algo = []
    records_real = []
    treatment_weight = [1.0]
    treatment_keys = [0]
    n_action = reader.n_action
    for data in trains:
        X, real_action, Y, B, lift, real_prob = data
        real_prob = np.ones(n_action).astype(np.floating) / n_action
        resp = {}
        resp[0] = Y
        algo_action, algo_prob = policy(X, lift)
        records_algo.append(
            [algo_action, real_action, resp, real_prob, algo_prob])
        records_real.append([algo_action, real_action, Y, B, lift])

    res = UMG(records=records_algo,
                    treatment_weight=treatment_weight, treatment_keys=treatment_keys, n_action=n_action)
    print('IPS')
    print('algo_total_lift', res['algo_total_lift'])
    print('algo_total_resp', res['algo_total_resp'])
    print('algo_total_base', res['algo_total_base'])
    print('algo_action_resp', res['algo_action_resp'])
    print('algo_action_base', res['algo_action_base'])

    # print('rand_lift', res['rand_lift'])
    # print('rand_total_lift', res['rand_total_lift'])
    # print('rand_total_resp', res['rand_total_resp'])
    # print('rand_total_base', res['rand_total_base'])
    # print('rand_action_resp', res['rand_action_resp'])

    # print('rand_lift', res['rand_lift'])
    # print('rand_total_lift', res['rand_total_lift'])
    # print('rand_total_resp', res['rand_total_resp'])
    # print('rand_total_base', res['rand_total_base'])
    # print('rand_action_resp', res['rand_action_resp'])
    # print('rand_action_base', res['rand_action_base'])
    # res['real_action_prob'] = sample_actions
    # res['real_action_resp'] = sample_treatment
    # print('real_lift', res['real_lift'])
    print('real_action_prob', res['real_action_prob'])
    print('real_action_resp', res['real_action_resp'])
    # print('real_action_resp', res['real_action_resp'])
    # print('real_action_nums', res['real_action_nums'])
    # print('real_action_prob', res['real_action_prob'])
    # print('algo_action_lift', res['algo_action_lift'])

    res = metric_real(records=records_real,
                      treatment_weight=treatment_weight, treatment_keys=[0], n_action=n_action)
    print('Real')
    print('algo_lift', res['algo_lift'])
    print('algo_resp', res['algo_resp'])
    print('algo_base', res['algo_base'])
    print('algo_action_resp', res['algo_action_resp'])
    print('algo_action_base', res['algo_action_base'])

    # datas = np.load('../output/train_max.npy')
    # print('probs', datas)
    # records = data2rec(datas)
    # print('algo')
    # algo_lift = same_diff(records)
    # records_random = []
    # for rec in records:
    #     algo = np.random.randint(2)
