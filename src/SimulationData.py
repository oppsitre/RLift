import numpy as np
import time
np.random.seed(int(time.time()))


class SimulationData:
    def __init__(self, n_feature=50, n_action=2, n_sizes=[2000, 2000], splits=[0.6, 0.2, 0.2], func=None, path_load=None, binary_respones=False, binary_actions=False, path_save=None):
        self.splits = splits
        self.datas = []
        self.name = 'Simulation'
        self.datas_train = None
        self.datas_test = None
        self.datas_valid = None
        if path_load is None:
            self.n_feature = n_feature
            self.n_sizes = n_sizes
            self.n_action = n_action
            self.alpha = 0.4
            self.generate_data()
        else:
            self.datas = np.load(path_load, encoding="latin1")
            # self.datas = np.load(path_load)   
            resp_mean = np.mean(self.datas[:, 2])
            self.n_feature = len(self.datas[0, 0])
            self.n_action = len(self.datas[0, 4]) + 1
            if binary_respones is True:
                self.datas[:, 2] = np.ones_like(
                    self.datas[:, 2]) * (self.datas[:, 2] >= resp_mean)

            if binary_actions is True:
                tmp = []
                for data in self.datas:
                    if data[1] < 2:
                        tmp.append(data)
                self.datas = tmp
            # for i, resp in enumerate(self.data[:, 2]):
            #     if resp > resp_mean:
            #         self.datas[i, 2] = 1
            #     else:
            #         self.datas[i, 2] = 0
            # print('n_size', self.n_sizes)

        print('n_feature', self.n_feature)
        print('n_action', self.n_action)

    # def policy(self, x):
    #     confound = int(x[1] * 100) % self.n_action
    #     prob = np.random.uniform(high=0.5, size=self.n_action)
    #     prob = np.ones((self.n_action)).astype('float')
    #     prob[confound] = 5.0
    #     prob /= np.sum(prob)
    #     algo_action = np.random.choice(self.n_action, size=1, p=prob)[0]
    #     return algo_action, prob
    def policy(self, x):
        prob = np.ones(self.n_action).astype(np.floating) / self.n_action
        action = np.random.choice(self.n_action)
        return action, prob

    def minibatch(self, datas=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if datas is None:
            datas = self.get_datas(tag='train')
        idx = np.random.choice(len(datas), batch_size)
        np.random.shuffle(idx)
        batch = []
        for x in idx[:batch_size]:
            batch.append(datas[x])
        return batch

    def extract_cf(self, datas):
        X = []
        Y = []
        A = []
        R = []
        P = []
        for data in datas:
            # print('data', data)
            # exit()
            X.append(data[0])
            A.append(data[1])
            Y.append(data[2])
            base = data[3]
            lift = np.hstack((0, data[4]))
            R.append(base + lift)
            # print('data)
            if data[5] is None:
                prob = np.ones(self.n_action).astype(np.float) / self.n_action
                P.append(prob)
            else:
                P.append(data[5])
        X = np.array(X)
        A = np.reshape(np.array(A), (-1, 1))
        Y = np.reshape(np.array(Y), (-1, 1))
        R = np.reshape(np.array(R), (-1, self.n_action))
        # P = np.array(P)
        # print('prob', P.shape)
        P = np.reshape(np.array(P), (-1, self.n_action))
        return X, A, Y, R, P
    # def policy(self, x, n_action):
    #     prob = np.ones(n_action) / n_action
    #     action = np.random.choice(n_ation)
        # a_lst = [[0], [0], [0], [0], [0]]
        # a_lst[4] = [0, 1, 2, 3, 4]
        # a_lst[3] = [5, 6, 7, 8]
        # a_lst[2] = [9, 10, 11]
        # a_lst[1] = [12, 13]
        # a_lst[0] = [14]
        # # print('x', x)
        # # print(x[1])
        # mod = int(x[1] * 100) % 15
        # confound = -1
        # for i in range(n_action):
        #     # print(i, a_lst[i])
        #     if mod in a_lst[i]:
        #         confound = i
        #         break
        # # print('mod', mod, 'confound', confound)
        # prob = np.random.uniform(high=0.5, size=n_action)
        # prob = np.ones((n_action)).astype('float')
        # prob[confound] = 5.0
        # prob /= np.sum(prob)
        # algo_action = np.random.choice(n_action, size=1, p=prob)[0]
        # return algo_action, prob

    def generate_data(self):
        Xs = self.base_features()
        bases = self.base_response(Xs) * 5
        lifts = self.treat_lift(Xs)
        print('lifts shape', lifts.shape)
        self.lifts = lifts
        # print('Xs', Xs.shape)
        # print('bases', bases.shape)
        # print('lifts', lifts.shape)
        print('bases', np.mean(bases, axis=0))
        print('lifts', np.mean(lifts, axis=0))
        actions = []
        probs = []
        for x in Xs:
            a, p = self.policy(x)
            actions.append(a)
            probs.append(p)
        # for a, sz in enumerate(self.n_sizes):
        #     actions.extend([a] * sz)
        actions = np.reshape(np.array(actions), (-1, 1))
        probs = np.array(probs)
        # print('actions', actions)
        respons = bases.copy()
        print('respons mean', np.mean(respons))
        self.datas = []
        for i, a in enumerate(actions):
            # val_lift = 0
            # print(i, respons.shape, lifts.shape)
            # print('action', a)
            if a > 0:
                respons[i] += lifts[i, a - 1]
                # val_lift = lifts[i, a - 1]
            self.datas.append(
                [Xs[i, :], a[0], respons[i, 0], bases[i, 0], lifts[i, :], probs[i, :]])
            # self.raw_data.append(
            #     [Xs[i, :], a[0], respons[i, 0], bases[i, 0], lifts[i, :]])
        self.save()

    def base_features(self):
        self.Xs = np.random.rand(np.sum(self.n_sizes), self.n_feature) * 10
        return self.Xs

    def base_response(self, Xs):
        a = np.random.rand(self.n_feature) * 10
        b = -np.random.rand(self.n_feature, self.n_feature) * 0.1
        c = np.random.rand(self.n_feature, self.n_feature) * 5
        responses = []
        for X in Xs:
            s1 = np.sum(b * np.abs(X - c), axis=1)
            s2 = np.sum(a * np.exp(s1))
            responses.append(s2)
        return np.reshape(np.array(responses), (-1, 1))

    def treat_lift(self, Xs, action=1):
        print('Xs', Xs.shape)
        # bases = np.random.rand(len(Xs))
        # bases = np.ones(len(Xs))
        lifts = None
        for a in range(self.n_action)[1:]:
            tmp = self.base_response(Xs)
            # tmp = np.reshape(
            #     self.alpha * Xs[:, a] * bases, (-1, 1))
            if lifts is None:
                lifts = tmp
            else:
                lifts = np.hstack((lifts, tmp))
        return np.array(lifts)

    def normalize(self, datas):
        '''
        Input: [Xs[i, :], a[0], respons[i, 0], bases[i, 0], lifts[i, :], probs[i, :]]
        Return: [feature, real_action, response, base, lifts, probs]
        '''
        features = np.array([data[0] for data in datas])
        features -= np.mean(features, axis=0)
        features /= np.std(features, axis=0)
        res = []
        for i, data in enumerate(datas):
            res.append([features[i, :], data[1], data[2],
                        data[3], data[4], None])
        return res

    def split_datas(self, splits=None):
        if self.datas_train is not None:
            return self.datas_train, self.datas_validate, self.datas_test

        if splits is None:
            # split is ratio form
            splits = self.splits
            n_all = len(self.datas)
            n_train = int(n_all * splits[0])
            self.n_train = n_train
            n_validate = int(n_all * splits[1])
            self.n_validate = n_validate
            n_test = n_all - self.n_train - self.n_validate
        else:
            # split is extact value form
            n_all = np.sum(splits)
            n_train = splits[0]
            n_validate = splits[1]
            n_test = splits[2]

        # print(type(self.datas))
        # print('n_all', n_all)
        # exit()
        datas = self.datas[:n_all]
        datas = self.normalize(datas)
        indexs = np.arange(n_all)
        np.random.shuffle(indexs)

        # print('n_train', n_train, 'n_validate',
        #       n_validate, 'n_test', n_test)
        # print('datas', self.datas[0])
        datas_train = [datas[i] for i in indexs[:n_train]]
        datas_test = [datas[i] for i in indexs[-n_test:]]
        datas_validate = [
            datas[i] for i in indexs[n_train:(n_train + n_validate)]]

        self.datas_train = datas_train
        self.datas_validate = datas_validate
        self.datas_test = datas_test
        return datas_train, datas_validate, datas_test

    def statistic(self, datas=None):
        if datas is None:
            datas = self.datas
        resp = {}
        base = {}
        for a in range(self.n_action):
            resp[a] = []
            base[a] = []
        for data in datas:
            a = data[1]
            r = data[2]
            b = data[3]
            resp[a].append(r)
            base[a].append(b)

        print('Response')
        for a in range(self.n_action):
            print('a', np.mean(resp[a]))
        print('Base')
        for a in range(self.n_action):
            print('a', np.mean(base[a]))

    def get_datas(self, tag='all', rep=None):
        if self.datas_train is None:
            self.split_datas()
        if tag == 'train':
            return self.datas_train
        if tag == 'validate':
            return self.datas_validate
        if tag == 'test':
            return self.datas_test

        if tag == 'all':
            return self.datas_train, self.datas_validate, self.datas_test

    def save(self):
        np.save('SimulationData_random.npy', np.array(self.datas))
        print('Data saved')

    # def load(self, path):


if __name__ == '__main__':
    n_action = 5
    # sd = SimulationData(n_feature=50, n_action=5, n_sizes=[500000] * n_action)

    reader = SimulationData(path_load='../dataset/SimulationData_random.npy')
    trains, tests, validates = reader.get_datas()
    reader.statistic()
    # print(sd.datas[:1000])
    # data = np.load('SimulationData.npy')
    # print('data', data.shape)
    # print(data[-1, :])
    # trains, validates, tests = sd.get_datas()
    # # sd.generate_data()
    # # print(sd.datas.shape)
    # sd.save()
    # print('train', len(trains))
    # print(trains[0])
    # print('validates', len(validates))
    # print(validates[0])
    # print('test', len(tests))
    # print(tests[0])
    # datas = sd.base_features()
    # sd.base_response(datas)
    # sd.treat_lift(datas)
