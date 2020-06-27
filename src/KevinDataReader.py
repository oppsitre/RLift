import sys
sys.path.append('../../')
import numpy as np


class DataReader:
    '''
    Attributes:
    '''

    def __init__(self, action, label, splits=[0.6, 0.2, 0.2], path='/home/lcc/code/python/rlift/dataset/advertise.csv'):
        if label == 'visit':
            lid = 9
        elif label == 'conversion':
            lid = 10
        elif label == 'spend':
            lid = 11
        else:
            print('label not found')
        self.n_feature = 8
        self.n_action = 2
        self.name = 'Kevin'
        self.actions = [0]
        if 'womens' in action:
            print('womens in action')
            self.actions.append(1)
        if 'mens' in action:
            print('mens in action')
            self.actions.append(2)
        if len(self.actions) <= 1:
            print('action not found')

        self.datas = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                line = [float(x) for x in line]
                if int(line[8]) not in self.actions:
                    continue
                # data = [features, action, label]
                data = [np.array(line[:8]), int(line[8]), int(line[lid])]
                self.datas.append(data)

        self.datas = self.normalize(self.datas)
        self.atts = ['recency', 'history_segment', 'history', 'mens', 'womens',
                     'zip_code', 'newbie', 'channel', 'segment', 'visit', 'conversion', 'spend']
        self.att2pos = {}
        for i in range(len(self.atts)):
            self.att2pos[self.atts[i]] = i

        self.n_all = len(self.datas)
        indexs = np.arange(self.n_all)
        np.random.shuffle(indexs)
        self.n_train = int(self.n_all * splits[0])
        self.n_validate = int(self.n_all * splits[1])
        self.n_test = self.n_all - self.n_train - self.n_validate
        # print('n_train', self.n_train, 'n_validate', self.n_validate, 'n_test', self.n_test)
        self.datas_train = [self.datas[i] for i in indexs[:self.n_train]]
        self.datas_validate = [
            self.datas[i] for i in indexs[self.n_train:(self.n_train + self.n_validate)]]
        self.datas_test = [self.datas[i] for i in indexs[-self.n_test:]]

    def normalize(self, datas):
        features = np.array([data[0] for data in datas])
        features -= np.mean(features, axis=0)
        features /= np.std(features, axis=0)
        res = []
        for i, data in enumerate(datas):
            res.append([features[i, :], data[1], data[2]])
        return res

    def get_datas(self):
        return self.datas_train, self.datas_validate, self.datas_test

    # def uplift(self):


if __name__ == '__main__':
    reader = DataReader(action=['womens'], label='visit')
    trains, validates, tests = reader.get_datas()
