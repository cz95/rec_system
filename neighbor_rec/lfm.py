#!/usr/bin/Python
# -*- coding: utf-8 -*-
import pickle
import math
import random
import os
import pandas as pd
import numpy as np
from time import time
from collections import Counter


class Corpus(object):
    item_dict_path = './data/lfm/lfm_items.dict'

    @classmethod
    def pre_process(cls, data):
        """
        预处理一下，获得每个用户的正负样本并存储
        :param data:
        :return:
        """
        if not os.path.exists('./data/lfm/'):
            os.makedirs('./data/lfm/')
        if os.path.exists(cls.item_dict_path):
            return
        cls.data = data
        cls.user_list = cls.data['userId'].unique()
        cls.item_list = cls.data['movieId'].unique()
        cls.item_count = Counter(cls.data['movieId'])
        cls.item_popular = [x[0] for x in cls.item_count.most_common()]
        cls.item_dict = {user_id: cls._get_pos_neg_item(user_id) for user_id in
                         cls.user_list}
        cls.save()

    @classmethod
    def _get_pos_neg_item(cls, user_id):
        """
        定义感兴趣的和不感兴趣的项目
        感兴趣为用户评分过的
        不感兴趣为用户没评分
        负样本/正样本 = 10（负样本选取很热门却没有行为的项目）
        :param user_id:
        :return:
        """
        user_data = cls.data[cls.data['userId'] == user_id][
            ['movieId', 'rating']]
        watched_item_data = user_data.set_index("movieId").to_dict()['rating']
        pos_item_list = set(user_data['movieId'])
        neg_item_list = [x for x in cls.item_popular if x not in pos_item_list]
        neg_item_list = list(neg_item_list)[:(len(pos_item_list) * 10)]
        item_dict = {}
        for item in pos_item_list:
            item_dict[item] = watched_item_data[item]
        for item in neg_item_list:
            item_dict[item] = 0
        return item_dict

    @classmethod
    def save(cls):
        f = open(cls.item_dict_path, 'wb')
        pickle.dump(cls.item_dict, f)
        f.close()
        pass

    @classmethod
    def load(cls):
        f = open(cls.item_dict_path, 'rb')
        item_dict = pickle.load(f)
        f.close()
        return item_dict


class LFM(object):
    lfm_dir = './data/lfm/lfm.neighbor_rec'

    def __init__(self, path):
        self.file_path = path
        self.feature_n = 50
        self.iter_count = 20
        self.lr = 0.02
        self.lamda = 0.01
        self._init_model()

    def _init_model(self):
        self.data = pd.read_csv(self.file_path)
        Corpus.pre_process(self.data)
        self.user_list = self.data['userId'].unique()
        self.item_list = self.data['movieId'].unique()
        self.item_dict = Corpus.load()
        if not os.path.exists(self.lfm_dir):
            array_p = np.random.randn(len(self.user_list), self.feature_n)
            array_q = np.random.randn(len(self.item_list), self.feature_n)
            self.p = pd.DataFrame(array_p, columns=range(0, self.feature_n),
                                  index=list(self.user_list))
            self.q = pd.DataFrame(array_q, columns=range(0, self.feature_n),
                                  index=list(self.item_list))
            self.train()

    def _predict(self, user_id, item_id):
        p = np.mat(self.p.ix[user_id].values)
        q = np.mat(self.q.ix[item_id].values).T
        pred = p * q
        return pred[0, 0]

    def _loss(self, user_id, item_id, y):
        e = y - self._predict(user_id, item_id)
        return e

    def _optimize(self, user_id, item_id, e):
        """
        SGD，误差为sse，RSVD
        e.g: E = 1/2 * (y - predict)^2, predict = matrix_p * matrix_q
             derivation(E, p) = -matrix_q*(y - predict)
             derivation(E, q) = -matrix_p*(y - predict)
             derivation（l2_square，p) = lam * p
             derivation（l2_square, q) = lam * q
             delta_p = lr * (derivation(E, p) + derivation（l2_square，p))
             delta_q = lr * (derivation(E, q) + derivation（l2_square, q))
        """
        grad_p = -e * self.q.ix[item_id].values
        l2_p = self.lamda * self.p.ix[user_id].values
        delta_p = self.lr * (grad_p + l2_p)

        grad_q = -e * self.p.ix[user_id].values
        l2_q = self.lamda * self.q.ix[item_id].values
        delta_q = self.lr * (grad_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    def train(self):
        for step in range(0, self.iter_count):
            print('Step: {}'.format(step))
            for user_id, item_dict in self.item_dict.items():
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    e = self._loss(user_id, item_id, item_dict[item_id])
                    self._optimize(user_id, item_id, e)
            self.lr *= 0.995
        rmse = 0
        count = 0
        for user_id, item_dict in self.item_dict.items():
            item_ids = list(item_dict.keys())
            random.shuffle(item_ids)
            for item_id in item_ids:
                e = self._loss(user_id, item_id, item_dict[item_id])
                self._optimize(user_id, item_id, e)
                rmse += e ** 2
                count += 1
        rmse = np.sqrt(rmse / count)
        print('最终, RMSE = {}'.format(rmse))
        self.save()

    def predict(self, user_id, top_n=10):
        self.load()
        item = set(
            self.data[self.data['userId'] == user_id]['movieId'].unique())
        other_item = set(self.item_list) ^ item
        interest_list = [self._predict(user_id, item_id) for item_id in
                         other_item]
        candi = sorted(zip(list(other_item), interest_list), key=lambda x: x[1],
                       reverse=True)
        return candi[:top_n]

    def save(self):
        if not os.path.exists('./data/lfm/'):
            os.makedirs('./data/lfm/')
        f = open(self.lfm_dir, 'wb')
        pickle.dump((self.p, self.q), f)
        f.close()

    def load(self):
        f = open(self.lfm_dir, 'rb')
        self.p, self.q = pickle.load(f)
        f.close()


if __name__ == '__main__':
    file_path = './data/ml-latest-small/ratings.csv'
    start = time()
    movies = LFM(file_path).predict(user_id=1)
    print(movies)
    print('用时：', time() - start)
