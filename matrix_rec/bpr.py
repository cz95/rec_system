#!/usr/bin/Python
# -*- coding: utf-8 -*-

import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from collections import Counter


class BPR(object):
    bpr_dir = './data/bpr/bpr.model'

    def __init__(self, path):
        self.file_path = path
        self.feature_n = 50
        self.iter_count = 20
        self.lr = 0.02
        self.lamda = 0.01
        self._init_model()

    def _init_model(self):
        self.data = pd.read_csv(self.file_path)
        if not os.path.exists('./data/bpr/'):
            os.makedirs('./data/bpr/')
        self.user_list = self.data['userId'].unique()
        self.item_list = self.data['movieId'].unique()
        item_len = len(self.item_list)
        user_len = len(self.user_list)
        self.item_count = Counter(self.data['movieId'])
        self.item_popular = [x[0] for x in self.item_count.most_common()]
        self.sample_dict = {user_id: self._get_pos_neg_item(user_id) for user_id
                            in self.user_list}
        self.item_index = {self.item_list[i]: i for i in range(item_len)}
        self.index_item = {i: self.item_list[i] for i in range(item_len)}
        self.user_index = {self.user_list[i]: i for i in range(user_len)}
        self.index_user = {i: self.user_list[i] for i in range(user_len)}
        if not os.path.exists(self.bpr_dir):
            self.p = np.random.randn(len(self.user_list), self.feature_n)
            self.q = np.random.randn(len(self.item_list), self.feature_n)
            self._train()

    def _get_pos_neg_item(self, user_id):
        """
        Define interesting and uninteresting items
        Interesting items is rated
        Uninteresting items in not rated
        Negative sample/positive sample = 1 (negative sample selects hot items with no behaviors)
        :param user_id:
        :return:
        """
        user_data = self.data[self.data['userId'] == user_id][
            ['movieId', 'rating']]
        pos_item_list = set(user_data['movieId'])
        neg_item_list = [x for x in self.item_popular if x not in pos_item_list]
        neg_item_list = list(neg_item_list)[:(len(pos_item_list))]
        item_np = {}
        item_np['pos'] = pos_item_list
        item_np['neg'] = neg_item_list
        return item_np

    def _predict(self, user_id, pos_item, neg_item):
        u = np.mat(self.p[user_id])
        pos_i = np.mat(self.q[pos_item]).T
        neg_i = np.mat(self.q[neg_item]).T
        pred_up = u * pos_i
        pred_un = u * neg_i
        return pred_up[0, 0] - pred_un[0, 0]

    def _loss(self, user_id, pos_item, neg_item):
        e = -1.0 / (1 + np.exp(self._predict(user_id, pos_item, neg_item)))
        return e

    def _optimize(self, user_id, pos_item, neg_item, e):
        self.p[user_id] -= self.lr * (
                e * (self.q[pos_item] + self.q[neg_item]) + self.lamda *
                self.p[user_id])
        self.q[pos_item] -= self.lr * (
                e * self.p[user_id] + self.lamda * self.q[pos_item])
        self.q[neg_item] -= self.lr * (
                - e * self.p[user_id] + self.lamda * self.q[neg_item])

    def _train(self):
        for _ in tqdm(range(0, self.iter_count)):
            for user_id, item_dict in self.sample_dict.items():
                pos_item_ids = list(item_dict['pos'])
                neg_item_ids = list(item_dict['neg'])
                random.shuffle(pos_item_ids)
                random.shuffle(neg_item_ids)
                for pos_item in pos_item_ids:
                    for neg_item in neg_item_ids:
                        pos_ix = self.item_index[pos_item]
                        neg_ix = self.item_index[neg_item]
                        user_ix = self.user_index[user_id]
                        e = self._loss(user_ix, pos_ix, neg_ix)
                        self._optimize(user_ix, pos_ix, neg_ix, e)
            self.lr *= 0.995

        # 计算auc
        count = 0
        score = 0
        predict_matrix = np.mat(self.p) * np.mat(self.q).T
        for user_id, item_dict in self.sample_dict.items():
            pos_item_ids = list(item_dict['pos'])
            neg_item_ids = list(item_dict['neg'])
            random.shuffle(pos_item_ids)
            random.shuffle(neg_item_ids)
            for pos_item in pos_item_ids:
                for neg_item in neg_item_ids:
                    pos_ix = self.item_index[pos_item]
                    neg_ix = self.item_index[neg_item]
                    user_ix = self.user_index[user_id]
                    score += self.score(predict_matrix[user_ix, pos_ix],
                                        predict_matrix[user_ix, neg_ix])
                    count += 1
                    e = self._loss(user_ix, pos_ix, neg_ix)
                    self._optimize(user_ix, pos_ix, neg_ix, e)
        auc = score / count
        print('The final, AUC = {}'.format(auc))
        self.save()

    def predict(self, user_id, top_n=10):
        self.load()
        item = set(
            self.data[self.data['userId'] == user_id]['movieId'].unique())
        other_item = set(self.item_list) ^ item
        user_ix = self.user_index[user_id]
        predict_matrix = np.mat(self.p) * np.mat(self.q).T
        pred_list = predict_matrix[user_ix]
        item_sore = {id: pred_list[0, self.item_index[id]] for id in other_item}
        candi = sorted(item_sore.items(), key=lambda x: x[1], reverse=True)
        return candi[:top_n]

    def score(self, pos, neg):
        sc = 0
        if pos > neg:
            sc = 1
        if pos == neg:
            sc = 0.5
        return sc

    def save(self):
        if not os.path.exists('./data/bpr/'):
            os.makedirs('./data/bpr/')
        f = open(self.bpr_dir, 'wb')
        pickle.dump((self.p, self.q, self.user_index, self.index_user,
                     self.item_index, self.index_item), f)
        f.close()

    def load(self):
        f = open(self.bpr_dir, 'rb')
        self.p, self.q, self.user_index, self.index_user, self.item_index, self.index_item = pickle.load(f)
        f.close()


if __name__ == '__main__':
    file_path = './data/ml-latest-small/ratings.csv'
    start = time()
    movies = BPR(file_path).predict(user_id=1)
    print(movies)
    print('Use time: ', time() - start)
