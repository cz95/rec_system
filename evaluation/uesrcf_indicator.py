#!/usr/bin/Python
# -*- coding: utf-8 -*-
import math
import random
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model.usercf import UserCF


class Indicators():
    def __init__(self, path):
        self.file_path = path
        self._init_data()
        self._split_data()

    def _init_data(self):
        self.data = pd.read_csv(self.file_path)
        self.user_n = 20
        self.item_n = 10

    def _split_data(self, test_size=0.2, seed=1):
        self.train, self.test = train_test_split(self.data, test_size=test_size,
                                                 random_state=seed)
        self.user_cf = UserCF(self.train)

    def _set_top(self, user_n, item_n):
        """
        设置topn值，找出user_n个最相似的用户，推荐item_n个电影
        :param user_n:
        :param item_n:
        :return:
        """
        self.user_n = user_n
        self.item_n = item_n

    def _get_recommend(self, user):
        """
        针对用户user拿到推荐item
        :param user:
        :return:
        """
        return self.user_cf.calculate(target_user_id=user, user_n=self.user_n,
                                      item_n=self.item_n, type=2)

    def precision(self, user_list):
        """
        计算精确率, 推荐正确item集合 / 所有推荐item集合
        :param user_list: 随机生成的user集合
        :return:
        """
        hit = 0
        all_recom = 0
        print('计算精确率：')
        for user in tqdm(user_list):
            recom_data = self._get_recommend(user)
            recom_item = set([data[0] for data in recom_data])
            user_item = set(
                self.test[self.test['userId'] == user]['movieId'].values)
            overlap = recom_item & user_item
            hit += len(overlap)
            all_recom += len(recom_item)
        print('\n精确率为：', hit / (all_recom * 1.0))
        return hit / (all_recom * 1.0)

    def recall(self, user_list):
        """
        计算召回率，推荐正确item集合 / 用户在测试集上喜欢的item集合
        :param user_list: 随机生成的user集合
        :return:
        """
        hit = 0
        like_item = 0
        print('\n计算召回率：')
        for user in tqdm(user_list):
            recom_data = self._get_recommend(user)
            recom_item = set([data[0] for data in recom_data])
            user_item = set(
                self.test[self.test['userId'] == user]['movieId'].values)
            overlap = recom_item & user_item
            hit += len(overlap)
            like_item += len(user_item)
        print('\n召回率为：', hit / (like_item * 1.0))
        return hit / (like_item * 1.0)

    def coverage(self, user_list):
        """
        计算覆盖率，最终推荐的item / 所有item
        :param user_list:
        :return:
        """
        all_recom_set = set()
        all_item = set(self.train['movieId'].values)
        print('\n计算覆盖率：')
        for user in tqdm(user_list):
            recom_data = self._get_recommend(user)
            recom_item = set([data[0] for data in recom_data])
            all_recom_set.update(recom_item)
        print('\n覆盖率为：', len(all_recom_set) / (len(all_item) * 1.0))
        return len(all_recom_set) / (len(all_item) * 1.0)

    def popularity(self, user_list):
        """
        计算新颖性，推荐物品越热门，新颖性越低
        :param user_list:
        :return:
        """
        item_popular = Counter(self.train['movieId'].values)
        ret = 0
        n = 0
        print('\n计算新颖度：')
        for user in tqdm(user_list):
            recom_data = self._get_recommend(user)
            for rec in set([data[0] for data in recom_data]):
                ret += math.log(1 + item_popular.get(rec))
                n += 1
        ret /= n * 1.0
        print('\n新颖度为：', ret)
        return ret

    def calculate(self, seed=1):
        """
        可以用于计算单个指标
        :param seed:
        :return:
        """
        self._split_data(seed=seed)
        test_user_list = list(set(self.test['userId'].unique()))
        user_list = [test_user_list[random.randint(0, len(test_user_list)) - 1]
                     for i in range(20)]
        # self.precision(user_list)
        # self.recall(user_list)
        # self.coverage(user_list)
        self.popularity(user_list)

    def calculate_total(self, calcu_user_n=20, user_n=20, item_n=10, seed=1):
        """
        计算所有指标
        :param calcu_user_n: 计算用户个数
        :param user_n:
        :param item_n:
        :param seed:
        :return:
        """
        self._split_data(seed=seed)
        self._set_top(user_n=user_n, item_n=item_n)
        test_user_list = list(set(self.test['userId'].unique()))
        user_list = [test_user_list[random.randint(0, len(test_user_list)) - 1]
                     for i in range(calcu_user_n)]
        hit = 0  # 击中长度
        all_recom = 0  # 所有用户推荐个数和，用于计算精确率
        like_item = 0  # 用户在测试集中喜欢的项目长度，用于计算召回率
        all_recom_set = set()
        all_item = set(self.train['movieId'].unique())
        item_popular = Counter(self.train['movieId'].values)
        ret = 0
        n = 0
        print('\n计算所有测评指标中...')
        for user in tqdm(user_list):
            recom_data = self._get_recommend(user, )
            recom_item = set([data[0] for data in recom_data])
            user_item = set(
                self.test[self.test['userId'] == user]['movieId'].values)
            overlap = recom_item & user_item
            hit += len(overlap)
            like_item += len(user_item)
            all_recom += len(recom_item)
            all_recom_set.update(recom_item)
            for rec in set([data[0] for data in recom_data]):
                ret += math.log(1 + item_popular.get(rec))
                n += 1
        print('\n计算完毕。')
        print('精确率为：', hit / (all_recom * 1.0))
        print('召回率为：', hit / (like_item * 1.0))
        print('覆盖率为：', len(all_recom_set) / (len(all_item) * 1.0))
        print('新颖度为：', (ret / n * 1.0))


if __name__ == '__main__':
    indic = Indicators(path='../data/ml-latest-small/ratings.csv')
    indic.calculate_total(40)
