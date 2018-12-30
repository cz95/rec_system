#!/usr/bin/Python
# -*- coding: utf-8 -*-
import math
import random
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from neighbor_rec.usercf import UserCF


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
        Set top-n value, find the most similar users with user_n,
        recommend item_n movies
        :param user_n:
        :param item_n:
        :return:
        """
        self.user_n = user_n
        self.item_n = item_n

    def _get_recommend(self, user):
        """
        Get the recommended item for the user
        :param user:
        :return:
        """
        return self.user_cf.calculate(target_user_id=user, user_n=self.user_n,
                                      item_n=self.item_n, type=2)

    def precision(self, user_list):
        """
        Calculate the precision rate
        (recommend the correct item) / (all recommended item sets)
        :param user_list:
        :return:
        """
        hit = 0
        all_recom = 0
        print('Calculate precision: ')
        for user in tqdm(user_list):
            recom_data = self._get_recommend(user)
            recom_item = set([data[0] for data in recom_data])
            user_item = set(
                self.test[self.test['userId'] == user]['movieId'].values)
            overlap = recom_item & user_item
            hit += len(overlap)
            all_recom += len(recom_item)
        print('\nprecision is: ', hit / (all_recom * 1.0))
        return hit / (all_recom * 1.0)

    def recall(self, user_list):
        """
        Calculate the recall rate
        (recommend the correct item) / (The collection of items that the user likes on the test set)
        :param user_list:
        :return:
        """
        hit = 0
        like_item = 0
        print('\nCalculate recall: ')
        for user in tqdm(user_list):
            recom_data = self._get_recommend(user)
            recom_item = set([data[0] for data in recom_data])
            user_item = set(
                self.test[self.test['userId'] == user]['movieId'].values)
            overlap = recom_item & user_item
            hit += len(overlap)
            like_item += len(user_item)
        print('\nrecall is: ', hit / (like_item * 1.0))
        return hit / (like_item * 1.0)

    def coverage(self, user_list):
        """
        Calculated coverage, (Final recommended items)/(all items)
        :param user_list:
        :return:
        """
        all_recom_set = set()
        all_item = set(self.train['movieId'].values)
        print('\nCalculated coverage: ')
        for user in tqdm(user_list):
            recom_data = self._get_recommend(user)
            recom_item = set([data[0] for data in recom_data])
            all_recom_set.update(recom_item)
        print('\nCoverage is: ', len(all_recom_set) / (len(all_item) * 1.0))
        return len(all_recom_set) / (len(all_item) * 1.0)

    def popularity(self, user_list):
        """
        Calculate popularity. The hotter the item, the less popularity
        :param user_list:
        :return:
        """
        item_popular = Counter(self.train['movieId'].values)
        ret = 0
        n = 0
        print('\nCalculate popularity: ')
        for user in tqdm(user_list):
            recom_data = self._get_recommend(user)
            for rec in set([data[0] for data in recom_data]):
                ret += math.log(1 + item_popular.get(rec))
                n += 1
        ret /= n * 1.0
        print('\npopularity: ', ret)
        return ret

    def calculate(self, seed=1):
        """
        Calculate a single metric
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
        Calculate all the indicators
        :param calcu_user_n: the number of users
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
        hit = 0  # Hit score
        all_recom = 0  # num of all recommendations, calculate the accuracy rate
        like_item = 0  # num of the item the user likes in the test set, calculate the recall rate
        all_recom_set = set()
        all_item = set(self.train['movieId'].unique())
        item_popular = Counter(self.train['movieId'].values)
        ret = 0
        n = 0
        print('\nCalculate all evaluation indicators...')
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
        print('\nCalculate over')
        print('Precision is: ', hit / (all_recom * 1.0))
        print('Recall is: ', hit / (like_item * 1.0))
        print('Coverage is: ', len(all_recom_set) / (len(all_item) * 1.0))
        print('Popularity is:', (ret / n * 1.0))


if __name__ == '__main__':
    indic = Indicators(path='./data/ml-latest-small/ratings.csv')
    indic.calculate_total(40)
