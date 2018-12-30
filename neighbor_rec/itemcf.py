#!/usr/bin/Python
# -*- coding: utf-8 -*-
import pickle
import os
import math
import pandas as pd
import numpy as np
from time import time
from tqdm import tqdm


class Similar(object):
    @classmethod
    def _cosine_sim(cls, target_user, other_user):
        """
        Calculate cosine similarity, regardless of scores
        :param target_user:
        :param other_user:
        :return:
        """
        target_set = set(target_user['userId'].unique())
        other_set = set(other_user['userId'].unique())
        union_len = len(target_set & other_set)
        if union_len == 0:
            return 0.0
        product = len(target_user) * len(other_user)
        return union_len / math.sqrt(product)

    @classmethod
    def _cosine_sim_score(cls, target_user, other_user):
        """
        Calculate cosine similarity, consider scores
        :param target_user:
        :param other_user:
        :return:
        """
        target_dict = target_user.set_index("userId").to_dict()['rating']
        user_dict = other_user.set_index("userId").to_dict()['rating']
        union_user = set(target_dict.keys()) & set(user_dict.keys())
        if len(union_user) == 0:
            return 0.0
        score_1 = 0
        for user in union_user:
            score_1 += (target_dict[user] * user_dict[user])
        rating_1 = sum(target_user['rating'].values ** 2)
        rating_2 = sum(other_user['rating'].values ** 2)
        score_2 = math.sqrt(rating_1 * rating_2)
        return score_1 / score_2

    @classmethod
    def get_sim(cls, target_user, other_user, sim_type=1):
        """
        Provide similarity function
        type=1: without considering the score
        type=2: consider the score
        :param target_user: Target item user group with rating
        :param other_user: Other item user group with rating
        :param sim_type: Similar function type
        :return:
        """
        if sim_type == 1:
            return cls._cosine_sim(target_user, other_user)
        else:
            return cls._cosine_sim_score(target_user, other_user)


class Matrix(object):
    itemcf_path = './data/itemcf/itemcf.matrix'
    itemcf_score_path = './data/itemcf/itemcf_score.matrix'

    @classmethod
    def pre_process(cls, data, type):
        if not os.path.exists('./data/itemcf/'):
            os.makedirs('./data/itemcf/')
        if type == 1:
            if not os.path.exists(cls.itemcf_path):
                cls._item_similarity(data)
        else:
            cls._item_similarity_scroe(data)
        pass

    @classmethod
    def _item_similarity(cls, data):
        """
        Calculate the similarity matrix between items,
        regardless of the score, only consider whether to watch
        :return sim_matrix:
        """
        item = dict()
        count = dict()
        item_unique = data['movieId'].unique()
        item_len = len(item_unique)
        sim_matrix = np.zeros([item_len, item_len])
        user_list = data['userId'].unique()
        cls.item_order = {item_unique[i]: i for i in range(item_len)}
        cls.order_item = {i: item_unique[i] for i in range(item_len)}
        for userId in tqdm(user_list):
            item_list = data[data['userId'] == userId]['movieId'].unique()
            for i in item_list:
                if i not in count:
                    count[i] = 0
                count[i] += 1
                for j in item_list:
                    if i != j:
                        if (i, j) not in item:
                            item[i, j] = 0
                        item[i, j] += 1
        for i, j in tqdm(item.keys()):
            sim_matrix[cls.item_order[i]][cls.item_order[j]] = \
                item[i, j] / math.sqrt(count[i] * count[j])
        cls.save(cls.itemcf_path, sim_matrix)

    @classmethod
    def _item_similarity_scroe(cls, data):
        """
        Calculate the similarity matrix between items, consider the score
        :param data:
        :return sim_matrix:
        """
        item_list = data['movieId'].unique()
        item_len = len(item_list)
        sim_matrix = np.zeros([item_len, item_len])
        cls.item_order = {item_list[i]: i for i in range(item_len)}
        cls.order_item = {i: item_list[i] for i in range(item_len)}
        # 针对每个item获取用户的打分情况
        for i in tqdm(item_list):
            for j in item_list:
                if sim_matrix[cls.item_order[i]][cls.item_order[j]] != 0:
                    continue
                if i == j:
                    continue
                sim = Similar.get_sim(data[data['movieId'] == i],
                                      data[data['movieId'] == j], sim_type=2)
                sim_matrix[cls.item_order[i]][cls.item_order[j]] = sim
                sim_matrix[cls.item_order[j]][cls.item_order[i]] = sim
        cls.save(cls.itemcf_score_path, sim_matrix)

    @classmethod
    def save(cls, path, sim_matrix):
        f = open(path, 'wb')
        pickle.dump(sim_matrix, f)
        f.close()

    @classmethod
    def load(cls, type):
        if type == 1:
            cls.item_dict_path = cls.itemcf_path
        else:
            cls.item_dict_path = cls.itemcf_score_path
        f = open(cls.item_dict_path, 'rb')
        item_dict = pickle.load(f)
        f.close()
        return item_dict


class ItemCF:
    def __init__(self, data):
        self.data = data

    def get_item_n(self, item_id, sim_matrix, item_n):
        """
        Calculate the top n items most similar to item_id based on the similarity matrix
        :param item_id:
        :param sim_matrix:
        :param item_n:
        :return:
        """
        item_list = self.data['movieId'].unique()
        self.item_order = {item_list[i]: i for i in range(len(item_list))}
        self.order_item = {i: item_list[i] for i in range(len(item_list))}
        candi_item = {}
        for j in range(sim_matrix.shape[0]):
            if sim_matrix[self.item_order[item_id]][j] != 0:
                item = self.order_item[j]
                if item not in candi_item.keys():
                    candi_item[item] = 0
                    candi_item[item] = sim_matrix[self.item_order[item_id]][j]
        sim_items = sorted(candi_item.items(), key=lambda x: x[1], reverse=True)
        return sim_items[:item_n]

    def get_top_n(self, item_similar, user_id, item_n, top_n):
        """
        Recommend top_n items
        :param item_similar:
        :param user_id:
        :param item_n:
        :param top_n:
        :return:
        """
        candi_items = dict()
        user_data = self.data[self.data['userId'] == user_id][
            ['movieId', 'rating']]
        watched_item_data = user_data.set_index("movieId").to_dict()['rating']
        watched_item_list = user_data['movieId'].unique()
        for movie_id in tqdm(watched_item_list):
            sim_items = self.get_item_n(movie_id, item_similar, item_n)
            for data in sim_items:
                if data[0] not in candi_items.keys():
                    candi_items[data[0]] = 0
                candi_items[data[0]] += data[1] * watched_item_data[movie_id]
        # 去掉看过的
        for item in watched_item_list:
            if item in candi_items.keys():
                candi_items.pop(item)
        recom_items = sorted(candi_items.items(), key=lambda x: x[1],
                             reverse=True)
        return recom_items[:top_n]

    def calculate(self, user_id=1, item_n=20, top_n=10, sim_type=1):
        """
        :param user_id:
        :param item_n:
        :param top_n:
        :param sim_type: type=1: without considering the score  type=2: consider the score
        :return:
        """
        Matrix.pre_process(self.data, sim_type)
        item_similar = Matrix.load(sim_type)
        top_n_item = self.get_top_n(item_similar, user_id, item_n, top_n)
        return top_n_item


if __name__ == "__main__":
    start = time()
    file_path = './data/ml-latest-small/ratings.csv'
    data = pd.read_csv(file_path)
    item_cf = ItemCF(data=data)
    print(item_cf.calculate())
    print('total time is:', time() - start)
