#!/usr/bin/Python
# -*- coding: utf-8 -*-
import pandas as pd
import math
from time import time


class Similar(object):
    @classmethod
    def _cosine_sim(cls, target_movies, other_movies):
        """
        计算余弦相似性， 看过为1 没看过为0
        :param target_movies:
        :param other_movies:
        :return:
        """
        target_set = set(target_movies['movieId'].unique())
        other_set = set(other_movies['movieId'].unique())
        union_len = len(target_set & other_set)
        if union_len == 0:
            return 0.0
        product = len(target_set) * len(other_set)
        return union_len / math.sqrt(product)

    @classmethod
    def _cosine_sim_score(cls, target_movies, other_movies):
        """
        计算余弦相似性 基于评分
        :param target_movies:
        :param other_movies:
        :return:
        """
        target_dict = target_movies.set_index("movieId").to_dict()['rating']
        movies_dict = other_movies.set_index("movieId").to_dict()['rating']
        union_movies = set(target_dict.keys()) & set(movies_dict.keys())
        if len(union_movies) == 0:
            return 0.0
        score_1 = 0
        for movie in union_movies:
            score_1 += (target_dict[movie] * movies_dict[movie])
        rating_1 = sum(target_movies['rating'].values ** 2)
        rating_2 = sum(other_movies['rating'].values ** 2)
        score_2 = math.sqrt(rating_1 * rating_2)
        return score_1 / score_2

    @classmethod
    def _iif_sim_(cls):
        """
        John S. Breese等人在1998年的工作中给出，两个用于对冷门物品采取过同样的行为更能说明他们兴趣的相似度
        e.g: 两人都买了《新华字典》，不能说明兴趣相似，但如果两人都买了《数据挖掘导论》，则认为兴趣相似
        :return:
        """

        pass

    @classmethod
    def get_sim(cls, target_movies, other_movies, sim_type=1):
        """
        提供相似性函数求
        :param target_movies: 当前用户列表
        :param other_movies: 其他用户列表
        :param sim_type: 相似函数类型
        :return:
        """
        if sim_type == 1:
            return cls._cosine_sim(target_movies, other_movies)
        else:
            return cls._cosine_sim_score(target_movies, other_movies)


class UserCF(object):
    def __init__(self, data):
        self.data = data
        # self.file_path = path
        # self._init_data()

    def _get_top_n_users(self, target_user_id, user_n):
        """
        计算目标用户与其他用户的相似性 基于评分
        :param target_user_id:
        :param user_n:
        :return:
        """
        target_movies = self.data[self.data['userId'] == target_user_id][
            ['movieId', 'rating']]
        other_users_id = set(self.data['userId'].unique()) - set(
            [target_user_id])
        # 二维矩阵，每一维包含当前用户看过的电影id
        other_movies = [
            self.data[self.data['userId'] == i][['movieId', 'rating']] for i in
            other_users_id]
        sim_list = [Similar.get_sim(target_movies, movies, self.type) for movies
                    in other_movies]
        sim_list = sorted(zip(other_users_id, sim_list), key=lambda x: x[1],
                          reverse=True)
        return sim_list[:user_n]

    def _get_candidates_items(self, target_user_id):
        """
        从源数据中找到与目标用户没有看过的所有电影
        :param target_user_id:
        :return:
        """
        target_user_movies = set(
            self.data[self.data['userId'] == target_user_id]['movieId'])
        candidates_movies = set(
            self.data['movieId'].unique()) - target_user_movies
        return candidates_movies

    def _get_top_m_items(self, top_n_users, candidates_movies, item_n):
        """
        计算候选movies中top n感兴趣的电影
        :param top_n_users:
        :param candidates_movies:
        :param item_n:
        :return:
        """
        top_n_user_data = [self.data[self.data['userId'] == k] for k, _ in
                           top_n_users]
        interest_list = []
        for movie_id in candidates_movies:
            temp = []
            for user_data in top_n_user_data:
                if movie_id in user_data['movieId'].values:
                    temp.append(user_data[user_data['movieId'] == movie_id][
                                    'rating'].values[0] / 5)
                else:
                    temp.append(0)
            interest = sum(
                [top_n_users[i][1] * temp[i] for i in range(len(top_n_users))])
            interest_list.append((movie_id, interest))
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:item_n]

    def calculate(self, target_user_id=1, user_n=20, item_n=10, type=2):
        """
        用userCF来做推荐
        :param target_user_id: 对目标用户进行推荐
        :param user_n: 找到最相似的20个用户
        :param item_n: 推荐Top item_n个
        :param type: 设置计算类型 1：只区别电影看过和没看过，忽视评分 2：电影看过的有评分
        :return:
        """
        # 设置类型
        self.type = type
        # 最相似的top N用户
        top_n_users = self._get_top_n_users(target_user_id, user_n)
        # 推荐系统的候选movies
        candidates_movies = self._get_candidates_items(target_user_id)
        # 最感兴趣的top M电影
        top_m_items = self._get_top_m_items(top_n_users, candidates_movies,
                                            item_n)
        return top_m_items


if __name__ == "__main__":
    start = time()
    file_path = './data/ml-latest-small/ratings.csv'
    data = pd.read_csv(file_path)
    user_cf = UserCF(data=data)
    # type = 1 只区别电影看过和没看过，忽视评分
    # type = 2 电影看过的有评分
    # type = 3
    print(user_cf.calculate(type=1))
    print('total time is:', time() - start)
