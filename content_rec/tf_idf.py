#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import jieba
import jieba.posseg as pseg
from math import log
from operator import itemgetter


class IDF(object):
    idf_dir = './data/idf.dict'

    @classmethod
    def _process(cls, corpus_dir):
        """
        计算idf的总负责
        :param corpus_dir:
        :return:
        """
        cls.num = 0  # 文章个数
        cls.article = {}  # 文章 存储该文章下所有分词的集合
        cls.idf = {}  # idf
        cls.word_set = set()
        cls._calculate_idf(corpus_dir)
        cls.save()

    @classmethod
    def _calculate_idf(cls, corpus_dir):
        """
        计算idf
        :param corpus_dir:
        :return:
        """
        content = open(corpus_dir, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            cls.num += 1
            words = set(jieba.cut(line))
            cls.article[cls.num] = words
            cls.word_set = cls.word_set | words
        for word in cls.word_set:
            n = 1.0
            for value in cls.article.values():
                if word in value:
                    n += 1.0
            cls.idf[word] = log(cls.num / n)

    @classmethod
    def save(cls):
        """
        保存idf模型
        :return:
        """
        with open(cls.idf_dir, 'wb') as f:
            pickle.dump(cls.idf, f)

    @classmethod
    def load(cls, corpus_dir):
        """
        对外接口，调用加载idf模型
        :param corpus_dir: 需要学习的语料库地址
        :return:
        """
        if not os.path.exists('./data/'):
            os.mkdir('./data/')
        if not os.path.exists(cls.idf_dir):
            cls._process(corpus_dir)
        f = open(cls.idf_dir, 'rb')
        idf = pickle.load(f)
        f.close()
        return idf


class TF_IDF(object):
    """
    tf_idf算法实现，需要停用词和语料库，语料库每一行为一篇文章。
    """

    # 允许的词性
    allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt',
                         'nz', 't', 'v', 'vd', 'vn', 'eng']

    def __init__(self, corpus_dir):
        self.stop_words = set()
        self._pro_stop_words()
        self.idf = IDF.load(corpus_dir)
        self.new_idf = np.mean(list(self.idf.values()))  # 对与新词 我们用均值

    def _pro_stop_words(self):
        """
        加入停用词
        :return:
        """
        stop_dir = os.path.join('./data/stop_words')
        content = open(stop_dir, 'rb').read().decode('utf-8')
        for line in content.splitlines():
            self.stop_words.add(line)

    def get_tf_idf(self, data, top_k):
        """
        计算tf_idf并获取topk关键词
        :param data:
        :param top_k:
        :return:
        """
        words = pseg.cut(data)
        # 过滤不要的词性
        words_filter = [w for w in words if w.flag in self.allow_speech_tags]
        tf = {}
        tf_idf = {}
        word_num = 0
        # 过滤停用词并计算tf
        for w, flag in words_filter:
            if len(w.strip()) < 2 or w.lower() in self.stop_words:
                continue
            word_num += 1
            tf[w] = tf.get(w, 0.0) + 1.0
        # 计算tf-idf
        for key in tf.keys():
            idf = self.new_idf
            if key in self.idf.keys():
                idf = self.idf[key]
            tf_idf[key] = tf_idf.get(key, 0.0) + tf[key] * idf
        tags = sorted(tf_idf.items(), key=itemgetter(1), reverse=True)
        return tags[:top_k]


if __name__ == '__main__':
    corpus_dir = './data/corpus_xueqiu.txt'  # 语料库，每一行表示一篇文章
    idf_test = './data/idf_test.txt'  # 测试文章
    text = open(idf_test, 'rb').read().decode('utf-8')
    a = TF_IDF(corpus_dir)
    print(a.get_tf_idf(text, 10))
