#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import jieba
import pickle
import numpy as np
from math import log
from operator import itemgetter


class IDF(object):
    idf_dir = './data/idf.dict'

    @classmethod
    def _process(cls, corpus_dir):
        cls.num = 0  # 文章个数
        cls.article = {}  # 文章 存储该文章下所有分词的集合
        cls.idf = {}  # idf
        cls.word_set = set()
        cls._calculate_idf(corpus_dir)
        cls.save()

    @classmethod
    def _calculate_idf(cls, corpus_dir):
        """
        加语料库
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
        f = open(cls.idf_dir, 'wb')
        pickle.dump(cls.idf, f)
        f.close()
        pass

    @classmethod
    def load(cls, corpus_dir):
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

    def __init__(self, corpus_dir):
        self.stop_words = set()
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
        # tf
        words = jieba.cut(data)
        tf = {}
        tf_idf = {}
        word_num = 0
        for w in words:
            if len(w.strip()) < 2 or w.lower() in self.stop_words:
                continue
            word_num += 1
            tf[w] = tf.get(w, 0.0) + 1.0
        for key in tf.keys():
            idf = self.new_idf
            if key in self.idf.keys():
                idf = self.idf[key]
            tf_idf[key] = tf_idf.get(key, 0.0) + tf[key] * idf
        tags = sorted(tf_idf.items(), key=itemgetter(1), reverse=True)
        return tags[:top_k]


if __name__ == '__main__':
    corpus_dir = './data/corpus_xueqiu.txt'
    corpus_dir = './data/corpus.txt'
    text = '搜狐体育讯 北京时间4月21日晚22点,在英超联赛第35轮的一场比赛中,曼城客场迎战联赛倒数第一的沃特福德,最终凭借瓦塞尔和普里什金分别为各自球队的进球,双方曼城客场和沃特福德1-01握手言和。上轮迎战阿森纳受伤下场的孙继海,本场比赛再次出现在首发阵容中,在孙继海复出之后参加了球队全部9轮联赛,其中8场比赛都是首发出场,在曼城主力位置不可撼动。比赛第85分钟孙继海被替换下场,还有伤病在身的孙继海显然是得到了教练的保护。 在本轮比赛之前曼城积41分,排在联赛第13位,本赛季已无降级之忧,曼城近一阶段战绩比较出色,一扫前一阶段联赛5连败的颓势,近6轮比赛取得11分,也正是这样的表现使得曼城一跃脱离降级区。反观沃特福德,本赛季表现及其糟糕,34轮联赛过后只积23分,排在倒数第一位,目前为止也只赢得了4场比赛的胜利,共进了25球,和曼城差不多,但是却失掉了56球,防守是球队主要的问题。 在英超的历史上,两队总共有23次交锋,曼城12胜6平5负占据优势,但是在沃特福德主场维卡里吉路球场,曼城并没有多少优势,11次交锋4胜4平3负,基本上打成了平手。本场比赛曼城的单箭头,能对沃特福德脆弱的防线形成多大威胁,将左右比赛的战局。中国球员孙继海再次出现在右前卫上,到底又会有怎样的发挥值得期待。 比赛开始之后,曼城表现比较积极,率先发动攻势。第一分钟,马洪对巴顿犯规,曼城获得任意球,哈曼将球打进禁区,被回防的马洪解围。第4分钟曼城再次获得前场任意球,哈曼再次将球发到禁区内,门将福斯特将球得到。第5分钟沃特福德获得角球机会,曼城门将伊萨克森将球解围。第10分钟,沃特福德获得射门机会,在禁区左肋12码处头球攻门,偏出。 曼城今天在防守中多采用造越位战术,沃特福德多次掉进越位陷阱。第19分钟,曼城再次获得前场任意球机会,哈曼第三次操刀,直接攻门被门将福斯特飞身救险。第24分钟,马龙·金再次越位,开场之后沃特福德已经三次越位。第24分钟,沃特福德后卫对巴顿犯规,曼城再次获得前场任意球,哈曼将球吊到禁区内,被后卫卡莱尔解围。第29分钟,瓦塞尔在进攻中和队员传跑没有形成默契,越位失去一次破门良机。第37分钟,瓦塞尔在前场犯规,福斯特将球直接吊到曼城禁区内,邓恩及时解围。 第38分钟,此前一直表现平淡的孙继海,在前场右路防守时手球犯规,沃特福德获得任意球机会,孙继海今天在右路功放表现一般,也许是受到了伤病的影响,而曼城在之前的比赛中也是一次射正都没有,今天曼城整体表现都不是很理想。第38分钟,沃特福德再次获得射门机会,布阿扎20码处的射门被伊萨克森得到。第42分钟,曼城获得前场任意球机会,哈曼将球打到禁区被后卫解围。 在上半场最后阶段曼城发起了一波进攻高潮,第44分钟,巴顿右路的角球被沃特福德后卫解围。第45分钟,孙继海在前场离球门25码处获得射门机会,大力低射从球门右边偏出。最后,上半场双方战成0-0. 易边再战,双方都没有对阵容作出调整。第48分钟,沃特福德获得角球机会,班古拉将球罚向禁区中路,哈曼将球解围。第53分钟沃特福德再次角球进攻,曼城回防的米勒将球解围。 第53分钟曼城终于迎来了首粒进球,瓦塞尔在在前场右侧离球门25米出起脚打门,球从右侧攻破了越过了福斯特的封堵,曼城客场1-0领先,曼城本场比赛第一脚打在门框范围内的射门,就得到一分,瓦塞尔也打进了本赛季个人第三粒进球。 进球之后曼城打得更加得心应手,逐渐开始控制场上局面,而沃特福德也没能组织起有效的攻势,场面开始变得异常沉闷。第55分钟,米勒对里纳尔迪犯规,沃特福德获得任意球,邓恩将球解围。第60分钟,哈曼对沃特福德班古拉恶意犯规,得到一张黄牌。 第75分钟沃特福德将比分扳平,下半场替补上场的普里什金建功,在禁区中路离球门12码处射门得手,场上比分1-1,双方再次回到同一起跑线。在另外一个赛场上,查尔顿在保级大战中,也被谢菲联顽强逼平,形势非常危急。第77分钟,巴顿在离球门12码处射门,被后卫挡住,之后巴顿在禁区线上传球,被班古拉解围。 第85分钟,曼城做出换人,孙继海被爱尔兰替换下场,这也是孙继海在近7场联赛中,首次因为战术调整而被替换下场,上次战术调整下场还是3月15日与切尔西的补赛中。 第87分钟,沃特福德再得破门良机,门前12码处射门从左侧偏出。第90分钟,萨马拉斯在进攻中越位,错失了最后一次得分机会。最终双方以1-1握手言和。 (风清扬) 沃特福德(433) 门将:26-福斯特 后卫:12-多伊利 、5-卡莱尔 、6-德梅里特 、11-鲍威尔 中场:20-班古拉 、7-弗朗西斯(24’15-卡巴) 、8-马洪 前锋:31-里纳尔迪(82’17-希图)、18-布阿扎、9-马龙·金 替补:16-理查德·李、23-马里亚帕 、17-希图 、15-卡巴、24-普里什金 曼城(451) 门将:1-伊萨克森 后卫:3-波尔、15-迪斯丁、22-邓恩、16-奥姆哈 中场:17-孙继海(85’7-爱尔兰)、33-约翰逊、8-巴顿、21-哈曼、24-比斯利(35’43-米勒) 前锋:11-瓦塞尔 替补:12-韦弗、7-爱尔兰、30-科拉迪、43-米勒、20-萨马拉斯 (责任编辑:Gedicht)'
    a = TF_IDF(corpus_dir)
    print(a.get_tf_idf(text, 10))
