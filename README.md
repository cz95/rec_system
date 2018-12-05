# movie_recommend
电影推荐系统，内含主流模型，不定期更新模型

## 数据集地址：
- 数据集（小）：http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- 数据集（大）：http://files.grouplens.org/datasets/movielens/ml-latest.zip

## 评价指标(indicators)
- 精确率：推荐正确item集合 / 所有推荐item集合
- 召回率：推荐正确item集合 / 用户在测试集上看的item集合
- 覆盖率：最终推荐的item / 所有item
- 新颖性：推荐物品越热门，新颖性越低

## 模型介绍(model)
### 完成
- usercf.py：基于用户的协同过滤来做推荐，利用余弦相似性
- itemcf.py：有了两种计算方法，利用余弦相似性，一种是分别求每个物品的最相似Top m的电影，然后基于此推荐Top n的电影；另一种是求矩阵直接求出所有物品间的相似性。后者速度更快一点，给单个用户推荐差不多十倍的差距，多用户推荐差距更大。
- lfm.py：利用梯度下降求解rsvd并保存，速度非常慢，可以通过调整feauture的个数和迭代次数来控制时间。
- prank.py：利用Peasonal Rank算法求解二元图，速度挺快的。
### 未完成
- usercf中改善用户相似性/物品相似性
- 完成itemcf与指标的接口
- itemcf中完成物品相似度归一化
- 完成itemcf中的IUF
