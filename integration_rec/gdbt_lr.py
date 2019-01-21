#!/usr/bin/Python
# -*- coding: utf-8 -*-
import gc
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class GDBT_LR(object):
    def __init__(self, data_dir):
        self.train, self.target, self.test = self._preprocess(data_dir)

    def _preprocess(self, data_dir):
        df_train = pd.read_csv(data_dir + 'train.csv')
        df_test = pd.read_csv(data_dir + 'test.csv')
        df_train.drop(['Id'], axis=1, inplace=True)
        df_test.drop(['Id'], axis=1, inplace=True)
        df_test['Label'] = -1
        data = pd.concat([df_train, df_test])
        data = data.fillna(-1)
        category_fe = ['C' + str(i + 1) for i in range(0, 26)]
        for col in category_fe:
            onehot_fe = pd.get_dummies(data[col], prefix=col)
            data.drop([col], axis=1, inplace=True)
            data = pd.concat([data, onehot_fe], axis=1)
        train = data[data['Label'] != -1]
        target = train.pop('Label')
        test = data[data['Label'] == -1]
        test.drop(['Label'], axis=1, inplace=True)
        return train, target, test

    def gdbt_lr_predict(self):
        x_train, x_val, y_train, y_val = train_test_split(self.train,
                                                          self.target,
                                                          test_size=0.2,
                                                          random_state=0)
        gbm = lgb.LGBMRegressor(objective='binary',
                                subsample=0.8,
                                min_child_weight=0.5,
                                colsample_bytree=0.7,
                                num_leaves=2**7 - 1,
                                max_depth=7,
                                learning_rate=0.05,
                                n_estimators=50)
        gbm.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
                eval_names=['train', 'val'], eval_metric='binary_logloss')
        model = gbm.booster_
        print('GDBT start train and get the leaves')
        gdbt_fe_train = model.predict(self.train, pred_leaf=True)
        gdbt_fe_test = model.predict(self.test, pred_leaf=True)
        gdbt_fe_name = ['leaf_' + str(i) for i in range(gdbt_fe_train.shape[1])]
        df_train_gbdt_feats = pd.DataFrame(gdbt_fe_train,
                                           columns=gdbt_fe_name)
        df_test_gbdt_feats = pd.DataFrame(gdbt_fe_test,
                                          columns=gdbt_fe_name)
        train = pd.concat([self.train, df_train_gbdt_feats], axis=1)
        test = pd.concat([self.test, df_test_gbdt_feats], axis=1)
        train_len = train.shape[0]
        data = pd.concat([train, test])
        del train
        del test
        gc.collect()
        print('one-hot start...')
        for col in gdbt_fe_name:
            print('this is feature:', col)
            onehot_feats = pd.get_dummies(data[col], prefix=col)
            data.drop([col], axis=1, inplace=True)
            data = pd.concat([data, onehot_feats], axis=1)
        print('one-hot end...')

        train = data[: train_len]
        test = data[train_len:]
        del data
        gc.collect()
        x_train, x_val, y_train, y_val = train_test_split(train, self.target,
                                                          test_size=0.2,
                                                          random_state=0)
        print('LR start')
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        tr_logloss = log_loss(y_train, lr.predict_proba(x_train)[:, 1])
        val_logloss = log_loss(y_val, lr.predict_proba(x_val)[:, 1])
        y_pred = lr.predict_proba(test)[:, 1]
        print('tr-logloss: ', tr_logloss)
        print('val-logloss: ', val_logloss)


if __name__ == '__main__':
    data_dir = 'data/'
    gdbt_lr = GDBT_LR(data_dir)
    gdbt_lr.gdbt_lr_predict()
