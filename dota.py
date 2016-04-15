# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 23:40:52 2016

@author: Vladimir
"""
#%%
#импортируем данные из выборок
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
train = pd.read_csv('features.csv')
test = pd.read_csv('features_test.csv')
columns_for_train = test.columns
#убираем колонкb которых нет в test
columns_for_train = columns_for_train.insert(0,'radiant_win')
train = train[columns_for_train]
#%%
#считаем колличество переменных
feature_count = train.apply(lambda x: x.count(), axis = 0)
feature_count[feature_count < train.shape[0]]
#%%
#удаляем нулевые значения
train = train.fillna(0)
from sklearn.cross_validation import KFold
X = train.drop('radiant_win', axis=1).as_matrix()
y = train['radiant_win'].as_matrix()
pred = []
#проводим кросс-валидацию
kf = KFold(len(X), n_folds=5, shuffle = True)
#%%
#обозначаем градиентный бустинг
import time
start_time = time.time()


for train_index, test_index in kf:
    clf = GradientBoostingClassifier(n_estimators=30)
    clf.fit(X[train_index], y[train_index])
    pred.append([clf.predict_proba(X[test_index])[:,1],test_index])
print("--- %s seconds ---" % (time.time() - start_time))
#%%
from sklearn.metrics import auc
from sklearn import metrics

#обучаем модель

i = 0
vstacked = []
for train_index, test_index in kf:
    vstacked.append(np.vstack(pred[i]).T)
    i = i+1
predvstacked = pd.DataFrame(np.vstack(vstacked))
predvstacked.columns = ['pred','ind']
predvstacked = predvstacked.set_index('ind')
data = pd.concat([predvstacked,train['radiant_win']], axis = 1)
tend = datetime.now() - tstart
print (tend)

fpr, tpr, thresholds = metrics.roc_curve(data['radiant_win'],data['pred'],pos_label=1)
metrics.auc(fpr, tpr)
#%%
import matplotlib
import matplotlib.pyplot as plt
#Подбираем лучшее колличество деревье
import time
start_time = time.time()
%matplotlib inline
metric_trees = []
kf = KFold(len(X), n_folds=5, shuffle = True)
for i in [10,20,30,40,50,60,70]:
    pred = []
    j = 0
    vstacked = []
    for train_index, test_index in kf:
        clf = GradientBoostingClassifier(n_estimators=i)
        clf.fit(X[train_index], y[train_index])
        pred.append([clf.predict_proba(X[test_index])[:,1],test_index])
        vstacked.append(np.vstack(pred[j]).T)
        j = j+1
    boost_pred_vstacked = pd.DataFrame(np.vstack(vstacked))
    boost_pred_vstacked.columns = ['pred','ind']
    boost_pred_vstacked = boost_pred_vstacked.set_index('ind')
    data = pd.concat([boost_pred_vstacked,train['radiant_win']], axis = 1)

    fpr, tpr, thresholds = metrics.roc_curve(data['radiant_win'],data['pred'],pos_label=1)
    metric_trees.append([metrics.auc(fpr, tpr),i])
res = pd.DataFrame(metric_trees)
print("--- %s seconds ---" % (time.time() - start_time))
res.sort(0, ascending=False).head(3)
#%%
#Второй подход через логистическую регрессию
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
logit_pred = []
kf = KFold(len(X), n_folds=5, shuffle = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
#%%
#подбираем наилучший С с шагом 0.1
metric_C = []
start_time = time.time()
penalty_list = np.linspace(0,10,101)[1:]
for i in penalty_list:
    logit_pred = []
    j = 0
    vstacked = []
    for train_index, test_index in kf:
        clf = LogisticRegression(penalty = 'l2', C = i)
        clf.fit(X[train_index], y[train_index])
        logit_pred.append([clf.predict_proba(X[test_index])[:,1],test_index])
        vstacked.append(np.vstack(logit_pred[j]).T)
        j = j+1
    logit_predvstacked = pd.DataFrame(np.vstack(vstacked))
    logit_predvstacked.columns = ['logit_pred','ind']
    logit_predvstacked = logit_predvstacked.set_index('ind')
    data = pd.concat([logit_predvstacked,train['radiant_win']], axis = 1)
    fpr, tpr, thresholds = metrics.roc_curve(data['radiant_win'],data['logit_pred'],pos_label=1)
    metric_C.append([metrics.auc(fpr, tpr),i])
res = pd.DataFrame(metric_C)
print("--- %s seconds ---" % (time.time() - start_time))
res[res[0]==res[0].max()]
#%%
X = train.drop(['radiant_win','match_id','lobby_type',
               'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
               'd1_hero','d2_hero','d3_hero','d4_hero','d5_hero'], axis=1).as_matrix()
y = train['radiant_win'].as_matrix()
start_time = time.time()
kf = KFold(len(X), n_folds=5, shuffle = True)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
logit = []
for train_index, test_index in kf:
    clf = LogisticRegression(penalty = 'l2', C = 0.1)
    clf.fit(X[train_index], y[train_index])
    logit.append([clf.predict_proba(X[test_index])[:,1],test_index])
i = 0
vstacked = []
for train_index, test_index in kf:
    vstacked.append(np.vstack(logit[i]).T)
    i = i+1
log_reg_vstacked = pd.DataFrame(np.vstack(vstacked))
log_reg_vstacked.columns = ['logit','ind']
log_reg_vstacked = log_reg_vstacked.set_index('ind')
data = pd.concat([log_reg_vstacked,train['radiant_win']], axis = 1)
print("--- %s seconds ---" % (time.time() - start_time))
fpr, tpr, thresholds = metrics.roc_curve(data['radiant_win'],data['logit'],pos_label=1)
metrics.auc(fpr, tpr)
#%%
train['r1_hero'].unique().shape
#%%
X = train
hero_c = [c for c in X.columns if 'hero' in c]
all_heroes_id = np.unique(X[hero_c])
wb = {}
for id in all_heroes_id:
    # Мы используем + 0 для автоматического приведения bool-->int.
    r = [(X['r%d_hero' % n] == id) + 0 for n in range(1, 6)]
    d = [(X['d%d_hero' % n] == id) + 0 for n in range(1, 6)]
    wb['hero%s' % id] = sum(r) - sum(d)
X_pick = X.assign(**wb)
#%%
X = X_pick.drop(['radiant_win','match_id','lobby_type','r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
           'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
# X = pd.concat([pd.DataFrame(X),dummy], axis=1).as_matrix()
y = train['radiant_win'].as_matrix()
kf = KFold(len(X), n_folds=5, shuffle = True)


logit_pred = []
for train_index, test_index in kf:
    clf = LogisticRegression(penalty = 'l2', C = 0.1)
    clf.fit(X[train_index], y[train_index])
    logit_pred.append([clf.predict_proba(X[test_index])[:,1],test_index])
i = 0
vstacked = []
for train_index, test_index in kf:
    vstacked.append(np.vstack(logit_pred[i]).T)
    i = i+1
log_reg_vstacked = pd.DataFrame(np.vstack(vstacked))
log_reg_vstacked.columns = ['logit_pred','ind']
log_reg_vstacked = log_reg_vstacked.set_index('ind')
data = pd.concat([log_reg_vstacked,train['radiant_win']], axis = 1)

fpr, tpr, thresholds = metrics.roc_curve(data['radiant_win'],data['logit_pred'],pos_label=1)
metrics.auc(fpr, tpr)
data['log_reg_pred'].min()
data['log_reg_pred'].max()