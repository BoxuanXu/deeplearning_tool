#--coding:utf-8--
#================================================================
#   Copyright (C) 2018 Seetatech. All rights reserved.
#   
#   文件名称：roc.py
#   创 建 者： seetatech & xuboxuan
#   创建日期：2018年11月02日
#   描    述：
#
#================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import cross_validation

iris = datasets.load_iris()
X = iris.data
y = iris.target

X, y = X[y != 2], y[y != 2]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3,random_state=0)

# Learn to predict each class against the other
svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)

###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = svm.fit(X_train, y_train).decision_function(X_test)
print(y_test)
print(y_score)
