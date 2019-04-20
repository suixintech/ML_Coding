#!/usr/bin/python3
#-*-coding:utf-8 -*-
# __Author__  : 随心
# __File__    : LogisticRegression.py
# __Software__: PyCharm


'''
    逻辑回归的实现及注意事项:
    1、要求label服从伯努利分布,属于广义线性模型中的一种
    2、使用多种优化方法实现逻辑回归(二分类 and 多分类)，使用以下数据集
    def getData():
        from sklearn.datasets import make_classification
        X1, y1 = make_classification(n_samples=200,n_features=10,n_classes=2,n_clusters_per_class=1,random_state=0)
        X2, y2 = make_classification(n_samples=200, n_features=10, n_classes=3, n_clusters_per_class=1, random_state=0)
        return X1,y1,X2,y2
'''
