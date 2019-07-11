#!/usr/bin/python3
#-*-coding:utf-8 -*-
# __Author__  : 随心
# __Time__    : 2019/7/11 09:30 PM
# __File__    : 感知机.py
# __Software__: PyCharm


'''
    感知机注意事项：
    1、错误样本更新；
    2、计算到超平面距离；
'''
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def getData():
    from sklearn.datasets import make_classification
    X1, y1 = make_classification(n_samples=200, n_features=10, n_classes=2, n_clusters_per_class=1, random_state=0)
    X2, y2 = make_classification(n_samples=200, n_features=10, n_classes=3, n_clusters_per_class=1, random_state=0)
    return X1, y1, X2, y2


def getTest():
    train=np.array([[1,1,1,2,3],[10,11,12,10,11],[1,2,1,2,3],[11,11,12,10,11],[1,1,3,2,3],[10,12,12,10,11],[100,200,300,500,600]])
    y_train=np.array([0,1,0,1,0,1,1])
    test = np.array([101, 202, 303, 503, 605])
    return train,y_train,test

class Perceptron:
    def __init__(self):
        # self.train,self.y_train,_,_=getData()
        self.train,self.y_train,_=getTest()
        self.y_train[self.y_train==0]=-1
        self.w=np.random.uniform(0,1,size=(self.train.shape[1]))
        self.b=0.1
        self.learning_rate=0.01
        self.maxiter=2000

    def costFunction(self,w,b,x,y):
        return -((w@x.T+b)*y).sum()


    def initTheta(self,w,b,x,y):
        wValue=np.sqrt((np.power(w,2)+np.power(b,2)).sum())
        result=(y*(w@x.T+b)/wValue)
        errSample=np.where(result<=0)
        return x[errSample],y[errSample]

    def wUpdate(self,x,y):
        w=self.learning_rate*x*y.sum()
        return w


    def bUpdate(self,y):
        b=self.learning_rate*y.sum()
        return b

    def fit(self):
        for i in range(self.maxiter):
            errtrain,errlabel=self.initTheta(self.w,self.b,self.train,self.y_train)
            initCost=self.costFunction(self.w, self.b, errtrain, errlabel)
            if errtrain.shape[0]==0:
                break
            self.w+=self.wUpdate(errtrain[0],errlabel[0])
            self.b+=self.bUpdate(errlabel[0])
            costValue=self.costFunction(self.w, self.b, errtrain, errlabel)
            if np.abs(costValue-initCost)<=1e-5:
                break
        return self.w,self.b

    def sign(self,val):
        return -1 if val<0 else 1

    def predict(self,x):
        w,b=self.fit()
        predictValue=self.sign(w@x.T+b)
        return predictValue




if __name__=="__main__":
    _,_,test=getTest()
    t=Perceptron()
    t.fit()
    print(t.predict(test))
