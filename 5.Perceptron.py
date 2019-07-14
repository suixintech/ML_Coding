#!/usr/bin/python3
#-*-coding:utf-8 -*-
# __Author__  : 随心
# __Time__    : 2019/7/13 09:30 PM
# __File__    : 感知机.py
# __Software__: PyCharm


'''
    感知机注意事项：
    1、感知机是一种二分类的线性模型；
    2、前提假设：假设样本线性可分；
    3、是神经网格与支持向量机的基础；
    4、损失函数是使用误分类点到超平面的距离；
    5、属于判别模型；
    6、本文使用梯度下降与粒子群算法优化实现；
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


    
#PSO求解感知机
class PerceptronPso:
    def __init__(self):
        self.train, self.y_train, _ = getTest()
        self.y_train[self.y_train == 0] = -1
        self.c1,self.c2=2,2
        self.r1,self.r2=0.6,0.5
        self.pN=2
        self.dim=self.train.shape[1]+1
        self.X=np.zeros((self.pN,self.dim))
        self.V=np.zeros((self.pN,self.dim))
        self.pbest=np.zeros((self.pN,self.dim))
        self.gbest=np.zeros((1,self.dim))
        self.p_fit=np.zeros(self.pN)
        self.fit=1e10
        self.maxiter = 1000
        self.w = 0.8

    def costFunction(self,w,x, y):
        return -((x @ w) * y).sum()

    def initTheta(self,w,x,y):
        x = np.insert(x, 0, 1, axis=1)
        wValue=np.sqrt((np.power(w,2)).sum())
        result=(y*(x@w)/wValue)
        errSample=np.where(result<=0)
        return x[errSample],y[errSample]


    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j]=np.random.uniform(0,1)
                self.V[i][j]=np.random.uniform(0,1)
            self.pbest[i]=self.X[i]
            self.train_data,self.train_label=self.initTheta(self.X[i],self.train,self.y_train)
            tmp=self.costFunction(self.X[i],self.train_data,self.train_label)
            self.p_fit[i]=tmp
            if (tmp<self.fit):
                self.fit=tmp
                self.gbest=self.X[i]

    def modelfit(self):
        self.init_Population()
        fitness=[]
        for t in range(self.maxiter):
            for i in range(self.pN):
                self.train_data, self.train_label = self.initTheta(self.X[i], self.train, self.y_train)
                tmp = self.costFunction(self.X[i], self.train_data, self.train_label)
                if (tmp<self.p_fit[i]):
                    self.p_fit[i]=tmp
                    self.pbest[i]=self.X[i]
                    if (self.p_fit[i]<self.fit):
                        self.gbest=self.X[i]
                        self.fit=self.p_fit[i]
            for k in range(self.pN):
                self.V[k]=self.w*self.V[k]+self.c1*self.r1*(self.pbest[k]-self.X[k])+self.c2*self.r2*(self.gbest-self.X[k])
                self.X[k]=self.X[k]+self.V[k]
            fitness.append(self.fit)
        return fitness,self.gbest

    def sign(self,x):
        return 1 if x>=0 else -1


    def predict(self,x):
        _,w=self.modelfit()
        return self.sign(np.insert(x,0,1)@w)


if __name__=="__main__":
    _,_,test=getTest()
    t=Perceptron()
    t.fit()
    print(t.predict(test))
